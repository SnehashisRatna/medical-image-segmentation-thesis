"""Integration test for volume visualizations using a CHAOS CT volume."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import re
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from pydicom.dataset import FileDataset

from src.data.dicom_reader import DICOMReader
from src.data.ground_truth_reader import GroundTruthReader
from src.visualization.volume_visualizer import VolumeVisualizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAOS_CT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CHAOS_Train_Sets"
    / "Train_Sets"
    / "CT"
)
DEBUG_OUTPUT = os.getenv("SAVE_TEST_OUTPUTS", "0") == "1"


class ChaosVolume(TypedDict):
    """Real CHAOS CT image and liver-mask volumes for integration testing."""

    image: np.ndarray
    mask: np.ndarray


def _slice_number_from_dicom_path(dicom_path: Path) -> int | None:
    """Extract the source slice number from a CHAOS DICOM filename."""
    match = re.fullmatch(r"i0*(\d+),.*", dicom_path.name)
    return int(match.group(1)) if match is not None else None


def _slice_number_from_mask_path(mask_path: Path) -> int | None:
    """Extract the source slice number from a CHAOS liver-mask filename."""
    match = re.fullmatch(r"liver_GT_(\d+)", mask_path.stem)
    return int(match.group(1)) if match is not None else None


def _find_complete_chaos_patient() -> tuple[dict[int, Path], dict[int, Path]] | None:
    """Find one CT patient with a liver mask for every DICOM slice."""
    if not CHAOS_CT_PATH.is_dir():
        return None

    for patient_path in sorted(CHAOS_CT_PATH.iterdir()):
        dicom_directory = patient_path / "DICOM_anon"
        ground_truth_directory = patient_path / "Ground"
        if not dicom_directory.is_dir() or not ground_truth_directory.is_dir():
            continue

        dicom_paths: dict[int, Path] = {}
        for dicom_path in sorted(dicom_directory.glob("*.dcm")):
            slice_number = _slice_number_from_dicom_path(dicom_path)
            if slice_number is None or slice_number in dicom_paths:
                dicom_paths = {}
                break
            dicom_paths[slice_number] = dicom_path

        mask_paths: dict[int, Path] = {}
        for mask_path in sorted(ground_truth_directory.glob("liver_GT_*.png")):
            slice_number = _slice_number_from_mask_path(mask_path)
            if slice_number is None or slice_number in mask_paths:
                mask_paths = {}
                break
            mask_paths[slice_number] = mask_path

        if dicom_paths and dicom_paths.keys() == mask_paths.keys():
            return dicom_paths, mask_paths

    return None


def _slice_sort_key(dataset: FileDataset, slice_number: int) -> tuple[int, float, int]:
    """Return a stable physical-order key for a CT DICOM slice."""
    image_position = dataset.get("ImagePositionPatient")
    if image_position is not None and len(image_position) >= 3:
        return (0, float(image_position[2]), slice_number)

    instance_number = dataset.get("InstanceNumber")
    if instance_number is not None:
        return (1, float(instance_number), slice_number)

    return (2, float(slice_number), slice_number)


@pytest.fixture
def chaos_volume() -> ChaosVolume:
    """Load matched image and liver-mask volumes from one real CHAOS patient."""
    sample_paths = _find_complete_chaos_patient()
    if sample_paths is None:
        pytest.skip(
            f"A complete CHAOS CT image and liver-mask volume is unavailable in "
            f"{CHAOS_CT_PATH}"
        )
    dicom_paths, mask_paths = sample_paths

    dicom_reader = DICOMReader()
    sorted_slices = sorted(
        (
            _slice_sort_key(dicom_reader.read(dicom_path), slice_number),
            slice_number,
            dicom_path,
        )
        for slice_number, dicom_path in dicom_paths.items()
    )
    ordered_slice_numbers = [slice_number for _, slice_number, _ in sorted_slices]
    image_slices = [
        dicom_reader.read_image(dicom_path) for _, _, dicom_path in sorted_slices
    ]
    mask_reader = GroundTruthReader()
    mask_slices = [
        mask_reader.read_mask(mask_paths[slice_number])
        for slice_number in ordered_slice_numbers
    ]

    if not image_slices or any(
        image.shape != mask.shape for image, mask in zip(image_slices, mask_slices)
    ):
        pytest.skip("The selected CHAOS CT volume does not align with liver masks.")

    image_volume = np.stack(image_slices)
    mask_volume = np.stack(mask_slices)
    if not np.any(mask_volume):
        pytest.skip("The selected CHAOS liver-mask volume has no foreground voxels.")

    return {"image": image_volume, "mask": mask_volume}


@pytest.fixture
def visualizer() -> VolumeVisualizer:
    """Provide a volume visualizer with default settings."""
    return VolumeVisualizer()


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close Matplotlib figures created by the integration test."""
    yield
    plt.close("all")


def _save_debug_output(figure: Figure, filename: str) -> None:
    """Optionally save a figure for manual visual inspection."""
    if not DEBUG_OUTPUT:
        return

    debug_directory = PROJECT_ROOT / "tests" / "output" / "volume"
    debug_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(debug_directory / filename, dpi=300, bbox_inches="tight")


def test_volume_visualizer_with_real_chaos_volume(
    chaos_volume: ChaosVolume,
    visualizer: VolumeVisualizer,
    tmp_path: Path,
) -> None:
    """Render and save every volume visualization for a real CHAOS CT patient."""
    # Arrange
    image_volume = chaos_volume["image"]
    liver_mask_volume = chaos_volume["mask"]
    axial_index = image_volume.shape[0] // 2
    assert image_volume.shape == liver_mask_volume.shape
    assert image_volume.ndim == 3

    # Act
    slice_figure = visualizer.render_slice(image_volume, axial_index)
    mask_figure = visualizer.render_mask(
        image_volume,
        liver_mask_volume,
        axial_index,
    )
    prediction_figure = visualizer.render_prediction(
        image_volume,
        liver_mask_volume,
        axial_index,
    )
    overlay_figure = visualizer.render_overlay(
        image_volume,
        liver_mask_volume,
        axial_index,
    )
    multiplanar_figure = visualizer.render_multiplanar(image_volume)

    _save_debug_output(slice_figure, "chaos_slice.png")
    _save_debug_output(mask_figure, "chaos_mask.png")
    _save_debug_output(prediction_figure, "chaos_prediction.png")
    _save_debug_output(overlay_figure, "chaos_overlay.png")
    _save_debug_output(multiplanar_figure, "chaos_multiplanar.png")

    output_path = tmp_path / "volume" / "chaos_slice.png"
    figure_number = slice_figure.number
    visualizer.save(slice_figure, str(output_path))

    # Assert
    assert isinstance(slice_figure, Figure)
    assert isinstance(mask_figure, Figure)
    assert isinstance(prediction_figure, Figure)
    assert isinstance(overlay_figure, Figure)
    assert isinstance(multiplanar_figure, Figure)
    assert len(slice_figure.axes) == 1
    assert slice_figure.axes[0].get_title() == "Axial Slice"
    assert len(mask_figure.axes) == 2
    assert len(prediction_figure.axes) == 2
    assert len(overlay_figure.axes) == 2
    assert len(multiplanar_figure.axes) == 3
    assert [axis.get_title() for axis in multiplanar_figure.axes] == [
        "Axial",
        "Coronal",
        "Sagittal",
    ]
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert not plt.fignum_exists(figure_number)
