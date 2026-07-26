"""Unit tests for three-dimensional medical image volume visualizations."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.core.volume_plane import VolumePlane
from src.visualization.volume_visualizer import VolumeVisualizer


@pytest.fixture
def sample_volume() -> np.ndarray:
    """Provide a deterministic three-dimensional image volume."""
    return np.arange(16 * 32 * 32, dtype=np.float32).reshape(16, 32, 32)


@pytest.fixture
def sample_mask(sample_volume: np.ndarray) -> np.ndarray:
    """Provide a deterministic binary segmentation mask."""
    return sample_volume % 2 == 0


@pytest.fixture
def sample_prediction(sample_mask: np.ndarray) -> np.ndarray:
    """Provide a deterministic segmentation prediction."""
    return np.roll(sample_mask, shift=2, axis=2)


@pytest.fixture
def visualizer() -> VolumeVisualizer:
    """Provide a volume visualizer with default settings."""
    return VolumeVisualizer()


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close Matplotlib figures created during a test."""
    yield
    plt.close("all")


def test_default_constructor_uses_expected_settings() -> None:
    """The default constructor stores the documented figure settings."""
    visualizer = VolumeVisualizer()

    assert visualizer._figsize == (6.0, 6.0)
    assert visualizer._dpi == 100


def test_constructor_accepts_custom_figsize() -> None:
    """The constructor accepts custom per-panel dimensions."""
    visualizer = VolumeVisualizer(figsize=(4.0, 3.0))

    assert visualizer._figsize == (4.0, 3.0)


def test_constructor_accepts_custom_dpi() -> None:
    """The constructor accepts a custom figure resolution."""
    visualizer = VolumeVisualizer(dpi=200)

    assert visualizer._dpi == 200


@pytest.mark.parametrize("dpi", [0, -100])
def test_constructor_rejects_non_positive_dpi(dpi: int) -> None:
    """Non-positive constructor resolutions are rejected."""
    with pytest.raises(ValueError):
        VolumeVisualizer(dpi=dpi)


@pytest.mark.parametrize(
    "figsize",
    [
        [6.0, 6.0],
        (6.0,),
    ],
)
def test_constructor_rejects_invalid_figsize_structure(
    figsize: object,
) -> None:
    """Figure sizes must be two-item tuples."""
    with pytest.raises(TypeError):
        VolumeVisualizer(figsize=figsize)  # type: ignore[arg-type]


@pytest.mark.parametrize("figsize", [(0.0, 6.0), (6.0, -1.0)])
def test_constructor_rejects_non_positive_figsize(
    figsize: tuple[float, float],
) -> None:
    """Figure dimensions must be positive."""
    with pytest.raises(ValueError):
        VolumeVisualizer(figsize=figsize)


@pytest.mark.parametrize(
    ("plane", "slice_index", "title"),
    [
        (VolumePlane.AXIAL, 4, "Axial Slice"),
        (VolumePlane.CORONAL, 12, "Coronal Slice"),
        (VolumePlane.SAGITTAL, 20, "Sagittal Slice"),
    ],
)
def test_render_slice_returns_titled_figure_for_each_plane(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
    plane: VolumePlane,
    slice_index: int,
    title: str,
) -> None:
    """Each anatomical plane renders one correctly titled slice panel."""
    figure = visualizer.render_slice(sample_volume, slice_index, plane)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert figure.axes[0].get_title() == title


def test_render_mask_returns_titled_two_panel_figure(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
    sample_mask: np.ndarray,
) -> None:
    """Mask rendering creates original-slice and mask panels."""
    figure = visualizer.render_mask(sample_volume, sample_mask, 4)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == ["Original Slice", "Mask"]


def test_render_prediction_returns_titled_two_panel_figure(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
    sample_prediction: np.ndarray,
) -> None:
    """Prediction rendering creates original-slice and prediction panels."""
    figure = visualizer.render_prediction(sample_volume, sample_prediction, 4)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == [
        "Original Slice",
        "Prediction",
    ]


def test_render_overlay_returns_titled_two_panel_figure(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
    sample_prediction: np.ndarray,
) -> None:
    """Overlay rendering creates original-slice and overlay panels."""
    figure = visualizer.render_overlay(sample_volume, sample_prediction, 4)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == [
        "Original Slice",
        "Prediction Overlay",
    ]


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_render_overlay_accepts_boundary_alpha_values(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
    sample_prediction: np.ndarray,
    alpha: float,
) -> None:
    """Overlay rendering accepts both inclusive alpha boundaries."""
    figure = visualizer.render_overlay(
        sample_volume,
        sample_prediction,
        4,
        alpha=alpha,
    )

    assert isinstance(figure, Figure)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_render_overlay_rejects_out_of_range_alpha(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
    sample_prediction: np.ndarray,
    alpha: float,
) -> None:
    """Overlay alpha values outside the inclusive range are rejected."""
    with pytest.raises(ValueError):
        visualizer.render_overlay(sample_volume, sample_prediction, 4, alpha=alpha)


def test_render_overlay_rejects_non_numeric_alpha(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
    sample_prediction: np.ndarray,
) -> None:
    """Non-numeric overlay alpha values are rejected."""
    with pytest.raises(TypeError):
        visualizer.render_overlay(
            sample_volume,
            sample_prediction,
            4,
            alpha="half",  # type: ignore[arg-type]
        )


def test_render_multiplanar_returns_titled_three_panel_figure(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
) -> None:
    """Multiplanar rendering creates axial, coronal, and sagittal panels."""
    figure = visualizer.render_multiplanar(sample_volume)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 3
    assert [axis.get_title() for axis in figure.axes] == [
        "Axial",
        "Coronal",
        "Sagittal",
    ]


def test_render_multiplanar_uses_center_slices_by_default(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
) -> None:
    """Multiplanar rendering selects each plane's central slice by default."""
    figure = visualizer.render_multiplanar(sample_volume)
    displayed_slices = [axis.get_images()[0].get_array() for axis in figure.axes]

    assert np.array_equal(displayed_slices[0], sample_volume[8, :, :])
    assert np.array_equal(displayed_slices[1], np.flipud(sample_volume[:, 16, :]))
    assert np.array_equal(displayed_slices[2], np.flipud(sample_volume[:, :, 16]))


def test_render_multiplanar_uses_custom_indices(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
) -> None:
    """Multiplanar rendering uses supplied slice indices for each plane."""
    figure = visualizer.render_multiplanar(
        sample_volume,
        axial_index=2,
        coronal_index=4,
        sagittal_index=6,
    )
    displayed_slices = [axis.get_images()[0].get_array() for axis in figure.axes]

    assert np.array_equal(displayed_slices[0], sample_volume[2, :, :])
    assert np.array_equal(displayed_slices[1], np.flipud(sample_volume[:, 4, :]))
    assert np.array_equal(displayed_slices[2], np.flipud(sample_volume[:, :, 6]))


def test_save_creates_output_file_and_closes_figure(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
    tmp_path: Path,
) -> None:
    """Saving creates parent directories, writes the image, and closes it."""
    figure = visualizer.render_slice(sample_volume, 4)
    figure_number = figure.number
    output_path = tmp_path / "figures" / "volume.png"

    visualizer.save(figure, str(output_path))

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert not plt.fignum_exists(figure_number)


def test_render_slice_rejects_non_ndarray_volume(
    visualizer: VolumeVisualizer,
) -> None:
    """Volumes that are not NumPy arrays are rejected."""
    with pytest.raises(TypeError):
        visualizer.render_slice([[[0]]], 0)  # type: ignore[arg-type]


def test_render_mask_rejects_non_ndarray_mask(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
) -> None:
    """Masks that are not NumPy arrays are rejected."""
    with pytest.raises(TypeError):
        visualizer.render_mask(sample_volume, [[[True]]], 0)  # type: ignore[arg-type]


def test_render_prediction_rejects_non_ndarray_prediction(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
) -> None:
    """Predictions that are not NumPy arrays are rejected."""
    with pytest.raises(TypeError):
        visualizer.render_prediction(
            sample_volume,
            [[[True]]],  # type: ignore[arg-type]
            0,
        )


@pytest.mark.parametrize("shape", [(32, 32), (1, 16, 32, 32)])
def test_render_slice_rejects_non_three_dimensional_volume(
    visualizer: VolumeVisualizer,
    shape: tuple[int, ...],
) -> None:
    """Two-dimensional and four-dimensional volumes are rejected."""
    volume = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError):
        visualizer.render_slice(volume, 0)


def test_render_mask_rejects_shape_mismatch(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
) -> None:
    """Masks with a different shape from the image volume are rejected."""
    mask = np.zeros((8, 32, 32), dtype=bool)

    with pytest.raises(ValueError):
        visualizer.render_mask(sample_volume, mask, 0)


def test_render_slice_rejects_invalid_plane(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
) -> None:
    """Plane selections outside the VolumePlane enumeration are rejected."""
    with pytest.raises(TypeError):
        visualizer.render_slice(
            sample_volume,
            0,
            plane="axial",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("slice_index", [-1, 16])
def test_render_slice_rejects_out_of_range_slice_index(
    visualizer: VolumeVisualizer,
    sample_volume: np.ndarray,
    slice_index: int,
) -> None:
    """Negative and too-large slice indices are rejected."""
    with pytest.raises(ValueError):
        visualizer.render_slice(sample_volume, slice_index)
