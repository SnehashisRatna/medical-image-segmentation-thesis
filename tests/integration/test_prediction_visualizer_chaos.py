"""Integration test for prediction visualizations using a CHAOS CT slice."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.data.dicom_reader import DICOMReader
from src.data.ground_truth_reader import GroundTruthReader
from src.visualization.prediction_visualizer import PredictionVisualizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DICOM_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CHAOS_Train_Sets"
    / "Train_Sets"
    / "CT"
    / "1"
    / "DICOM_anon"
    / "i0045,0000b.dcm"
)
MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CHAOS_Train_Sets"
    / "Train_Sets"
    / "CT"
    / "1"
    / "Ground"
    / "liver_GT_045.png"
)
DEBUG_OUTPUT = os.getenv("SAVE_TEST_OUTPUTS", "0") == "1"


@pytest.fixture
def visualizer() -> PredictionVisualizer:
    """Provide a prediction visualizer with default settings."""
    return PredictionVisualizer()


@pytest.fixture
def chaos_image() -> np.ndarray:
    """Load a real CHAOS CT slice."""
    if not DICOM_PATH.exists():
        pytest.skip("The required CHAOS CT image is unavailable.")

    return DICOMReader().read_image(DICOM_PATH)


@pytest.fixture
def ground_truth() -> np.ndarray:
    """Load the liver mask corresponding to the CHAOS CT slice."""
    if not MASK_PATH.exists():
        pytest.skip("The required CHAOS liver ground-truth mask is unavailable.")

    mask = GroundTruthReader().read_mask(MASK_PATH)
    if not np.any(mask):
        pytest.skip("The selected CHAOS ground-truth mask has no foreground pixels.")

    return mask


@pytest.fixture
def prediction(ground_truth: np.ndarray) -> np.ndarray:
    """Provide a deterministic synthetic prediction from the liver mask."""
    return np.roll(ground_truth, shift=5, axis=1)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close figures created by the integration test."""
    yield
    plt.close("all")


def _save_debug_output(figure: Figure, filename: str) -> None:
    """Optionally save a figure for manual visual inspection."""
    if not DEBUG_OUTPUT:
        return

    debug_dir = PROJECT_ROOT / "tests" / "output" / "prediction"
    debug_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        debug_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )


def test_prediction_visualizer_with_real_chaos_sample(
    visualizer: PredictionVisualizer,
    chaos_image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    tmp_path: Path,
) -> None:
    """Render and save every prediction visualization for a real CHAOS sample."""
    # Arrange
    assert chaos_image.shape == ground_truth.shape
    assert prediction.shape == ground_truth.shape

    # Act
    prediction_figure = visualizer.render_prediction(chaos_image, prediction)
    ground_truth_figure = visualizer.render_ground_truth(chaos_image, ground_truth)
    overlay_figure = visualizer.render_overlay(chaos_image, prediction)
    difference_figure = visualizer.render_difference(prediction, ground_truth)

    # Assert
    assert isinstance(prediction_figure, Figure)
    assert isinstance(ground_truth_figure, Figure)
    assert isinstance(overlay_figure, Figure)
    assert isinstance(difference_figure, Figure)
    assert len(prediction_figure.axes) == 2
    assert len(ground_truth_figure.axes) == 2
    assert len(overlay_figure.axes) == 2
    assert len(difference_figure.axes) == 1
    assert [axis.get_title() for axis in prediction_figure.axes] == [
        "Original Image",
        "Prediction",
    ]
    assert [axis.get_title() for axis in ground_truth_figure.axes] == [
        "Original Image",
        "Ground Truth",
    ]
    assert [axis.get_title() for axis in overlay_figure.axes] == [
        "Original Image",
        "Prediction Overlay",
    ]
    assert difference_figure.axes[0].get_title() == "Difference Map"

    _save_debug_output(prediction_figure, "chaos_prediction.png")
    _save_debug_output(ground_truth_figure, "chaos_ground_truth.png")
    _save_debug_output(overlay_figure, "chaos_overlay.png")
    _save_debug_output(difference_figure, "chaos_difference.png")

    output_path = tmp_path / "prediction" / "chaos_prediction.png"
    figure_number = prediction_figure.number
    visualizer.save(prediction_figure, str(output_path))

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert not plt.fignum_exists(figure_number)
