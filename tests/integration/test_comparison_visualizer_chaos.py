"""Integration test for comparison visualizations using a CHAOS CT slice."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.data.dicom_reader import DICOMReader
from src.data.ground_truth_reader import GroundTruthReader
from src.visualization.comparison_visualizer import ComparisonVisualizer


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


class ChaosSample(TypedDict):
    """Data used to render a real CHAOS qualitative comparison."""

    image: np.ndarray
    ground_truth: np.ndarray
    predictions: dict[str, np.ndarray]


@pytest.fixture
def chaos_sample() -> ChaosSample:
    """Load a matched CHAOS CT slice and derive deterministic predictions."""
    if not DICOM_PATH.exists() or not MASK_PATH.exists():
        pytest.skip("The required CHAOS CT image and ground-truth mask are unavailable.")

    image = DICOMReader().read_image(DICOM_PATH)
    ground_truth = GroundTruthReader().read_mask(MASK_PATH)
    if not np.any(ground_truth):
        pytest.skip("The selected CHAOS ground-truth mask has no foreground pixels.")

    shifted_prediction = np.roll(ground_truth, shift=3, axis=0)
    trimmed_prediction = ground_truth.copy()
    foreground_coordinates = np.argwhere(ground_truth != 0)
    center_y, center_x = foreground_coordinates[len(foreground_coordinates) // 2]
    trimmed_prediction[
        max(0, center_y - 4): center_y + 5,
        max(0, center_x - 4): center_x + 5,
    ] = 0

    return {
        "image": image,
        "ground_truth": ground_truth,
        "predictions": {
            "Shifted Mask": shifted_prediction,
            "Trimmed Mask": trimmed_prediction,
        },
    }


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close figures created by the integration test."""
    yield
    plt.close("all")


@pytest.fixture
def visualizer() -> ComparisonVisualizer:
    """Provide a comparison visualizer with default settings."""
    return ComparisonVisualizer()


def _save_debug_output(figure: Figure, filename: str) -> None:
    """Optionally save a figure for manual visual inspection."""
    if not DEBUG_OUTPUT:
        return

    debug_dir = PROJECT_ROOT / "tests" / "output" / "comparison"
    debug_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        debug_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )


def test_comparison_visualizer_with_real_chaos_sample(
    chaos_sample: ChaosSample,
    visualizer: ComparisonVisualizer,
    tmp_path: Path,
) -> None:
    """Render and save all qualitative comparisons for a real CHAOS sample."""
    image = chaos_sample["image"]
    ground_truth = chaos_sample["ground_truth"]
    predictions = chaos_sample["predictions"]

    assert image.shape == ground_truth.shape
    assert all(
        prediction.shape == ground_truth.shape
        for prediction in predictions.values()
    )

    prediction_figure = visualizer.compare_predictions(
        image,
        ground_truth,
        predictions,
    )
    model_figure = visualizer.compare_models(image, ground_truth, predictions)
    overlay_figure = visualizer.compare_overlays(image, ground_truth, predictions)
    difference_figure = visualizer.compare_difference(ground_truth, predictions)

    assert isinstance(prediction_figure, Figure)
    assert isinstance(model_figure, Figure)
    assert isinstance(overlay_figure, Figure)
    assert isinstance(difference_figure, Figure)
    assert len(prediction_figure.axes) == len(predictions) + 2
    assert len(model_figure.axes) == len(predictions) + 2
    assert len(overlay_figure.axes) == len(predictions) + 1
    assert len(difference_figure.axes) == len(predictions)

    output_path = tmp_path / "comparison" / "chaos_comparison.png"
    figure_number = prediction_figure.number
    _save_debug_output(prediction_figure, "chaos_comparison.png")
    visualizer.save(prediction_figure, str(output_path))

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert not plt.fignum_exists(figure_number)
