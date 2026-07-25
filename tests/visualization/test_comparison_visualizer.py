"""Unit tests for qualitative segmentation comparison visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.visualization.comparison_visualizer import ComparisonVisualizer


class SampleArrays(TypedDict):
    """Deterministic arrays used by comparison visualizer tests."""

    image: np.ndarray
    ground_truth: np.ndarray
    predictions: dict[str, np.ndarray]


@pytest.fixture
def sample_arrays() -> SampleArrays:
    """Provide a small image, ground truth mask, and two predictions."""
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    ground_truth = np.zeros((64, 64), dtype=bool)
    ground_truth[16:48, 16:48] = True

    prediction_one = np.zeros((64, 64), dtype=bool)
    prediction_one[18:46, 18:46] = True
    prediction_two = np.zeros((64, 64), dtype=bool)
    prediction_two[12:52, 20:44] = True

    return {
        "image": image,
        "ground_truth": ground_truth,
        "predictions": {
            "U-Net": prediction_one,
            "Attention U-Net": prediction_two,
        },
    }


@pytest.fixture
def visualizer() -> ComparisonVisualizer:
    """Provide a comparison visualizer with default settings."""
    return ComparisonVisualizer()


@pytest.fixture(autouse=True)
def close_figures() -> None:
    """Close Matplotlib figures created during a test."""
    yield
    plt.close("all")


def test_compare_predictions_returns_figure(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """Prediction comparisons return a figure with one panel per input."""
    figure = visualizer.compare_predictions(
        sample_arrays["image"],
        sample_arrays["ground_truth"],
        sample_arrays["predictions"],
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == len(sample_arrays["predictions"]) + 2


def test_compare_models_returns_figure(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """The model-comparison alias produces the prediction comparison layout."""
    prediction_figure = visualizer.compare_predictions(
        sample_arrays["image"],
        sample_arrays["ground_truth"],
        sample_arrays["predictions"],
    )
    model_figure = visualizer.compare_models(
        sample_arrays["image"],
        sample_arrays["ground_truth"],
        sample_arrays["predictions"],
    )

    assert isinstance(model_figure, Figure)
    assert len(model_figure.axes) == len(prediction_figure.axes)
    assert [axis.get_title() for axis in model_figure.axes] == [
        axis.get_title() for axis in prediction_figure.axes
    ]


def test_compare_overlays_returns_figure(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """Overlay comparisons return a figure with ground truth and predictions."""
    figure = visualizer.compare_overlays(
        sample_arrays["image"],
        sample_arrays["ground_truth"],
        sample_arrays["predictions"],
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == len(sample_arrays["predictions"]) + 1


def test_compare_difference_returns_figure(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """Difference comparisons create one panel for every prediction."""
    figure = visualizer.compare_difference(
        sample_arrays["ground_truth"],
        sample_arrays["predictions"],
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == len(sample_arrays["predictions"])


def test_save_creates_output_file(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
    tmp_path: Path,
) -> None:
    """Saving creates parent directories, writes the image, and closes it."""
    figure = visualizer.compare_predictions(
        sample_arrays["image"],
        sample_arrays["ground_truth"],
        sample_arrays["predictions"],
    )
    figure_number = figure.number
    output_path = tmp_path / "figures" / "comparison.png"

    visualizer.save(figure, str(output_path))

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert not plt.fignum_exists(figure_number)


def test_empty_predictions_raise_value_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """An empty prediction mapping is rejected."""
    with pytest.raises(ValueError):
        visualizer.compare_predictions(
            sample_arrays["image"],
            sample_arrays["ground_truth"],
            {},
        )


def test_none_image_raise_value_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """A missing image is rejected."""
    with pytest.raises(ValueError):
        visualizer.compare_predictions(
            None,
            sample_arrays["ground_truth"],
            sample_arrays["predictions"],
        )


def test_none_ground_truth_raise_value_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """A missing ground-truth mask is rejected."""
    with pytest.raises(ValueError):
        visualizer.compare_predictions(
            sample_arrays["image"],
            None,
            sample_arrays["predictions"],
        )


def test_prediction_shape_mismatch_raise_value_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """Predictions with a different shape are rejected."""
    predictions = {"U-Net": np.zeros((32, 32), dtype=bool)}

    with pytest.raises(ValueError):
        visualizer.compare_predictions(
            sample_arrays["image"],
            sample_arrays["ground_truth"],
            predictions,
        )


def test_image_ground_truth_shape_mismatch_raise_value_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """Images and ground-truth masks with different shapes are rejected."""
    image = np.zeros((32, 64), dtype=np.float32)

    with pytest.raises(ValueError):
        visualizer.compare_predictions(
            image,
            sample_arrays["ground_truth"],
            sample_arrays["predictions"],
        )


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_invalid_alpha_raise_value_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
    alpha: float,
) -> None:
    """Overlay alpha values outside the valid range are rejected."""
    with pytest.raises(ValueError):
        visualizer.compare_overlays(
            sample_arrays["image"],
            sample_arrays["ground_truth"],
            sample_arrays["predictions"],
            alpha=alpha,
        )


def test_invalid_alpha_type_raise_type_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """Non-numeric overlay alpha values are rejected."""
    with pytest.raises(TypeError):
        visualizer.compare_overlays(
            sample_arrays["image"],
            sample_arrays["ground_truth"],
            sample_arrays["predictions"],
            alpha="abc",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("dpi", [0, -10])
def test_invalid_dpi_raise_value_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
    tmp_path: Path,
    dpi: int,
) -> None:
    """Non-positive output resolutions are rejected."""
    figure = visualizer.compare_predictions(
        sample_arrays["image"],
        sample_arrays["ground_truth"],
        sample_arrays["predictions"],
    )

    with pytest.raises(ValueError):
        visualizer.save(figure, str(tmp_path / "comparison.png"), dpi=dpi)


def test_invalid_dpi_type_raise_type_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
    tmp_path: Path,
) -> None:
    """Non-integer output resolutions are rejected."""
    figure = visualizer.compare_predictions(
        sample_arrays["image"],
        sample_arrays["ground_truth"],
        sample_arrays["predictions"],
    )

    with pytest.raises(TypeError):
        visualizer.save(
            figure,
            str(tmp_path / "comparison.png"),
            dpi="300",  # type: ignore[arg-type]
        )


def test_prediction_not_ndarray_raise_type_error(
    visualizer: ComparisonVisualizer,
    sample_arrays: SampleArrays,
) -> None:
    """Prediction values that are not NumPy arrays are rejected."""
    predictions = {"U-Net": [[0, 1], [1, 0]]}

    with pytest.raises(TypeError):
        visualizer.compare_predictions(
            sample_arrays["image"],
            sample_arrays["ground_truth"],
            predictions,  # type: ignore[arg-type]
        )
