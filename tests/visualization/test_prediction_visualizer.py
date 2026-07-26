"""Unit tests for single-prediction segmentation visualizations."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.visualization.prediction_visualizer import PredictionVisualizer


@pytest.fixture
def sample_image() -> np.ndarray:
    """Provide a deterministic grayscale image."""
    return np.arange(64, dtype=np.float32).reshape(8, 8)


@pytest.fixture
def sample_prediction(sample_image: np.ndarray) -> np.ndarray:
    """Provide a deterministic prediction mask."""
    return sample_image % 2 == 0


@pytest.fixture
def sample_ground_truth(sample_image: np.ndarray) -> np.ndarray:
    """Provide a deterministic ground-truth mask."""
    return sample_image % 3 == 0


@pytest.fixture
def visualizer() -> PredictionVisualizer:
    """Provide a prediction visualizer with default settings."""
    return PredictionVisualizer()


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close Matplotlib figures created during a test."""
    yield
    plt.close("all")


def test_default_constructor_uses_expected_settings() -> None:
    """The default constructor stores the documented figure settings."""
    visualizer = PredictionVisualizer()

    assert visualizer._figsize == (6.0, 6.0)
    assert visualizer._dpi == 100


def test_constructor_accepts_custom_figsize() -> None:
    """The constructor accepts custom per-panel dimensions."""
    visualizer = PredictionVisualizer(figsize=(4.0, 3.0))

    assert visualizer._figsize == (4.0, 3.0)


def test_constructor_accepts_custom_dpi() -> None:
    """The constructor accepts a custom figure resolution."""
    visualizer = PredictionVisualizer(dpi=200)

    assert visualizer._dpi == 200


@pytest.mark.parametrize("dpi", [0, -100])
def test_invalid_constructor_dpi_raises_value_error(dpi: int) -> None:
    """Non-positive constructor resolutions are rejected."""
    with pytest.raises(ValueError):
        PredictionVisualizer(dpi=dpi)


def test_invalid_figsize_type_raises_type_error() -> None:
    """A non-tuple figure size is rejected."""
    with pytest.raises(TypeError):
        PredictionVisualizer(figsize=[6.0, 6.0])  # type: ignore[arg-type]


def test_invalid_figsize_length_raises_type_error() -> None:
    """A figure size without exactly two values is rejected."""
    with pytest.raises(TypeError):
        PredictionVisualizer(figsize=(6.0, 6.0, 6.0))  # type: ignore[arg-type]


@pytest.mark.parametrize("figsize", [(0.0, 6.0), (6.0, -1.0)])
def test_invalid_figsize_values_raise_value_error(
    figsize: tuple[float, float],
) -> None:
    """Non-positive figure dimensions are rejected."""
    with pytest.raises(ValueError):
        PredictionVisualizer(figsize=figsize)


def test_render_returns_titled_two_panel_figure(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
    sample_prediction: np.ndarray,
) -> None:
    """Rendering creates image and prediction panels."""
    figure = visualizer.render(sample_image, sample_prediction)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == [
        "Original Image",
        "Prediction",
    ]


def test_render_prediction_returns_titled_two_panel_figure(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
    sample_prediction: np.ndarray,
) -> None:
    """Prediction rendering creates image and prediction panels."""
    figure = visualizer.render_prediction(sample_image, sample_prediction)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == [
        "Original Image",
        "Prediction",
    ]


def test_render_ground_truth_returns_titled_two_panel_figure(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
    sample_ground_truth: np.ndarray,
) -> None:
    """Ground-truth rendering creates image and mask panels."""
    figure = visualizer.render_ground_truth(sample_image, sample_ground_truth)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == [
        "Original Image",
        "Ground Truth",
    ]


def test_render_overlay_returns_titled_two_panel_figure(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
    sample_prediction: np.ndarray,
) -> None:
    """Overlay rendering creates image and prediction-overlay panels."""
    figure = visualizer.render_overlay(sample_image, sample_prediction)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == [
        "Original Image",
        "Prediction Overlay",
    ]


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_render_overlay_accepts_boundary_alpha_values(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
    sample_prediction: np.ndarray,
    alpha: float,
) -> None:
    """Overlay rendering accepts both inclusive alpha boundaries."""
    figure = visualizer.render_overlay(sample_image, sample_prediction, alpha=alpha)

    assert isinstance(figure, Figure)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_render_overlay_rejects_out_of_range_alpha(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
    sample_prediction: np.ndarray,
    alpha: float,
) -> None:
    """Overlay alpha values outside the inclusive range are rejected."""
    with pytest.raises(ValueError):
        visualizer.render_overlay(sample_image, sample_prediction, alpha=alpha)


def test_render_overlay_rejects_non_numeric_alpha(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
    sample_prediction: np.ndarray,
) -> None:
    """Non-numeric overlay alpha values are rejected."""
    with pytest.raises(TypeError):
        visualizer.render_overlay(
            sample_image,
            sample_prediction,
            alpha="half",  # type: ignore[arg-type]
        )


def test_render_difference_returns_titled_single_panel_figure(
    visualizer: PredictionVisualizer,
    sample_prediction: np.ndarray,
    sample_ground_truth: np.ndarray,
) -> None:
    """Difference rendering creates one difference-map panel."""
    figure = visualizer.render_difference(sample_prediction, sample_ground_truth)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert figure.axes[0].get_title() == "Difference Map"


def test_save_creates_output_file_and_closes_figure(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
    sample_prediction: np.ndarray,
    tmp_path: Path,
) -> None:
    """Saving creates parent directories, writes the image, and closes it."""
    figure = visualizer.render_prediction(sample_image, sample_prediction)
    figure_number = figure.number
    output_path = tmp_path / "figures" / "prediction.png"

    visualizer.save(figure, str(output_path))

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert not plt.fignum_exists(figure_number)


def test_render_rejects_non_ndarray_image(
    visualizer: PredictionVisualizer,
    sample_prediction: np.ndarray,
) -> None:
    """Images that are not NumPy arrays are rejected."""
    with pytest.raises(TypeError):
        visualizer.render([[0, 1], [2, 3]], sample_prediction)  # type: ignore[arg-type]


def test_render_rejects_non_ndarray_prediction(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
) -> None:
    """Predictions that are not NumPy arrays are rejected."""
    with pytest.raises(TypeError):
        visualizer.render(sample_image, [[True, False]])  # type: ignore[arg-type]


def test_render_difference_rejects_non_ndarray_ground_truth(
    visualizer: PredictionVisualizer,
    sample_prediction: np.ndarray,
) -> None:
    """Ground-truth masks that are not NumPy arrays are rejected."""
    with pytest.raises(TypeError):
        visualizer.render_difference(
            sample_prediction,
            [[True, False]],  # type: ignore[arg-type]
        )


def test_render_rejects_shape_mismatch(
    visualizer: PredictionVisualizer,
    sample_image: np.ndarray,
) -> None:
    """Images and predictions with different shapes are rejected."""
    prediction = np.zeros((4, 4), dtype=bool)

    with pytest.raises(ValueError):
        visualizer.render(sample_image, prediction)


def test_render_rejects_three_dimensional_arrays(
    visualizer: PredictionVisualizer,
) -> None:
    """Three-dimensional rendering inputs are rejected."""
    image = np.zeros((8, 8, 1), dtype=np.float32)
    prediction = np.zeros((8, 8, 1), dtype=bool)

    with pytest.raises(ValueError):
        visualizer.render(image, prediction)


def test_render_rejects_one_dimensional_arrays(
    visualizer: PredictionVisualizer,
) -> None:
    """One-dimensional rendering inputs are rejected."""
    image = np.arange(8, dtype=np.float32)
    prediction = np.zeros(8, dtype=bool)

    with pytest.raises(ValueError):
        visualizer.render(image, prediction)
