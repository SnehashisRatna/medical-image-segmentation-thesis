"""Qualitative visualizations for a single segmentation prediction."""

from __future__ import annotations

from numbers import Real
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer


class PredictionVisualizer(BaseVisualizer):
    """Create qualitative visualizations for one segmentation prediction."""

    def __init__(
        self,
        figsize: tuple[float, float] = (6.0, 6.0),
        dpi: int = 100,
    ) -> None:
        """Initialize the prediction visualizer.

        Parameters
        ----------
        figsize : tuple[float, float], default=(6.0, 6.0)
            Width and height in inches for each displayed panel.
        dpi : int, default=100
            Resolution used when creating figures.

        Raises
        ------
        TypeError
            If ``figsize`` does not contain numeric values or ``dpi`` is not
            an integer.
        ValueError
            If a ``figsize`` value or ``dpi`` is not positive.
        """
        self._validate_figsize(figsize)
        self._validate_dpi(dpi)
        self._figsize = figsize
        self._dpi = dpi

    def render(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
    ) -> Figure:
        """Render an image alongside its segmentation prediction.

        Parameters
        ----------
        image : np.ndarray
            Source medical image.
        prediction : np.ndarray
            Predicted segmentation mask with the same shape as ``image``.

        Returns
        -------
        Figure
            Figure containing the original image and prediction.

        Raises
        ------
        TypeError
            If ``image`` or ``prediction`` is not a NumPy ndarray.
        ValueError
            If an input is not two-dimensional or the arrays have different
            shapes.
        """
        return self.render_prediction(image, prediction)

    def render_prediction(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
    ) -> Figure:
        """Render an image alongside its predicted segmentation mask.

        Parameters
        ----------
        image : np.ndarray
            Source medical image.
        prediction : np.ndarray
            Predicted segmentation mask with the same shape as ``image``.

        Returns
        -------
        Figure
            Figure containing the original image and prediction.

        Raises
        ------
        TypeError
            If ``image`` or ``prediction`` is not a NumPy ndarray.
        ValueError
            If an input is not two-dimensional or the arrays have different
            shapes.
        """
        self._validate_inputs(image, prediction)
        figure, axes = self._create_figure(panel_count=2)
        self._display_image(axes[0], image, "Original Image")
        self._display_mask(axes[1], prediction, "Prediction")
        self._apply_layout(figure)
        return figure

    def render_ground_truth(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Figure:
        """Render an image alongside its ground-truth segmentation mask.

        Parameters
        ----------
        image : np.ndarray
            Source medical image.
        ground_truth : np.ndarray
            Ground-truth segmentation mask with the same shape as ``image``.

        Returns
        -------
        Figure
            Figure containing the original image and ground-truth mask.

        Raises
        ------
        TypeError
            If ``image`` or ``ground_truth`` is not a NumPy ndarray.
        ValueError
            If an input is not two-dimensional or the arrays have different
            shapes.
        """
        self._validate_inputs(image, ground_truth)
        figure, axes = self._create_figure(panel_count=2)
        self._display_image(axes[0], image, "Original Image")
        self._display_mask(axes[1], ground_truth, "Ground Truth")
        self._apply_layout(figure)
        return figure

    def render_overlay(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        alpha: float = 0.5,
    ) -> Figure:
        """Render an image alongside an overlay of its prediction.

        Parameters
        ----------
        image : np.ndarray
            Source medical image.
        prediction : np.ndarray
            Predicted segmentation mask with the same shape as ``image``.
        alpha : float, default=0.5
            Opacity of the prediction overlay.

        Returns
        -------
        Figure
            Figure containing the original image and prediction overlay.

        Raises
        ------
        TypeError
            If ``image`` or ``prediction`` is not a NumPy ndarray, or if
            ``alpha`` is not numeric.
        ValueError
            If an input is not two-dimensional, the arrays have different
            shapes, or ``alpha`` is outside the inclusive range [0.0, 1.0].
        """
        self._validate_inputs(image, prediction)
        self._validate_alpha(alpha)
        figure, axes = self._create_figure(panel_count=2)
        self._display_image(axes[0], image, "Original Image")
        self._display_overlay(axes[1], image, prediction, "Prediction Overlay", alpha)
        self._apply_layout(figure)
        return figure

    def render_difference(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Figure:
        """Render the pixel-wise disagreement between two segmentation masks.

        Parameters
        ----------
        prediction : np.ndarray
            Predicted segmentation mask.
        ground_truth : np.ndarray
            Ground-truth segmentation mask with the same shape as
            ``prediction``.

        Returns
        -------
        Figure
            Figure containing a binary difference map.

        Raises
        ------
        TypeError
            If ``prediction`` or ``ground_truth`` is not a NumPy ndarray.
        ValueError
            If an input is not two-dimensional or the arrays have different
            shapes.
        """
        self._validate_masks(prediction, ground_truth)
        figure, axes = self._create_figure(panel_count=1)
        self._display_difference(axes[0], prediction, ground_truth)
        self._apply_layout(figure)
        return figure

    def save(
        self,
        figure: Figure,
        path: str,
    ) -> None:
        """Save and close a rendered prediction figure.

        Parameters
        ----------
        figure : Figure
            Matplotlib figure to save.
        path : str
            Destination image path. Missing parent directories are created.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``path`` is not a string or ``figure`` is not a Matplotlib
            figure.
        """
        if not isinstance(figure, Figure):
            raise TypeError("figure must be a Matplotlib Figure.")
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)

    def _validate_inputs(self, image: np.ndarray, prediction: np.ndarray) -> None:
        """Validate an image and a segmentation prediction."""
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy ndarray.")
        if not isinstance(prediction, np.ndarray):
            raise TypeError("prediction must be a NumPy ndarray.")
        if image.ndim != 2:
            raise ValueError("image must be a two-dimensional array.")
        if prediction.ndim != 2:
            raise ValueError("prediction must be a two-dimensional array.")
        if image.shape != prediction.shape:
            raise ValueError("image and prediction must have identical shapes.")

    def _validate_masks(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        """Validate prediction and ground-truth segmentation masks."""
        if not isinstance(prediction, np.ndarray):
            raise TypeError("prediction must be a NumPy ndarray.")
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("ground_truth must be a NumPy ndarray.")
        if prediction.ndim != 2:
            raise ValueError("prediction must be a two-dimensional array.")
        if ground_truth.ndim != 2:
            raise ValueError("ground_truth must be a two-dimensional array.")
        if prediction.shape != ground_truth.shape:
            raise ValueError("prediction and ground_truth must have identical shapes.")

    @staticmethod
    def _validate_alpha(alpha: float) -> None:
        """Validate the opacity used for prediction overlays."""
        if not isinstance(alpha, Real) or isinstance(alpha, bool):
            raise TypeError("alpha must be an int or float.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")

    @staticmethod
    def _validate_dpi(dpi: int) -> None:
        """Validate a Matplotlib resolution value."""
        if not isinstance(dpi, int) or isinstance(dpi, bool):
            raise TypeError("dpi must be an integer.")
        if dpi <= 0:
            raise ValueError("dpi must be positive.")

    @staticmethod
    def _validate_figsize(figsize: tuple[float, float]) -> None:
        """Validate dimensions used for each displayed panel."""
        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise TypeError("figsize must be a tuple containing two numeric values.")
        if any(
            not isinstance(value, Real) or isinstance(value, bool) for value in figsize
        ):
            raise TypeError("figsize must contain numeric values.")
        if any(value <= 0 for value in figsize):
            raise ValueError("figsize values must be positive.")

    def _create_figure(
        self,
        panel_count: int,
    ) -> tuple[Figure, tuple[Axes, ...]]:
        """Create a horizontal figure with one axes per visualization panel."""
        figure, axes = plt.subplots(
            1,
            panel_count,
            figsize=(self._figsize[0] * panel_count, self._figsize[1]),
            dpi=self._dpi,
            squeeze=False,
        )
        return figure, tuple(axes.flat)

    def _display_image(self, ax: Axes, image: np.ndarray, title: str) -> None:
        """Display a source image in a configured visualization panel."""
        ax.imshow(image, cmap="gray", interpolation="nearest")
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes=False)

    def _display_mask(self, ax: Axes, mask: np.ndarray, title: str) -> None:
        """Display a segmentation mask in a configured visualization panel."""
        ax.imshow(mask, cmap="gray", interpolation="nearest")
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes=False)

    def _display_overlay(
        self,
        ax: Axes,
        image: np.ndarray,
        prediction: np.ndarray,
        title: str,
        alpha: float,
    ) -> None:
        """Display a prediction overlaid on its source image."""
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.imshow(
            prediction,
            alpha=alpha,
            cmap="Reds",
            interpolation="nearest",
        )
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes=False)

    def _display_difference(
        self,
        ax: Axes,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        """Display the binary pixel-wise difference between two masks."""
        difference = prediction != ground_truth
        ax.imshow(difference, cmap="Reds", interpolation="nearest")
        self._apply_title(ax, "Difference Map")
        self._configure_axes(ax, show_axes=False)
