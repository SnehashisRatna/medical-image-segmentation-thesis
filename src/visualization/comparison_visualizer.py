"""Qualitative comparison visualizations for segmentation predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer
from .mask_visualizer import MaskVisualizer
from .overlay_visualizer import OverlayVisualizer


class ComparisonVisualizer(BaseVisualizer):
    """Create side-by-side qualitative segmentation comparisons."""

    def __init__(
        self,
        figsize: tuple[float, float] = (4.0, 4.5),
        dpi: int = 100,
    ) -> None:
        """Initialize visualizers and default figure settings.

        Parameters
        ----------
        figsize : tuple[float, float], default=(4.0, 4.5)
            Width and height in inches for an individual comparison panel.
            The figure width scales with the number of displayed panels.
        dpi : int, default=100
            Resolution used when creating comparison figures.

        Raises
        ------
        TypeError
            If ``dpi`` is not an integer.
        ValueError
            If ``dpi`` is not positive.
        """
        self._validate_dpi(dpi)
        self._figsize = figsize
        self._dpi = dpi
        self._mask_visualizer = MaskVisualizer()
        self._overlay_visualizer = OverlayVisualizer()

    def render(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        predictions: dict[str, np.ndarray],
    ) -> Figure:
        """Render the default side-by-side prediction comparison.

        Parameters
        ----------
        image : np.ndarray
            Source medical image.
        ground_truth : np.ndarray
            Ground-truth segmentation mask.
        predictions : dict[str, np.ndarray]
            Mapping of model names to their predicted masks.

        Returns
        -------
        Figure
            Figure containing the original image, ground truth, and predictions.
        """
        return self.compare_predictions(image, ground_truth, predictions)

    def compare_models(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        predictions: dict[str, np.ndarray],
    ) -> Figure:
        """Display a side-by-side comparison of segmentation models.

        This method is an alias for :meth:`compare_predictions`.

        Parameters
        ----------
        image : np.ndarray
            Source medical image.
        ground_truth : np.ndarray
            Ground-truth segmentation mask.
        predictions : dict[str, np.ndarray]
            Mapping of model names to their predicted masks.

        Returns
        -------
        Figure
            Side-by-side qualitative comparison figure.
        """
        return self.compare_predictions(image, ground_truth, predictions)

    def compare_predictions(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        predictions: dict[str, np.ndarray],
    ) -> Figure:
        """Display an image, its ground truth, and model predictions.

        Parameters
        ----------
        image : np.ndarray
            Source medical image.
        ground_truth : np.ndarray
            Ground-truth segmentation mask.
        predictions : dict[str, np.ndarray]
            Mapping of model names to their predicted masks.

        Returns
        -------
        Figure
            Side-by-side qualitative comparison figure.

        Raises
        ------
        ValueError
            If an input is ``None``, no predictions are supplied, or array
            shapes do not match.
        """
        self._validate_inputs(image, ground_truth, predictions)
        figure, axes = self._create_comparison_figure(len(predictions) + 2)

        self._display_array(axes[0], image, "Original Image", cmap="gray")
        self._display_mask(axes[1], ground_truth, "Ground Truth")
        for ax, (model_name, prediction) in zip(
            axes[2:], predictions.items(), strict=True
        ):
            self._display_mask(ax, prediction, model_name)

        self._apply_layout(figure)
        return figure

    def compare_overlays(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        predictions: dict[str, np.ndarray],
        alpha: float = 0.4,
    ) -> Figure:
        """Display ground-truth and prediction masks overlaid on an image.

        Parameters
        ----------
        image : np.ndarray
            Source medical image.
        ground_truth : np.ndarray
            Ground-truth segmentation mask.
        predictions : dict[str, np.ndarray]
            Mapping of model names to their predicted masks.
        alpha : float, default=0.4
            Opacity of each segmentation mask overlay.

        Returns
        -------
        Figure
            Figure containing the ground-truth and prediction overlays.

        Raises
        ------
        ValueError
            If an input is ``None``, no predictions are supplied, array
            shapes do not match, or ``alpha`` is invalid.
        """
        self._validate_inputs(image, ground_truth, predictions)
        self._validate_alpha(alpha)
        figure, axes = self._create_comparison_figure(len(predictions) + 1)

        self._display_overlay(
            axes[0], image, ground_truth, "Ground Truth Overlay", alpha
        )
        for ax, (model_name, prediction) in zip(
            axes[1:], predictions.items(), strict=True
        ):
            self._display_overlay(ax, image, prediction, f"{model_name} Overlay", alpha)

        self._apply_layout(figure)
        return figure

    def compare_difference(
        self,
        ground_truth: np.ndarray,
        predictions: dict[str, np.ndarray],
    ) -> Figure:
        """Display pixel-wise disagreement masks for each prediction.

        Parameters
        ----------
        ground_truth : np.ndarray
            Ground-truth segmentation mask.
        predictions : dict[str, np.ndarray]
            Mapping of model names to their predicted masks.

        Returns
        -------
        Figure
            Figure with one segmentation-difference panel per prediction.

        Raises
        ------
        ValueError
            If the ground truth is ``None``, no predictions are supplied, or
            a prediction shape differs from the ground truth.
        """
        self._validate_masks(ground_truth, predictions)
        figure, axes = self._create_comparison_figure(len(predictions))

        for ax, (model_name, prediction) in zip(
            axes, predictions.items(), strict=True
        ):
            difference = prediction != ground_truth
            self._display_array(
                ax,
                difference,
                f"{model_name} Difference",
                cmap="Reds",
            )

        self._apply_layout(figure)
        return figure

    def save(
        self,
        figure: Figure,
        output_path: str,
        dpi: int = 300,
    ) -> None:
        """Save and close a comparison figure.

        Parameters
        ----------
        figure : Figure
            Matplotlib figure to save.
        output_path : str
            Destination image path. Missing parent directories are created.
        dpi : int, default=300
            Output resolution in dots per inch.

        Raises
        ------
        TypeError
            If ``dpi`` is not an integer.
        ValueError
            If ``dpi`` is not positive.
        """
        self._validate_dpi(dpi)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)

    def _validate_inputs(
        self,
        image: np.ndarray | None,
        ground_truth: np.ndarray | None,
        predictions: Mapping[str, np.ndarray],
    ) -> None:
        """Validate arrays required for comparisons that include an image."""
        if image is None:
            raise ValueError("image must not be None.")
        self._validate_masks(ground_truth, predictions)
        if image.shape != ground_truth.shape:
            raise ValueError("image and ground_truth must have identical shapes.")

    def _validate_masks(
        self,
        ground_truth: np.ndarray | None,
        predictions: Mapping[str, np.ndarray],
    ) -> None:
        """Validate ground-truth and prediction mask shapes."""
        if ground_truth is None:
            raise ValueError("ground_truth must not be None.")
        if not predictions:
            raise ValueError("predictions must contain at least one mask.")

        for model_name, prediction in predictions.items():
            if not isinstance(prediction, np.ndarray):
                raise TypeError(
                    f"Prediction '{model_name}' must be a NumPy ndarray."
                )
            if prediction.shape != ground_truth.shape:
                raise ValueError(
                    f"Prediction '{model_name}' and ground_truth must have "
                    "identical shapes."
                )

    def _create_comparison_figure(
        self,
        panel_count: int,
    ) -> tuple[Figure, tuple[Axes, ...]]:
        """Create a horizontal figure with one axes for each panel."""
        figure, axes = plt.subplots(
            1,
            panel_count,
            figsize=(self._figsize[0] * panel_count, self._figsize[1]),
            dpi=self._dpi,
            squeeze=False,
        )
        return figure, tuple(axes.flat)

    def _display_array(
        self,
        ax: Axes,
        array: np.ndarray,
        title: str,
        *,
        cmap: str,
    ) -> None:
        """Display an array in a configured comparison panel."""
        ax.imshow(array, cmap=cmap, interpolation="nearest")
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes=False)

    def _display_mask(self, ax: Axes, mask: np.ndarray, title: str) -> None:
        """Render a mask with ``MaskVisualizer`` in a comparison panel."""
        source_figure = self._mask_visualizer.render(mask, cmap="gray")
        self._copy_images(source_figure.axes[0], ax)
        plt.close(source_figure)
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes=False)

    def _display_overlay(
        self,
        ax: Axes,
        image: np.ndarray,
        mask: np.ndarray,
        title: str,
        alpha: float,
    ) -> None:
        """Render an overlay with ``OverlayVisualizer`` in a comparison panel."""
        source_figure = self._overlay_visualizer.render(
            image,
            mask,
            mask_alpha=alpha,
        )
        self._copy_images(source_figure.axes[0], ax)
        plt.close(source_figure)
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes=False)

    @staticmethod
    def _copy_images(source: Axes, destination: Axes) -> None:
        """Copy displayed image layers from one axes to another.

        Notes
        -----
        This helper copies only image layers. Other Matplotlib artists, such
        as contours, annotations, and line plots, are not transferred.
        """
        for image in source.get_images():
            destination.imshow(
                image.get_array(),
                alpha=image.get_alpha(),
                cmap=image.get_cmap(),
                interpolation=image.get_interpolation(),
                origin=image.origin,
            )

    @staticmethod
    def _validate_alpha(alpha: float) -> None:
        """Validate the opacity value used for segmentation overlays."""
        if not isinstance(alpha, (int, float)):
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
