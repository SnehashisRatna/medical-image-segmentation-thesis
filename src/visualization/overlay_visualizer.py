"""Visualization component for image and segmentation mask overlays."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer


class OverlayVisualizer(BaseVisualizer):
    """Render a two-dimensional medical image with an overlaid mask."""

    def render(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        mask_alpha: float = 0.4,
        image_cmap: str = "gray",
        mask_cmap: str = "Reds",
        interpolation: str = "nearest",
        title: str | None = None,
        figsize: tuple[float, float] = (6.0, 6.0),
        dpi: int = 100,
        show_axes: bool = False,
    ) -> Figure:
        """Render a medical image with a segmentation mask overlay.

        Parameters
        ----------
        image : np.ndarray
            Two-dimensional medical image to display.
        mask : np.ndarray
            Two-dimensional segmentation mask to overlay on ``image``.
        mask_alpha : float, default=0.4
            Opacity of the mask layer.
        image_cmap : str, default="gray"
            Matplotlib colormap used to display the image.
        mask_cmap : str, default="Reds"
            Matplotlib colormap used to display the mask.
        interpolation : str, default="nearest"
            Matplotlib interpolation method used for both layers.
        title : str or None, default=None
            Optional title applied to the axes.
        figsize : tuple[float, float], default=(6.0, 6.0)
            Width and height of the figure in inches.
        dpi : int, default=100
            Resolution of the figure in dots per inch.
        show_axes : bool, default=False
            Whether to display the axes.

        Returns
        -------
        Figure
            Matplotlib figure containing the image and mask overlay.

        Raises
        ------
        TypeError
            If ``image`` or ``mask`` is not a NumPy ndarray, or if
            ``mask_alpha`` is not numeric.
        ValueError
            If either input is not two-dimensional, their shapes differ, or
            ``mask_alpha`` is outside the inclusive range [0.0, 1.0].
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy ndarray.")
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a NumPy ndarray.")
        if image.ndim != 2:
            raise ValueError("image must be a two-dimensional array.")
        if mask.ndim != 2:
            raise ValueError("mask must be a two-dimensional array.")
        if image.shape != mask.shape:
            raise ValueError("image and mask must have identical shapes.")
        if not isinstance(mask_alpha, (int, float)):
            raise TypeError("mask_alpha must be an int or float.")
        if not 0.0 <= mask_alpha <= 1.0:
            raise ValueError("mask_alpha must be between 0.0 and 1.0.")

        figure, ax = self._create_figure(figsize=figsize, dpi=dpi)
        ax.imshow(image, cmap=image_cmap, interpolation=interpolation)
        ax.imshow(
            mask,
            alpha=mask_alpha,
            cmap=mask_cmap,
            interpolation=interpolation,
        )
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes)
        self._apply_layout(figure)
        return figure
