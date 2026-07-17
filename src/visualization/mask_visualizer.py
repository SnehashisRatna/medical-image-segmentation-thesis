"""Visualization component for segmentation masks."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer


class MaskVisualizer(BaseVisualizer):
    """Render segmentation masks as two-dimensional images."""

    def render(
        self,
        mask: np.ndarray,
        *,
        title: str | None = None,
        cmap: str = "gray",
        figsize: tuple[float, float] = (6.0, 6.0),
        dpi: int = 100,
        show_axes: bool = False,
    ) -> Figure:
        """Render a segmentation mask.

        Parameters
        ----------
        mask : np.ndarray
            Mask to display.
        title : str or None, default=None
            Optional figure title.
        cmap : str, default="gray"
            Matplotlib colormap used to display the mask.
        figsize : tuple[float, float], default=(6.0, 6.0)
            Width and height of the figure in inches.
        dpi : int, default=100
            Resolution of the figure in dots per inch.
        show_axes : bool, default=False
            Whether to display the axes.

        Returns
        -------
        Figure
            Matplotlib figure containing the rendered mask.
        """
        figure, ax = self._create_figure(figsize=figsize, dpi=dpi)
        ax.imshow(mask, cmap=cmap)
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes)
        self._apply_layout(figure)
        return figure
