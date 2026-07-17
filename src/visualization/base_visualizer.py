"""Generic interface and shared helpers for visualization components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class BaseVisualizer(ABC):
    """Abstract base class for stateless visualization components.

    Notes
    -----
    Subclasses implement rendering behavior while this class supplies only
    generic Matplotlib setup helpers.
    """


    @abstractmethod
    def render(self, *args: Any, **kwargs: Any) -> Figure:
        """Render a single visualization.

        Returns
        -------
        Figure
            Matplotlib figure containing the visualization.
        """
        ...

    def _create_figure(
        self,
        figsize: tuple[float, float] = (6.0, 6.0),
        dpi: int = 100
    ) -> tuple[Figure, Axes]:
        """Create a Matplotlib figure and axes.

        Parameters
        ----------
        figsize : tuple[float, float], default=(6.0, 6.0)
            Width and height of the figure in inches.

        Returns
        -------
        tuple[Figure, Axes]
            Created figure and its single axes.
        """
        figure, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )
        return figure, ax

    def _apply_title(
        self,
        ax: Axes,
        title: str | None,
    ) -> None:
        """Apply an optional title to axes.

        Parameters
        ----------
        ax : Axes
            Axes that receives the title.
        title : str or None
            Title to apply. No title is set when ``None``.
        """
        if title:
            ax.set_title(title)

    def _configure_axes(
        self,
        ax: Axes,
        show_axes: bool,
    ) -> None:
        """Configure axes visibility.

        Parameters
        ----------
        ax : Axes
            Axes to configure.
        show_axes : bool
            Whether axes should be visible.
        """
        if show_axes:
            ax.set_axis_on()
        else:
            ax.set_axis_off()

    def _apply_layout(
        self,
        figure: Figure,
    ) -> None:
        """Apply layout adjustments to a figure.

        Parameters
        ----------
        figure : Figure
            Figure whose layout is adjusted.
        """
        figure.tight_layout()
