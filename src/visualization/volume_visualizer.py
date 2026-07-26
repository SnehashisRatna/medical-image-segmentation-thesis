"""Qualitative visualizations for three-dimensional medical image volumes."""

from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.core.volume_plane import VolumePlane

from .base_visualizer import BaseVisualizer


_SECONDARY_VOLUME_NOT_PROVIDED: Final[object] = object()


class VolumeVisualizer(BaseVisualizer):
    """Render anatomical slices and segmentation views from 3D volumes.

    The visualizer accepts only volumes with shape ``(D, H, W)``.  Axial,
    coronal, and sagittal slices are respectively extracted along the depth,
    height, and width axes.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (6.0, 6.0),
        dpi: int = 100,
    ) -> None:
        """Initialize the volume visualizer.

        Parameters
        ----------
        figsize : tuple[float, float], default=(6.0, 6.0)
            Width and height in inches for each visualization panel.
        dpi : int, default=100
            Resolution used when creating figures.

        Raises
        ------
        TypeError
            If ``figsize`` is not a two-item numeric tuple or ``dpi`` is not
            an integer.
        ValueError
            If a figure dimension or ``dpi`` is not positive.
        """
        self._validate_figsize(figsize)
        self._validate_dpi(dpi)
        self._figsize = figsize
        self._dpi = dpi

    def render(
        self,
        volume: np.ndarray,
        slice_index: int,
        plane: VolumePlane = VolumePlane.AXIAL,
    ) -> Figure:
        """Render a single anatomical slice.

        This implements the :class:`BaseVisualizer` rendering contract and is
        equivalent to :meth:`render_slice`.

        Parameters
        ----------
        volume : np.ndarray
            Three-dimensional image volume with shape ``(D, H, W)``.
        slice_index : int
            Index of the slice in the selected ``plane``.
        plane : VolumePlane, default=VolumePlane.AXIAL
            Anatomical plane from which to extract the slice.

        Returns
        -------
        Figure
            Figure containing the requested image slice.
        """
        return self.render_slice(volume, slice_index, plane)

    def render_slice(
        self,
        volume: np.ndarray,
        slice_index: int,
        plane: VolumePlane = VolumePlane.AXIAL,
    ) -> Figure:
        """Render one slice extracted from a 3D image volume.

        Parameters
        ----------
        volume : np.ndarray
            Three-dimensional image volume with shape ``(D, H, W)``.
        slice_index : int
            Index of the slice in the selected ``plane``.
        plane : VolumePlane, default=VolumePlane.AXIAL
            Anatomical plane from which to extract the slice.

        Returns
        -------
        Figure
            Figure containing the requested image slice.

        Raises
        ------
        TypeError
            If ``volume`` is not a NumPy array, ``slice_index`` is not an
            integer, or ``plane`` is not a :class:`VolumePlane`.
        ValueError
            If ``volume`` is not three-dimensional or ``slice_index`` is
            outside the valid range for ``plane``.
        """
        self._validate_inputs(volume)
        self._validate_plane(plane)
        self._validate_slice(volume, slice_index, plane)

        figure, axes = self._create_figure(panel_count=1)
        self._display_slice(
            axes[0],
            self._extract_slice(volume, slice_index, plane),
            f"{plane.value.title()} Slice",
        )
        self._apply_layout(figure)
        return figure

    def render_mask(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        slice_index: int,
        plane: VolumePlane = VolumePlane.AXIAL,
    ) -> Figure:
        """Render an image slice alongside its segmentation mask.

        Parameters
        ----------
        volume : np.ndarray
            Three-dimensional image volume with shape ``(D, H, W)``.
        mask : np.ndarray
            Three-dimensional segmentation mask matching ``volume``.
        slice_index : int
            Index of the slice in the selected ``plane``.
        plane : VolumePlane, default=VolumePlane.AXIAL
            Anatomical plane from which to extract both slices.

        Returns
        -------
        Figure
            Figure containing the original slice and its mask.

        Raises
        ------
        TypeError
            If an array is not a NumPy array, ``slice_index`` is not an
            integer, or ``plane`` is not a :class:`VolumePlane`.
        ValueError
            If a volume is not three-dimensional, their shapes differ, or the
            slice index is invalid for ``plane``.
        """
        self._validate_inputs(volume, mask, secondary_name="mask")
        self._validate_plane(plane)
        self._validate_slice(volume, slice_index, plane)

        image_slice = self._extract_slice(volume, slice_index, plane)
        mask_slice = self._extract_slice(mask, slice_index, plane)
        figure, axes = self._create_figure(panel_count=2)
        self._display_slice(axes[0], image_slice, "Original Slice")
        self._display_mask(axes[1], mask_slice, "Mask")
        self._apply_layout(figure)
        return figure

    def render_prediction(
        self,
        volume: np.ndarray,
        prediction: np.ndarray,
        slice_index: int,
        plane: VolumePlane = VolumePlane.AXIAL,
    ) -> Figure:
        """Render an image slice alongside its segmentation prediction.

        Parameters
        ----------
        volume : np.ndarray
            Three-dimensional image volume with shape ``(D, H, W)``.
        prediction : np.ndarray
            Three-dimensional prediction volume matching ``volume``.
        slice_index : int
            Index of the slice in the selected ``plane``.
        plane : VolumePlane, default=VolumePlane.AXIAL
            Anatomical plane from which to extract both slices.

        Returns
        -------
        Figure
            Figure containing the original slice and its prediction.

        Raises
        ------
        TypeError
            If an array is not a NumPy array, ``slice_index`` is not an
            integer, or ``plane`` is not a :class:`VolumePlane`.
        ValueError
            If a volume is not three-dimensional, their shapes differ, or the
            slice index is invalid for ``plane``.
        """
        self._validate_inputs(volume, prediction, secondary_name="prediction")
        self._validate_plane(plane)
        self._validate_slice(volume, slice_index, plane)

        image_slice = self._extract_slice(volume, slice_index, plane)
        prediction_slice = self._extract_slice(prediction, slice_index, plane)
        figure, axes = self._create_figure(panel_count=2)
        self._display_slice(axes[0], image_slice, "Original Slice")
        self._display_mask(axes[1], prediction_slice, "Prediction")
        self._apply_layout(figure)
        return figure

    def render_overlay(
        self,
        volume: np.ndarray,
        prediction: np.ndarray,
        slice_index: int,
        plane: VolumePlane = VolumePlane.AXIAL,
        alpha: float = 0.5,
    ) -> Figure:
        """Render an image slice and its overlaid segmentation prediction.

        Parameters
        ----------
        volume : np.ndarray
            Three-dimensional image volume with shape ``(D, H, W)``.
        prediction : np.ndarray
            Three-dimensional prediction volume matching ``volume``.
        slice_index : int
            Index of the slice in the selected ``plane``.
        plane : VolumePlane, default=VolumePlane.AXIAL
            Anatomical plane from which to extract both slices.
        alpha : float, default=0.5
            Opacity of the prediction overlay, from 0.0 through 1.0.

        Returns
        -------
        Figure
            Figure containing the original slice and prediction overlay.

        Raises
        ------
        TypeError
            If an array is not a NumPy array, ``slice_index`` is not an
            integer, ``plane`` is invalid, or ``alpha`` is not numeric.
        ValueError
            If a volume is not three-dimensional, their shapes differ, the
            slice index is invalid, or ``alpha`` is outside [0.0, 1.0].
        """
        self._validate_inputs(volume, prediction, secondary_name="prediction")
        self._validate_plane(plane)
        self._validate_slice(volume, slice_index, plane)
        self._validate_alpha(alpha)

        image_slice = self._extract_slice(volume, slice_index, plane)
        prediction_slice = self._extract_slice(prediction, slice_index, plane)
        figure, axes = self._create_figure(panel_count=2)
        self._display_slice(axes[0], image_slice, "Original Slice")
        self._display_overlay(
            axes[1], image_slice, prediction_slice, "Prediction Overlay", alpha
        )
        self._apply_layout(figure)
        return figure

    def render_multiplanar(
        self,
        volume: np.ndarray,
        axial_index: int | None = None,
        coronal_index: int | None = None,
        sagittal_index: int | None = None,
    ) -> Figure:
        """Render axial, coronal, and sagittal views of an image volume.

        Omitted indices select the central slice in their respective planes.

        Parameters
        ----------
        volume : np.ndarray
            Three-dimensional image volume with shape ``(D, H, W)``.
        axial_index : int or None, default=None
            Axial slice index. The central axial slice is used when omitted.
        coronal_index : int or None, default=None
            Coronal slice index. The central coronal slice is used when
            omitted.
        sagittal_index : int or None, default=None
            Sagittal slice index. The central sagittal slice is used when
            omitted.

        Returns
        -------
        Figure
            Figure containing axial, coronal, and sagittal views.

        Raises
        ------
        TypeError
            If ``volume`` is not a NumPy array or a supplied slice index is
            not an integer.
        ValueError
            If ``volume`` is not three-dimensional or a supplied slice index
            is outside its plane's valid range.
        """
        self._validate_inputs(volume)
        indices = (
            volume.shape[0] // 2 if axial_index is None else axial_index,
            volume.shape[1] // 2 if coronal_index is None else coronal_index,
            volume.shape[2] // 2 if sagittal_index is None else sagittal_index,
        )
        planes = (
            VolumePlane.AXIAL,
            VolumePlane.CORONAL,
            VolumePlane.SAGITTAL,
        )
        for slice_index, plane in zip(indices, planes, strict=True):
            self._validate_slice(volume, slice_index, plane)

        figure, axes = self._create_figure(panel_count=3)
        for ax, slice_index, plane in zip(axes, indices, planes, strict=True):
            self._display_slice(
                ax,
                self._extract_slice(volume, slice_index, plane),
                plane.value.title(),
            )
        figure.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
        return figure

    def save(self, figure: Figure, path: str) -> None:
        """Save and close a rendered volume figure.

        Parameters
        ----------
        figure : Figure
            Matplotlib figure to save.
        path : str
            Destination image path. Missing parent directories are created.

        Raises
        ------
        TypeError
            If ``figure`` is not a Matplotlib figure or ``path`` is not a
            string.
        """
        if not isinstance(figure, Figure):
            raise TypeError("figure must be a Matplotlib Figure.")
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(output_path, bbox_inches="tight")
        finally:
            plt.close(figure)

    def _validate_inputs(
        self,
        volume: np.ndarray,
        secondary_volume: np.ndarray | object = _SECONDARY_VOLUME_NOT_PROVIDED,
        *,
        secondary_name: str = "secondary_volume",
    ) -> None:
        """Validate a primary volume and an optional matching volume."""
        self._validate_volume(volume, "volume")
        if secondary_volume is _SECONDARY_VOLUME_NOT_PROVIDED:
            return

        self._validate_volume(secondary_volume, secondary_name)
        if volume.shape != secondary_volume.shape:
            raise ValueError(
                f"volume and {secondary_name} must have identical shapes."
            )

    @staticmethod
    def _validate_volume(volume: np.ndarray, name: str = "volume") -> None:
        """Validate that an input is a three-dimensional NumPy volume."""
        if not isinstance(volume, np.ndarray):
            raise TypeError(f"{name} must be a NumPy ndarray.")
        if volume.ndim != 3:
            raise ValueError(f"{name} must be a three-dimensional array.")

    @staticmethod
    def _validate_slice(
        volume: np.ndarray,
        slice_index: int,
        plane: VolumePlane,
    ) -> None:
        """Validate a slice index for a particular anatomical plane."""
        if not isinstance(slice_index, Integral) or isinstance(slice_index, bool):
            raise TypeError("slice_index must be an integer.")

        axis = {
            VolumePlane.AXIAL: 0,
            VolumePlane.CORONAL: 1,
            VolumePlane.SAGITTAL: 2,
        }[plane]
        if not 0 <= slice_index < volume.shape[axis]:
            raise ValueError(
                f"slice_index must be between 0 and {volume.shape[axis] - 1} "
                f"for the {plane.value} plane."
            )

    @staticmethod
    def _validate_plane(plane: VolumePlane) -> None:
        """Validate an anatomical plane selection."""
        if not isinstance(plane, VolumePlane):
            raise TypeError("plane must be a VolumePlane.")

    @staticmethod
    def _validate_alpha(alpha: float) -> None:
        """Validate the opacity used for segmentation prediction overlays."""
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
        """Validate figure dimensions used for individual display panels."""
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
        if panel_count == 3:
            figure = plt.figure(
                figsize=(self._figsize[0] * 2, self._figsize[1] * 2),
                dpi=self._dpi,
            )
            grid = figure.add_gridspec(2, 2, hspace=0.2, wspace=0.08)
            axes = (
                figure.add_subplot(grid[0, :]),
                figure.add_subplot(grid[1, 0]),
                figure.add_subplot(grid[1, 1]),
            )
            return figure, axes

        figure, axes = plt.subplots(
            1,
            panel_count,
            figsize=(self._figsize[0] * panel_count, self._figsize[1]),
            dpi=self._dpi,
            squeeze=False,
        )
        return figure, tuple(axes.flat)

    @staticmethod
    def _extract_slice(
        volume: np.ndarray,
        slice_index: int,
        plane: VolumePlane,
    ) -> np.ndarray:
        """Extract a two-dimensional slice from a validated image volume."""
        if plane is VolumePlane.AXIAL:
            return volume[slice_index, :, :]
        if plane is VolumePlane.CORONAL:
            return np.flipud(volume[:, slice_index, :])
        return np.flipud(volume[:, :, slice_index])

    def _display_slice(self, ax: Axes, image: np.ndarray, title: str) -> None:
        """Display an image slice in a configured visualization panel."""
        ax.imshow(image, cmap="gray", interpolation="nearest")
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes=False)

    def _display_mask(self, ax: Axes, mask: np.ndarray, title: str) -> None:
        """Display a segmentation mask in a configured visualization panel."""
        ax.imshow(mask, cmap="Reds", interpolation="nearest")
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
        """Display a prediction overlaid on an image slice."""
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.imshow(prediction, cmap="Reds", alpha=alpha, interpolation="nearest")
        self._apply_title(ax, title)
        self._configure_axes(ax, show_axes=False)
