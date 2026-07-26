"""Anatomical planes supported by volumetric visualizations."""

from __future__ import annotations

from enum import Enum


class VolumePlane(Enum):
    """Anatomical planes used to extract slices from a 3D image volume."""

    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"
