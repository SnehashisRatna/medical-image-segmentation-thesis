"""Binary liver mask preprocessing for the CHAOS dataset.

Frozen semantic mapping
-----------------------
MRI raw CHAOS labels:

    0   → 0  (background)
    63  → 1  (liver)
    126 → 0  (right kidney — mapped to non-liver background)
    189 → 0  (left kidney  — mapped to non-liver background)
    252 → 0  (spleen       — mapped to non-liver background)

CT:

    0        → 0  (background)
    non-zero → 1  (liver)

The binary projection belongs to the dataset-processing layer.
``GroundTruthReader`` is not modified; it continues to return raw
label arrays.

Notes
-----
Class 0 means **non-liver**, not exclusively anatomical background.
For MRI, kidney and spleen pixels are mapped to class 0 because the
current task is binary liver segmentation only.
"""

from __future__ import annotations

import numpy as np

from src.core.enums import Modality

# Frozen CHAOS MRI liver label value
_MRI_LIVER_LABEL: int = 63


def binarize_liver_mask(
    raw_mask: np.ndarray,
    modality: Modality,
) -> np.ndarray:
    """Convert a raw CHAOS segmentation mask to a binary liver target.

    Parameters
    ----------
    raw_mask : numpy.ndarray
        Raw mask array returned by ``GroundTruthReader.read_mask()``.
        For MRI this contains values in ``{0, 63, 126, 189, 252}``.
        For CT this contains values in ``{0}`` (background) and non-zero
        (liver).
    modality : Modality
        Imaging modality of the corresponding image slice.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of the same spatial shape as *raw_mask* with
        values only in ``{0, 1}``:

        * ``0`` — non-liver (background, kidney, spleen for MRI)
        * ``1`` — liver

    Notes
    -----
    Empty masks (all zeros in *raw_mask*) are valid inputs and produce
    an all-zero binary output, representing slices with no liver present.
    """
    if modality == Modality.MRI:
        binary = (raw_mask == _MRI_LIVER_LABEL).astype(np.uint8)
    else:
        # CT: CHAOS Ground directory contains liver-only masks;
        # any non-zero pixel is liver.
        binary = (raw_mask > 0).astype(np.uint8)
    return binary
