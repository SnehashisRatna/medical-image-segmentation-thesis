"""MRI-specific preprocessing for the CHAOS dataset.

Frozen pipeline
---------------
Raw DICOM pixel array
    → per-volume non-zero-pixel z-score normalization
    → background pixels (== 0 in the original) restored to exactly 0.0
    → float32

Volume-level statistics are computed once per ``(patient_id, modality,
sequence)`` group via ``compute_volume_stats``, then reused for every
slice in that volume via ``preprocess_mri``.

The accumulation is implemented with a numerically stable parallel
Welford update (Chan et al., 1979) so that only one DICOM slice needs
to be in memory at a time during the statistics pass.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Internal protocol — keeps this module independent of src.data
# ---------------------------------------------------------------------------


class _DicomImageReader(Protocol):
    """Minimal interface required by ``compute_volume_stats``."""

    def read_image(self, path: Path) -> np.ndarray:
        """Return the pixel array for the given DICOM path."""
        ...


# ---------------------------------------------------------------------------
# Welford accumulator helpers
# ---------------------------------------------------------------------------


def _merge_welford(
    n_a: int,
    mean_a: float,
    m2_a: float,
    n_b: int,
    mean_b: float,
    m2_b: float,
) -> tuple[int, float, float]:
    """Merge two partial Welford accumulators using the parallel update.

    Implements the Chan et al. (1979) parallel / batch Welford formula so
    that per-slice statistics can be combined into volume-level statistics
    without loading the entire volume into memory at once.

    Parameters
    ----------
    n_a, mean_a, m2_a : int, float, float
        Count, mean, and sum-of-squared-deviations of group A.
    n_b, mean_b, m2_b : int, float, float
        Count, mean, and sum-of-squared-deviations of group B.

    Returns
    -------
    tuple[int, float, float]
        Merged ``(n, mean, m2)`` for the combined group.
    """
    n = n_a + n_b
    if n == 0:
        return 0, 0.0, 0.0
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    m2 = m2_a + m2_b + delta**2 * n_a * n_b / n
    return n, mean, m2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_volume_stats(
    image_paths: list[Path],
    dicom_reader: _DicomImageReader,
) -> tuple[float, float]:
    """Compute the non-zero-pixel mean and population std for an MRI volume.

    Statistics are accumulated slice-by-slice; only one DICOM array is
    held in memory at a time.  The result is mathematically identical to
    collecting every non-zero pixel from the entire volume and computing
    the population mean and standard deviation in a single pass.

    Parameters
    ----------
    image_paths : list[pathlib.Path]
        Ordered DICOM paths for every slice that belongs to this volume.
    dicom_reader : _DicomImageReader
        Object with a ``read_image(path) -> np.ndarray`` method.

    Returns
    -------
    tuple[float, float]
        ``(mean, std)`` over all non-zero pixels in the complete volume,
        both as Python ``float``.

    Notes
    -----
    * Pixels equal to exactly ``0`` are excluded from the statistics.
    * If the entire volume is all-zero (edge case), returns ``(0.0, 0.0)``
      so that ``preprocess_mri`` can still apply its epsilon guard safely.
    * Standard deviation is the **population** std (``ddof=0``).
    """
    n_total: int = 0
    mean_total: float = 0.0
    m2_total: float = 0.0

    for path in image_paths:
        arr = dicom_reader.read_image(path).astype(np.float64)
        nonzero = arr[arr > 0]
        n_s = len(nonzero)
        if n_s == 0:
            continue

        mean_s = float(nonzero.mean())
        # Per-slice sum-of-squared-deviations (using float64 for precision)
        m2_s = float(((nonzero - mean_s) ** 2).sum())

        n_total, mean_total, m2_total = _merge_welford(
            n_total, mean_total, m2_total,
            n_s, mean_s, m2_s,
        )

    if n_total == 0:
        # All-zero volume: return safe defaults (epsilon in preprocess_mri
        # will keep the denominator non-zero)
        return 0.0, 0.0

    std = math.sqrt(m2_total / n_total) if n_total > 1 else 0.0
    return float(mean_total), float(std)


def preprocess_mri(
    pixel_array: np.ndarray,
    volume_mean: float,
    volume_std: float,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Apply per-volume z-score normalization to a single MRI slice.

    Parameters
    ----------
    pixel_array : numpy.ndarray
        Raw MRI slice pixel array (as returned by ``DICOMReader.read_image``).
    volume_mean : float
        Mean of all non-zero pixels in the MRI volume, obtained from
        ``compute_volume_stats``.
    volume_std : float
        Population standard deviation of all non-zero pixels in the volume.
    epsilon : float, optional
        Numerical stability constant added to the denominator.

    Returns
    -------
    numpy.ndarray
        Normalized slice as ``float32``.  Pixels that were exactly ``0``
        in *pixel_array* are set back to ``0.0`` after normalization.

    Notes
    -----
    The normalization formula is::

        normalized = (pixel - volume_mean) / (volume_std + epsilon)
        normalized[pixel == 0] = 0.0

    Do NOT use per-slice statistics; always pass volume-level statistics
    obtained from ``compute_volume_stats``.
    """
    background_mask = pixel_array == 0
    normalized = (pixel_array.astype(np.float32) - volume_mean) / (
        float(volume_std) + epsilon
    )
    normalized[background_mask] = 0.0
    return normalized.astype(np.float32)
