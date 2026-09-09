"""CT-specific preprocessing for the CHAOS dataset.

Frozen pipeline
---------------
Raw DICOM pixel array
    → HU conversion   (HU = pixel × RescaleSlope + RescaleIntercept)
    → clip to [-1000, 1000] HU
    → normalize to [0, 1]   ((clipped_hu + 1000) / 2000)
    → float32

DICOMReader remains generic; slope and intercept are read externally
via ``DICOMReader.read_metadata()`` and passed in here.
"""

from __future__ import annotations

import numpy as np


def preprocess_ct(
    pixel_array: np.ndarray,
    rescale_slope: float,
    rescale_intercept: float,
) -> np.ndarray:
    """Apply the frozen CT preprocessing pipeline to one DICOM slice.

    Parameters
    ----------
    pixel_array : numpy.ndarray
        Raw DICOM pixel array, typically ``float32`` as returned by
        ``DICOMReader.read_image()``.
    rescale_slope : float
        Value of the DICOM ``RescaleSlope`` tag.
    rescale_intercept : float
        Value of the DICOM ``RescaleIntercept`` tag.

    Returns
    -------
    numpy.ndarray
        Preprocessed slice: dtype ``float32``, values in ``[0.0, 1.0]``.
        Native spatial resolution is preserved (512×512 for CHAOS CT).

    Notes
    -----
    The conversion formula is::

        hu      = pixel * slope + intercept
        clipped = clip(hu, -1000, 1000)
        output  = (clipped + 1000) / 2000

    This maps −1000 HU → 0.0, 0 HU → 0.5, and +1000 HU → 1.0.
    """
    hu = pixel_array.astype(np.float32) * float(rescale_slope) + float(rescale_intercept)
    clipped = np.clip(hu, -1000.0, 1000.0)
    normalized = (clipped + 1000.0) / 2000.0
    return normalized.astype(np.float32)
