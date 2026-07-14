"""
Mask Statistics

Purpose
-------
Store descriptive statistics for a segmentation mask.

Author
------
Snehashis Ratna

Project
-------
Medical Image Segmentation Thesis
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MaskStatistics:
    """
    Store statistics describing a segmentation mask.
    """

    shape: tuple[int, int]

    dtype: np.dtype

    minimum: bool

    maximum: bool

    unique_labels: np.ndarray

    foreground_pixels: int

    background_pixels: int