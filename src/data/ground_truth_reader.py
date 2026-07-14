"""
Ground Truth Reader

Purpose
-------
Read and analyze segmentation masks from the CHAOS dataset.

Responsibilities
----------------
1. Read PNG masks
2. Extract mask statistics
3. Identify unique labels
4. Detect empty masks
5. Visualize segmentation masks

Author
------
Snehashis Ratna

Project
-------
Medical Image Segmentation Thesis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.types.mask_statistics import MaskStatistics

class GroundTruthReader:
   
    """
    Read and analyze segmentation masks.

    """

    def __init__(self) -> None:
        ...

    def read_mask(self, mask_path: str | Path) -> np.ndarray:
        """Read a segmentation mask from the CHAOS dataset.

        Parameters
        ----------
        mask_path : str | pathlib.Path
            Path to the PNG segmentation mask.

        Returns
        -------
        numpy.ndarray
            Mask array with the dtype and values stored in the image.

        Raises
        ------
        FileNotFoundError
            If ``mask_path`` does not exist.
        """
        # Convert to Path object
        path = Path(mask_path)

        # Verify the file exists
        if not path.exists():
           raise FileNotFoundError(path)

        # Read mask image
        with Image.open(path) as image:
           return np.array(image)

    def unique_labels(self, mask: np.ndarray) -> np.ndarray:
        """Return the unique label values in a segmentation mask.

        Parameters
        ----------
        mask : numpy.ndarray
            Segmentation mask containing label values.

        Returns
        -------
        numpy.ndarray
            Sorted unique label values with the same dtype as ``mask``.
        """
        return np.unique(mask)

    def mask_statistics(self, mask: np.ndarray) -> MaskStatistics:
        """Compute descriptive statistics for a segmentation mask.

        Parameters
        ----------
        mask : numpy.ndarray
            Segmentation mask containing background and foreground class labels.

        Returns
        -------
        MaskStatistics
            Statistics describing the mask's shape, dtype, value range,
            labels, and foreground and background pixel counts.
        """
        foreground_pixels = int(np.count_nonzero(mask))

        return MaskStatistics(
            shape=mask.shape,
            dtype=mask.dtype,
            minimum=mask.min(),
            maximum=mask.max(),
            unique_labels=self.unique_labels(mask),
            foreground_pixels=foreground_pixels,
            background_pixels=mask.size - foreground_pixels,
        )

    def is_empty_mask(self, mask: np.ndarray) -> bool:
        """Determine whether a segmentation mask has no foreground pixels.

        Parameters
        ----------
        mask : numpy.ndarray
            Segmentation mask containing background (zero) and foreground
            label values.

        Returns
        -------
        bool
            ``True`` if the mask contains no nonzero label values;
            otherwise, ``False``.
        """
        return not np.any(mask)

