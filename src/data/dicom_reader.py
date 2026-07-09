"""
DICOM Reader Module
===================

Purpose
-------
Reusable DICOM reader for the Medical Image Segmentation Framework.

Responsibilities
----------------
1. Read DICOM files
2. Read metadata
3. Read pixel arrays

Future Responsibilities
-----------------------
- Convert to Hounsfield Units
- Apply CT windowing
- Read complete volumes
- Handle MRI sequences

Author
------
Snehashis Ratna

Project
-------
Medical Image Segmentation Thesis
"""

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset


class DICOMReader:
    """
    Reusable DICOM Reader.

    This class provides a clean interface for reading
    DICOM images and metadata.
    """

    def read(self, dicom_path: Path) -> FileDataset:
        """
        Read a DICOM file.

        Parameters
        ----------
        dicom_path : Path
            Path to DICOM file.

        Returns
        -------
        FileDataset
            Loaded DICOM dataset.
        """

        if not dicom_path.exists():
            raise FileNotFoundError(
                f"DICOM file not found:\n{dicom_path}"
            )

        return pydicom.dcmread(dicom_path)

    def read_metadata(self, dicom_path: Path) -> dict[str, Any]:
        """
        Read important metadata from a DICOM file.
        """

        ds = self.read(dicom_path)

        metadata = {
            "patient_id": ds.get("PatientID", "Unknown"),
            "modality": ds.get("Modality", "Unknown"),
            "manufacturer": ds.get("Manufacturer", "Unknown"),
            "rows": ds.get("Rows"),
            "columns": ds.get("Columns"),
            "pixel_spacing": ds.get("PixelSpacing"),
            "slice_thickness": ds.get("SliceThickness"),
            "bits_allocated": ds.get("BitsAllocated"),
            "bits_stored": ds.get("BitsStored"),
            "rescale_slope": ds.get("RescaleSlope", 1),
            "rescale_intercept": ds.get("RescaleIntercept", 0),
        }

        return metadata

    def read_image(self, dicom_path: Path) -> np.ndarray:
        """
        Read image as NumPy array.

        Returns
        -------
        np.ndarray
        """

        ds = self.read(dicom_path)

        return ds.pixel_array.astype(np.float32)

    def image_statistics(self, image: np.ndarray) -> dict[str, float]:
        """
        Compute basic image statistics.
        """

        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min": float(image.min()),
            "max": float(image.max()),
            "mean": float(image.mean()),
            "std": float(image.std()),
        }