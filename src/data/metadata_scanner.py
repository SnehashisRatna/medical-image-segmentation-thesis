"""
Metadata Scanner
================

Purpose
-------
Scan the complete CHAOS dataset and generate
a structured metadata report.

Responsibilities
----------------
1. Scan CT patients
2. Scan MRI patients
3. Read metadata using DICOMReader
4. Generate dataset summary CSV

Author
------
Snehashis Ratna

Project
-------
Medical Image Segmentation Thesis
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dicom_reader import DICOMReader


class MetadataScanner:
    """
    Scan the CHAOS dataset and collect metadata.
    """

    def __init__(self, dataset_root: str) -> None:
        self.dataset_root: Path = Path(dataset_root)
        self.reader: DICOMReader = DICOMReader()
        self.records: list[dict[str, Any]] = []

    def scan(self) -> None:
        """
        Scan CT and MR datasets.
        """

        self.records = []

        self._scan_ct()

        self._scan_mr()

    def _scan_ct(self) -> None:
        """
        Scan CT patient folders and collect volume metadata.
        """

        ct_root = self.dataset_root / "CT"

        if not ct_root.exists():
            return

        for patient in sorted(ct_root.iterdir()):

            if not patient.is_dir():
                continue

            dicom_folder = patient / "DICOM_anon"

            if not dicom_folder.exists():
                continue

            dicom_files = sorted(dicom_folder.glob("*.dcm"))

            if len(dicom_files) == 0:
                continue

            metadata = self.reader.read_metadata(dicom_files[0])

            self.records.append(
                {
                    "Patient": metadata["patient_id"] or patient.name,
                    "Modality": "CT",
                    "Sequence": "CT",
                    "Slices": len(dicom_files),
                    "Rows": metadata["rows"],
                    "Columns": metadata["columns"],
                    "Pixel Spacing": metadata["pixel_spacing"],
                    "Slice Thickness": metadata["slice_thickness"],
                    "Bits Stored": metadata["bits_stored"],
                    "Rescale Slope": metadata["rescale_slope"],
                    "Rescale Intercept": metadata["rescale_intercept"],
                }
            )

    def _scan_mr(self) -> None:
        """Scan MR patient folders when MR sequence handling is implemented."""

        # TODO: CHAOS MR patients contain T1DUAL/DICOM_anon/InPhase and
        # T1DUAL/DICOM_anon/OutPhase, plus T2SPIR/DICOM_anon. Scan each
        # sequence separately and record its metadata and slice count.
        pass

    def to_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(self.records)

    def save_csv(self, output_path: str) -> None:

        output = Path(output_path)

        output.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()

        df.to_csv(output, index=False)

        print(f"\nDataset summary saved to:\n{output}")
