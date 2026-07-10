"""
Dataset Explorer for the CHAOS Dataset

Purpose
-------
Explore the CHAOS dataset structure and verify that the dataset
is correctly organized before preprocessing and training.

Current Features
----------------
1. Count CT patients
2. Count MR patients
3. Read metadata from the first CT DICOM file

Author: Snehashis Ratna
Project: Medical Image Segmentation Thesis
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.data.dicom_reader import DICOMReader

class DatasetExplorer:
    """
    Explore the CHAOS dataset.
    """

    def __init__(self, dataset_root: str):
        """
        Parameters
        ----------
        dataset_root : str
            Path to Train_Sets folder.
        """

        self.dataset_root = Path(dataset_root)

        self.ct_path = self.dataset_root / "CT"
        self.mr_path = self.dataset_root / "MR"

        self.reader = DICOMReader()

    def count_patients(self, modality_path: Path) -> int:
        """
        Count patient folders.

        Parameters
        ----------
        modality_path : Path

        Returns
        -------
        int
        """

        if not modality_path.exists():
            return 0

        return len(
            [
                folder
                for folder in modality_path.iterdir()
                if folder.is_dir()
            ]
        )

    def print_summary(self):
        """
        Print dataset summary.
        """

        ct_patients = self.count_patients(self.ct_path)
        mr_patients = self.count_patients(self.mr_path)

        print("\n" + "=" * 60)
        print("CHAOS DATASET SUMMARY")
        print("=" * 60)

        print(f"Dataset Root : {self.dataset_root}")
        print()

        print(f"CT Patients  : {ct_patients}")
        print(f"MR Patients  : {mr_patients}")
        print(f"Total        : {ct_patients + mr_patients}")

        print("=" * 60)

    def read_first_ct_metadata(self):
        """
        Read metadata from the first CT DICOM image.
        """

        if not self.ct_path.exists():
            print("\nCT folder not found.")
            return

        # -----------------------------
        # First patient
        # -----------------------------
        patient_folders = sorted(
            [
                folder
                for folder in self.ct_path.iterdir()
                if folder.is_dir()
            ]
        )

        if len(patient_folders) == 0:
            print("\nNo CT patients found.")
            return

        first_patient = patient_folders[0]

        # -----------------------------
        # DICOM folder
        # -----------------------------
        dicom_folder = first_patient / "DICOM_anon"

        if not dicom_folder.exists():
            print("\nDICOM_anon folder not found.")
            return

        # -----------------------------
        # First DICOM file
        # -----------------------------
        dicom_files = sorted(dicom_folder.glob("*.dcm"))

        if len(dicom_files) == 0:
            print("\nNo DICOM files found.")
            return

        first_dicom = dicom_files[0]

        # -----------------------------
        # Read metadata
        # -----------------------------
        metadata = self.reader.read_metadata(first_dicom)

        print("\n" + "=" * 60)
        print("FIRST CT DICOM METADATA")
        print("=" * 60)

        print(f"Patient Folder      : {first_patient.name}")
        print(f"DICOM File          : {first_dicom.name}")
        print()

        print(f"Modality            : {metadata['modality']}")
        print(f"Manufacturer        : {metadata['manufacturer']}")

        print()

        print(f"Rows                : {metadata['rows']}")
        print(f"Columns             : {metadata['columns']}")

        print()

        print(f"Pixel Spacing (mm)  : {metadata['pixel_spacing']}")
        print(f"Slice Thickness(mm) : {metadata['slice_thickness']}")

        print()

        print(f"Bits Allocated      : {metadata['bits_allocated']}")
        print(f"Bits Stored         : {metadata['bits_stored']}")

        print()

        print(f"Rescale Slope       : {metadata['rescale_slope']}")
        print(f"Rescale Intercept   : {metadata['rescale_intercept']}")

        print("=" * 60)

    def visualize_first_ct_slice(self):
        """
        Read and visualize the first CT slice.
        """

        patient_folders = sorted(
        folder
        for folder in self.ct_path.iterdir()
        if folder.is_dir()
        )

        first_patient = patient_folders[0]

        dicom_folder = first_patient / "DICOM_anon"

        first_dicom = sorted(dicom_folder.glob("*.dcm"))[0]

        image = self.reader.read_image(first_dicom)

        stats = self.reader.image_statistics(image)

        print("\n" + "=" * 60)
        print("IMAGE INFORMATION")
        print("=" * 60)

        print(f"Shape       : {stats['shape']}")
        print(f"Data Type   : {stats['dtype']}")
        print(f"Minimum     : {stats['min']}")
        print(f"Maximum     : {stats['max']}")
        print(f"Mean        : {stats['mean']:.2f}")
        print(f"Std Dev     : {stats['std']:.2f}")

        print("=" * 60)

        plt.figure(figsize=(6, 6))

        plt.imshow(image, cmap="gray")

        plt.title(f"CT Patient {first_patient.name}")

        plt.axis("off")

        plt.show()



def main():

    explorer = DatasetExplorer(
        dataset_root="data/raw/CHAOS_Train_Sets/Train_Sets"
    )

    explorer.print_summary()

    explorer.read_first_ct_metadata()

    explorer.visualize_first_ct_slice()

if __name__ == "__main__":
    main()