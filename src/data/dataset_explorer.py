"""
Dataset Explorer for the CHAOS Dataset

Purpose:
--------
Explore the CHAOS dataset structure and collect metadata.

Author: Snehashis Ratna
Project: Medical Image Segmentation Thesis
"""

from pathlib import Path


class DatasetExplorer:

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)

        self.ct_path = self.dataset_root / "CT"
        self.mr_path = self.dataset_root / "MR"

    def count_patients(self, modality_path: Path) -> int:
        """
        Count patient folders inside a modality.
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

        ct_patients = self.count_patients(self.ct_path)
        mr_patients = self.count_patients(self.mr_path)

        print("=" * 50)
        print("CHAOS DATASET SUMMARY")
        print("=" * 50)

        print(f"Dataset Root : {self.dataset_root}")
        print()

        print(f"CT Patients  : {ct_patients}")
        print(f"MR Patients  : {mr_patients}")
        print(f"Total        : {ct_patients + mr_patients}")

        print("=" * 50)


if __name__ == "__main__":

    explorer = DatasetExplorer(
        dataset_root="data/raw/CHAOS_Train_Sets/Train_Sets"
    )

    explorer.print_summary()