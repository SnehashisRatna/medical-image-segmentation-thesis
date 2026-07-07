"""
Dataset Explorer for the CHAOS Dataset

Purpose:
--------
Explore the CHAOS dataset structure and collect metadata.

This module is intentionally independent from the dataset loader.
It helps verify dataset integrity before preprocessing or training.

Author: Snehashis Ratna
Project: Medical Image Segmentation Thesis
"""

from pathlib import Path


class DatasetExplorer:
    """
    Explore the CHAOS dataset directory structure.
    """

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)

    def print_structure(self):
        """
        Print the top-level directory structure.
        """

        print(f"\nDataset Root: {self.dataset_root}\n")

        for item in sorted(self.dataset_root.iterdir()):
            print(item.name)


if __name__ == "__main__":

    dataset = DatasetExplorer(
        dataset_root="data/raw/CHAOS_Train_Sets/Train_Sets"
    )

    dataset.print_structure()