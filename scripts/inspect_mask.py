"""
Inspect Ground Truth Mask

Purpose
-------
Inspect a single segmentation mask from the CHAOS dataset.

This script helps us understand:

1. Image size
2. Data type
3. Pixel value range
4. Unique label values
5. Visual appearance

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


def main() -> None:
    """
    Inspect one ground truth mask.
    """

    mask_path = Path(
    "data/raw/CHAOS_Train_Sets/Train_Sets/CT/1/Ground/liver_GT_045.png"
    )

    # -----------------------------
    # Read PNG
    # -----------------------------
    print(type(Image.open(mask_path)))
    print(Image.open(mask_path).mode)
    mask = np.array(Image.open(mask_path))

    # -----------------------------
    # Print information
    # -----------------------------
    print("\n" + "=" * 60)
    print("GROUND TRUTH MASK INFORMATION")
    print("=" * 60)

    print(f"File           : {mask_path.name}")
    print(f"Shape          : {mask.shape}")
    print(f"Data Type      : {mask.dtype}")

    print(f"Minimum Value  : {mask.min()}")
    print(f"Maximum Value  : {mask.max()}")

    print(f"Unique Values  : {np.unique(mask)}")

    print("=" * 60)

    # -----------------------------
    # Display mask
    # -----------------------------
    plt.figure(figsize=(6, 6))

    plt.imshow(mask, cmap="gray")

    plt.title("CHAOS CT Ground Truth Mask")

    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()