"""
Test GroundTruthReader

Purpose
-------
Verify that GroundTruthReader correctly reads
segmentation masks from the CHAOS dataset.

Author
------
Snehashis Ratna

Project
-------
Medical Image Segmentation Thesis
"""

from src.data.ground_truth_reader import GroundTruthReader


def main() -> None:
    """
    Test the read_mask() method.
    """

    reader = GroundTruthReader()

    mask = reader.read_mask(
        "data/raw/CHAOS_Train_Sets/Train_Sets/CT/1/Ground/liver_GT_045.png"
    )

    labels = reader.unique_labels(mask)

    stats = reader.mask_statistics(mask)

    print("\n" + "=" * 60)
    print("MASK STATISTICS")
    print("=" * 60)

    print(f"Shape               : {stats.shape}")
    print(f"Data Type           : {stats.dtype}")
    print(f"Minimum             : {stats.minimum}")
    print(f"Maximum             : {stats.maximum}")
    print(f"Unique Labels       : {stats.unique_labels}")
    print(f"Foreground Pixels   : {stats.foreground_pixels}")
    print(f"Background Pixels   : {stats.background_pixels}")

    print("=" * 60)

    print(f"Unique Labels : {labels}")

    print("\n" + "=" * 60)
    print("GROUND TRUTH READER TEST")
    print("=" * 60)

    print(f"Type          : {type(mask)}")
    print(f"Shape         : {mask.shape}")
    print(f"Data Type     : {mask.dtype}")
    print(f"Minimum Value : {mask.min()}")
    print(f"Maximum Value : {mask.max()}")

    print("=" * 60)

    print("\n" + "=" * 60)
    print("EMPTY MASK TEST")
    print("=" * 60)

    print(f"Is Empty Mask : {reader.is_empty_mask(mask)}")

    print("=" * 60)


if __name__ == "__main__":
    main()