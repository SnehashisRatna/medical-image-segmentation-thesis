"""Loss functions for medical image segmentation."""

from src.losses.segmentation import CombinedSegmentationLoss, DiceLoss

__all__ = ["DiceLoss", "CombinedSegmentationLoss"]
