"""Plain 2D U-Net baseline package.

Exports
-------
UNet
    The plain 2D U-Net model for medical image segmentation.
"""

from src.models.unet.model import UNet

__all__ = ["UNet"]
