"""Plain 2D U-Net baseline for medical image segmentation.

Architecture summary
--------------------
Encoder  : 4 stages (64, 128, 256, 512 channels).
           Each stage: Conv2d(3x3, pad=1) -> ReLU -> Conv2d(3x3, pad=1) -> ReLU,
           then MaxPool2d(2x2).
Bottleneck: 512 -> 1024 channels, same double-conv structure.
Decoder  : 4 stages (512, 256, 128, 64 channels).
           Each stage: ConvTranspose2d(2x2, stride=2) -> concat skip -> DoubleConv.
Output   : Conv2d(1x1), in=64, out=num_classes.  Raw logits, no activation.

Design constraints (frozen specification)
-----------------------------------------
* Input:  [B, 1, H, W]   (single-channel, H and W divisible by 16)
* Output: [B, num_classes, H, W]  -- raw logits
* No BatchNorm / LayerNorm / GroupNorm / InstanceNorm.
* No attention, residual, transformer, or patch-embedding components.
* ReLU activation after every 3x3 convolution.
* Skip connections use concatenation (not addition).
* Upsampling uses ConvTranspose2d (not bilinear/nearest interpolation).
* Padding=1 on every 3x3 convolution preserves spatial dimensions.
"""

import torch
import torch.nn as nn

from src.models.unet.blocks import DecoderBlock, DoubleConvBlock


class UNet(nn.Module):
    """Plain 2D U-Net for medical image segmentation.

    Implements the encoder-bottleneck-decoder architecture with four
    skip connections using concatenation.  Returns raw logits suitable
    for use with any external loss function.

    Parameters
    ----------
    in_channels : int, optional
        Number of input image channels.  Default is ``1`` (grayscale / single
        modality slice).
    num_classes : int
        Number of segmentation classes (controls the output channel count).

    Examples
    --------
    >>> model = UNet(in_channels=1, num_classes=2)
    >>> x = torch.zeros(1, 1, 256, 256)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([1, 2, 256, 256])
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder1 = DoubleConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        self.bottleneck = DoubleConvBlock(512, 1024)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        self.decoder4 = DecoderBlock(in_channels=1024, skip_channels=512, out_channels=512)
        self.decoder3 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.decoder2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder1 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # ------------------------------------------------------------------
        # Final 1x1 projection to class logits
        # ------------------------------------------------------------------
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``[B, in_channels, H, W]``.  ``H`` and ``W``
            must each be divisible by 16 to ensure that the four MaxPool
            operations and four transposed-convolution upsamplings align
            perfectly.

        Returns
        -------
        torch.Tensor
            Raw logit tensor of shape ``[B, num_classes, H, W]``.
            No activation is applied; call ``softmax`` or ``sigmoid``
            externally if required.
        """
        # Encoder
        x1 = self.encoder1(x)
        p1 = self.pool1(x1)

        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)

        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)

        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder (each stage upsample -> concat skip -> double-conv)
        d4 = self.decoder4(b, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        # Output projection
        return self.final_conv(d1)
