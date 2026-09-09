"""Building blocks for the plain 2D U-Net baseline.

This module provides two block types used by the U-Net:

``DoubleConvBlock``
    Two consecutive Conv2d -> ReLU pairs with ``kernel_size=3`` and
    ``padding=1``.  Used in every encoder stage, the bottleneck, and every
    decoder stage.

``DecoderBlock``
    Transposed-convolution upsampling (``kernel_size=2``, ``stride=2``)
    followed by channel-wise concatenation with the matching encoder skip
    feature map and then a ``DoubleConvBlock``.

Design constraints (frozen specification)
-----------------------------------------
* No normalisation layers (no BatchNorm, LayerNorm, GroupNorm, InstanceNorm).
* No attention, residual, or transformer components.
* ReLU activation only.
* 3x3 convolutions use ``padding=1`` to preserve spatial dimensions.
* Upsampling uses ``ConvTranspose2d`` exclusively.
"""

import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    """Two consecutive Conv2d -> ReLU pairs.

    Each convolution uses ``kernel_size=3`` and ``padding=1`` so that the
    spatial dimensions of the feature map are unchanged.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels produced by both convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double-convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, in_channels, H, W]``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[B, out_channels, H, W]``.
        """
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsampling step of the U-Net decoder.

    Applies:

    1. ``ConvTranspose2d`` (``kernel_size=2``, ``stride=2``) to double the
       spatial dimensions **and** reduce channels from ``in_channels`` to
       ``out_channels``.
    2. Concatenation of the upsampled tensor with the corresponding encoder
       skip-connection feature map along the channel dimension, giving
       ``out_channels + skip_channels`` total channels.
    3. A ``DoubleConvBlock`` to mix the concatenated channels back down to
       ``out_channels``.

    Channel flow example for the first decoder stage::

        1024                          # from bottleneck
          -> ConvTranspose2d -> 512   # channel reduction + spatial doubling
          -> cat(512 encoder skip)    # 512 + 512 = 1024
          -> DoubleConvBlock -> 512   # back to out_channels

    Parameters
    ----------
    in_channels : int
        Number of channels entering the transposed convolution (from the
        previous decoder stage or the bottleneck).
    skip_channels : int
        Number of channels in the encoder skip-connection feature map.
    out_channels : int
        Number of output channels produced by both the transposed convolution
        and the final ``DoubleConvBlock``.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConvBlock(out_channels + skip_channels, out_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        """Apply the decoder step.

        Parameters
        ----------
        x : torch.Tensor
            Tensor from the previous decoder stage or bottleneck, shape
            ``[B, in_channels, H, W]``.
        skip : torch.Tensor
            Encoder skip-connection feature map, shape
            ``[B, skip_channels, 2*H, 2*W]``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[B, out_channels, 2*H, 2*W]``.
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
