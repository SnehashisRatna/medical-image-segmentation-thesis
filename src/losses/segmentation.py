"""Loss functions for binary liver segmentation.

Frozen loss formulation
-----------------------
    L_total = 0.5 × CrossEntropyLoss + 0.5 × ForegroundDiceLoss

* ``CrossEntropyLoss``  — standard 2-class CE, no class weights.
* ``ForegroundDiceLoss`` — softmax applied to raw logits; liver/foreground
  probability (class 1) compared with binary liver target; numerical
  epsilon for stability; empty-foreground masks handled safely.

The U-Net returns **raw logits** ``[B, 2, H, W]``.  Softmax is applied
inside this module; it must NOT be added to the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss on the foreground (liver) class.

    Applies softmax to raw logits, extracts the class-1 (liver) probability
    channel, and computes the Dice coefficient against the binary target.
    The returned loss is ``1 − mean_batch_Dice``.

    Parameters
    ----------
    epsilon : float, optional
        Numerical smoothing constant added to both numerator and
        denominator.  Keeps the loss well-defined for empty-foreground
        (all-background) slices.  Default ``1e-6``.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute foreground Dice loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model output, shape ``[B, 2, H, W]``.
        targets : torch.Tensor
            Binary ground-truth mask, shape ``[B, H, W]``, dtype
            ``torch.long``, values in ``{0, 1}``.

        Returns
        -------
        torch.Tensor
            Scalar loss value ``1 − mean_batch_Dice``.
        """
        probs = torch.softmax(logits, dim=1)   # [B, 2, H, W]
        foreground = probs[:, 1, :, :]         # [B, H, W]  liver probs
        targets_f = targets.float()            # [B, H, W]

        # Sum over spatial dimensions; keep batch axis for per-sample Dice
        intersection = (foreground * targets_f).sum(dim=(1, 2))
        cardinality = foreground.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2))

        dice_per_sample = (2.0 * intersection + self.epsilon) / (
            cardinality + self.epsilon
        )
        return 1.0 - dice_per_sample.mean()


class CombinedSegmentationLoss(nn.Module):
    """Combined cross-entropy and Dice loss for binary liver segmentation.

    Computes::

        L = 0.5 × CrossEntropyLoss + 0.5 × DiceLoss

    Parameters
    ----------
    epsilon : float, optional
        Epsilon forwarded to :class:`DiceLoss`.  Default ``1e-6``.

    Notes
    -----
    * No class weights are applied to the cross-entropy term.
    * Softmax is applied inside ``DiceLoss``; the U-Net must continue
      to return raw logits with no internal activation.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(epsilon=epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined segmentation loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model output, shape ``[B, 2, H, W]``.
        targets : torch.Tensor
            Binary ground-truth mask, shape ``[B, H, W]``, dtype
            ``torch.long``, values in ``{0, 1}``.

        Returns
        -------
        torch.Tensor
            Scalar combined loss.
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return 0.5 * ce + 0.5 * dice
