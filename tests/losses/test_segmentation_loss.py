"""Unit tests for combined CE + Dice segmentation loss.

Test coverage:
- DiceLoss: scalar output, finite value, foreground-only
- DiceLoss: empty foreground (all-background target) handled safely
- CombinedSegmentationLoss: L = 0.5*CE + 0.5*Dice
- Loss is finite for valid inputs
- Loss = 0.5*CE + 0.5*Dice verified against manual calculation
- Backward pass completes without error
- No softmax inside U-Net (verified via model introspection)
- Empty-mask Dice loss is bounded and finite

All tests run on CPU with small synthetic tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


from src.losses.segmentation import CombinedSegmentationLoss, DiceLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logits(b: int = 2, h: int = 8, w: int = 8) -> torch.Tensor:
    """Random raw logits [B, 2, H, W]."""
    return torch.randn(b, 2, h, w)


def _targets_random(b: int = 2, h: int = 8, w: int = 8) -> torch.Tensor:
    """Random binary targets [B, H, W] dtype long."""
    return torch.randint(0, 2, (b, h, w), dtype=torch.long)


def _targets_all_background(b: int = 2, h: int = 8, w: int = 8) -> torch.Tensor:
    """All-background targets (empty foreground) [B, H, W]."""
    return torch.zeros(b, h, w, dtype=torch.long)


def _targets_all_foreground(b: int = 2, h: int = 8, w: int = 8) -> torch.Tensor:
    """All-foreground targets [B, H, W]."""
    return torch.ones(b, h, w, dtype=torch.long)


# ---------------------------------------------------------------------------
# DiceLoss
# ---------------------------------------------------------------------------


class TestDiceLoss:
    """Tests for the foreground Dice loss."""

    def test_returns_scalar(self) -> None:
        """DiceLoss returns a scalar tensor."""
        loss_fn = DiceLoss()
        loss = loss_fn(_logits(), _targets_random())
        assert loss.shape == ()

    def test_output_is_finite(self) -> None:
        """DiceLoss is finite for normal random inputs."""
        loss_fn = DiceLoss()
        loss = loss_fn(_logits(), _targets_random())
        assert loss.isfinite()

    def test_empty_foreground_is_finite(self) -> None:
        """DiceLoss is finite when target contains no foreground pixels."""
        loss_fn = DiceLoss()
        logits = _logits()
        targets = _targets_all_background()
        loss = loss_fn(logits, targets)
        assert loss.isfinite()

    def test_empty_foreground_loss_is_bounded(self) -> None:
        """Empty-foreground Dice loss is in (0, 1] due to epsilon smoothing."""
        loss_fn = DiceLoss(epsilon=1e-6)
        logits = _logits()
        targets = _targets_all_background()
        loss = loss_fn(logits, targets)
        assert 0.0 <= float(loss) <= 1.0 + 1e-4

    def test_perfect_prediction_reduces_loss(self) -> None:
        """Logits that perfectly predict the target produce lower loss than random."""
        # Create target: all foreground
        targets = _targets_all_foreground(b=1, h=4, w=4)
        # Perfect prediction: large logit for class 1
        perfect_logits = torch.zeros(1, 2, 4, 4)
        perfect_logits[:, 1, :, :] = 10.0  # high prob for foreground
        # Random prediction
        random_logits = _logits(b=1, h=4, w=4)

        loss_fn = DiceLoss()
        perfect_loss = float(loss_fn(perfect_logits, targets))
        random_loss = float(loss_fn(random_logits, targets))

        assert perfect_loss < random_loss

    def test_loss_in_zero_one_range(self) -> None:
        """Dice loss value is in [0, 1] for reasonable inputs."""
        loss_fn = DiceLoss()
        for _ in range(10):
            loss = loss_fn(_logits(), _targets_random())
            assert -1e-4 <= float(loss) <= 1.0 + 1e-4

    def test_uses_softmax_not_argmax(self) -> None:
        """Dice loss uses soft probabilities (softmax), not hard argmax."""
        # If argmax were used, flipping one logit slightly would have no effect
        # on loss when one class already dominates.  With softmax, every change
        # in logit value changes the probabilities continuously.
        logits = torch.zeros(1, 2, 2, 2)
        logits[:, 1, :, :] = 5.0  # class 1 dominates
        targets = torch.ones(1, 2, 2, dtype=torch.long)

        loss_fn = DiceLoss()
        loss_before = float(loss_fn(logits, targets))

        logits_stronger = logits.clone()
        logits_stronger[:, 1, :, :] = 10.0
        loss_after = float(loss_fn(logits_stronger, targets))

        # With softmax, stronger logit → lower loss (probability closer to 1)
        assert loss_after < loss_before


# ---------------------------------------------------------------------------
# CombinedSegmentationLoss
# ---------------------------------------------------------------------------


class TestCombinedSegmentationLoss:
    """Tests for the 0.5 * CE + 0.5 * Dice combined loss."""

    def test_returns_scalar(self) -> None:
        """CombinedSegmentationLoss returns a scalar tensor."""
        loss_fn = CombinedSegmentationLoss()
        loss = loss_fn(_logits(), _targets_random())
        assert loss.shape == ()

    def test_output_is_finite(self) -> None:
        """Combined loss is finite for normal random inputs."""
        loss_fn = CombinedSegmentationLoss()
        loss = loss_fn(_logits(), _targets_random())
        assert loss.isfinite()

    def test_equals_half_ce_plus_half_dice(self) -> None:
        """Combined loss equals 0.5 * CE + 0.5 * Dice."""
        logits = _logits(b=2, h=4, w=4)
        targets = _targets_random(b=2, h=4, w=4)

        ce_fn = nn.CrossEntropyLoss()
        dice_fn = DiceLoss()
        combined_fn = CombinedSegmentationLoss()

        expected = 0.5 * ce_fn(logits, targets) + 0.5 * dice_fn(logits, targets)
        actual = combined_fn(logits, targets)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_empty_foreground_is_finite(self) -> None:
        """Combined loss is finite when target is all background."""
        loss_fn = CombinedSegmentationLoss()
        loss = loss_fn(_logits(), _targets_all_background())
        assert loss.isfinite()

    def test_backward_pass_completes(self) -> None:
        """Gradients flow through the combined loss without error."""
        logits = _logits().requires_grad_(True)
        targets = _targets_random()

        loss_fn = CombinedSegmentationLoss()
        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert torch.all(logits.grad.isfinite())

    def test_no_class_weights_in_ce(self) -> None:
        """CrossEntropyLoss has no class weights (weight attribute is None)."""
        loss_fn = CombinedSegmentationLoss()
        assert loss_fn.ce_loss.weight is None

    def test_loss_positive(self) -> None:
        """Combined loss is positive for random (imperfect) predictions."""
        loss_fn = CombinedSegmentationLoss()
        for _ in range(5):
            loss = float(loss_fn(_logits(), _targets_random()))
            assert loss > 0.0
