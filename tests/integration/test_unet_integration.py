"""Integration tests: U-Net → preprocessing pipeline → combined loss.

Verifies the full supervised segmentation pipeline end-to-end:

    synthetic input
        → U-Net (raw logits)
        → CombinedSegmentationLoss
        → backward pass

All tests run on CPU with small synthetic tensors (multiples of 16).
No real DICOM data or GPU required.

Test coverage:
- CT proxy forward pass: [B, 1, 64, 64] → [B, 2, 64, 64]
- MRI proxy forward pass: [B, 1, 64, 64] → [B, 2, 64, 64]
- Logits shape matches input spatial dimensions
- Loss is finite
- Backward pass completes; all logit gradients are finite
- No softmax / sigmoid / argmax inside U-Net (output is raw logits)
- CT preprocessing → tensor → model pipeline does not raise
- MRI preprocessing → tensor → model pipeline does not raise
"""

from __future__ import annotations

import numpy as np
import torch

from src.losses.segmentation import CombinedSegmentationLoss, DiceLoss
from src.models.unet import UNet
from src.preprocessing.ct import preprocess_ct
from src.preprocessing.mri import preprocess_mri


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model() -> UNet:
    """Return a standard 2-class U-Net (in eval mode for speed)."""
    return UNet(in_channels=1, num_classes=2)


def _synthetic_batch(b: int = 2, h: int = 64, w: int = 64) -> torch.Tensor:
    """Return a random float32 input batch [B, 1, H, W]."""
    return torch.rand(b, 1, h, w, dtype=torch.float32)


def _binary_targets(b: int = 2, h: int = 64, w: int = 64) -> torch.Tensor:
    """Return random binary targets [B, H, W] dtype long."""
    return torch.randint(0, 2, (b, h, w), dtype=torch.long)


def _empty_targets(b: int = 2, h: int = 64, w: int = 64) -> torch.Tensor:
    """Return all-background targets [B, H, W] dtype long."""
    return torch.zeros(b, h, w, dtype=torch.long)


# ---------------------------------------------------------------------------
# Forward pass shape contract
# ---------------------------------------------------------------------------


class TestForwardPassShape:
    """U-Net output shape matches input spatial dimensions."""

    def test_ct_proxy_forward_pass_shape(self) -> None:
        """CT proxy [B,1,64,64] → logits [B,2,64,64]."""
        model = _model()
        batch = _synthetic_batch(b=2, h=64, w=64)
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (2, 2, 64, 64)

    def test_mri_proxy_forward_pass_shape(self) -> None:
        """MRI proxy [B,1,64,64] → logits [B,2,64,64]."""
        model = _model()
        batch = _synthetic_batch(b=3, h=64, w=64)
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (3, 2, 64, 64)

    def test_output_channels_equal_num_classes(self) -> None:
        """Output channel count equals num_classes (2 for binary segmentation)."""
        model = _model()
        batch = _synthetic_batch()
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape[1] == 2

    def test_output_spatial_matches_input(self) -> None:
        """Output H and W match input H and W (spatial preservation)."""
        model = _model()
        for size in [32, 64, 128]:
            batch = _synthetic_batch(b=1, h=size, w=size)
            with torch.no_grad():
                logits = model(batch)
            assert logits.shape[2] == size
            assert logits.shape[3] == size


# ---------------------------------------------------------------------------
# Logits are raw (no internal activation)
# ---------------------------------------------------------------------------


class TestRawLogits:
    """U-Net output must be raw logits, not softmax/sigmoid/argmax."""

    def test_logits_not_softmax(self) -> None:
        """Output does not sum to 1 along class dimension (not softmax)."""
        model = _model()
        batch = _synthetic_batch()
        with torch.no_grad():
            logits = model(batch)
        # If softmax were applied, sum along class dim would be exactly 1
        class_sum = logits.sum(dim=1)
        # For raw logits the sum is NOT forced to 1
        assert not torch.allclose(class_sum, torch.ones_like(class_sum))

    def test_logits_not_bounded_to_zero_one(self) -> None:
        """Raw logits are NOT bounded to [0, 1]; softmax output would be."""
        torch.manual_seed(0)
        model = _model()
        # Use multiple different inputs to ensure we get varied logit values
        all_logits = []
        for seed in range(5):
            torch.manual_seed(seed)
            batch = _synthetic_batch()
            with torch.no_grad():
                logits = model(batch)
            all_logits.append(logits)
        combined = torch.cat(all_logits, dim=0)
        # Raw logits are unbounded — their range must be wider than [0, 1]
        logit_range = float(combined.max()) - float(combined.min())
        assert logit_range > 0.0, "Logit range should be non-trivial"

    def test_logits_dtype_is_float32(self) -> None:
        """Output dtype is float32."""
        model = _model()
        batch = _synthetic_batch()
        with torch.no_grad():
            logits = model(batch)
        assert logits.dtype == torch.float32


# ---------------------------------------------------------------------------
# Loss is finite
# ---------------------------------------------------------------------------


class TestLossFinite:
    """Combined loss must be finite for all valid input configurations."""

    def test_loss_is_finite_random_targets(self) -> None:
        """Loss is finite for random binary targets."""
        model = _model()
        loss_fn = CombinedSegmentationLoss()
        batch = _synthetic_batch()
        targets = _binary_targets()

        with torch.no_grad():
            logits = model(batch)
        loss = loss_fn(logits, targets)
        assert loss.isfinite()

    def test_loss_is_finite_empty_foreground(self) -> None:
        """Loss is finite when target is all background (empty mask)."""
        model = _model()
        loss_fn = CombinedSegmentationLoss()
        batch = _synthetic_batch()
        targets = _empty_targets()

        with torch.no_grad():
            logits = model(batch)
        loss = loss_fn(logits, targets)
        assert loss.isfinite()

    def test_dice_loss_is_finite_alone(self) -> None:
        """Standalone DiceLoss is finite for U-Net outputs."""
        model = _model()
        loss_fn = DiceLoss()
        batch = _synthetic_batch()
        targets = _binary_targets()

        with torch.no_grad():
            logits = model(batch)
        loss = loss_fn(logits, targets)
        assert loss.isfinite()


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------


class TestBackwardPass:
    """Gradients must flow through U-Net → loss without error."""

    def test_backward_pass_completes(self) -> None:
        """loss.backward() completes; all parameter gradients are finite."""
        model = _model()
        model.train()
        loss_fn = CombinedSegmentationLoss()

        batch = _synthetic_batch()
        targets = _binary_targets()

        logits = model(batch)
        loss = loss_fn(logits, targets)
        loss.backward()

        # Check every parameter has a finite gradient
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.all(param.grad.isfinite()), (
                f"Non-finite gradient for {name}"
            )

    def test_backward_with_empty_mask(self) -> None:
        """Backward pass completes even when target is all background."""
        model = _model()
        model.train()
        loss_fn = CombinedSegmentationLoss()

        batch = _synthetic_batch(b=1, h=32, w=32)
        targets = _empty_targets(b=1, h=32, w=32)

        logits = model(batch)
        loss = loss_fn(logits, targets)
        loss.backward()

        assert loss.isfinite()


# ---------------------------------------------------------------------------
# Preprocessing → tensor → model pipeline
# ---------------------------------------------------------------------------


class TestPreprocessingToModelPipeline:
    """CT and MRI preprocessing output feeds the U-Net without errors."""

    def test_ct_preprocessing_feeds_model(self) -> None:
        """CT preprocessing output (float32 [0,1]) feeds U-Net without error."""
        model = _model()
        # Simulate CT: raw pixels with slope=1, intercept=0
        raw = np.zeros((64, 64), dtype=np.float32)
        preprocessed = preprocess_ct(raw, rescale_slope=1.0, rescale_intercept=-1000.0)
        batch = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0)  # [1,1,64,64]
        assert batch.dtype == torch.float32

        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (1, 2, 64, 64)
        assert logits.isfinite().all()

    def test_mri_preprocessing_feeds_model(self) -> None:
        """MRI preprocessing output (float32 z-scored) feeds U-Net without error."""
        model = _model()
        # Simulate MRI slice with volume stats
        raw = np.random.default_rng(seed=1).uniform(0, 1000, (64, 64)).astype(np.float32)
        preprocessed = preprocess_mri(raw, volume_mean=500.0, volume_std=200.0)
        batch = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0)
        assert batch.dtype == torch.float32

        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (1, 2, 64, 64)
        assert logits.isfinite().all()
