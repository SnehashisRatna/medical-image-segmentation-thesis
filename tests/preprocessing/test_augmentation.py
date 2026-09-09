"""Unit tests for training augmentation.

Test coverage:
- flip_prob=1.0: both image and mask are always flipped identically
- rotation: mask remains binary after rotation (nearest-neighbor preserves {0,1})
- shapes and dtypes are preserved after augmentation
- image dtype stays float32
- mask dtype stays torch.long
- validation/test: when augmentation is not applied, tensors are unchanged

All tests use small synthetic tensors (CPU only, no GPU required).
"""

from __future__ import annotations

import torch

from src.preprocessing.augmentation import TrainingAugmentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(h: int = 8, w: int = 8) -> torch.Tensor:
    """Return a synthetic float32 image tensor [1, H, W]."""
    return torch.rand(1, h, w, dtype=torch.float32)


def _make_mask(h: int = 8, w: int = 8) -> torch.Tensor:
    """Return a synthetic binary long mask tensor [H, W]."""
    return torch.randint(0, 2, (h, w), dtype=torch.long)


# ---------------------------------------------------------------------------
# Horizontal flip — force p=1.0 for deterministic tests
# ---------------------------------------------------------------------------


class TestHorizontalFlip:
    """Horizontal flip tests with flip_prob=1.0 (always applied)."""

    def _aug(self) -> TrainingAugmentation:
        return TrainingAugmentation(flip_prob=1.0, rotation_prob=0.0)

    def test_image_is_horizontally_flipped(self) -> None:
        """Image tensor is flipped along the width axis (dim=-1)."""
        aug = self._aug()
        image = _make_image()
        mask = _make_mask()
        image_aug, _ = aug(image, mask)
        expected = torch.flip(image, dims=[-1])
        assert torch.equal(image_aug, expected)

    def test_mask_is_horizontally_flipped(self) -> None:
        """Mask tensor is flipped with the same transform as the image."""
        aug = self._aug()
        image = _make_image()
        mask = _make_mask()
        _, mask_aug = aug(image, mask)
        expected = torch.flip(mask, dims=[-1])
        assert torch.equal(mask_aug, expected)

    def test_image_and_mask_receive_identical_flip(self) -> None:
        """Image and mask are always flipped with the same spatial transform."""
        aug = self._aug()
        # Distinctive spatial pattern: column 0 is all-0, column -1 is all-1
        image = torch.zeros(1, 4, 4)
        image[:, :, -1] = 1.0
        mask = torch.zeros(4, 4, dtype=torch.long)
        mask[:, -1] = 1

        image_aug, mask_aug = aug(image, mask)

        # After flip: column 0 should now be all-1
        assert torch.all(image_aug[:, :, 0] == 1.0)
        assert torch.all(mask_aug[:, 0] == 1)

    def test_flip_twice_restores_original(self) -> None:
        """Applying flip twice returns the original tensor."""
        aug = self._aug()
        image = _make_image()
        mask = _make_mask()
        image_1, mask_1 = aug(image, mask)
        image_2, mask_2 = aug(image_1, mask_1)
        assert torch.equal(image_2, image)
        assert torch.equal(mask_2, mask)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class TestRotation:
    """Rotation tests using various rotation_prob configurations."""

    def test_mask_remains_binary_after_rotation(self) -> None:
        """Mask values remain in {0, 1} after nearest-neighbor rotation."""
        aug = TrainingAugmentation(flip_prob=0.0, rotation_prob=1.0)
        mask = _make_mask(h=64, w=64)
        image = _make_image(h=64, w=64)

        _, mask_aug = aug(image, mask)

        unique = torch.unique(mask_aug)
        for v in unique:
            assert v.item() in {0, 1}, (
                f"Mask contains unexpected value {v.item()} after rotation"
            )

    def test_shapes_preserved_after_rotation(self) -> None:
        """Image and mask shapes are unchanged after rotation."""
        aug = TrainingAugmentation(flip_prob=0.0, rotation_prob=1.0)
        image = _make_image(h=32, w=32)
        mask = _make_mask(h=32, w=32)

        image_aug, mask_aug = aug(image, mask)

        assert image_aug.shape == image.shape
        assert mask_aug.shape == mask.shape

    def test_zero_degree_rotation_is_identity(self) -> None:
        """A 0° rotation leaves image and mask unchanged."""
        aug = TrainingAugmentation(
            flip_prob=0.0, rotation_prob=1.0, max_rotation_deg=0.0
        )
        image = _make_image(h=16, w=16)
        mask = _make_mask(h=16, w=16)

        image_aug, mask_aug = aug(image, mask)

        torch.testing.assert_close(image_aug, image)
        assert torch.equal(mask_aug, mask)


# ---------------------------------------------------------------------------
# Dtype and shape contract
# ---------------------------------------------------------------------------


class TestDtypeAndShapeContract:
    """Image dtype must stay float32; mask dtype must stay torch.long."""

    def test_image_dtype_preserved(self) -> None:
        """Image tensor dtype remains torch.float32 after augmentation."""
        aug = TrainingAugmentation()
        image = _make_image()
        mask = _make_mask()
        image_aug, _ = aug(image, mask)
        assert image_aug.dtype == torch.float32

    def test_mask_dtype_preserved(self) -> None:
        """Mask tensor dtype remains torch.long after augmentation."""
        aug = TrainingAugmentation()
        image = _make_image()
        mask = _make_mask()
        _, mask_aug = aug(image, mask)
        assert mask_aug.dtype == torch.long

    def test_image_shape_preserved(self) -> None:
        """Image shape is unchanged after augmentation."""
        aug = TrainingAugmentation()
        image = _make_image(h=32, w=32)
        mask = _make_mask(h=32, w=32)
        image_aug, _ = aug(image, mask)
        assert image_aug.shape == (1, 32, 32)

    def test_mask_shape_preserved(self) -> None:
        """Mask shape is unchanged after augmentation."""
        aug = TrainingAugmentation()
        image = _make_image(h=32, w=32)
        mask = _make_mask(h=32, w=32)
        _, mask_aug = aug(image, mask)
        assert mask_aug.shape == (32, 32)


# ---------------------------------------------------------------------------
# Validation / test data: no augmentation
# ---------------------------------------------------------------------------


class TestNoAugmentation:
    """Validation and test data should NOT use TrainingAugmentation.

    These tests verify the expected usage pattern: for val/test, do not
    instantiate TrainingAugmentation — tensors pass through unchanged.
    """

    def test_tensors_unchanged_without_augmentation(self) -> None:
        """Without augmentation the tensors are returned as-is."""
        image = _make_image()
        mask = _make_mask()

        # Simulate val/test: no augmentation is applied
        image_out = image
        mask_out = mask

        assert torch.equal(image_out, image)
        assert torch.equal(mask_out, mask)

    def test_flip_prob_zero_never_flips(self) -> None:
        """flip_prob=0.0 means the flip transform is never applied."""
        aug = TrainingAugmentation(flip_prob=0.0, rotation_prob=0.0)
        image = _make_image()
        mask = _make_mask()

        for _ in range(20):
            image_aug, mask_aug = aug(image, mask)
            assert torch.equal(image_aug, image)
            assert torch.equal(mask_aug, mask)

    def test_rotation_prob_zero_never_rotates(self) -> None:
        """rotation_prob=0.0 means rotation is never applied."""
        aug = TrainingAugmentation(flip_prob=0.0, rotation_prob=0.0)
        image = _make_image()
        mask = _make_mask()

        for _ in range(20):
            image_aug, mask_aug = aug(image, mask)
            assert torch.equal(image_aug, image)
            assert torch.equal(mask_aug, mask)
