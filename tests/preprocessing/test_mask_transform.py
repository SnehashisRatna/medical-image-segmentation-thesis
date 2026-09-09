"""Unit tests for binary liver mask preprocessing.

Tests the frozen CHAOS label → binary target mapping:

MRI:
    0   → 0 (background)
    63  → 1 (liver)
    126 → 0 (right kidney)
    189 → 0 (left kidney)
    252 → 0 (spleen)

CT:
    0        → 0 (background)
    non-zero → 1 (liver)

All tests use synthetic numpy arrays. No real DICOM or mask files required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.enums import Modality
from src.preprocessing.mask import binarize_liver_mask


# ---------------------------------------------------------------------------
# MRI label mapping
# ---------------------------------------------------------------------------


class TestMriLabelMapping:
    """Frozen CHAOS MRI raw label → binary liver target."""

    def test_mri_background_zero_maps_to_zero(self) -> None:
        """MRI label 0 (background) maps to 0."""
        mask = np.array([[0]], dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.MRI)
        assert int(result[0, 0]) == 0

    def test_mri_label_63_maps_to_liver(self) -> None:
        """MRI label 63 (liver) maps to 1."""
        mask = np.array([[63]], dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.MRI)
        assert int(result[0, 0]) == 1

    def test_mri_label_126_maps_to_background(self) -> None:
        """MRI label 126 (right kidney) maps to 0 — non-liver background."""
        mask = np.array([[126]], dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.MRI)
        assert int(result[0, 0]) == 0

    def test_mri_label_189_maps_to_background(self) -> None:
        """MRI label 189 (left kidney) maps to 0 — non-liver background."""
        mask = np.array([[189]], dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.MRI)
        assert int(result[0, 0]) == 0

    def test_mri_label_252_maps_to_background(self) -> None:
        """MRI label 252 (spleen) maps to 0 — non-liver background."""
        mask = np.array([[252]], dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.MRI)
        assert int(result[0, 0]) == 0

    def test_mri_all_labels_in_one_mask(self) -> None:
        """Full CHAOS MRI label set maps correctly in a single mask."""
        mask = np.array([[0, 63, 126, 189, 252]], dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.MRI)
        expected = np.array([[0, 1, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_mri_empty_mask_all_background(self) -> None:
        """An all-zero MRI mask (no organ present) produces all zeros."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.MRI)
        assert int(result.sum()) == 0

    def test_mri_liver_only_mask(self) -> None:
        """An all-liver MRI mask produces all ones."""
        mask = np.full((4, 4), fill_value=63, dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.MRI)
        assert int(result.sum()) == 16


# ---------------------------------------------------------------------------
# CT label mapping
# ---------------------------------------------------------------------------


class TestCtLabelMapping:
    """CT ground truth → binary liver target."""

    def test_ct_zero_maps_to_background(self) -> None:
        """CT label 0 (background) maps to 0."""
        mask = np.array([[0]], dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.CT)
        assert int(result[0, 0]) == 0

    def test_ct_nonzero_maps_to_liver(self) -> None:
        """Any non-zero CT pixel maps to 1 (liver)."""
        for val in [1, 127, 255]:
            mask = np.array([[val]], dtype=np.uint8)
            result = binarize_liver_mask(mask, Modality.CT)
            assert int(result[0, 0]) == 1, f"Expected 1 for CT pixel value {val}"

    def test_ct_boolean_mask_false_maps_to_zero(self) -> None:
        """Boolean False (0) maps to 0."""
        mask = np.array([[False, True]], dtype=bool)
        result = binarize_liver_mask(mask, Modality.CT)
        assert int(result[0, 0]) == 0
        assert int(result[0, 1]) == 1

    def test_ct_empty_mask_all_background(self) -> None:
        """An all-zero CT mask produces all zeros."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = binarize_liver_mask(mask, Modality.CT)
        assert int(result.sum()) == 0


# ---------------------------------------------------------------------------
# Output contract (both modalities)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modality", [Modality.CT, Modality.MRI])
def test_output_dtype_is_uint8(modality: Modality) -> None:
    """Output dtype is uint8 for both modalities."""
    mask = np.array([[0, 63]], dtype=np.uint8)
    result = binarize_liver_mask(mask, modality)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("modality", [Modality.CT, Modality.MRI])
def test_output_values_only_zero_and_one(modality: Modality) -> None:
    """Output contains only values in {0, 1} for both modalities."""
    # MRI: all known CHAOS labels
    mask = np.array([[0, 63, 126, 189, 252]], dtype=np.uint8)
    result = binarize_liver_mask(mask, modality)
    unique = set(result.flatten().tolist())
    assert unique.issubset({0, 1})


@pytest.mark.parametrize("modality", [Modality.CT, Modality.MRI])
def test_output_shape_matches_input(modality: Modality) -> None:
    """Output spatial shape equals input shape for both modalities."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    result = binarize_liver_mask(mask, modality)
    assert result.shape == (256, 256)


def test_empty_mask_is_valid_input() -> None:
    """Empty masks (all-zero) are valid and do not raise errors."""
    empty_mask = np.zeros((128, 128), dtype=np.uint8)
    # MRI
    result_mri = binarize_liver_mask(empty_mask, Modality.MRI)
    assert result_mri.sum() == 0
    # CT
    result_ct = binarize_liver_mask(empty_mask, Modality.CT)
    assert result_ct.sum() == 0
