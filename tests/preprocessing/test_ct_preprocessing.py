"""Unit tests for CT preprocessing.

All tests use synthetic pixel arrays and known slope/intercept values.
No real DICOM files or GPU are required.

Test coverage:
- Correct HU conversion (pixel * slope + intercept)
- Clipping at -1000 HU
- Clipping at +1000 HU
- Normalization: -1000 HU → 0.0, 0 HU → 0.5, +1000 HU → 1.0
- Output range is in [0.0, 1.0]
- Output dtype is float32
- Identity slope/intercept (slope=1, intercept=0)
- Slope and intercept are applied before clipping
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.ct import preprocess_ct


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_pixels() -> np.ndarray:
    """Pixel array that maps directly to HU when slope=1, intercept=0."""
    return np.array([[-1000.0, 0.0, 1000.0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# HU conversion
# ---------------------------------------------------------------------------


def test_hu_conversion_identity_slope_intercept() -> None:
    """With slope=1 and intercept=0 the HU value equals the raw pixel."""
    pixels = np.array([[0.0, 500.0, -500.0]], dtype=np.float32)
    result = preprocess_ct(pixels, rescale_slope=1.0, rescale_intercept=0.0)
    # After no HU offset: 0 → 0.5, 500 → 0.75, -500 → 0.25
    np.testing.assert_allclose(result[0, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(result[0, 1], 0.75, atol=1e-6)
    np.testing.assert_allclose(result[0, 2], 0.25, atol=1e-6)


def test_hu_conversion_applies_slope_and_intercept() -> None:
    """HU = pixel * slope + intercept is applied before normalization."""
    # pixel=0, slope=1, intercept=-1024 → HU = -1024, clipped → -1000 → 0.0
    pixels = np.array([[0.0]], dtype=np.float32)
    result = preprocess_ct(pixels, rescale_slope=1.0, rescale_intercept=-1024.0)
    np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-6)


def test_hu_conversion_non_unit_slope() -> None:
    """Slope != 1.0 scales pixels before the intercept is added."""
    # pixel=500, slope=2, intercept=0 → HU = 1000, clipped → 1000 → 1.0
    pixels = np.array([[500.0]], dtype=np.float32)
    result = preprocess_ct(pixels, rescale_slope=2.0, rescale_intercept=0.0)
    np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


def test_clipping_below_minus_1000() -> None:
    """Pixels below -1000 HU are clipped to -1000 HU → normalized to 0.0."""
    pixels = np.array([[-5000.0, -2000.0]], dtype=np.float32)
    result = preprocess_ct(pixels, rescale_slope=1.0, rescale_intercept=0.0)
    np.testing.assert_allclose(result, np.array([[0.0, 0.0]]), atol=1e-6)


def test_clipping_above_plus_1000() -> None:
    """Pixels above +1000 HU are clipped to +1000 HU → normalized to 1.0."""
    pixels = np.array([[5000.0, 2000.0]], dtype=np.float32)
    result = preprocess_ct(pixels, rescale_slope=1.0, rescale_intercept=0.0)
    np.testing.assert_allclose(result, np.array([[1.0, 1.0]]), atol=1e-6)


def test_clipping_at_exact_boundaries() -> None:
    """Values exactly at -1000 and +1000 HU are NOT clipped away."""
    pixels = np.array([[-1000.0, 1000.0]], dtype=np.float32)
    result = preprocess_ct(pixels, rescale_slope=1.0, rescale_intercept=0.0)
    np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(result[0, 1], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalization_minus_1000_maps_to_zero(identity_pixels: np.ndarray) -> None:
    """-1000 HU maps to 0.0 after normalization."""
    result = preprocess_ct(identity_pixels, rescale_slope=1.0, rescale_intercept=0.0)
    np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-6)


def test_normalization_zero_hu_maps_to_half(identity_pixels: np.ndarray) -> None:
    """0 HU maps to 0.5 after normalization."""
    result = preprocess_ct(identity_pixels, rescale_slope=1.0, rescale_intercept=0.0)
    np.testing.assert_allclose(result[0, 1], 0.5, atol=1e-6)


def test_normalization_plus_1000_maps_to_one(identity_pixels: np.ndarray) -> None:
    """+1000 HU maps to 1.0 after normalization."""
    result = preprocess_ct(identity_pixels, rescale_slope=1.0, rescale_intercept=0.0)
    np.testing.assert_allclose(result[0, 2], 1.0, atol=1e-6)


def test_output_range_is_zero_to_one() -> None:
    """All output values are within [0, 1] for arbitrary realistic inputs."""
    rng = np.random.default_rng(seed=0)
    pixels = rng.uniform(-4096.0, 4096.0, size=(16, 16)).astype(np.float32)
    result = preprocess_ct(pixels, rescale_slope=1.0, rescale_intercept=-1024.0)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


# ---------------------------------------------------------------------------
# Output dtype
# ---------------------------------------------------------------------------


def test_output_dtype_is_float32() -> None:
    """preprocess_ct always returns float32 regardless of input dtype."""
    pixels_f64 = np.zeros((4, 4), dtype=np.float64)
    result = preprocess_ct(pixels_f64, rescale_slope=1.0, rescale_intercept=0.0)
    assert result.dtype == np.float32


def test_output_shape_matches_input() -> None:
    """Output spatial shape equals input shape."""
    pixels = np.zeros((512, 512), dtype=np.float32)
    result = preprocess_ct(pixels, rescale_slope=1.0, rescale_intercept=-1000.0)
    assert result.shape == (512, 512)


# ---------------------------------------------------------------------------
# Formula consistency check (manual calculation)
# ---------------------------------------------------------------------------


def test_formula_consistency() -> None:
    """Verify the full formula: HU → clip → normalize against manual calc."""
    # pixel=2048, slope=0.5, intercept=-1024
    # HU = 2048 * 0.5 + (-1024) = 1024 - 1024 = 0
    # clipped = clip(0, -1000, 1000) = 0
    # normalized = (0 + 1000) / 2000 = 0.5
    pixels = np.array([[2048.0]], dtype=np.float32)
    result = preprocess_ct(pixels, rescale_slope=0.5, rescale_intercept=-1024.0)
    np.testing.assert_allclose(result[0, 0], 0.5, atol=1e-6)
