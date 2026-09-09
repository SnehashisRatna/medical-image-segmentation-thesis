"""Unit tests for MRI preprocessing.

Key test: VOLUME-LEVEL normalization.

The critical property under test is that all slices belonging to the same
MRI volume are normalized using statistics computed across the ENTIRE volume,
not per-slice.  Two explicit tests prove this:

1. ``test_volume_level_not_per_slice`` — constructs two synthetic slices with
   deliberately different intensity distributions, computes volume stats
   manually, and asserts that both slices are normalized with the volume
   stats (not per-slice stats).

2. ``test_independent_volumes_get_independent_stats`` — constructs two separate
   MRI volumes and verifies their statistics differ.

All tests use mocked DICOMReader-like objects; no real data required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np

from src.preprocessing.mri import compute_volume_stats, preprocess_mri


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reader(path_to_array: dict[Path, np.ndarray]) -> Mock:
    """Return a mock reader whose read_image dispatches by Path."""
    reader = Mock()
    reader.read_image.side_effect = lambda p: path_to_array[p]
    return reader


def _population_stats(arrays: list[np.ndarray]) -> tuple[float, float]:
    """Compute reference population mean/std of all non-zero pixels."""
    nonzero = np.concatenate([a[a > 0].astype(np.float64) for a in arrays])
    if len(nonzero) == 0:
        return 0.0, 0.0
    mean = float(nonzero.mean())
    std = float(nonzero.std())   # ddof=0 (population)
    return mean, std


# ---------------------------------------------------------------------------
# compute_volume_stats
# ---------------------------------------------------------------------------


class TestComputeVolumeStats:
    """Tests for the incremental volume statistics accumulator."""

    def test_single_slice_nonzero_pixels(self) -> None:
        """Stats of a single-slice volume match numpy reference."""
        arr = np.array([[100.0, 200.0, 0.0, 0.0]], dtype=np.float32)
        path = Path("s0.dcm")
        reader = _make_reader({path: arr})

        mean, std = compute_volume_stats([path], reader)

        ref_mean, ref_std = _population_stats([arr])
        assert abs(mean - ref_mean) < 1e-5
        assert abs(std - ref_std) < 1e-5

    def test_multi_slice_stats_match_reference(self) -> None:
        """Incremental accumulator matches all-at-once numpy reference."""
        arr1 = np.array([[10.0, 20.0, 0.0]], dtype=np.float32)
        arr2 = np.array([[30.0, 40.0, 50.0]], dtype=np.float32)
        arr3 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # all background
        paths = [Path("s0.dcm"), Path("s1.dcm"), Path("s2.dcm")]
        reader = _make_reader(dict(zip(paths, [arr1, arr2, arr3])))

        mean, std = compute_volume_stats(paths, reader)

        ref_mean, ref_std = _population_stats([arr1, arr2, arr3])
        assert abs(mean - ref_mean) < 1e-5
        assert abs(std - ref_std) < 1e-5

    def test_all_zero_volume_returns_safe_defaults(self) -> None:
        """All-zero volume returns (0.0, 0.0) without raising."""
        arr = np.zeros((4, 4), dtype=np.float32)
        path = Path("s0.dcm")
        reader = _make_reader({path: arr})

        mean, std = compute_volume_stats([path], reader)

        assert mean == 0.0
        assert std == 0.0

    def test_returns_python_floats(self) -> None:
        """Return values are plain Python floats, not numpy scalars."""
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        path = Path("s0.dcm")
        reader = _make_reader({path: arr})

        mean, std = compute_volume_stats([path], reader)

        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_large_volume_stats_accuracy(self) -> None:
        """Accumulator stays accurate for many slices (precision test)."""
        rng = np.random.default_rng(seed=42)
        arrays = [rng.uniform(1.0, 1000.0, size=(32, 32)).astype(np.float32)
                  for _ in range(20)]
        paths = [Path(f"s{i}.dcm") for i in range(20)]
        reader = _make_reader(dict(zip(paths, arrays)))

        mean, std = compute_volume_stats(paths, reader)

        ref_mean, ref_std = _population_stats(arrays)
        assert abs(mean - ref_mean) < 1e-2
        assert abs(std - ref_std) < 1e-2


# ---------------------------------------------------------------------------
# CRITICAL: volume-level vs per-slice normalization
# ---------------------------------------------------------------------------


class TestVolumeLevelNormalization:
    """Prove normalization uses volume-level statistics, not per-slice."""

    def test_volume_level_not_per_slice(self) -> None:
        """Slices in the same volume are normalized with volume stats.

        Constructs two slices with different intensity distributions.
        Volume stats differ from per-slice stats for each slice.
        Verifies that preprocess_mri with volume stats produces values
        that CANNOT be reproduced with per-slice stats.
        """
        # Slice 1: non-zero pixels [100, 200] → per-slice mean=150, std=50
        # Slice 2: non-zero pixels [300, 400] → per-slice mean=350, std=50
        # Volume:  non-zero pixels [100, 200, 300, 400]
        #          volume_mean = 250.0
        #          volume_std  = std([100,200,300,400]) = sqrt(12500) ≈ 111.803
        arr1 = np.array([[100.0, 200.0], [0.0, 0.0]], dtype=np.float32)
        arr2 = np.array([[300.0, 400.0], [0.0, 0.0]], dtype=np.float32)

        # Compute expected volume stats manually
        all_nonzero = np.array([100.0, 200.0, 300.0, 400.0])
        expected_mean = float(all_nonzero.mean())   # 250.0
        expected_std = float(all_nonzero.std())     # ≈ 111.803

        paths = [Path("s0.dcm"), Path("s1.dcm")]
        reader = _make_reader({paths[0]: arr1, paths[1]: arr2})

        mean, std = compute_volume_stats(paths, reader)

        assert abs(mean - expected_mean) < 1e-4, (
            f"Volume mean: expected {expected_mean:.4f}, got {mean:.4f}"
        )
        assert abs(std - expected_std) < 1e-4, (
            f"Volume std: expected {expected_std:.4f}, got {std:.4f}"
        )

        # Now normalize slice 1 with volume stats
        normalized_s1 = preprocess_mri(arr1, mean, std)
        expected_pixel_00 = (100.0 - expected_mean) / (expected_std + 1e-6)
        actual_pixel_00 = float(normalized_s1[0, 0])

        assert abs(actual_pixel_00 - expected_pixel_00) < 1e-4, (
            f"Normalized slice-1[0,0]: expected (volume) {expected_pixel_00:.4f}, "
            f"got {actual_pixel_00:.4f}"
        )

        # Confirm it does NOT equal the per-slice result
        per_slice_mean = 150.0
        per_slice_std = 50.0
        per_slice_pixel_00 = (100.0 - per_slice_mean) / (per_slice_std + 1e-6)
        # Volume result must differ from per-slice result (they differ because
        # 250 ≠ 150 and 111.8 ≠ 50)
        assert abs(actual_pixel_00 - per_slice_pixel_00) > 0.1, (
            "Volume-level and per-slice results are unexpectedly equal; "
            "normalization may be using per-slice statistics."
        )

    def test_independent_volumes_get_independent_stats(self) -> None:
        """Two MRI volumes with different intensities get independent stats."""
        # Volume A: pixels concentrated around 100
        arr_a = np.array([[100.0, 110.0, 90.0]], dtype=np.float32)
        # Volume B: pixels concentrated around 1000
        arr_b = np.array([[1000.0, 1100.0, 900.0]], dtype=np.float32)

        path_a = Path("vol_a.dcm")
        path_b = Path("vol_b.dcm")

        reader_a = _make_reader({path_a: arr_a})
        reader_b = _make_reader({path_b: arr_b})

        mean_a, std_a = compute_volume_stats([path_a], reader_a)
        mean_b, std_b = compute_volume_stats([path_b], reader_b)

        # Stats must differ between volumes
        assert abs(mean_a - mean_b) > 100.0, (
            "Volumes A and B should have very different means"
        )
        # Volume A mean ≈ 100, Volume B mean ≈ 1000
        assert abs(mean_a - 100.0) < 10.0
        assert abs(mean_b - 1000.0) < 100.0

    def test_same_volume_same_stats_for_all_slices(self) -> None:
        """All slices of one volume use exactly the same statistics."""
        arr1 = np.array([[50.0, 150.0]], dtype=np.float32)
        arr2 = np.array([[250.0, 350.0]], dtype=np.float32)
        paths = [Path("s0.dcm"), Path("s1.dcm")]
        reader = _make_reader({paths[0]: arr1, paths[1]: arr2})

        mean, std = compute_volume_stats(paths, reader)

        # Both slices should be normalized with the SAME mean and std
        norm1 = preprocess_mri(arr1, mean, std)
        norm2 = preprocess_mri(arr2, mean, std)

        # Manually verify: pixel 50 and pixel 250 use the same mean/std
        expected_50 = (50.0 - mean) / (std + 1e-6)
        expected_250 = (250.0 - mean) / (std + 1e-6)
        assert abs(float(norm1[0, 0]) - expected_50) < 1e-4
        assert abs(float(norm2[0, 0]) - expected_250) < 1e-4


# ---------------------------------------------------------------------------
# preprocess_mri
# ---------------------------------------------------------------------------


class TestPreprocessMri:
    """Tests for the per-slice normalization function."""

    def test_zero_background_preserved(self) -> None:
        """Pixels that were 0 in the input are 0.0 in the output."""
        arr = np.array([[100.0, 0.0], [0.0, 200.0]], dtype=np.float32)
        result = preprocess_mri(arr, volume_mean=150.0, volume_std=50.0)
        assert float(result[0, 1]) == 0.0
        assert float(result[1, 0]) == 0.0

    def test_nonzero_pixels_are_normalized(self) -> None:
        """Non-zero pixels are z-score normalized with provided stats."""
        arr = np.array([[150.0]], dtype=np.float32)
        result = preprocess_mri(arr, volume_mean=150.0, volume_std=50.0)
        expected = (150.0 - 150.0) / (50.0 + 1e-6)
        np.testing.assert_allclose(result[0, 0], expected, atol=1e-5)

    def test_output_dtype_is_float32(self) -> None:
        """Output is always float32 regardless of input dtype."""
        arr = np.array([[100.0, 200.0]], dtype=np.float64)
        result = preprocess_mri(arr, volume_mean=150.0, volume_std=50.0)
        assert result.dtype == np.float32

    def test_output_shape_matches_input(self) -> None:
        """Output spatial shape equals input shape."""
        arr = np.zeros((256, 256), dtype=np.float32)
        result = preprocess_mri(arr, volume_mean=0.0, volume_std=0.0)
        assert result.shape == (256, 256)

    def test_no_fixed_intensity_window(self) -> None:
        """The function is purely parametric — no hard-coded HU or range."""
        # Very large intensities (e.g., 10000) should be normalized without
        # any clipping to a fixed window
        arr = np.array([[10000.0, 20000.0, 0.0]], dtype=np.float32)
        result = preprocess_mri(arr, volume_mean=15000.0, volume_std=5000.0)
        # If a fixed window existed (e.g., clip to 255), these would be wrong
        expected_0 = (10000.0 - 15000.0) / (5000.0 + 1e-6)
        expected_1 = (20000.0 - 15000.0) / (5000.0 + 1e-6)
        np.testing.assert_allclose(result[0, 0], expected_0, atol=1e-4)
        np.testing.assert_allclose(result[0, 1], expected_1, atol=1e-4)
        assert float(result[0, 2]) == 0.0  # background preserved

    def test_all_zero_slice_stays_all_zero(self) -> None:
        """An all-zero slice produces an all-zero output."""
        arr = np.zeros((32, 32), dtype=np.float32)
        result = preprocess_mri(arr, volume_mean=100.0, volume_std=50.0)
        assert float(result.sum()) == 0.0

    def test_zero_std_handled_safely(self) -> None:
        """std=0 with epsilon guard does not raise ZeroDivisionError."""
        arr = np.array([[100.0, 100.0]], dtype=np.float32)
        # Should not raise; epsilon prevents division by zero
        result = preprocess_mri(arr, volume_mean=100.0, volume_std=0.0)
        assert np.all(np.isfinite(result))
