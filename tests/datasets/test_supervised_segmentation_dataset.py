"""Unit tests for SupervisedSegmentationDataset.

All tests use mocked DICOMReader and GroundTruthReader instances so that
no real CHAOS data is required.

Test coverage:
- Only labeled samples (mask_path != None) are included
- Samples with mask_path=None are excluded
- Patient-level split has no patient overlap (DatasetSplitter integration)
- CT image output shape [1, 512, 512]
- MRI image output shape [1, 256, 256]
- mask shape [H, W]
- mask dtype torch.long
- mask values only {0, 1}
- MRI volume-level stats cached correctly
- Volume stats are volume-level (proved by two-slice test)
- Independent MRI volumes get independent statistics
- augment=False: no augmentation applied
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from src.core.enums import Modality
from src.datasets.dataset_splitter import DatasetSplitter, SplitConfiguration
from src.datasets.sample import Sample
from src.datasets.supervised_segmentation_dataset import SupervisedSegmentationDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    patient_id: str,
    modality: Modality,
    sequence: str,
    image_path: Path,
    mask_path: Path | None,
    slice_index: int = 0,
) -> Sample:
    return Sample(
        patient_id=patient_id,
        modality=modality,
        sequence=sequence,
        image_path=image_path,
        mask_path=mask_path,
        slice_index=slice_index,
    )


def _ct_sample(
    patient_id: str = "CT01",
    image_path: Path = Path("ct.dcm"),
    mask_path: Path | None = Path("ct_mask.png"),
    slice_index: int = 0,
) -> Sample:
    return _make_sample(patient_id, Modality.CT, "CT", image_path, mask_path, slice_index)


def _mri_sample(
    patient_id: str = "MR01",
    sequence: str = "T1DUAL/InPhase",
    image_path: Path = Path("mri.dcm"),
    mask_path: Path | None = Path("mri_mask.png"),
    slice_index: int = 0,
) -> Sample:
    return _make_sample(patient_id, Modality.MRI, sequence, image_path, mask_path, slice_index)


def _mock_ct_reader(shape: tuple[int, int] = (512, 512)) -> Mock:
    """Mock DICOMReader for CT: read_image returns zeros, metadata returns identity."""
    reader = Mock()
    reader.read_image.return_value = np.zeros(shape, dtype=np.float32)
    reader.read_metadata.return_value = {
        "rescale_slope": 1.0,
        "rescale_intercept": 0.0,
    }
    return reader


def _mock_mri_reader(
    path_to_array: dict[Path, np.ndarray] | None = None,
    default_shape: tuple[int, int] = (256, 256),
) -> Mock:
    """Mock DICOMReader for MRI: read_image returns arrays by path."""
    reader = Mock()
    if path_to_array:
        reader.read_image.side_effect = lambda p: path_to_array[p]
    else:
        reader.read_image.return_value = (
            np.ones(default_shape, dtype=np.float32) * 100.0
        )
    return reader


def _mock_mask_reader(shape: tuple[int, int] = (256, 256)) -> Mock:
    """Mock GroundTruthReader returning an all-liver MRI mask."""
    reader = Mock()
    mask = np.full(shape, fill_value=63, dtype=np.uint8)  # all liver (MRI label)
    reader.read_mask.return_value = mask
    return reader


def _mock_binary_mask_reader(shape: tuple[int, int] = (512, 512)) -> Mock:
    """Mock GroundTruthReader returning a CT binary mask."""
    reader = Mock()
    mask = np.ones(shape, dtype=np.uint8)  # all liver (CT: non-zero)
    reader.read_mask.return_value = mask
    return reader


# ---------------------------------------------------------------------------
# Filtering: mask_path=None samples excluded
# ---------------------------------------------------------------------------


class TestLabeledSampleFiltering:
    """Only labeled samples (mask_path != None) must be included."""

    def test_all_labeled_samples_included(self) -> None:
        """Labeled samples are all retained."""
        samples = [_ct_sample(slice_index=i) for i in range(5)]
        ds = SupervisedSegmentationDataset(
            samples,
            dicom_reader=_mock_ct_reader(),
            mask_reader=_mock_binary_mask_reader(),
        )
        assert len(ds) == 5

    def test_unlabeled_samples_excluded(self) -> None:
        """Samples with mask_path=None are silently excluded."""
        labeled = _ct_sample(patient_id="P1", image_path=Path("img1.dcm"))
        unlabeled = _ct_sample(
            patient_id="P2",
            image_path=Path("img2.dcm"),
            mask_path=None,
        )
        ds = SupervisedSegmentationDataset(
            [labeled, unlabeled],
            dicom_reader=_mock_ct_reader(),
            mask_reader=_mock_binary_mask_reader(),
        )
        assert len(ds) == 1

    def test_all_unlabeled_produces_empty_dataset(self) -> None:
        """A list of only unlabeled samples yields an empty dataset."""
        samples = [
            _ct_sample(
                patient_id=f"P{i}",
                image_path=Path(f"img{i}.dcm"),
                mask_path=None,
            )
            for i in range(3)
        ]
        ds = SupervisedSegmentationDataset(
            samples,
            dicom_reader=_mock_ct_reader(),
            mask_reader=_mock_binary_mask_reader(),
        )
        assert len(ds) == 0

    def test_mri_outphase_unlabeled_is_excluded(self) -> None:
        """T1DUAL/OutPhase samples (mask_path=None) are excluded."""
        labeled = _mri_sample(sequence="T1DUAL/InPhase", mask_path=Path("m.png"))
        unlabeled = _mri_sample(sequence="T1DUAL/OutPhase", mask_path=None)

        arr = np.ones((4, 4), dtype=np.float32) * 100.0
        reader = Mock()
        reader.read_image.return_value = arr

        ds = SupervisedSegmentationDataset(
            [labeled, unlabeled],
            dicom_reader=reader,
            mask_reader=_mock_mask_reader(shape=(4, 4)),
        )
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# Output tensor contracts
# ---------------------------------------------------------------------------


class TestOutputContracts:
    """Output tensors must satisfy the frozen shape/dtype contract."""

    def test_ct_image_shape_one_h_w(self) -> None:
        """CT image tensor has shape [1, H, W] matching native resolution."""
        sample = _ct_sample(image_path=Path("ct.dcm"))
        ds = SupervisedSegmentationDataset(
            [sample],
            dicom_reader=_mock_ct_reader(shape=(512, 512)),
            mask_reader=_mock_binary_mask_reader(shape=(512, 512)),
        )
        image, _ = ds[0]
        assert image.shape == (1, 512, 512)

    def test_mri_image_shape_one_h_w(self) -> None:
        """MRI image tensor has shape [1, H, W] matching native resolution."""
        sample = _mri_sample(image_path=Path("mri.dcm"))
        reader = Mock()
        reader.read_image.return_value = np.ones((256, 256), dtype=np.float32) * 100.0
        ds = SupervisedSegmentationDataset(
            [sample],
            dicom_reader=reader,
            mask_reader=_mock_mask_reader(shape=(256, 256)),
        )
        image, _ = ds[0]
        assert image.shape == (1, 256, 256)

    def test_mask_shape_h_w(self) -> None:
        """Mask tensor has shape [H, W]."""
        sample = _ct_sample(image_path=Path("ct.dcm"))
        ds = SupervisedSegmentationDataset(
            [sample],
            dicom_reader=_mock_ct_reader(shape=(512, 512)),
            mask_reader=_mock_binary_mask_reader(shape=(512, 512)),
        )
        _, mask = ds[0]
        assert mask.shape == (512, 512)

    def test_image_dtype_is_float32(self) -> None:
        """Image tensor dtype is torch.float32."""
        sample = _ct_sample(image_path=Path("ct.dcm"))
        ds = SupervisedSegmentationDataset(
            [sample],
            dicom_reader=_mock_ct_reader(),
            mask_reader=_mock_binary_mask_reader(),
        )
        image, _ = ds[0]
        assert image.dtype == torch.float32

    def test_mask_dtype_is_torch_long(self) -> None:
        """Mask tensor dtype is torch.long (int64)."""
        sample = _ct_sample(image_path=Path("ct.dcm"))
        ds = SupervisedSegmentationDataset(
            [sample],
            dicom_reader=_mock_ct_reader(),
            mask_reader=_mock_binary_mask_reader(),
        )
        _, mask = ds[0]
        assert mask.dtype == torch.long

    def test_mask_values_only_zero_and_one(self) -> None:
        """Mask tensor values are only in {0, 1}."""
        # MRI mask: raw labels 0, 63, 126, 189, 252 in one row
        sample = _mri_sample(image_path=Path("mri.dcm"), mask_path=Path("mask.png"))
        reader = Mock()
        reader.read_image.return_value = np.ones((1, 5), dtype=np.float32) * 100.0
        mask_reader = Mock()
        mask_reader.read_mask.return_value = np.array(
            [[0, 63, 126, 189, 252]], dtype=np.uint8
        )
        ds = SupervisedSegmentationDataset(
            [sample],
            dicom_reader=reader,
            mask_reader=mask_reader,
        )
        _, mask = ds[0]
        unique = set(mask.flatten().tolist())
        assert unique.issubset({0, 1})


# ---------------------------------------------------------------------------
# MRI volume-level statistics
# ---------------------------------------------------------------------------


class TestMriVolumeStats:
    """Volume-level statistics are cached and correctly keyed."""

    def test_volume_stats_are_precomputed_at_construction(self) -> None:
        """_volume_stats is populated during __init__, not lazily."""
        path = Path("mri.dcm")
        reader = Mock()
        reader.read_image.return_value = np.ones((4, 4), dtype=np.float32) * 200.0

        sample = _mri_sample(image_path=path, patient_id="P1")
        ds = SupervisedSegmentationDataset(
            [sample],
            dicom_reader=reader,
            mask_reader=_mock_mask_reader(shape=(4, 4)),
        )

        key = ("P1", Modality.MRI, "T1DUAL/InPhase")
        assert key in ds._volume_stats

    def test_volume_level_not_per_slice_normalization(self) -> None:
        """Both slices of the same volume are normalized with volume stats.

        Slice 1 has non-zero pixels [100, 200]; per-slice mean=150.
        Slice 2 has non-zero pixels [300, 400]; per-slice mean=350.
        Volume mean = 250.  Verifies slice 1 uses volume mean (250), not
        its own per-slice mean (150).
        """
        path1 = Path("s1.dcm")
        path2 = Path("s2.dcm")
        mask_path = Path("mask.png")

        arr1 = np.array([[100.0, 200.0], [0.0, 0.0]], dtype=np.float32)
        arr2 = np.array([[300.0, 400.0], [0.0, 0.0]], dtype=np.float32)

        reader = Mock()
        reader.read_image.side_effect = lambda p: arr1 if p == path1 else arr2

        mask_reader = Mock()
        mask_reader.read_mask.return_value = np.zeros((2, 2), dtype=np.uint8)

        # Both slices: same (patient_id, modality, sequence) → same volume
        samples = [
            _mri_sample(patient_id="P1", image_path=path1, mask_path=mask_path, slice_index=0),
            _mri_sample(patient_id="P1", image_path=path2, mask_path=mask_path, slice_index=1),
        ]
        ds = SupervisedSegmentationDataset(
            samples, augment=False,
            dicom_reader=reader, mask_reader=mask_reader,
        )

        # Expected volume stats: all non-zero = [100, 200, 300, 400]
        all_nonzero = np.array([100.0, 200.0, 300.0, 400.0])
        expected_mean = float(all_nonzero.mean())  # 250.0
        expected_std = float(all_nonzero.std())    # ≈111.803

        key = ("P1", Modality.MRI, "T1DUAL/InPhase")
        cached_mean, cached_std = ds._volume_stats[key]
        assert abs(cached_mean - expected_mean) < 1e-4
        assert abs(cached_std - expected_std) < 1e-4

        # Normalize slice 1 using volume stats → pixel[0,0] = 100
        image1, _ = ds[0]
        expected_px = (100.0 - expected_mean) / (expected_std + 1e-6)
        actual_px = float(image1[0, 0, 0])
        assert abs(actual_px - expected_px) < 1e-4, (
            f"Pixel was {actual_px:.4f}; expected volume-normalized {expected_px:.4f}"
        )

        # Confirm it is NOT per-slice normalized (per-slice mean for arr1 = 150)
        per_slice_px = (100.0 - 150.0) / (50.0 + 1e-6)
        assert abs(actual_px - per_slice_px) > 0.05

    def test_independent_volumes_get_independent_stats(self) -> None:
        """Two MRI volumes with different intensities get independent stats."""
        path_a = Path("vol_a.dcm")
        path_b = Path("vol_b.dcm")
        mask_path = Path("mask.png")

        arr_a = np.ones((4, 4), dtype=np.float32) * 100.0    # mean ≈ 100
        arr_b = np.ones((4, 4), dtype=np.float32) * 1000.0   # mean ≈ 1000

        reader = Mock()
        reader.read_image.side_effect = lambda p: arr_a if p == path_a else arr_b

        mask_reader = Mock()
        mask_reader.read_mask.return_value = np.zeros((4, 4), dtype=np.uint8)

        # Different patient IDs → different volume keys
        sample_a = _mri_sample(
            patient_id="PA", image_path=path_a, mask_path=mask_path
        )
        sample_b = _mri_sample(
            patient_id="PB", image_path=path_b, mask_path=mask_path
        )
        ds = SupervisedSegmentationDataset(
            [sample_a, sample_b],
            dicom_reader=reader,
            mask_reader=mask_reader,
        )

        key_a = ("PA", Modality.MRI, "T1DUAL/InPhase")
        key_b = ("PB", Modality.MRI, "T1DUAL/InPhase")

        mean_a, _ = ds._volume_stats[key_a]
        mean_b, _ = ds._volume_stats[key_b]

        assert abs(mean_a - 100.0) < 1e-4
        assert abs(mean_b - 1000.0) < 1e-4
        assert abs(mean_a - mean_b) > 100.0


# ---------------------------------------------------------------------------
# Augmentation flag
# ---------------------------------------------------------------------------


class TestAugmentationFlag:
    """augment=False means no augmentation is applied."""

    def test_augment_false_does_not_apply_transforms(self) -> None:
        """With augment=False, image/mask tensors are reproducible."""
        sample = _ct_sample(image_path=Path("ct.dcm"))
        reader = _mock_ct_reader()
        mask_reader = _mock_binary_mask_reader()

        ds = SupervisedSegmentationDataset(
            [sample], augment=False,
            dicom_reader=reader, mask_reader=mask_reader,
        )
        img1, mask1 = ds[0]
        img2, mask2 = ds[0]

        assert torch.equal(img1, img2)
        assert torch.equal(mask1, mask2)


# ---------------------------------------------------------------------------
# Patient-level split has no patient overlap
# ---------------------------------------------------------------------------


class TestPatientLevelSplit:
    """Integration: DatasetSplitter → SupervisedSegmentationDataset."""

    def test_no_patient_overlap_across_splits(self) -> None:
        """Patient identities are disjoint across train/val/test splits."""
        # Create 10 patients × 3 slices each
        samples = []
        for pid in range(10):
            for sidx in range(3):
                samples.append(
                    _ct_sample(
                        patient_id=f"P{pid:02d}",
                        image_path=Path(f"P{pid}_{sidx}.dcm"),
                        mask_path=Path(f"P{pid}_{sidx}.png"),
                        slice_index=sidx,
                    )
                )

        config = SplitConfiguration(
            train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, seed=42
        )
        split = DatasetSplitter(config).split(samples)

        def patient_ids(lst: list[Sample]) -> set[str]:
            return {s.patient_id for s in lst}

        train_pids = patient_ids(split.train)
        val_pids = patient_ids(split.validation)
        test_pids = patient_ids(split.test)

        assert train_pids.isdisjoint(val_pids), "Train/val patient overlap"
        assert train_pids.isdisjoint(test_pids), "Train/test patient overlap"
        assert val_pids.isdisjoint(test_pids), "Val/test patient overlap"
