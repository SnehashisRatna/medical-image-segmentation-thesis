"""
Tests for MedicalImageDataset
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.core.enums import Modality
from src.datasets.medical_image_dataset import DatasetItem, MedicalImageDataset
from src.datasets.sample import Sample


@pytest.fixture
def dummy_image() -> np.ndarray:
    return np.ones((256, 256), dtype=np.float32)


@pytest.fixture
def dummy_mask() -> np.ndarray:
    return np.zeros((256, 256), dtype=np.uint8)


@pytest.fixture
def mock_dicom_reader(dummy_image: np.ndarray) -> Mock:
    reader = Mock()
    reader.read_image.return_value = dummy_image
    return reader


@pytest.fixture
def mock_mask_reader(dummy_mask: np.ndarray) -> Mock:
    reader = Mock()
    reader.read_mask.return_value = dummy_mask
    return reader


@pytest.fixture
def labeled_sample() -> Sample:
    return Sample(
        patient_id="PAT_01",
        modality=Modality.CT,
        sequence="seq",
        image_path=Path("image.dcm"),
        mask_path=Path("mask.png"),
        slice_index=0,
    )


@pytest.fixture
def unlabeled_sample() -> Sample:
    return Sample(
        patient_id="PAT_02",
        modality=Modality.MRI,
        sequence="seq2",
        image_path=Path("image2.dcm"),
        mask_path=None,
        slice_index=1,
    )


def test_dataset_construction(labeled_sample: Sample):
    """Test dataset can be constructed with minimum requirements."""
    dataset = MedicalImageDataset(samples=[labeled_sample])
    assert len(dataset) == 1


def test_dataset_length(labeled_sample: Sample, unlabeled_sample: Sample):
    """Test __len__ returns the exact number of samples."""
    dataset = MedicalImageDataset(samples=[labeled_sample, unlabeled_sample])
    assert len(dataset) == 2

    empty_dataset = MedicalImageDataset(samples=[])
    assert len(empty_dataset) == 0


def test_labeled_sample_loading(
    labeled_sample: Sample,
    mock_dicom_reader: Mock,
    mock_mask_reader: Mock,
    dummy_image: np.ndarray,
    dummy_mask: np.ndarray,
):
    """Test image and mask are loaded for a labeled sample."""
    dataset = MedicalImageDataset(
        samples=[labeled_sample],
        dicom_reader=mock_dicom_reader,
        mask_reader=mock_mask_reader,
    )

    item = dataset[0]

    mock_dicom_reader.read_image.assert_called_once_with(labeled_sample.image_path)
    mock_mask_reader.read_mask.assert_called_once_with(labeled_sample.mask_path)

    assert isinstance(item, DatasetItem)
    np.testing.assert_array_equal(item.image, dummy_image)
    np.testing.assert_array_equal(item.mask, dummy_mask)


def test_unlabeled_sample_loading(
    unlabeled_sample: Sample,
    mock_dicom_reader: Mock,
    mock_mask_reader: Mock,
    dummy_image: np.ndarray,
):
    """Test mask is None for an unlabeled sample."""
    dataset = MedicalImageDataset(
        samples=[unlabeled_sample],
        dicom_reader=mock_dicom_reader,
        mask_reader=mock_mask_reader,
    )

    item = dataset[0]

    mock_dicom_reader.read_image.assert_called_once_with(unlabeled_sample.image_path)
    mock_mask_reader.read_mask.assert_not_called()

    assert item.mask is None
    np.testing.assert_array_equal(item.image, dummy_image)


def test_metadata_preserved(
    labeled_sample: Sample, mock_dicom_reader: Mock, mock_mask_reader: Mock
):
    """Test sample metadata is correctly preserved in DatasetItem."""
    dataset = MedicalImageDataset(
        samples=[labeled_sample],
        dicom_reader=mock_dicom_reader,
        mask_reader=mock_mask_reader,
    )

    item = dataset[0]

    assert item.patient_id == labeled_sample.patient_id
    assert item.modality == labeled_sample.modality
    assert item.sequence == labeled_sample.sequence
    assert item.slice_index == labeled_sample.slice_index
    assert item.image_path == labeled_sample.image_path
    assert item.mask_path == labeled_sample.mask_path


def test_transform_boundary(
    labeled_sample: Sample, mock_dicom_reader: Mock, mock_mask_reader: Mock
):
    """Test optional transform boundary works."""
    # A dummy transform that replaces the image
    transformed_image = np.zeros((128, 128))

    def dummy_transform(item: DatasetItem) -> DatasetItem:
        return DatasetItem(
            image=transformed_image,
            mask=item.mask,
            patient_id=item.patient_id,
            modality=item.modality,
            sequence=item.sequence,
            slice_index=item.slice_index,
            image_path=item.image_path,
            mask_path=item.mask_path,
        )

    dataset = MedicalImageDataset(
        samples=[labeled_sample],
        transform=dummy_transform,
        dicom_reader=mock_dicom_reader,
        mask_reader=mock_mask_reader,
    )

    item = dataset[0]
    np.testing.assert_array_equal(item.image, transformed_image)


def test_reader_error_propagation(labeled_sample: Sample):
    """Test reader errors propagate up appropriately."""
    failing_dicom_reader = Mock()
    failing_dicom_reader.read_image.side_effect = FileNotFoundError("Missing DICOM")

    dataset = MedicalImageDataset(
        samples=[labeled_sample],
        dicom_reader=failing_dicom_reader,
    )

    with pytest.raises(FileNotFoundError, match="Missing DICOM"):
        _ = dataset[0]


def test_original_sample_not_modified(
    labeled_sample: Sample, mock_dicom_reader: Mock, mock_mask_reader: Mock
):
    """Test original sample object is not mutated by the dataset or reader."""
    original_dict = {
        "patient_id": labeled_sample.patient_id,
        "modality": labeled_sample.modality,
        "sequence": labeled_sample.sequence,
        "image_path": labeled_sample.image_path,
        "mask_path": labeled_sample.mask_path,
        "slice_index": labeled_sample.slice_index,
    }

    dataset = MedicalImageDataset(
        samples=[labeled_sample],
        dicom_reader=mock_dicom_reader,
        mask_reader=mock_mask_reader,
    )

    _ = dataset[0]

    assert labeled_sample.patient_id == original_dict["patient_id"]
    assert labeled_sample.modality == original_dict["modality"]
    assert labeled_sample.sequence == original_dict["sequence"]
    assert labeled_sample.image_path == original_dict["image_path"]
    assert labeled_sample.mask_path == original_dict["mask_path"]
    assert labeled_sample.slice_index == original_dict["slice_index"]


def test_deterministic_indexing(
    labeled_sample: Sample,
    unlabeled_sample: Sample,
    mock_dicom_reader: Mock,
    mock_mask_reader: Mock,
):
    """Test multiple Samples maintain deterministic indexing/order."""
    dataset = MedicalImageDataset(
        samples=[labeled_sample, unlabeled_sample],
        dicom_reader=mock_dicom_reader,
        mask_reader=mock_mask_reader,
    )

    item0 = dataset[0]
    item1 = dataset[1]

    assert item0.patient_id == "PAT_01"
    assert item1.patient_id == "PAT_02"
