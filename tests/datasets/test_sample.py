"""Unit tests for the :class:`Sample` data model."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.core.enums import Modality
from src.datasets import Sample


@pytest.fixture
def sample_kwargs() -> dict[str, object]:
    """Return valid constructor arguments for a sample."""
    return {
        "patient_id": "patient-001",
        "modality": Modality.MRI,
        "sequence": "T1DUAL_IN",
        "image_path": Path("unavailable/image.dcm"),
        "mask_path": Path("unavailable/mask.png"),
        "slice_index": 12,
    }


def test_valid_sample_stores_all_fields(sample_kwargs: dict[str, object]) -> None:
    """A valid sample stores each supplied descriptor unchanged."""
    sample = Sample(**sample_kwargs)

    assert sample.patient_id == "patient-001"
    assert sample.modality is Modality.MRI
    assert sample.sequence == "T1DUAL_IN"
    assert sample.image_path == Path("unavailable/image.dcm")
    assert sample.mask_path == Path("unavailable/mask.png")
    assert sample.slice_index == 12


def test_accepts_none_mask_path(sample_kwargs: dict[str, object]) -> None:
    """A sample without a segmentation mask is valid."""
    sample = Sample(**(sample_kwargs | {"mask_path": None}))
    assert sample.mask_path is None


def test_accepts_nonexistent_paths_without_loading_or_accessing_them(
    sample_kwargs: dict[str, object],
) -> None:
    """Paths remain references even when their files do not exist."""
    image_path = Path("unavailable/image.dcm")
    mask_path = Path("unavailable/mask.png")
    sample = Sample(
        **(sample_kwargs | {"image_path": image_path, "mask_path": mask_path})
    )

    assert not image_path.exists()
    assert not mask_path.exists()
    assert sample.image_path == image_path
    assert sample.mask_path == mask_path


@pytest.mark.parametrize("patient_id", ["", "   ", "\t"])
def test_rejects_empty_patient_id(
    sample_kwargs: dict[str, object], patient_id: str
) -> None:
    """Patient identifiers cannot be empty or whitespace-only."""
    with pytest.raises(ValueError, match="patient_id"):
        Sample(**(sample_kwargs | {"patient_id": patient_id}))


@pytest.mark.parametrize("sequence", ["", "   ", "\n"])
def test_rejects_empty_sequence(
    sample_kwargs: dict[str, object], sequence: str
) -> None:
    """Sequence identifiers cannot be empty or whitespace-only."""
    with pytest.raises(ValueError, match="sequence"):
        Sample(**(sample_kwargs | {"sequence": sequence}))


def test_rejects_negative_slice_index(sample_kwargs: dict[str, object]) -> None:
    """Slice indices must be non-negative."""
    with pytest.raises(ValueError, match="slice_index"):
        Sample(**(sample_kwargs | {"slice_index": -1}))


@pytest.mark.parametrize("slice_index", ["12", 1.5, None, True])
def test_rejects_non_integer_slice_index(
    sample_kwargs: dict[str, object], slice_index: object
) -> None:
    """Slice indices must be integer values, excluding booleans."""
    with pytest.raises(TypeError, match="slice_index"):
        Sample(**(sample_kwargs | {"slice_index": slice_index}))


def test_sample_is_immutable_and_has_value_equality(
    sample_kwargs: dict[str, object],
) -> None:
    """Samples are immutable value objects with dataclass representations."""
    sample = Sample(**sample_kwargs)

    assert sample == Sample(**sample_kwargs)
    assert "Sample(" in repr(sample)
    with pytest.raises(FrozenInstanceError):
        sample.sequence = "T2SPIR"  # type: ignore[misc]
