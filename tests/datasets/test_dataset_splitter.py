"""Tests for the DatasetSplitter module."""

from pathlib import Path
import pytest

from src.core.enums import Modality
from src.datasets.sample import Sample
from src.datasets.dataset_splitter import (
    DatasetSplitter,
    SplitConfiguration,
    DatasetSplit,
)


@pytest.fixture
def dummy_samples() -> list[Sample]:
    """Generate a list of dummy samples for testing."""
    samples = []
    # 10 CT patients, 5 slices each
    for i in range(10):
        patient_id = f"CT_Patient_{i}"
        for j in range(5):
            samples.append(
                Sample(
                    patient_id=patient_id,
                    modality=Modality.CT,
                    sequence="CT",
                    image_path=Path(f"fake/ct/{patient_id}_{j}.dcm"),
                    mask_path=None,
                    slice_index=j,
                )
            )

    # 10 MR patients, 5 slices each
    # Some MR patients have the same patient_id as CT, verifying multimodal key separation
    for i in range(10):
        patient_id = f"Patient_{i}" if i < 5 else f"MR_Patient_{i}"
        for j in range(5):
            samples.append(
                Sample(
                    patient_id=patient_id,
                    modality=Modality.MRI,
                    sequence="T2SPIR",
                    image_path=Path(f"fake/mr/{patient_id}_{j}.dcm"),
                    mask_path=None,
                    slice_index=j,
                )
            )

    # Also reuse the "Patient_x" for CT to ensure they are treated as distinct keys
    for i in range(5):
        patient_id = f"Patient_{i}"
        for j in range(5):
            samples.append(
                Sample(
                    patient_id=patient_id,
                    modality=Modality.CT,
                    sequence="CT",
                    image_path=Path(f"fake/ct2/{patient_id}_{j}.dcm"),
                    mask_path=None,
                    slice_index=j,
                )
            )
    return samples


def test_split_conservation_and_exclusivity(dummy_samples: list[Sample]) -> None:
    config = SplitConfiguration(train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15)
    splitter = DatasetSplitter(config)
    split = splitter.split(dummy_samples)

    # Sample conservation
    assert len(split.train) + len(split.validation) + len(split.test) == len(
        dummy_samples
    )

    # Re-collect all samples to ensure none were modified
    all_split_samples = split.train + split.validation + split.test
    assert set(all_split_samples) == set(dummy_samples)

    # Patient Exclusivity
    train_patients = {(s.modality, s.patient_id) for s in split.train}
    val_patients = {(s.modality, s.patient_id) for s in split.validation}
    test_patients = {(s.modality, s.patient_id) for s in split.test}

    assert not (train_patients & val_patients)
    assert not (train_patients & test_patients)
    assert not (val_patients & test_patients)


def test_deterministic_behavior(dummy_samples: list[Sample]) -> None:
    config1 = SplitConfiguration(seed=42)
    splitter1 = DatasetSplitter(config1)
    split1 = splitter1.split(dummy_samples)

    config2 = SplitConfiguration(seed=42)
    splitter2 = DatasetSplitter(config2)
    split2 = splitter2.split(dummy_samples)

    assert split1.train == split2.train
    assert split1.validation == split2.validation
    assert split1.test == split2.test


def test_different_seeds(dummy_samples: list[Sample]) -> None:
    config1 = SplitConfiguration(seed=42)
    splitter1 = DatasetSplitter(config1)
    split1 = splitter1.split(dummy_samples)

    config2 = SplitConfiguration(seed=99)
    splitter2 = DatasetSplitter(config2)
    split2 = splitter2.split(dummy_samples)

    # Given the number of patients, it's highly improbable that two seeds yield identical splits
    assert split1.train != split2.train


def test_configurable_ratios(dummy_samples: list[Sample]) -> None:
    config = SplitConfiguration(train_ratio=0.5, validation_ratio=0.5, test_ratio=0.0)
    splitter = DatasetSplitter(config)
    split = splitter.split(dummy_samples)

    assert len(split.test) == 0
    assert len(split.train) > 0
    assert len(split.validation) > 0

    # 15 CT patients, 10 MR patients -> 25 total keys
    # 50% split per modality: CT(15)-> 8/7, MR(10)-> 5/5
    # Total train patients = 13, val = 12
    train_patients = {(s.modality, s.patient_id) for s in split.train}
    val_patients = {(s.modality, s.patient_id) for s in split.validation}
    assert len(train_patients) + len(val_patients) == 25


def test_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfiguration(train_ratio=0.5, validation_ratio=0.5, test_ratio=0.5)

    with pytest.raises(ValueError, match="non-negative"):
        SplitConfiguration(train_ratio=-0.1, validation_ratio=0.6, test_ratio=0.5)


def test_empty_sequence() -> None:
    config = SplitConfiguration()
    splitter = DatasetSplitter(config)

    with pytest.raises(ValueError, match="Cannot split an empty sequence"):
        splitter.split([])


def test_insufficient_identities_one_patient() -> None:
    sample = Sample(
        patient_id="Single",
        modality=Modality.CT,
        sequence="CT",
        image_path=Path("fake.dcm"),
        mask_path=None,
        slice_index=0,
    )

    config = SplitConfiguration(train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15)
    splitter = DatasetSplitter(config)

    with pytest.raises(ValueError, match="cannot be satisfied with 1 patient"):
        splitter.split([sample])


def test_insufficient_identities_two_patients() -> None:
    samples = [
        Sample(
            patient_id="P1",
            modality=Modality.CT,
            sequence="CT",
            image_path=Path("1.dcm"),
            mask_path=None,
            slice_index=0,
        ),
        Sample(
            patient_id="P2",
            modality=Modality.CT,
            sequence="CT",
            image_path=Path("2.dcm"),
            mask_path=None,
            slice_index=0,
        ),
    ]

    config = SplitConfiguration(train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15)
    splitter = DatasetSplitter(config)

    with pytest.raises(ValueError, match="cannot be satisfied with 2 patient"):
        splitter.split(samples)


def test_valid_small_dataset_all_positive_ratios() -> None:
    # 7 patients are required for 70/15/15 to yield at least 1 in each (round(7*0.15)=1)
    samples = [
        Sample(
            patient_id=f"P{i}",
            modality=Modality.CT,
            sequence="CT",
            image_path=Path(f"{i}.dcm"),
            mask_path=None,
            slice_index=0,
        )
        for i in range(7)
    ]

    config = SplitConfiguration(train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15)
    splitter = DatasetSplitter(config)
    split = splitter.split(samples)

    assert len(split.train) > 0
    assert len(split.validation) > 0
    assert len(split.test) > 0


def test_zero_ratio_split_allowed() -> None:
    # 1 patient with train_ratio=1.0, val=0, test=0 should not throw
    sample = Sample(
        patient_id="Single",
        modality=Modality.CT,
        sequence="CT",
        image_path=Path("fake.dcm"),
        mask_path=None,
        slice_index=0,
    )

    config = SplitConfiguration(train_ratio=1.0, validation_ratio=0.0, test_ratio=0.0)
    splitter = DatasetSplitter(config)
    split = splitter.split([sample])

    assert len(split.train) == 1
    assert len(split.validation) == 0
    assert len(split.test) == 0


def test_stratify_by_modality_true_vs_false(dummy_samples: list[Sample]) -> None:
    # In dummy_samples we have 15 CT keys, 10 MR keys. Total = 25.

    # Stratified: CT and MR split individually.
    config_stratified = SplitConfiguration(stratify_by_modality=True, seed=42)
    splitter_stratified = DatasetSplitter(config_stratified)
    split_stratified = splitter_stratified.split(dummy_samples)

    # Global: 25 keys split collectively.
    config_global = SplitConfiguration(stratify_by_modality=False, seed=42)
    splitter_global = DatasetSplitter(config_global)
    split_global = splitter_global.split(dummy_samples)

    assert split_stratified.train != split_global.train


def test_multimodal_identity(dummy_samples: list[Sample]) -> None:
    config = SplitConfiguration()
    splitter = DatasetSplitter(config)
    split = splitter.split(dummy_samples)

    # Patient_0 has both CT and MR modalities in dummy_samples
    train_patients = {(s.modality, s.patient_id) for s in split.train}
    val_patients = {(s.modality, s.patient_id) for s in split.validation}
    test_patients = {(s.modality, s.patient_id) for s in split.test}

    all_assigned = train_patients | val_patients | test_patients
    assert (Modality.CT, "Patient_0") in all_assigned
    assert (Modality.MRI, "Patient_0") in all_assigned

    # The split must not group them together merely because patient_id is "Patient_0"
    # Although possible they end up in the same split by chance, we just verify they
    # are tracked as independent keys.


def test_sample_unchanged(dummy_samples: list[Sample]) -> None:
    config = SplitConfiguration()
    splitter = DatasetSplitter(config)
    split = splitter.split(dummy_samples)

    for sample in split.train:
        assert isinstance(sample, Sample)
        assert sample.image_path is not None


def test_validation_fails_on_leakage(dummy_samples: list[Sample]) -> None:
    config = SplitConfiguration()
    splitter = DatasetSplitter(config)
    split = splitter.split(dummy_samples)

    # Create artificial leakage
    leaked_split = DatasetSplit(
        train=split.train + [split.validation[0]],
        validation=split.validation,
        test=split.test,
    )

    with pytest.raises(ValueError, match="Patient leakage"):
        splitter._validate_result(leaked_split, dummy_samples)


def test_validation_fails_on_duplicates(dummy_samples: list[Sample]) -> None:
    config = SplitConfiguration()
    splitter = DatasetSplitter(config)
    split = splitter.split(dummy_samples)

    # Artificial duplicates
    duplicated_split = DatasetSplit(
        train=split.train + [split.train[0]],
        validation=split.validation,
        test=split.test,
    )

    with pytest.raises(ValueError, match="Duplicate samples"):
        splitter._validate_result(duplicated_split, dummy_samples + [split.train[0]])


def test_cross_process_determinism() -> None:
    import subprocess
    import sys

    script = """
import sys
from pathlib import Path
from src.core.enums import Modality
from src.datasets.sample import Sample
from src.datasets.dataset_splitter import DatasetSplitter, SplitConfiguration

samples = []
for i in range(25):
    samples.append(Sample(
        patient_id=f"Patient_{i}",
        modality=Modality.CT,
        sequence="CT",
        image_path=Path(f"{i}.dcm"),
        mask_path=None,
        slice_index=0
    ))

config = SplitConfiguration(seed=42)
splitter = DatasetSplitter(config)
split = splitter.split(samples)

train_ids = [s.patient_id for s in split.train]
val_ids = [s.patient_id for s in split.validation]
test_ids = [s.patient_id for s in split.test]

print("TRAIN:", train_ids)
print("VAL:", val_ids)
print("TEST:", test_ids)
"""

    # Run in separate processes to test if set iteration nondeterminism affects output
    # Python >= 3.3 has hash randomization on by default
    out1 = subprocess.check_output([sys.executable, "-c", script], text=True)
    out2 = subprocess.check_output([sys.executable, "-c", script], text=True)

    assert out1 == out2
    assert "TRAIN:" in out1
