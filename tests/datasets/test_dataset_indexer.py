from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset

from src.core.enums import Modality
from src.datasets.dataset_indexer import DatasetIndexError, DatasetIndexer


def _write_dicom(path: Path, *, instance: int, z: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.InstanceNumber = instance
    dataset.ImagePositionPatient = [0.0, 0.0, z]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.save_as(path, enforce_file_format=True)


def _add_acquisition(
    dicom_directory: Path, ground_directory: Path | None, stems: list[str]
) -> None:
    for index, stem in enumerate(stems):
        _write_dicom(
            dicom_directory / f"{stem}.dcm", instance=20 - index, z=float(index)
        )
        if ground_directory is not None:
            ground_directory.mkdir(parents=True, exist_ok=True)
            (ground_directory / f"{stem}.png").touch()


def _add_mr_patient(
    root: Path, patient_id: str = "38", *, labeled: bool = True
) -> None:
    patient = root / "MR" / patient_id
    t1_ground = patient / "T1DUAL" / "Ground" if labeled else None
    _add_acquisition(
        patient / "T1DUAL" / "DICOM_anon" / "InPhase", t1_ground, ["in-2", "in-1"]
    )
    _add_acquisition(
        patient / "T1DUAL" / "DICOM_anon" / "OutPhase", None, ["out-2", "out-1"]
    )
    _add_acquisition(
        patient / "T2SPIR" / "DICOM_anon",
        patient / "T2SPIR" / "Ground" if labeled else None,
        ["t2-2", "t2-1"],
    )


def test_indexes_ct_with_stem_matched_masks_and_spatial_order(tmp_path: Path) -> None:
    _add_acquisition(
        tmp_path / "CT" / "1" / "DICOM_anon",
        tmp_path / "CT" / "1" / "Ground",
        ["slice-b", "slice-a"],
    )

    samples = DatasetIndexer(tmp_path).index()

    assert [sample.patient_id for sample in samples] == ["1", "1"]
    assert [sample.modality for sample in samples] == [Modality.CT, Modality.CT]
    assert [sample.sequence for sample in samples] == ["CT", "CT"]
    assert [sample.image_path.stem for sample in samples] == ["slice-b", "slice-a"]
    assert [sample.mask_path.stem for sample in samples if sample.mask_path] == [
        "slice-b",
        "slice-a",
    ]
    assert [sample.slice_index for sample in samples] == [0, 1]


def test_indexes_mri_sequences_without_collapsing_t1_phases(tmp_path: Path) -> None:
    _add_mr_patient(tmp_path)

    samples = DatasetIndexer(tmp_path).index()

    assert {sample.patient_id for sample in samples} == {"38"}
    assert {sample.modality for sample in samples} == {Modality.MRI}
    assert [sample.sequence for sample in samples] == [
        "T1DUAL/InPhase",
        "T1DUAL/InPhase",
        "T1DUAL/OutPhase",
        "T1DUAL/OutPhase",
        "T2SPIR",
        "T2SPIR",
    ]
    in_phase = [sample for sample in samples if sample.sequence == "T1DUAL/InPhase"]
    out_phase = [sample for sample in samples if sample.sequence == "T1DUAL/OutPhase"]
    t2spir = [sample for sample in samples if sample.sequence == "T2SPIR"]
    assert all(sample.mask_path is not None for sample in in_phase)
    assert all(sample.mask_path is None for sample in out_phase)
    assert all(sample.mask_path is not None for sample in t2spir)


def test_unlabeled_acquisitions_have_no_mask(tmp_path: Path) -> None:
    _add_mr_patient(tmp_path, labeled=False)

    samples = DatasetIndexer(tmp_path).index()

    assert len(samples) == 6
    assert all(sample.mask_path is None for sample in samples)


def test_indexing_is_deterministic(tmp_path: Path) -> None:
    _add_mr_patient(tmp_path)

    first = DatasetIndexer(tmp_path).index()
    second = DatasetIndexer(tmp_path).index()

    assert first == second


def test_rejects_invalid_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a directory"):
        DatasetIndexer(tmp_path / "missing").index()


def test_rejects_missing_expected_structure(tmp_path: Path) -> None:
    (tmp_path / "MR" / "38" / "T1DUAL").mkdir(parents=True)

    with pytest.raises(DatasetIndexError, match="T1DUAL and T2SPIR"):
        DatasetIndexer(tmp_path).index()


def test_rejects_missing_mask_in_labeled_acquisition(tmp_path: Path) -> None:
    dicom = tmp_path / "CT" / "1" / "DICOM_anon"
    ground = tmp_path / "CT" / "1" / "Ground"
    _add_acquisition(dicom, ground, ["present"])
    _write_dicom(dicom / "missing.dcm", instance=30, z=4.0)

    with pytest.raises(DatasetIndexError, match="Missing expected mask"):
        DatasetIndexer(tmp_path).index()


def test_rejects_orphan_mask_in_labeled_acquisition(tmp_path: Path) -> None:
    dicom = tmp_path / "CT" / "1" / "DICOM_anon"
    ground = tmp_path / "CT" / "1" / "Ground"
    _add_acquisition(dicom, ground, ["present"])
    (ground / "orphan.png").touch()

    with pytest.raises(DatasetIndexError, match="no corresponding DICOM image"):
        DatasetIndexer(tmp_path).index()
