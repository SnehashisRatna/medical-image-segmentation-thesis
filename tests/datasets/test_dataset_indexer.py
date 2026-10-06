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


def test_indexes_ct_with_standard_filenames_and_spatial_order(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "CT" / "1" / "DICOM_anon"
    ground_dir = tmp_path / "CT" / "1" / "Ground"
    ground_dir.mkdir(parents=True)

    # Spatial Z=0.0 -> InstanceNumber=2 -> Mask liver_GT_001.png
    _write_dicom(dicom_dir / "i0005,0000b.dcm", instance=2, z=0.0)
    (ground_dir / "liver_GT_001.png").touch()

    # Spatial Z=1.0 -> InstanceNumber=1 -> Mask liver_GT_000.png
    _write_dicom(dicom_dir / "i0002,0000b.dcm", instance=1, z=1.0)
    (ground_dir / "liver_GT_000.png").touch()

    samples = DatasetIndexer(tmp_path).index()

    assert [sample.patient_id for sample in samples] == ["1", "1"]
    assert [sample.modality for sample in samples] == [Modality.CT, Modality.CT]
    assert [sample.sequence for sample in samples] == ["CT", "CT"]
    # Spatial sort orders by z=0.0 then z=1.0
    assert [sample.image_path.stem for sample in samples] == ["i0005,0000b", "i0002,0000b"]
    # Masks matched by InstanceNumber-1
    assert [sample.mask_path.stem for sample in samples if sample.mask_path] == [
        "liver_GT_001",
        "liver_GT_000",
    ]
    assert [sample.slice_index for sample in samples] == [0, 1]


def test_indexes_ct_with_img_filenames_and_reversed_z_order(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "CT" / "21" / "DICOM_anon"
    ground_dir = tmp_path / "CT" / "21" / "Ground"
    ground_dir.mkdir(parents=True)

    # Spatial Z=10.0 (high z, sorted last) -> Instance=1 -> liver_GT_000.png
    _write_dicom(dicom_dir / "IMG-0001-00001.dcm", instance=1, z=10.0)
    (ground_dir / "liver_GT_000.png").touch()

    # Spatial Z=5.0 (low z, sorted first) -> Instance=2 -> liver_GT_001.png
    _write_dicom(dicom_dir / "IMG-0001-00002.dcm", instance=2, z=5.0)
    (ground_dir / "liver_GT_001.png").touch()

    samples = DatasetIndexer(tmp_path).index()

    # Z-order sort: Z=5.0 (Instance 2) then Z=10.0 (Instance 1)
    assert [sample.image_path.stem for sample in samples] == ["IMG-0001-00002", "IMG-0001-00001"]
    assert [sample.mask_path.stem for sample in samples if sample.mask_path] == [
        "liver_GT_001",
        "liver_GT_000",
    ]


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


def test_rejects_missing_mask_in_labeled_ct_acquisition(tmp_path: Path) -> None:
    dicom = tmp_path / "CT" / "1" / "DICOM_anon"
    ground = tmp_path / "CT" / "1" / "Ground"
    ground.mkdir(parents=True)
    _write_dicom(dicom / "i0001,0000b.dcm", instance=1, z=0.0)
    (ground / "liver_GT_000.png").touch()

    _write_dicom(dicom / "i0002,0000b.dcm", instance=2, z=1.0)
    # Missing liver_GT_001.png for Instance 2 (mask index 1)

    with pytest.raises(DatasetIndexError, match="Missing masks for CT DICOM slice numbers: \\[1\\]"):
        DatasetIndexer(tmp_path).index()


def test_rejects_orphan_mask_in_labeled_ct_acquisition(tmp_path: Path) -> None:
    dicom = tmp_path / "CT" / "1" / "DICOM_anon"
    ground = tmp_path / "CT" / "1" / "Ground"
    ground.mkdir(parents=True)
    _write_dicom(dicom / "i0001,0000b.dcm", instance=1, z=0.0)
    (ground / "liver_GT_000.png").touch()

    (ground / "liver_GT_001.png").touch() # Orphan mask index 1

    with pytest.raises(DatasetIndexError, match="Extra masks found without corresponding CT DICOMs for slice numbers: \\[1\\]"):
        DatasetIndexer(tmp_path).index()


def test_rejects_invalid_ct_mask_filename(tmp_path: Path) -> None:
    dicom = tmp_path / "CT" / "1" / "DICOM_anon"
    ground = tmp_path / "CT" / "1" / "Ground"
    ground.mkdir(parents=True)
    _write_dicom(dicom / "i0001,0000b.dcm", instance=1, z=0.0)
    (ground / "invalid.png").touch()

    with pytest.raises(DatasetIndexError, match="Unexpected CT mask filename format: invalid"):
        DatasetIndexer(tmp_path).index()


def test_rejects_duplicate_instance_number(tmp_path: Path) -> None:
    dicom = tmp_path / "CT" / "1" / "DICOM_anon"
    ground = tmp_path / "CT" / "1" / "Ground"
    ground.mkdir(parents=True)
    _write_dicom(dicom / "i0001,0000b.dcm", instance=1, z=0.0)
    _write_dicom(dicom / "i0002,0000b.dcm", instance=1, z=1.0)  # Duplicate InstanceNumber
    (ground / "liver_GT_000.png").touch()

    with pytest.raises(DatasetIndexError, match="Duplicate mask index 0 derived from InstanceNumber"):
        DatasetIndexer(tmp_path).index()


def test_rejects_duplicate_mask_index(tmp_path: Path) -> None:
    dicom = tmp_path / "CT" / "1" / "DICOM_anon"
    ground = tmp_path / "CT" / "1" / "Ground"
    ground.mkdir(parents=True)
    _write_dicom(dicom / "i0001,0000b.dcm", instance=1, z=0.0)
    (ground / "liver_GT_000.png").touch()
    (ground / "liver_GT_00.png").touch() # Evaluates to mask index 0 too

    with pytest.raises(DatasetIndexError, match="Duplicate mask slice number 0 in CT acquisition"):
        DatasetIndexer(tmp_path).index()


def test_rejects_missing_instance_number(tmp_path: Path) -> None:
    dicom = tmp_path / "CT" / "1" / "DICOM_anon"
    ground = tmp_path / "CT" / "1" / "Ground"
    ground.mkdir(parents=True)

    # Write DICOM without InstanceNumber
    path = dicom / "i0001,0000b.dcm"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
    dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.save_as(path, enforce_file_format=True)

    (ground / "liver_GT_000.png").touch()

    with pytest.raises(DatasetIndexError, match="Missing InstanceNumber in CT DICOM"):
        DatasetIndexer(tmp_path).index()
