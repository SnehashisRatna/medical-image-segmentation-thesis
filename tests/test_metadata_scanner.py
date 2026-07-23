from pathlib import Path

import pydicom
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset

from src.data.metadata_scanner import MetadataScanner


def _write_dicom(path: Path, modality: str, patient_id: str) -> None:
    """Write a minimal DICOM file for a CHAOS-layout scanner test."""
    path.parent.mkdir(parents=True, exist_ok=True)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = patient_id
    ds.Modality = modality
    ds.Rows = 128
    ds.Columns = 256
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 2.0
    ds.BitsStored = 16
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.save_as(path, enforce_file_format=True)


def test_scan_collects_ct_and_mr_metadata(tmp_path: Path) -> None:
    """Scan CT and T2SPIR DICOM files arranged as in the CHAOS dataset."""
    _write_dicom(
        tmp_path / "CT" / "patient_ct" / "DICOM_anon" / "slice.dcm",
        "CT",
        "patient_ct",
    )
    _write_dicom(
        tmp_path / "MR" / "patient_mr" / "T2SPIR" / "DICOM_anon" / "slice.dcm",
        "MR",
        "patient_mr",
    )

    scanner = MetadataScanner(str(tmp_path))
    scanner.scan()

    df = scanner.to_dataframe()

    assert len(df) == 2
    assert set(df["Modality"]) == {"CT", "MR"}
    assert df.loc[df["Modality"] == "CT", "Patient_ID"].iloc[0] == "patient_ct"
    assert df.loc[df["Modality"] == "MR", "Patient_ID"].iloc[0] == "patient_mr"
    assert df.loc[df["Modality"] == "MR", "Sequence"].iloc[0] == "T2SPIR"

    output_csv = tmp_path / "summary.csv"
    scanner.save_csv(output_csv)
    assert output_csv.exists()
    assert len(pd.read_csv(output_csv)) == 2
