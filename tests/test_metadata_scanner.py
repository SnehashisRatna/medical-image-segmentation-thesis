from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset

from src.data.metadata_scanner import MetadataScanner


def _write_dicom(path: Path, modality: str, patient_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ds = Dataset()
    ds.PatientID = patient_id
    ds.Modality = modality
    ds.Rows = 128
    ds.Columns = 256
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 2.0
    ds.BitsStored = 16
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    file_ds = FileDataset(str(path), ds, file_meta=ds.file_meta, preamble=b"\x00" * 128)
    file_ds.is_little_endian = True
    file_ds.is_implicit_VR = False
    pydicom.dcmwrite(path, file_ds, write_like_original=False)


def test_scan_collects_ct_and_mr_metadata(tmp_path: Path) -> None:
    _write_dicom(tmp_path / "CT" / "patient_ct" / "DICOM_anon" / "slice.dcm", "CT", "patient_ct")
    _write_dicom(tmp_path / "MR" / "patient_mr" / "DICOM_anon" / "slice.dcm", "MR", "patient_mr")

    scanner = MetadataScanner(str(tmp_path))
    scanner.scan()

    df = scanner.to_dataframe()

    assert len(df) == 2
    assert set(df["Modality"]) == {"CT", "MR"}
    assert df.loc[df["Modality"] == "CT", "Patient"].iloc[0] == "patient_ct"
    assert df.loc[df["Modality"] == "MR", "Patient"].iloc[0] == "patient_mr"

    output_csv = tmp_path / "summary.csv"
    scanner.save_csv(output_csv)
    assert output_csv.exists()
