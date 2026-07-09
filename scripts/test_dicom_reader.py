from pathlib import Path

from src.data.dicom_reader import DICOMReader


def main():

    reader = DICOMReader()

    dicom_path = Path(
        "data/raw/CHAOS_Train_Sets/Train_Sets/CT/1/DICOM_anon/i0000,0000b.dcm"
    )

    metadata = reader.read_metadata(dicom_path)

    image = reader.read_image(dicom_path)

    stats = reader.image_statistics(image)

    print("=" * 60)
    print("METADATA")
    print("=" * 60)

    for key, value in metadata.items():
        print(f"{key:20}: {value}")

    print("\n" + "=" * 60)
    print("IMAGE STATISTICS")
    print("=" * 60)

    for key, value in stats.items():
        print(f"{key:20}: {value}")


if __name__ == "__main__":
    main()