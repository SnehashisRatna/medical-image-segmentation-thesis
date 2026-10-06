"""Deterministically index the CHAOS directory layout into :class:`Sample`s."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pydicom

from src.core.enums import Modality
from src.datasets.sample import Sample


class DatasetIndexError(ValueError):
    """Raised when a supplied CHAOS dataset cannot be indexed safely."""


@dataclass(frozen=True, slots=True)
class _DicomSlice:
    """The small amount of header information needed to order one slice."""

    path: Path
    position: tuple[float, float, float] | None
    orientation: tuple[float, float, float, float, float, float] | None
    instance_number: int | None


class DatasetIndexer:
    """Discover CHAOS CT and MR slices below an explicitly supplied root.

    The indexer only reads DICOM headers.  It never accesses ``pixel_array``.
    A ``Ground`` directory makes the associated acquisition labeled; every
    DICOM in that acquisition must then have exactly one PNG with the same
    filename stem.  CHAOS T1DUAL Ground applies only to InPhase.
    """

    def __init__(self, dataset_root: Path) -> None:
        if not isinstance(dataset_root, Path):
            raise TypeError("dataset_root must be a pathlib.Path instance.")
        self.dataset_root = dataset_root

    def index(self) -> list[Sample]:
        """Return all CHAOS samples in a stable modality/patient/series order."""
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"CHAOS dataset root is not a directory: {self.dataset_root}"
            )

        ct_root = self.dataset_root / "CT"
        mr_root = self.dataset_root / "MR"
        if not ct_root.is_dir() and not mr_root.is_dir():
            raise DatasetIndexError(
                "Unsupported CHAOS dataset structure: expected a CT and/or MR directory "
                f"below {self.dataset_root}."
            )

        samples: list[Sample] = []
        if ct_root.exists():
            if not ct_root.is_dir():
                raise DatasetIndexError(f"CT path is not a directory: {ct_root}")
            samples.extend(self._index_ct(ct_root))
        if mr_root.exists():
            if not mr_root.is_dir():
                raise DatasetIndexError(f"MR path is not a directory: {mr_root}")
            samples.extend(self._index_mr(mr_root))
        return samples

    def _index_ct(self, ct_root: Path) -> list[Sample]:
        samples: list[Sample] = []
        for patient in self._directories(ct_root):
            samples.extend(
                self._index_acquisition(
                    patient_id=patient.name,
                    modality=Modality.CT,
                    sequence="CT",
                    dicom_directory=patient / "DICOM_anon",
                    ground_directory=patient / "Ground",
                )
            )
        return samples

    def _index_mr(self, mr_root: Path) -> list[Sample]:
        samples: list[Sample] = []
        for patient in self._directories(mr_root):
            t1dual = patient / "T1DUAL"
            t2spir = patient / "T2SPIR"
            if not t1dual.is_dir() or not t2spir.is_dir():
                raise DatasetIndexError(
                    f"MR patient {patient.name!r} must contain T1DUAL and T2SPIR directories."
                )

            t1_dicom = t1dual / "DICOM_anon"
            if not t1_dicom.is_dir():
                raise DatasetIndexError(f"Missing expected DICOM directory: {t1_dicom}")
            t1_ground = t1dual / "Ground"
            samples.extend(
                self._index_acquisition(
                    patient_id=patient.name,
                    modality=Modality.MRI,
                    sequence="T1DUAL/InPhase",
                    dicom_directory=t1_dicom / "InPhase",
                    ground_directory=t1_ground,
                )
            )
            samples.extend(
                self._index_acquisition(
                    patient_id=patient.name,
                    modality=Modality.MRI,
                    sequence="T1DUAL/OutPhase",
                    dicom_directory=t1_dicom / "OutPhase",
                    ground_directory=None,
                )
            )
            samples.extend(
                self._index_acquisition(
                    patient_id=patient.name,
                    modality=Modality.MRI,
                    sequence="T2SPIR",
                    dicom_directory=t2spir / "DICOM_anon",
                    ground_directory=t2spir / "Ground",
                )
            )
        return samples

    def _index_acquisition(
        self,
        *,
        patient_id: str,
        modality: Modality,
        sequence: str,
        dicom_directory: Path,
        ground_directory: Path | None,
    ) -> list[Sample]:
        if not dicom_directory.is_dir():
            raise DatasetIndexError(
                f"Missing expected DICOM directory: {dicom_directory}"
            )

        images = self._dicom_files(dicom_directory)
        if not images:
            raise DatasetIndexError(
                f"No DICOM files found in expected directory: {dicom_directory}"
            )

        masks = (
            self._mask_map(ground_directory)
            if ground_directory is not None and ground_directory.exists()
            else None
        )
        ordered_images = self._order_slices(images)

        ct_mask_mapping: dict[Path, Path] | None = None
        if masks is not None:
            if modality == Modality.CT:
                ct_image_numbers: dict[int, Path] = {}
                for dicom_slice in ordered_images:
                    if dicom_slice.instance_number is None:
                        raise DatasetIndexError(
                            f"Missing InstanceNumber in CT DICOM: {dicom_slice.path}"
                        )
                    mask_index = dicom_slice.instance_number - 1
                    if mask_index in ct_image_numbers:
                        raise DatasetIndexError(f"Duplicate mask index {mask_index} derived from InstanceNumber in CT acquisition.")
                    ct_image_numbers[mask_index] = dicom_slice.path

                ct_mask_numbers: dict[int, Path] = {}
                for stem, path in masks.items():
                    if not stem.startswith("liver_gt_"):
                        raise DatasetIndexError(f"Unexpected CT mask filename format: {stem}")
                    try:
                        num = int(stem[9:])
                    except ValueError:
                        raise DatasetIndexError(f"Cannot extract slice number from CT mask: {stem}")
                    if num in ct_mask_numbers:
                        raise DatasetIndexError(f"Duplicate mask slice number {num} in CT acquisition.")
                    ct_mask_numbers[num] = path

                missing_masks = set(ct_image_numbers.keys()) - set(ct_mask_numbers.keys())
                if missing_masks:
                    raise DatasetIndexError(
                        f"Missing masks for CT DICOM slice numbers: {sorted(missing_masks)}"
                    )

                extra_masks = set(ct_mask_numbers.keys()) - set(ct_image_numbers.keys())
                if extra_masks:
                    raise DatasetIndexError(
                        f"Extra masks found without corresponding CT DICOMs for slice numbers: {sorted(extra_masks)}"
                    )

                ct_mask_mapping = {
                    img_path: ct_mask_numbers[num]
                    for num, img_path in ct_image_numbers.items()
                }
            else:
                image_stems = {image.stem.casefold() for image in images}
                orphan_stems = sorted(set(masks) - image_stems)
                if orphan_stems:
                    raise DatasetIndexError(
                        "Ground mask has no corresponding DICOM image in labeled acquisition "
                        f"{dicom_directory}: {orphan_stems[0]!r}."
                    )

        samples: list[Sample] = []
        for slice_index, image in enumerate(ordered_images):
            if masks is None:
                mask = None
            elif modality == Modality.CT:
                assert ct_mask_mapping is not None
                mask = ct_mask_mapping[image.path]
            else:
                mask = masks.get(image.path.stem.casefold())
                if mask is None:
                    raise DatasetIndexError(
                        "Missing expected mask for labeled image "
                        f"{image.path} in {ground_directory}."
                    )

            samples.append(
                Sample(patient_id, modality, sequence, image.path, mask, slice_index)
            )
        return samples

    @staticmethod
    def _directories(root: Path) -> list[Path]:
        return sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )

    @staticmethod
    def _dicom_files(directory: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.casefold() == ".dcm"
            ),
            key=lambda path: path.name,
        )

    @staticmethod
    def _mask_map(ground_directory: Path) -> dict[str, Path]:
        if not ground_directory.is_dir():
            raise DatasetIndexError(
                f"Ground path is not a directory: {ground_directory}"
            )
        masks: dict[str, Path] = {}
        for mask in sorted(ground_directory.iterdir(), key=lambda path: path.name):
            if not mask.is_file() or mask.suffix.casefold() != ".png":
                continue
            stem = mask.stem.casefold()
            if stem in masks:
                raise DatasetIndexError(
                    f"Ambiguous masks with filename stem {mask.stem!r} in {ground_directory}."
                )
            masks[stem] = mask
        return masks

    def _order_slices(self, paths: Iterable[Path]) -> list[_DicomSlice]:
        slices = [self._read_slice_header(path) for path in paths]
        if len({item.path.stem.casefold() for item in slices}) != len(slices):
            raise DatasetIndexError("Duplicate DICOM filename stem in one acquisition.")

        if all(item.position is not None for item in slices):
            coordinates = [self._spatial_coordinate(item) for item in slices]
            if len(set(coordinates)) != len(coordinates):
                raise DatasetIndexError(
                    "Duplicate DICOM spatial slice identity in one acquisition."
                )
            return sorted(
                slices,
                key=lambda item: (self._spatial_coordinate(item), item.path.name),
            )

        if all(item.instance_number is not None for item in slices):
            instances = [item.instance_number for item in slices]
            if len(set(instances)) != len(instances):
                raise DatasetIndexError(
                    "Duplicate DICOM InstanceNumber in one acquisition."
                )
            return sorted(
                slices, key=lambda item: (item.instance_number, item.path.name)
            )

        raise DatasetIndexError(
            "Cannot establish deterministic DICOM slice order: every slice needs "
            "ImagePositionPatient or InstanceNumber."
        )

    @staticmethod
    def _read_slice_header(path: Path) -> _DicomSlice:
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=[
                    "ImagePositionPatient",
                    "ImageOrientationPatient",
                    "InstanceNumber",
                ],
            )
        except Exception as error:  # pydicom exposes several parse exception types.
            raise DatasetIndexError(f"Cannot read DICOM header: {path}") from error

        position = DatasetIndexer._float_tuple(dataset.get("ImagePositionPatient"), 3)
        orientation = DatasetIndexer._float_tuple(
            dataset.get("ImageOrientationPatient"), 6
        )
        instance = dataset.get("InstanceNumber")
        try:
            instance_number = None if instance is None else int(instance)
        except (TypeError, ValueError) as error:
            raise DatasetIndexError(
                f"Invalid InstanceNumber in DICOM file: {path}"
            ) from error
        return _DicomSlice(path, position, orientation, instance_number)

    @staticmethod
    def _float_tuple(value: object, expected_length: int) -> tuple[float, ...] | None:
        if value is None:
            return None
        try:
            result = tuple(float(item) for item in value)  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return None
        return result if len(result) == expected_length else None

    @staticmethod
    def _spatial_coordinate(item: _DicomSlice) -> float:
        assert item.position is not None
        if item.orientation is None:
            return item.position[2]
        row = item.orientation[:3]
        column = item.orientation[3:]
        normal = (
            row[1] * column[2] - row[2] * column[1],
            row[2] * column[0] - row[0] * column[2],
            row[0] * column[1] - row[1] * column[0],
        )
        return sum(
            component * position for component, position in zip(normal, item.position)
        )
