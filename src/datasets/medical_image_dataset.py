"""
Medical Image Dataset
=====================

Purpose
-------
Provide a lightweight 2D sample-retrieval layer for medical images.

Responsibilities
----------------
1. Orchestrate sample retrieval
2. Return DatasetItem structures
3. Integrate DICOM and mask reading

Author
------
Snehashis Ratna

Project
-------
Medical Image Segmentation Thesis
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from src.core.enums import Modality
from src.data.dicom_reader import DICOMReader
from src.data.ground_truth_reader import GroundTruthReader
from src.datasets.sample import Sample


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """A loaded medical image slice and its metadata.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D image array.
    mask : numpy.ndarray | None
        The 2D segmentation mask array, or None if not available.
    patient_id : str
        The patient identifier.
    modality : Modality
        The imaging modality.
    sequence : str
        The imaging sequence.
    slice_index : int
        The slice index.
    image_path : pathlib.Path
        The source image path.
    mask_path : pathlib.Path | None
        The source mask path, if available.
    """

    image: np.ndarray
    mask: np.ndarray | None
    patient_id: str
    modality: Modality
    sequence: str
    slice_index: int
    image_path: Path
    mask_path: Path | None


class MedicalImageDataset(Dataset):
    """
    Medical Image Dataset for 2D slice retrieval.

    This dataset acts as an orchestration layer connecting
    indexed Samples to data loaders.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable[[Any], Any] | None = None,
        dicom_reader: DICOMReader | None = None,
        mask_reader: GroundTruthReader | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        samples : Sequence[Sample]
            A sequence of Sample objects to serve.
        transform : Callable | None, optional
            An optional transformation to apply to each DatasetItem.
        dicom_reader : DICOMReader | None, optional
            A custom DICOMReader instance for dependency injection.
        mask_reader : GroundTruthReader | None, optional
            A custom GroundTruthReader instance for dependency injection.
        """
        self._samples = samples
        self._transform = transform
        self._dicom_reader = dicom_reader or DICOMReader()
        self._mask_reader = mask_reader or GroundTruthReader()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._samples)

    def __getitem__(self, index: int) -> Any:
        """
        Retrieve and load a sample by index.

        Parameters
        ----------
        index : int
            The index of the sample to retrieve.

        Returns
        -------
        Any
            The loaded DatasetItem, or transformed item.
        """
        sample = self._samples[index]

        image = self._dicom_reader.read_image(sample.image_path)

        mask = None
        if sample.mask_path is not None:
            mask = self._mask_reader.read_mask(sample.mask_path)

        item = DatasetItem(
            image=image,
            mask=mask,
            patient_id=sample.patient_id,
            modality=sample.modality,
            sequence=sample.sequence,
            slice_index=sample.slice_index,
            image_path=sample.image_path,
            mask_path=sample.mask_path,
        )

        if self._transform is not None:
            return self._transform(item)

        return item
