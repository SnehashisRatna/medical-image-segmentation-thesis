"""Lightweight references to individual medical-image slices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.enums import Modality


@dataclass(frozen=True, slots=True)
class Sample:
    """Describe one image slice in a dataset.

    ``Sample`` is a lightweight dataset reference: it stores paths and metadata
    rather than loaded image or mask arrays. The patient identifier identifies
    the patient, ``modality`` identifies the imaging modality, and ``sequence``
    identifies the imaging sequence. ``image_path`` identifies the source image;
    ``mask_path`` is optional because test and inference data may not have
    labels. ``slice_index`` identifies the represented slice.

    This data model will later be consumed by dataset indexing and dataset
    infrastructure.

    Parameters
    ----------
    patient_id : str
        Non-empty identifier of the patient associated with this slice.
    modality : Modality
        Imaging modality of the slice.
    sequence : str
        Non-empty identifier of the imaging sequence.
    image_path : pathlib.Path
        Reference to the source image file. The path is not accessed.
    mask_path : pathlib.Path | None
        Optional reference to the segmentation mask. The path is not accessed.
    slice_index : int
        Non-negative index of the represented slice.
    """

    patient_id: str
    modality: Modality
    sequence: str
    image_path: Path
    mask_path: Path | None
    slice_index: int

    def __post_init__(self) -> None:
        """Validate the sample metadata without accessing the filesystem."""
        if not isinstance(self.patient_id, str):
            raise TypeError("patient_id must be a str.")
        if not isinstance(self.modality, Modality):
            raise TypeError("modality must be a Modality instance.")
        if not isinstance(self.sequence, str):
            raise TypeError("sequence must be a str.")
        if not isinstance(self.image_path, Path):
            raise TypeError("image_path must be a pathlib.Path instance.")
        if self.mask_path is not None and not isinstance(self.mask_path, Path):
            raise TypeError("mask_path must be a pathlib.Path instance or None.")
        if isinstance(self.slice_index, bool) or not isinstance(self.slice_index, int):
            raise TypeError("slice_index must be an int.")

        if not self.patient_id.strip():
            raise ValueError("patient_id must not be empty or whitespace.")
        if not self.sequence.strip():
            raise ValueError("sequence must not be empty or whitespace.")
        if self.slice_index < 0:
            raise ValueError("slice_index must be greater than or equal to zero.")
