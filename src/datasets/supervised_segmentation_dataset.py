"""Supervised segmentation dataset for binary liver segmentation.

Orchestrates the complete 2D data pipeline:

    Sample  (slice reference from DatasetIndexer)
        → filter out mask_path=None          (unlabeled samples excluded)
        → DICOMReader                        (raw pixel array per slice)
        → CT or MRI preprocessing           (modality-specific)
        → GroundTruthReader                  (raw PNG mask)
        → binarize_liver_mask               (binary {0, 1} liver target)
        → TrainingAugmentation (optional)   (training only)
        → torch tensors

MRI volume-level statistics
----------------------------
Per-volume non-zero-pixel z-score normalization is required (frozen spec).
At construction time this dataset groups all MRI samples by their volume
key ``(patient_id, modality, sequence)``, reads every DICOM in the group
one slice at a time using a Welford accumulator, and caches the resulting
``(mean, std)`` for each volume.  Every ``__getitem__`` call looks up the
pre-computed statistics for the sample's volume and applies them.

CT preprocessing
-----------------
HU conversion using per-file ``RescaleSlope`` / ``RescaleIntercept`` tags,
followed by clipping to [−1000, 1000] HU and linear normalization to [0, 1].

This module does NOT modify:
    Sample, DatasetIndexer, DatasetSplitter, DICOMReader, GroundTruthReader,
    MedicalImageDataset, or the U-Net architecture.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.enums import Modality
from src.data.dicom_reader import DICOMReader
from src.data.ground_truth_reader import GroundTruthReader
from src.datasets.sample import Sample
from src.preprocessing.augmentation import TrainingAugmentation
from src.preprocessing.ct import preprocess_ct
from src.preprocessing.mask import binarize_liver_mask
from src.preprocessing.mri import compute_volume_stats, preprocess_mri

# Volume identity key: (patient_id, modality, sequence)
_VolumeKey = tuple[str, Modality, str]


class SupervisedSegmentationDataset(Dataset):
    """PyTorch Dataset for supervised binary liver segmentation.

    Accepts a sequence of :class:`~src.datasets.sample.Sample` objects,
    silently filters out any sample whose ``mask_path`` is ``None``
    (unlabeled / ``T1DUAL/OutPhase``), and returns ``(image, mask)``
    tensor pairs suitable for training a 2D U-Net.

    Parameters
    ----------
    samples : Sequence[Sample]
        All candidate samples (typically from ``DatasetSplitter``).
        Samples with ``mask_path=None`` are excluded at construction time.
    augment : bool, optional
        If ``True``, apply the frozen training-time geometric augmentation
        (random horizontal flip p=0.5, random rotation ±10° p=0.5).
        Must be ``False`` for validation and test splits.
    dicom_reader : DICOMReader | None, optional
        Concrete reader; defaults to ``DICOMReader()``.  Accepts any
        object with ``read_image`` and ``read_metadata`` methods for
        dependency injection in tests.
    mask_reader : GroundTruthReader | None, optional
        Concrete mask reader; defaults to ``GroundTruthReader()``.

    Notes
    -----
    CT and MRI preprocessing are strictly separated.  CT and MRI samples
    must not be mixed in a single ``DataLoader`` batch because their native
    spatial resolutions differ (512×512 vs 256×256).  Use separate dataset
    instances (and separate DataLoaders) for each modality.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        augment: bool = False,
        dicom_reader: DICOMReader | None = None,
        mask_reader: GroundTruthReader | None = None,
    ) -> None:
        # ------------------------------------------------------------------ #
        # 1. Filter: retain only labeled samples                             #
        # ------------------------------------------------------------------ #
        self._samples: list[Sample] = [s for s in samples if s.mask_path is not None]

        # ------------------------------------------------------------------ #
        # 2. Shared infrastructure                                           #
        # ------------------------------------------------------------------ #
        self._dicom_reader: DICOMReader = dicom_reader or DICOMReader()
        self._mask_reader: GroundTruthReader = mask_reader or GroundTruthReader()
        self._augmentation: TrainingAugmentation | None = (
            TrainingAugmentation() if augment else None
        )

        # ------------------------------------------------------------------ #
        # 3. Pre-compute per-volume MRI statistics (once at construction)    #
        # ------------------------------------------------------------------ #
        self._volume_stats: dict[_VolumeKey, tuple[float, float]] = {}
        self._precompute_mri_volume_stats()

    # ---------------------------------------------------------------------- #
    # Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _precompute_mri_volume_stats(self) -> None:
        """Accumulate non-zero-pixel mean and std for every MRI volume.

        Groups the labeled MRI samples by ``(patient_id, modality, sequence)``
        and calls :func:`~src.preprocessing.mri.compute_volume_stats` for
        each unique volume.  Only one DICOM slice is held in memory at a
        time during the accumulation pass.

        CT samples are skipped entirely; they do not use volume statistics.
        """
        # Collect image paths grouped by volume key
        mri_volume_paths: dict[_VolumeKey, list[Path]] = defaultdict(list)
        for sample in self._samples:
            if sample.modality == Modality.MRI:
                key: _VolumeKey = (
                    sample.patient_id,
                    sample.modality,
                    sample.sequence,
                )
                mri_volume_paths[key].append(sample.image_path)

        # Compute and cache stats for each MRI volume
        for key, paths in mri_volume_paths.items():
            mean, std = compute_volume_stats(paths, self._dicom_reader)
            self._volume_stats[key] = (mean, std)

    # ---------------------------------------------------------------------- #
    # Dataset protocol                                                        #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Return the number of labeled (supervised) samples."""
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, preprocess, and optionally augment one sample.

        Parameters
        ----------
        index : int
            Index into the filtered labeled-sample list.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            * **image** — shape ``[1, H, W]``, dtype ``torch.float32``
            * **mask**  — shape ``[H, W]``,    dtype ``torch.long``,
              values in ``{0, 1}``

        Notes
        -----
        CT natural resolution: 512×512.  MRI natural resolution: 256×256.
        No spatial resampling is applied.
        """
        sample = self._samples[index]

        # ------------------------------------------------------------------ #
        # Image: raw pixels → modality-specific preprocessing → float32     #
        # ------------------------------------------------------------------ #
        raw_pixels = self._dicom_reader.read_image(sample.image_path)

        if sample.modality == Modality.CT:
            metadata = self._dicom_reader.read_metadata(sample.image_path)
            slope = float(metadata["rescale_slope"])
            intercept = float(metadata["rescale_intercept"])
            preprocessed = preprocess_ct(raw_pixels, slope, intercept)
        else:
            # MRI: look up cached volume-level statistics
            key: _VolumeKey = (
                sample.patient_id,
                sample.modality,
                sample.sequence,
            )
            volume_mean, volume_std = self._volume_stats[key]
            preprocessed = preprocess_mri(raw_pixels, volume_mean, volume_std)

        # [H, W] → [1, H, W] float32
        image_tensor: torch.Tensor = torch.from_numpy(preprocessed).unsqueeze(0)

        # ------------------------------------------------------------------ #
        # Mask: raw PNG → binary liver projection → torch.long              #
        # ------------------------------------------------------------------ #
        raw_mask = self._mask_reader.read_mask(sample.mask_path)
        binary_mask = binarize_liver_mask(raw_mask, sample.modality)
        mask_tensor: torch.Tensor = torch.from_numpy(
            binary_mask.astype(np.int64)
        )

        # ------------------------------------------------------------------ #
        # Augmentation (training splits only)                                #
        # ------------------------------------------------------------------ #
        if self._augmentation is not None:
            image_tensor, mask_tensor = self._augmentation(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
