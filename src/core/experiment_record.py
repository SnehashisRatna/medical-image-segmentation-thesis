"""Immutable experiment inference-result data contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from uuid import UUID

import numpy as np

from src.core.enums import Modality


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Store the result of one model performing inference on one sample.

    Parameters
    ----------
    experiment_id : UUID
        Unique identifier for the experiment record.
    sample_id : str
        Unique identifier for the processed sample.
    dataset_name : str
        Name of the source dataset.
    modality : Modality
        Imaging modality of the sample.
    image : np.ndarray
        Two-dimensional input medical image.
    ground_truth : np.ndarray
        Two-dimensional ground-truth segmentation mask.
    prediction : np.ndarray
        Two-dimensional model prediction.
    metrics : Mapping[str, float]
        Evaluation metric names and numeric values.
    inference_time : float
        Inference duration in seconds.
    model_name : str
        Name of the model that produced the prediction.
    timestamp : datetime
        Time at which the record was created.
    patient_id : str | None, default=None
        Optional patient identifier.
    slice_index : int | None, default=None
        Optional non-negative slice index.
    probability_map : np.ndarray | None, default=None
        Optional two-dimensional probability map.
    memory_usage : float | None, default=None
        Optional peak memory usage in megabytes.
    epoch : int | None, default=None
        Optional non-negative training epoch.
    fold : int | None, default=None
        Optional non-negative cross-validation fold.
    notes : str | None, default=None
        Optional free-form notes.
    """

    experiment_id: UUID
    sample_id: str
    dataset_name: str
    modality: Modality
    image: np.ndarray
    ground_truth: np.ndarray
    prediction: np.ndarray
    metrics: Mapping[str, float]
    inference_time: float
    model_name: str
    timestamp: datetime
    patient_id: str | None = None
    slice_index: int | None = None
    probability_map: np.ndarray | None = None
    memory_usage: float | None = None
    epoch: int | None = None
    fold: int | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate the experiment record according to the frozen contract."""
        # Type validation
        if not isinstance(self.experiment_id, UUID):
            raise TypeError("experiment_id must be a UUID instance.")
        if not isinstance(self.sample_id, str):
            raise TypeError("sample_id must be a str.")
        if not isinstance(self.dataset_name, str):
            raise TypeError("dataset_name must be a str.")
        if not isinstance(self.modality, Modality):
            raise TypeError("modality must be a Modality instance.")
        if self.patient_id is not None and not isinstance(self.patient_id, str):
            raise TypeError("patient_id must be a str or None.")
        if self.slice_index is not None and not isinstance(self.slice_index, int):
            raise TypeError("slice_index must be an int or None.")
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray.")
        if not isinstance(self.ground_truth, np.ndarray):
            raise TypeError("ground_truth must be a numpy.ndarray.")
        if not isinstance(self.prediction, np.ndarray):
            raise TypeError("prediction must be a numpy.ndarray.")
        if self.probability_map is not None and not isinstance(
            self.probability_map, np.ndarray
        ):
            raise TypeError("probability_map must be a numpy.ndarray or None.")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must implement Mapping.")
        self._validate_metrics(validate_values=False)
        if not isinstance(self.inference_time, Real):
            raise TypeError("inference_time must be a real number.")
        if self.memory_usage is not None and not isinstance(
            self.memory_usage, Real
        ):
            raise TypeError("memory_usage must be a real number or None.")
        if not isinstance(self.model_name, str):
            raise TypeError("model_name must be a str.")
        if self.epoch is not None and not isinstance(self.epoch, int):
            raise TypeError("epoch must be an int or None.")
        if self.fold is not None and not isinstance(self.fold, int):
            raise TypeError("fold must be an int or None.")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime instance.")
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("notes must be a str or None.")

        # Structure validation
        if self.image.ndim != 2:
            raise ValueError("image must be two-dimensional.")
        if self.ground_truth.ndim != 2:
            raise ValueError("ground_truth must be two-dimensional.")
        if self.prediction.ndim != 2:
            raise ValueError("prediction must be two-dimensional.")
        if self.image.shape != self.ground_truth.shape:
            raise ValueError("image and ground_truth must have the same shape.")
        if self.ground_truth.shape != self.prediction.shape:
            raise ValueError("ground_truth and prediction must have the same shape.")
        if self.probability_map is not None:
            if self.probability_map.ndim != 2:
                raise ValueError("probability_map must be two-dimensional.")
            if self.probability_map.shape != self.prediction.shape:
                raise ValueError(
                    "probability_map and prediction must have the same shape."
                )

        # Value validation
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty or whitespace.")
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must not be empty or whitespace.")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty or whitespace.")
        if self.patient_id is not None and not self.patient_id.strip():
            raise ValueError("patient_id must not be empty or whitespace.")
        if self.slice_index is not None and self.slice_index < 0:
            raise ValueError("slice_index must be greater than or equal to zero.")
        if self.inference_time < 0:
            raise ValueError("inference_time must be greater than or equal to zero.")
        if self.memory_usage is not None and self.memory_usage < 0:
            raise ValueError("memory_usage must be greater than or equal to zero.")
        if self.epoch is not None and self.epoch < 0:
            raise ValueError("epoch must be greater than or equal to zero.")
        if self.fold is not None and self.fold < 0:
            raise ValueError("fold must be greater than or equal to zero.")
        self._validate_metrics(validate_values=True)

    def _validate_metrics(self, *, validate_values: bool) -> None:
        """Validate metric types or values for the active validation stage.

        Parameters
        ----------
        validate_values : bool
            Whether to apply value validation after metric types are verified.
        """
        for key, value in self.metrics.items():
            if not validate_values:
                if not isinstance(key, str):
                    raise TypeError("metric keys must be str instances.")
                if not isinstance(value, Real):
                    raise TypeError("metric values must be real numbers.")
                continue
            if not key.strip():
                raise ValueError("metric keys must not be empty or whitespace.")
            if not math.isfinite(value):
                raise ValueError("metric values must be finite.")
