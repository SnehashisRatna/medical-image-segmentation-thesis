"""Comparison utilities for experiment records from a single sample."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.core.enums import Modality
from src.core.experiment_record import ExperimentRecord


@dataclass(slots=True)
class ExperimentComparison:
    """Organize experiment records for one dataset sample.

    The class stores results that have already been produced and evaluated. It
    deliberately does not perform inference, calculate metrics, visualize
    results, or persist data.

    Parameters
    ----------
    sample_id : str
        Identifier of the dataset sample shared by all records.
    dataset_name : str
        Name of the dataset shared by all records.
    modality : Modality
        Imaging modality shared by all records.
    experiments : dict[str, ExperimentRecord], default={}
        Records keyed by their model names.
    """

    sample_id: str
    dataset_name: str
    modality: Modality
    experiments: dict[str, ExperimentRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate comparison metadata and any initial experiment records."""
        self._validate_metadata()

        if not isinstance(self.experiments, dict):
            raise TypeError("experiments must be a dict of model names to records.")

        for model_name, record in self.experiments.items():
            if not isinstance(model_name, str):
                raise TypeError("experiment model names must be str instances.")
            self._validate_record(record)
            if model_name != record.model_name:
                raise ValueError(
                    "Experiment dictionary keys must match their record model names."
                )

    def add(self, record: ExperimentRecord) -> None:
        """Add an experiment record after checking its compatibility.

        Parameters
        ----------
        record : ExperimentRecord
            Evaluated record to add.

        Raises
        ------
        TypeError
            If ``record`` is not an :class:`ExperimentRecord`.
        ValueError
            If the record has inconsistent metadata or its model already exists.
        """
        self._validate_record(record)
        if record.model_name in self.experiments:
            raise ValueError(
                f"An experiment for model '{record.model_name}' already exists."
            )
        self.experiments[record.model_name] = record

    def remove(self, model_name: str) -> ExperimentRecord:
        """Remove the record associated with a model name.

        Parameters
        ----------
        model_name : str
            Name of the model to remove.

        Returns
        -------
        ExperimentRecord
            The removed record.

        Raises
        ------
        KeyError
            If no record exists for ``model_name``.
        """
        if model_name not in self.experiments:
            raise KeyError(f"No experiment exists for model '{model_name}'.")
        return self.experiments.pop(model_name)

    def get(self, model_name: str) -> ExperimentRecord:
        """Return the record associated with a model name.

        Parameters
        ----------
        model_name : str
            Name of the model to retrieve.

        Returns
        -------
        ExperimentRecord
            The stored record for ``model_name``.

        Raises
        ------
        KeyError
            If no record exists for ``model_name``.
        """
        try:
            return self.experiments[model_name]
        except KeyError as error:
            raise KeyError(f"No experiment exists for model '{model_name}'.") from error

    def exists(self, model_name: str) -> bool:
        """Return whether a model has a stored experiment record.

        Parameters
        ----------
        model_name : str
            Name of the model to look up.

        Returns
        -------
        bool
            ``True`` when a record exists for ``model_name``.
        """
        return model_name in self.experiments

    def models(self) -> list[str]:
        """Return model names in insertion order.

        Returns
        -------
        list[str]
            Names of all stored models.
        """
        return list(self.experiments)

    def metrics(self) -> list[str]:
        """Return every unique metric available across stored records.

        Returns
        -------
        list[str]
            Alphabetically sorted metric names, or an empty list when no
            metrics are available.
        """
        return sorted(
            {
                metric_name
                for record in self.experiments.values()
                for metric_name in record.metrics
            }
        )

    def metric_table(self, metric_name: str) -> dict[str, float]:
        """Collect one metric for every stored model.

        Parameters
        ----------
        metric_name : str
            Name of the metric to collect.

        Returns
        -------
        dict[str, float]
            Metric values keyed by model name.

        Raises
        KeyError
            If any record does not contain ``metric_name``.
        """
        return {
            model_name: self._metric_value(record, metric_name)
            for model_name, record in self.experiments.items()
        }

    def rank_by(
        self, metric_name: str, descending: bool = True
    ) -> list[ExperimentRecord]:
        """Return records sorted by an existing metric without mutating storage.

        Parameters
        ----------
        metric_name : str
            Name of the metric used for ranking.
        descending : bool, default=True
            Whether higher values should appear first.

        Returns
        -------
        list[ExperimentRecord]
            Records ordered by the requested metric.

        Raises
        KeyError
            If any record does not contain ``metric_name``.
        """
        metric_values = self.metric_table(metric_name)
        return sorted(
            self.experiments.values(),
            key=lambda record: metric_values[record.model_name],
            reverse=descending,
        )

    def best(self, metric_name: str) -> ExperimentRecord:
        """Return the highest-scoring record for a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric used for comparison.

        Returns
        -------
        ExperimentRecord
            The record with the largest metric value.

        Raises
        ValueError
            If no experiments are available.
        KeyError
            If any record does not contain ``metric_name``.
        """
        self._require_experiments()
        return self.rank_by(metric_name)[0]

    def worst(self, metric_name: str) -> ExperimentRecord:
        """Return the lowest-scoring record for a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric used for comparison.

        Returns
        -------
        ExperimentRecord
            The record with the smallest metric value.

        Raises
        ValueError
            If no experiments are available.
        KeyError
            If any record does not contain ``metric_name``.
        """
        self._require_experiments()
        return self.rank_by(metric_name, descending=False)[0]

    def summary(self) -> dict[str, Any]:
        """Return lightweight metadata about the comparison.

        Returns
        -------
        dict[str, Any]
            Sample metadata, model information, available metrics, and the best
            model for every available metric when experiments are present.
        """
        available_metrics = self.metrics()
        summary = {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "modality": self.modality.value,
            "number_of_models": len(self),
            "model_names": self.models(),
            "available_metrics": available_metrics,
        }
        if self.experiments:
            summary["best_models"] = {
                metric_name: self._best_available_metric(metric_name).model_name
                for metric_name in available_metrics
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Return a nested, serialization-friendly comparison representation.

        NumPy arrays, UUIDs, datetimes, and enum values are converted to Python
        values that standard serializers can consume.

        Returns
        -------
        dict[str, Any]
            Comparison metadata and serialized records keyed by model name.
        """
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "modality": self.modality.value,
            "experiments": {
                model_name: self._record_to_dict(record)
                for model_name, record in self.experiments.items()
            },
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return experiment metadata and metric values in a DataFrame.

        Models without a particular metric receive a missing value in that
        metric column. Model names are included in a regular ``model_name``
        column. No files are written.

        Returns
        -------
        pandas.DataFrame
            One row per model with model metadata and available metric columns.
        """
        columns = ["model_name", *self.metrics(), "inference_time", "memory_usage"]
        rows = [
            {
                "model_name": record.model_name,
                **dict(record.metrics),
                "inference_time": float(record.inference_time),
                "memory_usage": record.memory_usage,
            }
            for record in self.experiments.values()
        ]
        return pd.DataFrame(rows, columns=columns)

    def __len__(self) -> int:
        """Return the number of stored experiment records."""
        return len(self.experiments)

    def __iter__(self) -> Iterator[ExperimentRecord]:
        """Iterate over stored records in model insertion order."""
        return iter(self.experiments.values())

    def __contains__(self, model_name: object) -> bool:
        """Return whether a model name is present in the comparison."""
        return model_name in self.experiments

    def _validate_metadata(self) -> None:
        """Validate metadata shared by every record in the comparison."""
        if not isinstance(self.sample_id, str):
            raise TypeError("sample_id must be a str.")
        if not isinstance(self.dataset_name, str):
            raise TypeError("dataset_name must be a str.")
        if not isinstance(self.modality, Modality):
            raise TypeError("modality must be a Modality instance.")
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty or whitespace.")
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must not be empty or whitespace.")

    def _validate_record(self, record: ExperimentRecord) -> None:
        """Validate one record against this comparison's shared metadata."""
        if not isinstance(record, ExperimentRecord):
            raise TypeError("record must be an ExperimentRecord instance.")
        if record.sample_id != self.sample_id:
            raise ValueError("record sample_id must match the comparison sample_id.")
        if record.dataset_name != self.dataset_name:
            raise ValueError(
                "record dataset_name must match the comparison dataset_name."
            )
        if record.modality != self.modality:
            raise ValueError("record modality must match the comparison modality.")

    @staticmethod
    def _metric_value(record: ExperimentRecord, metric_name: str) -> float:
        """Return a metric value or raise an informative missing-metric error."""
        try:
            value = record.metrics[metric_name]
        except KeyError as error:
            raise KeyError(
                f"Metric '{metric_name}' is not available for model "
                f"'{record.model_name}'."
            ) from error
        return float(value)

    def _require_experiments(self) -> None:
        """Raise a clear error when an operation requires at least one record."""
        if not self.experiments:
            raise ValueError("At least one experiment is required for comparison.")

    def _best_available_metric(self, metric_name: str) -> ExperimentRecord:
        """Return the best record among models that provide a metric.

        ``best`` intentionally requires every stored model to contain its
        requested metric. Summaries instead report every metric available in
        the comparison, so a temporary compatible subset is used when metrics
        are not shared by all models.
        """
        records_with_metric = {
            model_name: record
            for model_name, record in self.experiments.items()
            if metric_name in record.metrics
        }
        if len(records_with_metric) == len(self.experiments):
            return self.best(metric_name)
        return ExperimentComparison(
            sample_id=self.sample_id,
            dataset_name=self.dataset_name,
            modality=self.modality,
            experiments=records_with_metric,
        ).best(metric_name)

    @staticmethod
    def _record_to_dict(record: ExperimentRecord) -> dict[str, Any]:
        """Convert an immutable record into serializer-compatible primitives."""
        return {
            "experiment_id": str(record.experiment_id),
            "sample_id": record.sample_id,
            "dataset_name": record.dataset_name,
            "modality": record.modality.value,
            "image": record.image.tolist(),
            "ground_truth": record.ground_truth.tolist(),
            "prediction": record.prediction.tolist(),
            "metrics": dict(record.metrics),
            "inference_time": float(record.inference_time),
            "model_name": record.model_name,
            "timestamp": record.timestamp.isoformat(),
            "patient_id": record.patient_id,
            "slice_index": record.slice_index,
            "probability_map": (
                None
                if record.probability_map is None
                else record.probability_map.tolist()
            ),
            "memory_usage": (
                None if record.memory_usage is None else float(record.memory_usage)
            ),
            "epoch": record.epoch,
            "fold": record.fold,
            "notes": record.notes,
        }
