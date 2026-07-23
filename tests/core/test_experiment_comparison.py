"""Unit tests for :class:`ExperimentComparison`."""

from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest

from src.core.enums import Modality
from src.core.experiment_comparison import ExperimentComparison
from src.core.experiment_record import ExperimentRecord


@pytest.fixture
def make_record() -> Callable[..., ExperimentRecord]:
    """Return a factory for compatible experiment records."""

    def _factory(**overrides: object) -> ExperimentRecord:
        values: dict[str, object] = {
            "experiment_id": uuid4(),
            "sample_id": "sample-001",
            "dataset_name": "CHAOS",
            "modality": Modality.CT,
            "image": np.zeros((2, 2)),
            "ground_truth": np.zeros((2, 2)),
            "prediction": np.ones((2, 2)),
            "metrics": {"dice": 0.9, "iou": 0.8},
            "inference_time": 0.1,
            "model_name": "UNet",
            "timestamp": datetime(2025, 1, 2, 3, 4, 5),
        }
        values.update(overrides)
        return ExperimentRecord(**values)

    return _factory


@pytest.fixture
def comparison() -> ExperimentComparison:
    """Return an empty valid comparison."""
    return ExperimentComparison("sample-001", "CHAOS", Modality.CT)


def test_add_retrieve_and_remove_record(
    comparison: ExperimentComparison, make_record: Callable[..., ExperimentRecord]
) -> None:
    """Records can be added, found, iterated, and removed by model name."""
    record = make_record()
    comparison.add(record)

    assert comparison.exists("UNet")
    assert "UNet" in comparison
    assert comparison.models() == ["UNet"]
    assert comparison.get("UNet") is record
    assert list(comparison) == [record]
    assert len(comparison) == 1

    assert comparison.remove("UNet") is record
    assert len(comparison) == 0
    with pytest.raises(KeyError, match="No experiment exists"):
        comparison.get("UNet")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sample_id", "other-sample"),
        ("dataset_name", "other-dataset"),
        ("modality", Modality.MRI),
    ],
)
def test_add_rejects_incompatible_records(
    comparison: ExperimentComparison,
    make_record: Callable[..., ExperimentRecord],
    field_name: str,
    value: object,
) -> None:
    """Records must share sample, dataset, and modality metadata."""
    with pytest.raises(ValueError):
        comparison.add(make_record(**{field_name: value}))


def test_add_rejects_duplicate_model_name(
    comparison: ExperimentComparison, make_record: Callable[..., ExperimentRecord]
) -> None:
    """Only one record per model can be stored."""
    comparison.add(make_record())
    with pytest.raises(ValueError, match="already exists"):
        comparison.add(make_record(experiment_id=UUID(int=1)))


def test_metrics_ranking_and_extremes(
    comparison: ExperimentComparison, make_record: Callable[..., ExperimentRecord]
) -> None:
    """Existing metrics can be collected and used without changing storage."""
    unet = make_record(model_name="UNet", metrics={"dice": 0.9})
    unetr = make_record(model_name="UNETR", metrics={"dice": 0.95})
    comparison.add(unet)
    comparison.add(unetr)

    assert comparison.metric_table("dice") == {"UNet": 0.9, "UNETR": 0.95}
    assert [record.model_name for record in comparison.rank_by("dice")] == [
        "UNETR",
        "UNet",
    ]
    assert comparison.models() == ["UNet", "UNETR"]
    assert comparison.best("dice") is unetr
    assert comparison.worst("dice") is unet
    assert comparison.summary()["best_models"] == {"dice": "UNETR"}
    with pytest.raises(KeyError, match="not available"):
        comparison.metric_table("iou")


def test_metrics_returns_unique_sorted_names(
    comparison: ExperimentComparison, make_record: Callable[..., ExperimentRecord]
) -> None:
    """Metric names are unique and sorted across all stored records."""
    comparison.add(make_record(metrics={"recall": 0.85, "dice": 0.9}))
    comparison.add(
        make_record(
            model_name="UNETR",
            metrics={"precision": 0.92, "iou": 0.8, "dice": 0.95},
        )
    )

    assert comparison.metrics() == ["dice", "iou", "precision", "recall"]
    assert comparison.summary()["best_models"] == {
        "dice": "UNETR",
        "iou": "UNETR",
        "precision": "UNETR",
        "recall": "UNet",
    }


def test_metrics_is_empty_without_experiments(
    comparison: ExperimentComparison,
) -> None:
    """An empty comparison exposes no metric names."""
    assert comparison.metrics() == []


def test_summary_serialization_and_dataframe(
    comparison: ExperimentComparison, make_record: Callable[..., ExperimentRecord]
) -> None:
    """Exports expose metadata, serializer-compatible values, and metrics."""
    comparison.add(
        make_record(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            probability_map=np.full((2, 2), 0.5),
        )
    )

    assert comparison.summary() == {
        "sample_id": "sample-001",
        "dataset_name": "CHAOS",
        "modality": "CT",
        "number_of_models": 1,
        "model_names": ["UNet"],
        "available_metrics": ["dice", "iou"],
        "best_models": {"dice": "UNet", "iou": "UNet"},
    }
    exported = comparison.to_dict()
    assert exported["modality"] == "CT"
    assert exported["experiments"]["UNet"]["experiment_id"] == (
        "12345678-1234-5678-1234-567812345678"
    )
    assert exported["experiments"]["UNet"]["image"] == [[0.0, 0.0], [0.0, 0.0]]

    frame = comparison.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == [
        "model_name",
        "dice",
        "iou",
        "inference_time",
        "memory_usage",
    ]
    assert frame.loc[0, "model_name"] == "UNet"
    assert frame.loc[0, "dice"] == 0.9
    assert frame.loc[0, "inference_time"] == 0.1
    assert pd.isna(frame.loc[0, "memory_usage"])


def test_empty_comparison_cannot_have_best_or_worst(
    comparison: ExperimentComparison,
) -> None:
    """Extrema operations require at least one record."""
    with pytest.raises(ValueError, match="At least one experiment"):
        comparison.best("dice")
    with pytest.raises(ValueError, match="At least one experiment"):
        comparison.worst("dice")
