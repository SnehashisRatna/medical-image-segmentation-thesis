"""Unit tests for the frozen :class:`ExperimentRecord` contract."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import datetime
from uuid import UUID, uuid4

import numpy as np
import pytest

from src.core.enums import Modality
from src.core.experiment_record import ExperimentRecord


########################################################################
# Fixtures
########################################################################


@pytest.fixture
def valid_uuid() -> UUID:
    """Return a deterministic UUID for a valid record."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def valid_timestamp() -> datetime:
    """Return a valid timestamp for a record."""
    return datetime(2025, 1, 2, 3, 4, 5)


@pytest.fixture
def valid_image() -> np.ndarray:
    """Return a valid two-dimensional input image."""
    return np.arange(9, dtype=np.float32).reshape(3, 3)


@pytest.fixture
def valid_ground_truth() -> np.ndarray:
    """Return a valid two-dimensional ground-truth mask."""
    return np.zeros((3, 3), dtype=np.uint8)


@pytest.fixture
def valid_prediction() -> np.ndarray:
    """Return a valid two-dimensional prediction mask."""
    return np.ones((3, 3), dtype=np.uint8)


@pytest.fixture
def valid_probability_map() -> np.ndarray:
    """Return a valid two-dimensional probability map."""
    return np.full((3, 3), 0.5, dtype=np.float32)


@pytest.fixture
def valid_metrics() -> dict[str, float]:
    """Return valid evaluation metrics."""
    return {"dice": 0.95, "iou": 0.91}


@pytest.fixture
def valid_record_kwargs(
    valid_uuid: UUID,
    valid_timestamp: datetime,
    valid_image: np.ndarray,
    valid_ground_truth: np.ndarray,
    valid_prediction: np.ndarray,
    valid_probability_map: np.ndarray,
    valid_metrics: dict[str, float],
) -> dict[str, object]:
    """Return every valid constructor argument for ``ExperimentRecord``."""
    return {
        "experiment_id": valid_uuid,
        "sample_id": "sample-001",
        "dataset_name": "CHAOS",
        "modality": Modality.CT,
        "patient_id": "patient-001",
        "slice_index": 4,
        "image": valid_image,
        "ground_truth": valid_ground_truth,
        "prediction": valid_prediction,
        "probability_map": valid_probability_map,
        "metrics": valid_metrics,
        "inference_time": 1.25,
        "memory_usage": 128.0,
        "model_name": "U-Net",
        "epoch": 10,
        "fold": 2,
        "timestamp": valid_timestamp,
        "notes": "valid record",
    }


@pytest.fixture
def make_record(
    valid_record_kwargs: dict[str, object],
) -> Callable[..., ExperimentRecord]:
    """Return a factory for valid records with per-test field overrides."""
    def _factory(**overrides: object) -> ExperimentRecord:
        kwargs = valid_record_kwargs.copy()
        kwargs.update(overrides)
        return ExperimentRecord(**kwargs)

    return _factory


@pytest.fixture
def valid_record(make_record: Callable[..., ExperimentRecord]) -> ExperimentRecord:
    """Construct and return a valid experiment record."""
    return make_record()


########################################################################
# Happy Path
########################################################################


def test_valid_record_stores_every_field(
    valid_record: ExperimentRecord,
    valid_record_kwargs: dict[str, object],
) -> None:
    """A valid record stores all supplied fields unchanged."""
    assert valid_record.experiment_id == valid_record_kwargs["experiment_id"]
    assert valid_record.sample_id == valid_record_kwargs["sample_id"]
    assert valid_record.dataset_name == valid_record_kwargs["dataset_name"]
    assert valid_record.modality is valid_record_kwargs["modality"]
    assert valid_record.patient_id == valid_record_kwargs["patient_id"]
    assert valid_record.slice_index == valid_record_kwargs["slice_index"]
    np.testing.assert_array_equal(valid_record.image, valid_record_kwargs["image"])
    np.testing.assert_array_equal(
        valid_record.ground_truth, valid_record_kwargs["ground_truth"]
    )
    np.testing.assert_array_equal(
        valid_record.prediction, valid_record_kwargs["prediction"]
    )
    np.testing.assert_array_equal(
        valid_record.probability_map, valid_record_kwargs["probability_map"]
    )
    assert valid_record.metrics == valid_record_kwargs["metrics"]
    assert valid_record.inference_time == valid_record_kwargs["inference_time"]
    assert valid_record.memory_usage == valid_record_kwargs["memory_usage"]
    assert valid_record.model_name == valid_record_kwargs["model_name"]
    assert valid_record.epoch == valid_record_kwargs["epoch"]
    assert valid_record.fold == valid_record_kwargs["fold"]
    assert valid_record.timestamp == valid_record_kwargs["timestamp"]
    assert valid_record.notes == valid_record_kwargs["notes"]


def test_accepts_random_uuid(make_record: Callable[..., ExperimentRecord]) -> None:
    """Accept any valid UUID rather than a particular UUID value."""
    random_uuid = uuid4()
    record = make_record(experiment_id=random_uuid)
    assert record.experiment_id == random_uuid


def test_preserves_modality_identity(valid_record: ExperimentRecord) -> None:
    """Preserve the exact Modality enum member supplied to the record."""
    assert valid_record.modality is Modality.CT


def test_accepts_alternate_array_dtypes(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Accept valid arrays independently of their NumPy dtypes."""
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    ground_truth = np.zeros((3, 3), dtype=np.int16)
    prediction = np.ones((3, 3), dtype=np.int16)
    probability_map = np.full((3, 3), 0.5, dtype=np.float64)
    record = make_record(
        image=image,
        ground_truth=ground_truth,
        prediction=prediction,
        probability_map=probability_map,
    )
    assert record.image.dtype == np.float64
    assert record.ground_truth.dtype == np.int16
    assert record.prediction.dtype == np.int16
    assert record.probability_map is not None
    assert record.probability_map.dtype == np.float64


def test_valid_record_without_probability_map(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Accept the common inference case without a probability map."""
    record = make_record(probability_map=None)
    assert record.probability_map is None


########################################################################
# Immutability
########################################################################


def test_record_is_frozen(valid_record: ExperimentRecord) -> None:
    """A record rejects assignment to its fields."""
    with pytest.raises(FrozenInstanceError):
        valid_record.sample_id = "changed"  # type: ignore[misc]


########################################################################
# Type Validation
########################################################################


def test_rejects_invalid_experiment_id(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-UUID experiment identifier."""
    with pytest.raises(TypeError):
        make_record(experiment_id="not-a-uuid")


def test_rejects_invalid_sample_id(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-string sample identifier."""
    with pytest.raises(TypeError):
        make_record(sample_id=1)


def test_rejects_invalid_dataset_name(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-string dataset name."""
    with pytest.raises(TypeError):
        make_record(dataset_name=1)


def test_rejects_invalid_modality(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-Modality value."""
    with pytest.raises(TypeError):
        make_record(modality="CT")


def test_rejects_invalid_patient_id(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-string patient identifier."""
    with pytest.raises(TypeError):
        make_record(patient_id=1)


def test_rejects_invalid_image(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-array input image."""
    with pytest.raises(TypeError):
        make_record(image=[[0, 1], [1, 0]])


def test_rejects_invalid_ground_truth(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-array ground-truth mask."""
    with pytest.raises(TypeError):
        make_record(ground_truth=[[0, 1], [1, 0]])


def test_rejects_invalid_prediction(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-array prediction mask."""
    with pytest.raises(TypeError):
        make_record(prediction=[[0, 1], [1, 0]])


def test_rejects_invalid_probability_map(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a non-array probability map."""
    with pytest.raises(TypeError):
        make_record(probability_map=[[0.5, 0.5], [0.5, 0.5]])


def test_rejects_invalid_metrics(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-Mapping metrics value."""
    with pytest.raises(TypeError):
        make_record(metrics=[("dice", 0.9)])


def test_rejects_invalid_inference_time(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a non-real inference time."""
    with pytest.raises(TypeError):
        make_record(inference_time="1.25")


def test_rejects_invalid_memory_usage(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a non-real memory usage value."""
    with pytest.raises(TypeError):
        make_record(memory_usage="128")


def test_rejects_invalid_model_name(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-string model name."""
    with pytest.raises(TypeError):
        make_record(model_name=1)


def test_rejects_invalid_epoch(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-integer epoch."""
    with pytest.raises(TypeError):
        make_record(epoch=1.5)


def test_rejects_invalid_fold(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-integer fold."""
    with pytest.raises(TypeError):
        make_record(fold=1.5)


def test_rejects_invalid_timestamp(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a non-datetime timestamp."""
    with pytest.raises(TypeError):
        make_record(timestamp="2025-01-02T03:04:05")


def test_rejects_invalid_notes(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject non-string notes."""
    with pytest.raises(TypeError):
        make_record(notes=1)


########################################################################
# Structure Validation
########################################################################


def test_rejects_non_2d_image(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject an image whose number of dimensions is not two."""
    with pytest.raises(ValueError):
        make_record(image=np.zeros((1, 3, 3)))


def test_rejects_non_2d_ground_truth(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a ground-truth array whose number of dimensions is not two."""
    with pytest.raises(ValueError):
        make_record(ground_truth=np.zeros((1, 3, 3)))


def test_rejects_non_2d_prediction(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a prediction array whose number of dimensions is not two."""
    with pytest.raises(ValueError):
        make_record(prediction=np.zeros((1, 3, 3)))


def test_rejects_non_2d_probability_map(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a probability map whose number of dimensions is not two."""
    with pytest.raises(ValueError):
        make_record(probability_map=np.zeros((1, 3, 3)))


def test_rejects_image_ground_truth_shape_mismatch(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject an image and ground-truth shape mismatch."""
    with pytest.raises(ValueError):
        make_record(ground_truth=np.zeros((2, 3)))


def test_rejects_ground_truth_prediction_shape_mismatch(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a ground-truth and prediction shape mismatch."""
    with pytest.raises(ValueError):
        make_record(prediction=np.zeros((2, 3)))


def test_rejects_probability_map_prediction_shape_mismatch(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a probability-map and prediction shape mismatch."""
    with pytest.raises(ValueError):
        make_record(probability_map=np.zeros((2, 3)))


########################################################################
# Value Validation
########################################################################


def test_rejects_empty_sample_id(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject an empty sample identifier."""
    with pytest.raises(ValueError):
        make_record(sample_id="   ")


def test_rejects_empty_dataset_name(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject an empty dataset name."""
    with pytest.raises(ValueError):
        make_record(dataset_name="   ")


def test_rejects_empty_model_name(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject an empty model name."""
    with pytest.raises(ValueError):
        make_record(model_name="   ")


def test_rejects_empty_patient_id(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject an empty patient identifier."""
    with pytest.raises(ValueError):
        make_record(patient_id="   ")


@pytest.mark.parametrize("whitespace", ["\t", "\n", "\r\n"])
@pytest.mark.parametrize(
    "field_name", ["sample_id", "dataset_name", "model_name", "patient_id"]
)
def test_rejects_additional_whitespace_only_values(
    make_record: Callable[..., ExperimentRecord],
    field_name: str,
    whitespace: str,
) -> None:
    """Reject tab and newline-only values for constrained string fields."""
    with pytest.raises(ValueError):
        make_record(**{field_name: whitespace})


def test_rejects_negative_slice_index(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a negative slice index."""
    with pytest.raises(ValueError):
        make_record(slice_index=-1)


def test_rejects_negative_inference_time(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a negative inference time."""
    with pytest.raises(ValueError):
        make_record(inference_time=-1.0)


def test_rejects_negative_memory_usage(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject a negative memory usage value."""
    with pytest.raises(ValueError):
        make_record(memory_usage=-1.0)


def test_rejects_negative_epoch(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a negative epoch."""
    with pytest.raises(ValueError):
        make_record(epoch=-1)


def test_rejects_negative_fold(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a negative fold."""
    with pytest.raises(ValueError):
        make_record(fold=-1)


def test_rejects_empty_metric_key(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject an empty metric key."""
    with pytest.raises(ValueError):
        make_record(metrics={"   ": 0.9})


def test_rejects_nan_metric_value(make_record: Callable[..., ExperimentRecord]) -> None:
    """Reject a NaN metric value."""
    with pytest.raises(ValueError):
        make_record(metrics={"dice": np.nan})


def test_rejects_infinite_metric_value(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject an infinite metric value."""
    with pytest.raises(ValueError):
        make_record(metrics={"dice": np.inf})


########################################################################
# Metrics
########################################################################


def test_accepts_valid_metrics(make_record: Callable[..., ExperimentRecord]) -> None:
    """Accept a mapping containing valid metrics."""
    metrics = {"dice": 0.95, "iou": 0.91}
    record = make_record(metrics=metrics)
    assert record.metrics == metrics


def test_accepts_empty_metrics(make_record: Callable[..., ExperimentRecord]) -> None:
    """Accept an empty metrics mapping because no metric is required."""
    record = make_record(metrics={})
    assert record.metrics == {}


def test_rejects_non_string_metric_key(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject metrics with a non-string key."""
    with pytest.raises(TypeError):
        make_record(metrics={1: 0.9})


def test_rejects_non_numeric_metric_value(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Reject metrics with a non-numeric value."""
    with pytest.raises(TypeError):
        make_record(metrics={"dice": "0.9"})


########################################################################
# Optional Fields
########################################################################


def test_optional_fields_may_all_be_none(
    make_record: Callable[..., ExperimentRecord],
) -> None:
    """Allow all optional fields to be explicitly set to None."""
    record = make_record(
        patient_id=None,
        slice_index=None,
        probability_map=None,
        memory_usage=None,
        epoch=None,
        fold=None,
        notes=None,
    )
    assert record.patient_id is None
    assert record.slice_index is None
    assert record.probability_map is None
    assert record.memory_usage is None
    assert record.epoch is None
    assert record.fold is None
    assert record.notes is None
