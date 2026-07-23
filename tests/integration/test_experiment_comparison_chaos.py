"""Integration test for comparisons built from real CHAOS CT data."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from src.core.enums import Modality
from src.core.experiment_comparison import ExperimentComparison
from src.core.experiment_record import ExperimentRecord
from src.data.dataset_explorer import DatasetExplorer
from src.data.dicom_reader import DICOMReader
from src.data.ground_truth_reader import GroundTruthReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAOS_ROOT = PROJECT_ROOT / "data" / "raw" / "CHAOS_Train_Sets" / "Train_Sets"


def _find_ct_image_and_liver_mask() -> tuple[Path, Path] | None:
    """Return one matched CHAOS CT DICOM image and liver mask, if available."""
    explorer = DatasetExplorer(str(CHAOS_ROOT))
    if explorer.count_patients(explorer.ct_path) == 0:
        return None

    ground_truth_reader = GroundTruthReader()
    for patient_directory in sorted(explorer.ct_path.iterdir()):
        if not patient_directory.is_dir():
            continue

        dicom_directory = patient_directory / "DICOM_anon"
        ground_truth_directory = patient_directory / "Ground"
        if not dicom_directory.is_dir() or not ground_truth_directory.is_dir():
            continue

        dicoms_by_slice: dict[int, Path] = {}
        for dicom_path in sorted(dicom_directory.glob("*.dcm")):
            match = re.fullmatch(r"i0*(\d+),.*", dicom_path.name)
            if match is not None:
                dicoms_by_slice.setdefault(int(match.group(1)), dicom_path)

        for mask_path in sorted(ground_truth_directory.glob("liver_GT_*.png")):
            match = re.fullmatch(r"liver_GT_(\d+)", mask_path.stem)
            if match is None:
                continue
            dicom_path = dicoms_by_slice.get(int(match.group(1)))
            if dicom_path is not None and np.any(
                ground_truth_reader.read_mask(mask_path)
            ):
                return dicom_path, mask_path

    return None


def _prediction_with_removed_region(mask: np.ndarray, region_size: int) -> np.ndarray:
    """Create a deterministic prediction by removing a foreground region."""
    foreground_coordinates = np.argwhere(mask != 0)
    if foreground_coordinates.size == 0:
        pytest.skip("The selected CHAOS liver mask contains no foreground pixels.")

    center_y, center_x = foreground_coordinates[len(foreground_coordinates) // 2]
    half_size = region_size // 2
    start_y = max(0, int(center_y) - half_size)
    end_y = min(mask.shape[0], int(center_y) + half_size + 1)
    start_x = max(0, int(center_x) - half_size)
    end_x = min(mask.shape[1], int(center_x) + half_size + 1)

    prediction = mask.copy()
    prediction[start_y:end_y, start_x:end_x] = 0
    return prediction


def _create_record(
    *,
    experiment_id: UUID,
    model_name: str,
    prediction: np.ndarray,
    metrics: dict[str, float],
    inference_time: float,
    memory_usage: float,
    common_fields: dict[str, object],
) -> ExperimentRecord:
    """Create one experiment record sharing a real CHAOS sample's fields."""
    return ExperimentRecord(
        experiment_id=experiment_id,
        prediction=prediction,
        metrics=metrics,
        inference_time=inference_time,
        memory_usage=memory_usage,
        model_name=model_name,
        **common_fields,
    )


def test_experiment_comparison_with_real_chaos_ct_sample() -> None:
    """Compare deterministic model records constructed from a real CHAOS sample."""
    # Stage 1: discover a matched CT image and liver mask, or skip gracefully.
    sample_paths = _find_ct_image_and_liver_mask()
    if sample_paths is None:
        pytest.skip(
            f"A matched CHAOS CT image and liver mask are unavailable in {CHAOS_ROOT}"
        )
    dicom_path, mask_path = sample_paths

    # Stage 2: load the real image and mask through the project's readers.
    image = DICOMReader().read_image(dicom_path)
    ground_truth = GroundTruthReader().read_mask(mask_path)
    assert isinstance(image, np.ndarray)
    assert isinstance(ground_truth, np.ndarray)
    assert image.ndim == 2
    assert ground_truth.ndim == 2
    assert image.shape == ground_truth.shape

    # Stage 3: create deterministic predictions with progressively larger errors.
    # Fixed foreground edits, rather than random noise, make the integration
    # test reproducible while still simulating realistic segmentation variation.
    unetr_prediction = ground_truth.copy()
    attention_unet_prediction = _prediction_with_removed_region(ground_truth, 3)
    unet_prediction = _prediction_with_removed_region(ground_truth, 7)
    assert not np.array_equal(attention_unet_prediction, ground_truth)
    assert not np.array_equal(unet_prediction, ground_truth)

    # Stage 4: construct realistic records with known, monotonically improving metrics.
    common_record_fields = {
        "sample_id": f"CHAOS_CT_{dicom_path.parent.parent.name}_{mask_path.stem}",
        "dataset_name": "CHAOS",
        "modality": Modality.CT,
        "image": image,
        "ground_truth": ground_truth,
        "timestamp": datetime(2025, 1, 1, 12, 0, 0),
    }
    unet = _create_record(
        experiment_id=UUID("11111111-1111-1111-1111-111111111111"),
        prediction=unet_prediction,
        metrics={"dice": 0.91, "iou": 0.84, "precision": 0.90, "recall": 0.92},
        inference_time=0.045,
        memory_usage=512.0,
        model_name="UNet",
        common_fields=common_record_fields,
    )
    attention_unet = _create_record(
        experiment_id=UUID("22222222-2222-2222-2222-222222222222"),
        prediction=attention_unet_prediction,
        metrics={"dice": 0.94, "iou": 0.88, "precision": 0.93, "recall": 0.95},
        inference_time=0.052,
        memory_usage=640.0,
        model_name="AttentionUNet",
        common_fields=common_record_fields,
    )
    unetr = _create_record(
        experiment_id=UUID("33333333-3333-3333-3333-333333333333"),
        prediction=unetr_prediction,
        metrics={"dice": 0.97, "iou": 0.93, "precision": 0.96, "recall": 0.98},
        inference_time=0.071,
        memory_usage=896.0,
        model_name="UNETR",
        common_fields=common_record_fields,
    )

    # Stage 5: build the comparison object with all evaluated model records.
    comparison = ExperimentComparison(
        sample_id=unet.sample_id,
        dataset_name="CHAOS",
        modality=Modality.CT,
    )
    comparison.add(unet)
    comparison.add(attention_unet)
    comparison.add(unetr)

    # Stage 6: verify comparison, ranking, summary, and export operations.
    assert len(comparison) == 3
    assert comparison.models() == ["UNet", "AttentionUNet", "UNETR"]
    assert comparison.metrics() == ["dice", "iou", "precision", "recall"]
    assert comparison.metric_table("dice") == {
        "UNet": 0.91,
        "AttentionUNet": 0.94,
        "UNETR": 0.97,
    }
    assert comparison.best("dice").model_name == "UNETR"
    assert comparison.worst("dice").model_name == "UNet"
    assert [record.model_name for record in comparison.rank_by("dice")] == [
        "UNETR",
        "AttentionUNet",
        "UNet",
    ]
    for metric_name in ("iou", "precision", "recall"):
        assert comparison.best(metric_name).model_name == "UNETR"
        assert comparison.worst(metric_name).model_name == "UNet"

    summary = comparison.summary()
    assert summary["number_of_models"] == 3
    assert summary["available_metrics"] == ["dice", "iou", "precision", "recall"]
    assert summary["best_models"]["dice"] == "UNETR"

    exported = comparison.to_dict()
    assert exported["sample_id"] == comparison.sample_id
    assert exported["dataset_name"] == "CHAOS"
    assert exported["modality"] == Modality.CT.value
    assert set(exported["experiments"]) == {"UNet", "AttentionUNet", "UNETR"}

    dataframe = comparison.to_dataframe()
    assert isinstance(dataframe, pd.DataFrame)
    assert {"model_name", "inference_time", "memory_usage"}.issubset(dataframe.columns)
    assert len(dataframe) == 3
    assert dataframe["model_name"].nunique() == 3

    unetr_row = dataframe.loc[dataframe["model_name"] == "UNETR"].iloc[0]
    assert unetr_row["dice"] == 0.97
    assert unetr_row["iou"] == 0.93
    assert unetr_row["precision"] == 0.96
    assert unetr_row["recall"] == 0.98
    assert unetr_row["inference_time"] == 0.071
    assert unetr_row["memory_usage"] == 896.0
