"""Tests for SegmentationMetricTracker."""

from pathlib import Path

import torch

from src.core.enums import Modality
from src.datasets.sample import Sample
from src.metrics.tracker import SegmentationMetricTracker


def test_metric_tracker_perfect_match():
    tracker = SegmentationMetricTracker()

    # 2 samples in batch
    predictions = torch.tensor([
        [[0, 1], [1, 0]],
        [[1, 1], [0, 0]]
    ])
    targets = torch.tensor([
        [[0, 1], [1, 0]],
        [[1, 1], [0, 0]]
    ])

    samples = [
        Sample(image_path=Path("1.dcm"), mask_path=Path("1.png"), patient_id="p1", modality=Modality.CT, sequence="ct", slice_index=1),
        Sample(image_path=Path("2.dcm"), mask_path=Path("2.png"), patient_id="p2", modality=Modality.CT, sequence="ct", slice_index=2),
    ]

    tracker.update(predictions, targets, samples)
    metrics = tracker.compute()

    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_metric_tracker_empty_volume_policy():
    tracker = SegmentationMetricTracker()

    # True Negative (Empty GT, Empty Pred)
    pred_tn = torch.tensor([[[0, 0], [0, 0]]])
    tgt_tn = torch.tensor([[[0, 0], [0, 0]]])
    s_tn = [Sample(image_path=Path(""), mask_path=Path(""), patient_id="tn", modality=Modality.CT, sequence="ct", slice_index=1)]

    # False Positive (Empty GT, Non-empty Pred)
    pred_fp = torch.tensor([[[1, 0], [0, 0]]])
    tgt_fp = torch.tensor([[[0, 0], [0, 0]]])
    s_fp = [Sample(image_path=Path(""), mask_path=Path(""), patient_id="fp", modality=Modality.CT, sequence="ct", slice_index=1)]

    # False Negative (Non-empty GT, Empty Pred)
    pred_fn = torch.tensor([[[0, 0], [0, 0]]])
    tgt_fn = torch.tensor([[[1, 0], [0, 0]]])
    s_fn = [Sample(image_path=Path(""), mask_path=Path(""), patient_id="fn", modality=Modality.CT, sequence="ct", slice_index=1)]

    tracker.update(pred_tn, tgt_tn, s_tn)
    tracker.update(pred_fp, tgt_fp, s_fp)
    tracker.update(pred_fn, tgt_fn, s_fn)

    metrics = tracker.compute()

    # 3 patients. TN -> 1.0, FP -> 0.0, FN -> 0.0. Average = 1.0 / 3 = 0.3333
    assert abs(metrics["dice"] - 0.3333333) < 1e-5


def test_metric_tracker_volume_accumulation():
    tracker = SegmentationMetricTracker()

    # Slice 1 of patient p1: TP=1, FP=1, FN=0
    pred1 = torch.tensor([[[1, 1], [0, 0]]])
    tgt1  = torch.tensor([[[1, 0], [0, 0]]])
    s1 = [Sample(image_path=Path(""), mask_path=Path(""), patient_id="p1", modality=Modality.CT, sequence="ct", slice_index=1)]

    # Slice 2 of patient p1: TP=0, FP=0, FN=1
    pred2 = torch.tensor([[[0, 0], [0, 0]]])
    tgt2  = torch.tensor([[[0, 1], [0, 0]]])
    s2 = [Sample(image_path=Path(""), mask_path=Path(""), patient_id="p1", modality=Modality.CT, sequence="ct", slice_index=2)]

    tracker.update(pred1, tgt1, s1)
    tracker.update(pred2, tgt2, s2)

    metrics = tracker.compute()

    # Total for p1: TP=1, FP=1, FN=1
    # Dice = 2TP / (2TP+FP+FN) = 2 / (2+1+1) = 2/4 = 0.5
    assert metrics["dice"] == 0.5
