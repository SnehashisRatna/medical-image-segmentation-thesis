"""Segmentation metric tracking and accumulation."""

from __future__ import annotations

from typing import Iterable

import torch

from src.datasets.sample import Sample


class SegmentationMetricTracker:
    """Tracks True Positives, False Positives, and False Negatives per volume.

    Accumulates batch-level predictions across multiple batches, grouping counts
    by the volume identity ``(patient_id, modality, sequence)``.
    """

    def __init__(self) -> None:
        """Initialize an empty metric tracker."""
        self._counts: dict[tuple, dict[str, int]] = {}
        self.reset()

    def reset(self) -> None:
        """Clear the internal state to start a new tracking epoch."""
        self._counts = {}

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        samples: Iterable[Sample],
    ) -> None:
        """Update counts from a batch of binary predictions and targets.

        Parameters
        ----------
        predictions : torch.Tensor
            Binary predictions of shape ``[B, H, W]``.
        targets : torch.Tensor
            Binary ground truth masks of shape ``[B, H, W]``.
        samples : Iterable[Sample]
            The corresponding ``Sample`` objects for the batch.

        Raises
        ------
        ValueError
            If the batch dimension of tensors does not match the length of samples.
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        b = predictions.shape[0]
        sample_list = list(samples)
        if b != len(sample_list):
            raise ValueError("Batch dimension must match number of samples.")

        # Ensure boolean for correct intersection/union math
        pred_bool = predictions > 0
        tgt_bool = targets > 0

        for i in range(b):
            sample = sample_list[i]
            key = (sample.patient_id, sample.modality, sample.sequence)

            p = pred_bool[i]
            t = tgt_bool[i]

            tp = (p & t).sum().item()
            fp = (p & ~t).sum().item()
            fn = (~p & t).sum().item()

            if key not in self._counts:
                self._counts[key] = {"TP": 0, "FP": 0, "FN": 0}

            self._counts[key]["TP"] += int(tp)
            self._counts[key]["FP"] += int(fp)
            self._counts[key]["FN"] += int(fn)

    def compute(self) -> dict[str, float]:
        """Compute the unweighted macro-average metrics across all tracked volumes.

        Calculates Dice, IoU, Precision, and Recall for each volume, applying
        the exact empty-volume policy, then averages them.

        Empty-Volume Policy:
        - GT empty + Prediction empty (TP=0, FP=0, FN=0) -> 1.0
        - GT empty + Prediction non-empty (TP=0, FP>0, FN=0) -> 0.0
        - GT non-empty + Prediction empty (TP=0, FP=0, FN>0) -> 0.0

        Returns
        -------
        dict[str, float]
            A dictionary containing "dice", "iou", "precision", and "recall".
            Returns 0.0 for all if no samples were tracked.
        """
        if not self._counts:
            return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

        total_dice = 0.0
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        n = len(self._counts)

        for counts in self._counts.values():
            tp = counts["TP"]
            fp = counts["FP"]
            fn = counts["FN"]

            if tp == 0 and fp == 0 and fn == 0:
                # Perfect background prediction
                dice = 1.0
                iou = 1.0
                precision = 1.0
                recall = 1.0
            else:
                dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                iou = (tp) / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                precision = (tp) / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = (tp) / (tp + fn) if (tp + fn) > 0 else 0.0

            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall

        return {
            "dice": total_dice / n,
            "iou": total_iou / n,
            "precision": total_precision / n,
            "recall": total_recall / n,
        }
