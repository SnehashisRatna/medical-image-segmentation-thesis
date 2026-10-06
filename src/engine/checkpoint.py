"""Model checkpointing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_metric: float,
    training_history: dict[str, list[float]],
    scheduler: LRScheduler | None = None,
) -> None:
    """Save a full training checkpoint to disk.

    Parameters
    ----------
    path : str | Path
        Destination file path. Missing parent directories are created.
    model : nn.Module
        The neural network model.
    optimizer : Optimizer
        The active optimizer.
    epoch : int
        The current training epoch.
    best_metric : float
        The best validation metric achieved so far.
    training_history : dict[str, list[float]]
        The history of training and validation metrics.
    scheduler : LRScheduler | None, optional
        The active learning rate scheduler, if any.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "training_history": training_history,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, output_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint from disk.

    Parameters
    ----------
    path : str | Path
        Path to the saved checkpoint file.
    model : nn.Module
        The neural network model to populate.
    optimizer : Optimizer | None, optional
        The optimizer to populate, if resuming training.
    scheduler : LRScheduler | None, optional
        The learning rate scheduler to populate, if resuming training.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the loaded metadata ('epoch', 'best_metric',
        'training_history').
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric", 0.0),
        "training_history": checkpoint.get("training_history", {}),
    }
