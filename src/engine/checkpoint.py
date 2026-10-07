"""Model checkpointing utilities."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


def capture_rng_state() -> dict[str, Any]:
    """Capture the global RNG states of Python, NumPy, and PyTorch.

    The returned structure contains only tensors and plain Python types so
    that it remains loadable with ``torch.load(..., weights_only=True)``.

    Returns
    -------
    dict[str, Any]
        RNG states keyed by ``'python'``, ``'numpy'``, ``'torch'`` and,
        when CUDA is available, ``'cuda'``.
    """
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(np_name),
            "keys": torch.from_numpy(np.asarray(np_keys, dtype=np.int64)),
            "pos": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore global RNG states previously produced by ``capture_rng_state``.

    Missing entries are skipped, so partial states are tolerated.

    Parameters
    ----------
    state : dict[str, Any]
        RNG states as returned by :func:`capture_rng_state`.
    """
    if "python" in state:
        version, internal, gauss_next = state["python"]
        random.setstate((version, tuple(internal), gauss_next))

    if "numpy" in state:
        np_state = state["numpy"]
        keys = np_state["keys"]
        if isinstance(keys, torch.Tensor):
            keys = keys.numpy()
        np.random.set_state(
            (
                np_state["bit_generator"],
                np.asarray(keys, dtype=np.uint32),
                np_state["pos"],
                np_state["has_gauss"],
                np_state["cached_gaussian"],
            )
        )

    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())

    cuda_states = state.get("cuda")
    if cuda_states and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(cuda_states) != device_count:
            logger.warning(
                "Checkpoint has CUDA RNG state for %d device(s) but %d are "
                "available; restoring the overlapping devices only.",
                len(cuda_states),
                device_count,
            )
        for device_index, cuda_state in enumerate(cuda_states[:device_count]):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device_index)
    elif cuda_states:
        logger.warning("Checkpoint contains CUDA RNG state but CUDA is unavailable; skipping it.")


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
        "rng_state": capture_rng_state(),
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, output_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    restore_rng: bool = False,
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
    restore_rng : bool, default=False
        If True, restore the global Python/NumPy/PyTorch/CUDA RNG states
        stored in the checkpoint. Checkpoints without RNG state (created
        before RNG state was recorded) are still loaded; RNG is then left
        untouched.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the loaded metadata ('epoch', 'best_metric',
        'training_history') and 'rng_state_restored' (bool).
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    rng_state_restored = False
    if restore_rng:
        rng_state = checkpoint.get("rng_state")
        if rng_state:
            restore_rng_state(rng_state)
            rng_state_restored = True
        else:
            logger.warning("Checkpoint %s has no RNG state; RNG states were not restored.", path)

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric", 0.0),
        "training_history": checkpoint.get("training_history", {}),
        "rng_state_restored": rng_state_restored,
    }
