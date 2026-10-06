"""Tests for checkpointing utilities."""

from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import SGD

from src.engine.checkpoint import save_checkpoint, load_checkpoint


def test_save_and_load_checkpoint(tmp_path: Path):
    model = nn.Linear(10, 2)
    optimizer = SGD(model.parameters(), lr=0.1)

    # Modify state to ensure it's loaded correctly
    model.weight.data.fill_(1.0)
    optimizer.step()

    ckpt_path = tmp_path / "model.pt"

    history = {"train_loss": [0.5]}

    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=5,
        best_metric=0.8,
        training_history=history,
    )

    assert ckpt_path.exists()

    # Load into new instances
    new_model = nn.Linear(10, 2)
    new_optimizer = SGD(new_model.parameters(), lr=0.1)

    metadata = load_checkpoint(
        path=ckpt_path,
        model=new_model,
        optimizer=new_optimizer,
    )

    assert metadata["epoch"] == 5
    assert metadata["best_metric"] == 0.8
    assert metadata["training_history"] == history

    assert torch.allclose(new_model.weight.data, torch.tensor(1.0))
