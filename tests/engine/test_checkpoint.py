"""Tests for checkpointing utilities."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from src.engine.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


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


def _trained_model_and_optimizer() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.randn(8, 4)).sum().backward()
        optimizer.step()
    return model, optimizer


def test_optimizer_and_scheduler_state_restored(tmp_path: Path):
    model, optimizer = _trained_model_and_optimizer()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    scheduler.step()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, 3, 0.5, {}, scheduler=scheduler)

    new_model = nn.Linear(4, 2)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.5)
    load_checkpoint(ckpt_path, new_model, new_optimizer, new_scheduler)

    for p_old, p_new in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p_old, p_new)
        old_state = optimizer.state[p_old]
        new_state = new_optimizer.state[p_new]
        assert torch.equal(old_state["exp_avg"], new_state["exp_avg"])
        assert torch.equal(old_state["exp_avg_sq"], new_state["exp_avg_sq"])
        assert int(old_state["step"]) == int(new_state["step"])

    assert new_scheduler.last_epoch == scheduler.last_epoch
    assert new_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_checkpoint_contains_rng_state_and_is_weights_only_loadable(tmp_path: Path):
    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, 1, 0.1, {"train_loss": [1.0]})

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert {"python", "numpy", "torch"} <= set(checkpoint["rng_state"])


def test_rng_state_restoration(tmp_path: Path):
    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    ckpt_path = tmp_path / "ckpt.pt"

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    save_checkpoint(ckpt_path, model, optimizer, 1, 0.1, {})

    expected = (random.random(), np.random.rand(3), torch.rand(3))

    # Perturb all generators
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    metadata = load_checkpoint(ckpt_path, nn.Linear(2, 2), restore_rng=True)
    assert metadata["rng_state_restored"] is True

    assert random.random() == expected[0]
    assert np.array_equal(np.random.rand(3), expected[1])
    assert torch.equal(torch.rand(3), expected[2])


def test_rng_not_restored_by_default(tmp_path: Path):
    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    ckpt_path = tmp_path / "ckpt.pt"

    torch.manual_seed(1)
    save_checkpoint(ckpt_path, model, optimizer, 1, 0.1, {})
    after_save = torch.rand(1)

    target_model = nn.Linear(2, 2)  # built before seeding: init consumes RNG
    torch.manual_seed(1)
    metadata = load_checkpoint(ckpt_path, target_model)
    assert metadata["rng_state_restored"] is False
    # Global RNG untouched: still in the freshly seeded state
    assert torch.equal(torch.rand(1), after_save)


def test_legacy_checkpoint_without_rng_state_is_loadable(tmp_path: Path):
    model = nn.Linear(3, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    model.weight.data.fill_(2.0)
    history = {"train_loss": [0.9, 0.7], "val_dice": [0.5, 0.6]}

    # Exactly the field set written before RNG state was introduced
    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 2,
            "best_metric": 0.6,
            "training_history": history,
        },
        legacy_path,
    )

    new_model = nn.Linear(3, 2)  # built before seeding: init consumes RNG
    new_optimizer = SGD(new_model.parameters(), lr=0.1)

    torch.manual_seed(7)
    expected_next = torch.rand(1)
    torch.manual_seed(7)

    metadata = load_checkpoint(legacy_path, new_model, new_optimizer, restore_rng=True)

    assert metadata["epoch"] == 2
    assert metadata["best_metric"] == 0.6
    assert metadata["training_history"] == history
    assert metadata["rng_state_restored"] is False
    assert torch.allclose(new_model.weight.data, torch.tensor(2.0))
    assert torch.equal(torch.rand(1), expected_next)


def test_restore_rng_state_tolerates_partial_state():
    torch.manual_seed(5)
    state = capture_rng_state()
    expected = torch.rand(2)

    torch.manual_seed(42)
    restore_rng_state({"torch": state["torch"]})
    assert torch.equal(torch.rand(2), expected)
