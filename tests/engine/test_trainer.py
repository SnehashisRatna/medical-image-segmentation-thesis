"""Tests for Trainer using mocked components."""

import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.core.config import ExperimentConfig
from src.core.enums import Modality
from src.datasets.sample import Sample
from src.engine.checkpoint import save_checkpoint
from src.engine.trainer import Trainer
from src.utils.reproducibility import set_seed


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # [B, 1, H, W] -> [B, 2, H, W]
        b, _, h, w = x.shape
        return torch.randn(b, 2, h, w)


def test_trainer_initialization():
    model = MockModel()
    config = ExperimentConfig(modality=Modality.CT, device="cpu")
    optimizer = SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        criterion=MagicMock(),
        optimizer=optimizer,
        config=config,
    )

    assert trainer.device == torch.device("cpu")
    assert trainer.best_metric == 0.0


def test_train_epoch():
    model = MockModel()
    config = ExperimentConfig(modality=Modality.CT, device="cpu")
    optimizer = SGD(model.parameters(), lr=0.1)

    criterion = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))

    # Fake batch: images, masks, samples
    batch = (torch.randn(2, 1, 16, 16), torch.zeros(2, 16, 16, dtype=torch.long), [])
    train_loader = [batch, batch]

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=MagicMock(),
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )

    loss = trainer.train_epoch()
    assert loss == 0.5


def test_validate_epoch():
    model = MockModel()
    config = ExperimentConfig(modality=Modality.CT, device="cpu")
    optimizer = SGD(model.parameters(), lr=0.1)

    criterion = MagicMock(return_value=torch.tensor(0.5))

    s1 = Sample(image_path=Path(""), mask_path=Path(""), patient_id="p1", modality=Modality.CT, sequence="ct", slice_index=1)
    batch = (torch.randn(1, 1, 16, 16), torch.zeros(1, 16, 16, dtype=torch.long), [s1])
    val_loader = [batch]

    trainer = Trainer(
        model=model,
        train_loader=MagicMock(),
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )

    metrics = trainer.validate_epoch()

    assert "val_loss" in metrics
    assert "dice" in metrics
    assert metrics["val_loss"] == 0.5


@patch("src.engine.trainer.save_checkpoint")
def test_trainer_run(mock_save_checkpoint):
    model = MockModel()
    config = ExperimentConfig(modality=Modality.CT, device="cpu", epochs=2)
    optimizer = SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        criterion=MagicMock(),
        optimizer=optimizer,
        config=config,
    )

    trainer.train_epoch = MagicMock(return_value=0.5)
    trainer.validate_epoch = MagicMock(return_value={"val_loss": 0.4, "dice": 0.8, "iou": 0.7, "precision": 0.9, "recall": 0.8})

    trainer.run()

    assert trainer.train_epoch.call_count == 2
    assert trainer.validate_epoch.call_count == 2

    # Save checkpoint should be called 3 times total (best & latest for epoch 1, only latest for epoch 2)
    assert mock_save_checkpoint.call_count == 3

    assert trainer.best_metric == 0.8


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

_FIXED_METRICS = {"val_loss": 0.4, "dice": 0.8, "iou": 0.7, "precision": 0.9, "recall": 0.8}


def _make_trainer(model: nn.Module, optimizer, epochs: int, output_dir: Path, **kwargs) -> Trainer:
    config = ExperimentConfig(
        modality=Modality.CT, device="cpu", epochs=epochs, output_dir=str(output_dir)
    )
    return Trainer(
        model=model,
        train_loader=kwargs.pop("train_loader", MagicMock()),
        val_loader=MagicMock(),
        criterion=kwargs.pop("criterion", MagicMock()),
        optimizer=optimizer,
        config=config,
        **kwargs,
    )


def _epochs_saved(mock_save_checkpoint: MagicMock) -> list[int]:
    return [
        c.kwargs["epoch"]
        for c in mock_save_checkpoint.call_args_list
        if Path(c.kwargs["path"]).name == "latest_model.pt"
    ]


@patch("src.engine.trainer.save_checkpoint")
def test_fresh_training_starts_at_epoch_one(mock_save_checkpoint, tmp_path: Path):
    model = MockModel()
    trainer = _make_trainer(model, SGD(model.parameters(), lr=0.1), 3, tmp_path)
    trainer.train_epoch = MagicMock(return_value=0.5)
    trainer.validate_epoch = MagicMock(return_value=_FIXED_METRICS)

    assert trainer.start_epoch == 1
    trainer.run()

    assert _epochs_saved(mock_save_checkpoint) == [1, 2, 3]
    assert len(trainer.training_history["train_loss"]) == 3


def _save_reference_checkpoint(path: Path, epoch: int) -> tuple[nn.Module, torch.optim.Optimizer, dict]:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.randn(3, 4)).sum().backward()
        optimizer.step()
    history = {
        "train_loss": [0.9, 0.8, 0.7][:epoch],
        "val_loss": [0.95, 0.85, 0.75][:epoch],
        "val_dice": [0.5, 0.6, 0.65][:epoch],
        "val_iou": [0.4, 0.5, 0.55][:epoch],
    }
    save_checkpoint(path, model, optimizer, epoch, 0.65, history)
    return model, optimizer, history


def test_resume_restores_state_and_continues_at_next_epoch(tmp_path: Path):
    ckpt_path = tmp_path / "latest_model.pt"
    ref_model, ref_optimizer, ref_history = _save_reference_checkpoint(ckpt_path, epoch=3)

    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = _make_trainer(model, optimizer, 10, tmp_path)

    resume_epoch = trainer.resume_from_checkpoint(ckpt_path)

    assert resume_epoch == 4
    assert trainer.start_epoch == 4
    assert trainer.best_metric == 0.65
    assert trainer.training_history == ref_history

    for p_ref, p_new in zip(ref_model.parameters(), model.parameters()):
        assert torch.equal(p_ref, p_new)
        ref_state, new_state = ref_optimizer.state[p_ref], optimizer.state[p_new]
        assert torch.equal(ref_state["exp_avg"], new_state["exp_avg"])
        assert torch.equal(ref_state["exp_avg_sq"], new_state["exp_avg_sq"])
        assert int(ref_state["step"]) == int(new_state["step"])


@patch("src.engine.trainer.save_checkpoint")
def test_epochs_is_total_target_when_resuming(mock_save_checkpoint, tmp_path: Path):
    ckpt_path = tmp_path / "ckpt.pt"
    _save_reference_checkpoint(ckpt_path, epoch=3)

    model = nn.Linear(4, 2)
    trainer = _make_trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), 5, tmp_path)
    trainer.resume_from_checkpoint(ckpt_path)
    trainer.train_epoch = MagicMock(return_value=0.5)
    trainer.validate_epoch = MagicMock(return_value=_FIXED_METRICS)

    trainer.run()

    # Epochs 4 and 5 only -- not another 5 epochs
    assert trainer.train_epoch.call_count == 2
    assert _epochs_saved(mock_save_checkpoint) == [4, 5]
    # History continues from the restored history
    assert trainer.training_history["train_loss"] == [0.9, 0.8, 0.7, 0.5, 0.5]
    # Restored best metric (0.65) is beaten by 0.8 on epoch 4 only
    best_epochs = [
        c.kwargs["epoch"]
        for c in mock_save_checkpoint.call_args_list
        if Path(c.kwargs["path"]).name == "best_model.pt"
    ]
    assert best_epochs == [4]


@patch("src.engine.trainer.save_checkpoint")
def test_restored_best_metric_prevents_spurious_best_save(mock_save_checkpoint, tmp_path: Path):
    ckpt_path = tmp_path / "ckpt.pt"
    _save_reference_checkpoint(ckpt_path, epoch=3)  # best_metric = 0.65

    model = nn.Linear(4, 2)
    trainer = _make_trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), 4, tmp_path)
    trainer.resume_from_checkpoint(ckpt_path)
    trainer.train_epoch = MagicMock(return_value=0.5)
    trainer.validate_epoch = MagicMock(return_value={**_FIXED_METRICS, "dice": 0.6})

    trainer.run()

    saved_names = [Path(c.kwargs["path"]).name for c in mock_save_checkpoint.call_args_list]
    assert saved_names == ["latest_model.pt"]
    assert trainer.best_metric == 0.65


def test_resume_when_target_already_reached_runs_nothing(tmp_path: Path):
    ckpt_path = tmp_path / "ckpt.pt"
    _save_reference_checkpoint(ckpt_path, epoch=3)

    model = nn.Linear(4, 2)
    trainer = _make_trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), 3, tmp_path)
    trainer.resume_from_checkpoint(ckpt_path)
    trainer.train_epoch = MagicMock(return_value=0.5)

    trainer.run()

    trainer.train_epoch.assert_not_called()


def test_resume_from_legacy_checkpoint_without_rng_state(tmp_path: Path):
    model = nn.Linear(4, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 7,
            "best_metric": 0.42,
            "training_history": {"train_loss": [1.0] * 7},
        },
        legacy_path,
    )

    new_model = nn.Linear(4, 2)
    trainer = _make_trainer(new_model, SGD(new_model.parameters(), lr=0.1), 10, tmp_path)
    assert trainer.resume_from_checkpoint(legacy_path) == 8
    assert trainer.best_metric == 0.42
    assert trainer.training_history["train_loss"] == [1.0] * 7
    # Missing history keys are back-filled so run() can append to them
    assert trainer.training_history["val_dice"] == []


def _noisy_collate(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    # Consume torch, NumPy and Python RNG like a stochastic augmentation would
    images = images + 0.1 * torch.randn_like(images)
    images = images * float(np.random.uniform(0.9, 1.1)) * random.uniform(0.9, 1.1)
    return images, masks, []


def _run_real_training(output_dir: Path, epochs: int, resume_from: Path | None = None) -> nn.Module:
    set_seed(0)
    data_gen = torch.Generator().manual_seed(123)
    images = torch.randn(12, 1, 4, 4, generator=data_gen)
    masks = (images[:, 0] > 0).long()
    loader = DataLoader(
        TensorDataset(images, masks), batch_size=4, shuffle=True, collate_fn=_noisy_collate
    )

    model = nn.Conv2d(1, 2, kernel_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = _make_trainer(
        model,
        optimizer,
        epochs,
        output_dir,
        train_loader=loader,
        criterion=nn.CrossEntropyLoss(),
    )
    trainer.validate_epoch = MagicMock(return_value=_FIXED_METRICS)

    if resume_from is not None:
        trainer.resume_from_checkpoint(resume_from)
    trainer.run()
    return model


def test_split_run_matches_uninterrupted_run(tmp_path: Path):
    """2 epochs + resume to 4 must reproduce an uninterrupted 4-epoch run."""
    continuous = _run_real_training(tmp_path / "continuous", epochs=4)

    _run_real_training(tmp_path / "split", epochs=2)
    resumed = _run_real_training(
        tmp_path / "split", epochs=4, resume_from=tmp_path / "split" / "latest_model.pt"
    )

    for p_cont, p_res in zip(continuous.parameters(), resumed.parameters()):
        assert torch.equal(p_cont, p_res)
