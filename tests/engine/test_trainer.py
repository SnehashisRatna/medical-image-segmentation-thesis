"""Tests for Trainer using mocked components."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.optim import SGD

from src.core.config import ExperimentConfig
from src.core.enums import Modality
from src.datasets.sample import Sample
from src.engine.trainer import Trainer


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
