"""Model-agnostic training engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.core.config import ExperimentConfig
from src.engine.checkpoint import load_checkpoint, save_checkpoint
from src.metrics.tracker import SegmentationMetricTracker

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the training and validation loops."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: Optimizer,
        config: ExperimentConfig,
        scheduler: LRScheduler | None = None,
    ) -> None:
        """Initialize the Trainer.

        Parameters
        ----------
        model : nn.Module
            The neural network model to train.
        train_loader : DataLoader
            DataLoader for the training set.
        val_loader : DataLoader
            DataLoader for the validation set.
        criterion : Callable
            The loss function (e.g., CrossEntropy + Dice).
        optimizer : Optimizer
            The optimizer.
        config : ExperimentConfig
            The experiment configuration.
        scheduler : LRScheduler | None
            Optional learning rate scheduler.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)

        self.metric_tracker = SegmentationMetricTracker()
        # First epoch to run; ``config.epochs`` is the total (inclusive) target.
        self.start_epoch = 1
        self.best_metric = 0.0
        self.training_history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
        }

    def resume_from_checkpoint(self, path: str | Path) -> int:
        """Restore training state from a checkpoint to continue training.

        Restores model weights, optimizer state, scheduler state (if a
        scheduler is configured), RNG states (if stored in the checkpoint),
        the best metric, and the training history. Training will continue at
        ``checkpoint_epoch + 1`` up to ``config.epochs`` (the total target).

        Parameters
        ----------
        path : str | Path
            Path to a checkpoint produced by :func:`save_checkpoint`.

        Returns
        -------
        int
            The epoch at which training will resume.
        """
        metadata = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            restore_rng=True,
        )

        completed_epoch = int(metadata["epoch"])
        self.start_epoch = completed_epoch + 1
        self.best_metric = float(metadata["best_metric"])

        restored_history = {
            key: list(values) for key, values in metadata["training_history"].items()
        }
        for key in self.training_history:
            restored_history.setdefault(key, [])
        self.training_history = restored_history

        logger.info(
            f"Resumed from checkpoint {path} (completed epoch {completed_epoch}, "
            f"best Dice {self.best_metric:.4f}, "
            f"RNG restored: {metadata['rng_state_restored']}). "
            f"Continuing at epoch {self.start_epoch}/{self.config.epochs}."
        )
        if self.start_epoch > self.config.epochs:
            logger.warning(
                f"Checkpoint epoch {completed_epoch} already reaches the target of "
                f"{self.config.epochs} epochs; no further training will be run."
            )

        return self.start_epoch

    def train_epoch(self) -> float:
        """Execute one full pass over the training dataset.

        Returns
        -------
        float
            The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)

            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate_epoch(self) -> dict[str, float]:
        """Execute one full pass over the validation dataset.

        Returns
        -------
        dict[str, float]
            A dictionary containing the validation loss and metrics (dice, iou,
            precision, recall).
        """
        self.model.eval()
        self.metric_tracker.reset()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        for images, masks, samples in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            total_loss += loss.item()

            # The U-Net outputs [B, 2, H, W] logits.
            # Predictions are the argmax over the channel dimension (dim=1).
            predictions = logits.argmax(dim=1)

            # Move back to CPU for metric tracking to save VRAM, or keep on GPU.
            # Keeping on GPU is fine for tracker if it supports it, but tracker
            # does `.item()`, which moves to CPU implicitly.
            self.metric_tracker.update(predictions, masks, samples)

        metrics = self.metric_tracker.compute()
        metrics["val_loss"] = total_loss / max(num_batches, 1)

        return metrics

    def run(self) -> None:
        """Execute the complete training loop."""
        logger.info(f"Starting training on device {self.device}")
        output_dir = Path(self.config.output_dir)

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_metrics["val_loss"])
            self.training_history["val_dice"].append(val_metrics["dice"])
            self.training_history["val_iou"].append(val_metrics["iou"])

            logger.info(
                f"Epoch [{epoch}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f}"
            )

            # Save checkpoint if current Dice is strictly better
            if val_metrics["dice"] > self.best_metric:
                self.best_metric = val_metrics["dice"]
                save_checkpoint(
                    path=output_dir / "best_model.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    training_history=self.training_history,
                    scheduler=self.scheduler,
                )
                logger.info(f"Saved new best model with Dice {self.best_metric:.4f}")

            # Save latest checkpoint
            save_checkpoint(
                path=output_dir / "latest_model.pt",
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_metric=self.best_metric,
                training_history=self.training_history,
                scheduler=self.scheduler,
            )

        logger.info("Training complete.")
