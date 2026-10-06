"""Tiny-overfit test for the U-Net baseline on CHAOS.

Demonstrates that the model can perfectly memorize a very small dataset
(e.g. 4 CT slices) and that the loss goes to 0 / Dice goes to 1 without NaN.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.core.config import ExperimentConfig
from src.core.enums import Modality
from src.datasets.dataset_indexer import DatasetIndexer
from src.datasets.supervised_segmentation_dataset import SupervisedSegmentationDataset
from src.datasets.collate import segmentation_collate_fn
from src.engine.trainer import Trainer
from src.losses.segmentation import CombinedSegmentationLoss
from src.models.unet import UNet
from src.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_tiny_overfit() -> None:
    """Run a tiny overfitting test on 4 slices."""
    set_seed(42)

    # 1. Configuration
    # We use a very low number of max epochs for safety, but check early stopping.
    config = ExperimentConfig(
        modality=Modality.CT,
        batch_size=4,
        epochs=100,  # Max safety limit; we expect to stop much earlier.
        learning_rate=1e-4,  # Reduced LR for stability diagnostic
        seed=42,
        device="auto",
        num_workers=0,  # No workers needed for 4 samples
        output_dir="outputs/overfit_tiny",
    )

    out_dir = Path(config.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 2. Get samples
    indexer = DatasetIndexer(Path("data/raw/CHAOS_Train_Sets/Train_Sets"))
    all_samples = indexer.index()
    ct_samples = [s for s in all_samples if s.modality == Modality.CT and s.mask_path is not None]

    if len(ct_samples) < 4:
        raise ValueError("Not enough CT samples to run tiny overfit test.")

    # Select the 4 slices with the most foreground (liver) pixels.
    # Counting mask foreground here is diagnostic-script-only logic;
    # it does not change DatasetIndexer, Sample, or any frozen component.
    def _foreground_pixel_count(sample: object) -> int:
        mask_arr = np.array(Image.open(sample.mask_path).convert("L"))
        return int(np.count_nonzero(mask_arr))

    ct_samples_ranked = sorted(ct_samples, key=_foreground_pixel_count, reverse=True)
    tiny_samples = ct_samples_ranked[:4]

    logger.info(f"Selected {len(tiny_samples)} samples for tiny overfit test.")
    for s in tiny_samples:
        count = _foreground_pixel_count(s)
        logger.info(
            f"  patient={s.patient_id}  seq={s.sequence}  "
            f"slice={s.slice_index}  fg_pixels={count}"
        )

    # 3. Datasets (No augmentation for overfit test)
    dataset = SupervisedSegmentationDataset(tiny_samples, augment=False)

    # We use the same dataset for train and validation
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,  # No shuffle needed for exact memorization tracking
        drop_last=False,
        num_workers=config.num_workers,
        collate_fn=segmentation_collate_fn,
    )

    # 4. Model, Loss, Optimizer
    model = UNet(in_channels=1, num_classes=2)
    criterion = CombinedSegmentationLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # 5. Trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )

    # Run custom training loop to break early upon memorization.
    # Patience counter is a script-local safety mechanism only;
    # it does NOT alter the frozen Trainer, ExperimentConfig, or MetricTracker.
    _PATIENCE = 20  # epochs without Dice improvement before aborting
    logger.info("Starting tiny overfit training...")

    success = False
    best_dice = -1.0
    no_improve_count = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = trainer.train_epoch()
        val_metrics = trainer.validate_epoch()

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f}"
        )

        # Check for NaN
        if torch.isnan(torch.tensor(train_loss)):
            raise RuntimeError("Training loss became NaN.")

        # Check for convergence
        if val_metrics["dice"] > 0.99 and train_loss < 0.05:
            logger.info("Model successfully memorized the tiny dataset!")
            success = True
            break

        # Script-local patience safety stop
        if val_metrics["dice"] > best_dice + 1e-4:
            best_dice = val_metrics["dice"]
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= _PATIENCE:
            logger.warning(
                f"No Dice improvement for {_PATIENCE} consecutive epochs. "
                "Aborting tiny-overfit diagnostic."
            )
            break

    if not success:
        logger.error("Failed to memorize the dataset within the epoch limit.")
        raise RuntimeError("Tiny overfit test failed.")


if __name__ == "__main__":
    run_tiny_overfit()
