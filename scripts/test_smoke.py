"""Smoke tests for the complete U-Net baseline pipeline.

Executes three completely independent training runs for CT, T1 InPhase, and T2SPIR.
These runs are limited to a minimal subset of data and epochs to verify that
data loading, training, validation, metrics, and checkpointing execute without
crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Sequence

from torch.utils.data import DataLoader
from torch.optim import Adam

from src.core.config import ExperimentConfig
from src.core.enums import Modality
from src.datasets.dataset_indexer import DatasetIndexer
from src.datasets.dataset_splitter import DatasetSplitter, SplitConfiguration
from src.datasets.supervised_segmentation_dataset import SupervisedSegmentationDataset
from src.datasets.collate import segmentation_collate_fn
from src.datasets.sample import Sample
from src.engine.trainer import Trainer
from src.losses.segmentation import CombinedSegmentationLoss
from src.models.unet import UNet
from src.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_modality_smoke_test(modality: Modality, sequence_name: str, samples: Sequence[Sample], base_out: Path) -> None:
    """Run a smoke test for a single modality using a tiny data slice."""
    logger.info(f"--- Starting smoke test for {sequence_name} ---")

    out_dir = base_out / sequence_name.replace("/", "_")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    config = ExperimentConfig(
        modality=modality,
        batch_size=2,
        epochs=2,
        learning_rate=1e-4,
        seed=42,
        device="auto",
        num_workers=0,
        output_dir=str(out_dir),
    )
    set_seed(config.seed)

    # 1. Split data (we just take a few patients to speed up the smoke test)
    # The DatasetSplitter will group by patient.
    split_config = SplitConfiguration(train_ratio=0.5, validation_ratio=0.5, test_ratio=0.0, seed=config.seed)
    splitter = DatasetSplitter(split_config)
    split = splitter.split(samples)
    train_samples = split.train
    val_samples = split.validation

    # Trim to bare minimum for smoke test (e.g. 4 slices train, 4 slices val)
    train_samples = train_samples[:4]
    val_samples = val_samples[:4]

    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    if len(train_samples) == 0 or len(val_samples) == 0:
        logger.warning(f"Not enough data for {sequence_name} smoke test. Skipping.")
        return

    # 2. Datasets & Loaders
    train_dataset = SupervisedSegmentationDataset(train_samples, augment=True)
    val_dataset = SupervisedSegmentationDataset(val_samples, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Mandatory per instructions
        drop_last=False, # Mandatory per instructions
        num_workers=config.num_workers,
        collate_fn=segmentation_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        collate_fn=segmentation_collate_fn,
    )

    # 3. Model, Loss, Optimizer
    model = UNet(in_channels=1, num_classes=2)
    criterion = CombinedSegmentationLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # 4. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )

    # 5. Run
    trainer.run()

    # 6. Verify outputs exist
    # latest_model.pt is written unconditionally every epoch.
    # best_model.pt is written only when Dice strictly improves above 0.0;
    # with a freshly-initialised model and only 2 epochs this is not guaranteed,
    # so we do not assert it here — checkpointing logic is covered by unit tests.
    assert (out_dir / "latest_model.pt").exists(), "latest_model.pt not saved."

    logger.info(f"--- Finished smoke test for {sequence_name} successfully. ---")


def main() -> None:
    """Run smoke tests for all baseline modalities."""
    base_out = Path("outputs/smoke_test")

    indexer = DatasetIndexer(Path("data/raw/CHAOS_Train_Sets/Train_Sets"))
    all_samples = indexer.index()

    # Separate modalities
    ct_samples = [s for s in all_samples if s.modality == Modality.CT and s.mask_path is not None]
    t1_samples = [s for s in all_samples if s.modality == Modality.MRI and s.sequence == "T1DUAL/InPhase" and s.mask_path is not None]
    t2_samples = [s for s in all_samples if s.modality == Modality.MRI and s.sequence == "T2SPIR" and s.mask_path is not None]

    run_modality_smoke_test(Modality.CT, "CT", ct_samples, base_out)
    run_modality_smoke_test(Modality.MRI, "T1_InPhase", t1_samples, base_out)
    run_modality_smoke_test(Modality.MRI, "T2SPIR", t2_samples, base_out)

    logger.info("All smoke tests completed without crashing.")


if __name__ == "__main__":
    main()
