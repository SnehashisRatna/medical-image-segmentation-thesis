import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.core.config import ExperimentConfig
from src.core.enums import Modality
from src.datasets.collate import segmentation_collate_fn
from src.datasets.dataset_indexer import DatasetIndexer
from src.datasets.dataset_splitter import DatasetSplitter, SplitConfiguration
from src.datasets.supervised_segmentation_dataset import SupervisedSegmentationDataset
from src.engine.trainer import Trainer
from src.losses.segmentation import CombinedSegmentationLoss
from src.models.unet.model import UNet
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D U-Net for liver segmentation on CHAOS.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to the CHAOS dataset root.")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Directory to save checkpoints and logs.")
    parser.add_argument("--modality", type=str, choices=["CT", "MRI"], required=True, help="Target modality (CT or MRI).")
    parser.add_argument("--sequence", type=str, required=True, help="Target sequence (e.g., 'CT', 'T1DUAL/InPhase', 'T2SPIR').")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of subprocesses for data loading.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu').")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Reproducibility
    set_seed(args.seed)

    # 2. Configuration
    modality = Modality[args.modality]
    config = ExperimentConfig(
        modality=modality,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer="Adam",
        scheduler=None,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )

    # Resolve target device explicitly before creating optimizer
    if config.device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(config.device)

    logger.info(f"Target device resolved to: {target_device}")

    # 3. Data Indexing and Filtering
    logger.info(f"Indexing dataset at {args.data_root}")
    indexer = DatasetIndexer(args.data_root)
    all_samples = indexer.index()

    # Filter to requested modality and sequence
    # Unlabeled samples (like T1DUAL/OutPhase) will be naturally skipped by SupervisedSegmentationDataset
    experiment_samples = [
        s for s in all_samples if s.modality == modality and s.sequence == args.sequence
    ]

    if not experiment_samples:
        logger.error(f"No samples found for modality {modality.name} and sequence {args.sequence}.")
        sys.exit(1)

    logger.info(f"Found {len(experiment_samples)} total slices for {modality.name} ({args.sequence}).")

    # 4. Patient-level Splitting
    split_config = SplitConfiguration(
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )
    splitter = DatasetSplitter(split_config)
    dataset_split = splitter.split(experiment_samples)

    logger.info(f"Splits (patients) - Train: {len(set(s.patient_id for s in dataset_split.train))}, "
                f"Val: {len(set(s.patient_id for s in dataset_split.validation))}, "
                f"Test: {len(set(s.patient_id for s in dataset_split.test))}")

    # 5. Datasets and DataLoaders
    train_dataset = SupervisedSegmentationDataset(dataset_split.train, augment=True)
    val_dataset = SupervisedSegmentationDataset(dataset_split.validation, augment=False)

    logger.info(f"Train slices (labeled): {len(train_dataset)}")
    logger.info(f"Val slices (labeled): {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=segmentation_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=segmentation_collate_fn,
        pin_memory=True,
    )

    # 6. Model, Loss, Optimizer
    model = UNet(in_channels=1, num_classes=2)
    # Move model to target device BEFORE instantiating the optimizer
    model.to(target_device)

    criterion = CombinedSegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 7. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )

    # 8. Run Experiment
    logger.info("Starting formal training...")
    trainer.run()

    logger.info(f"Experiment completed. Best Val Dice: {trainer.best_metric:.4f}")
    logger.info(f"Checkpoints saved in: {config.output_dir}")

if __name__ == "__main__":
    main()
