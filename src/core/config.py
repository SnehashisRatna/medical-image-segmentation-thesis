"""Experiment configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.enums import Modality


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Hyperparameters and configuration for a segmentation training run.

    Parameters
    ----------
    modality : Modality
        Target modality for the model (CT, MRI).
    batch_size : int
        Number of samples per training and validation batch. Defaults to 16.
    epochs : int
        Number of full passes over the training dataset. Defaults to 100.
    learning_rate : float
        Optimization step size. Defaults to 1e-4.
    optimizer : str
        Name of the optimizer to use (e.g., 'Adam', 'AdamW'). Defaults to 'Adam'.
    scheduler : str | None
        Name of the learning rate scheduler to use, or None. Defaults to None.
    seed : int
        Global random seed for reproducibility. Defaults to 42.
    device : str
        Target compute device ('auto', 'cuda', 'cpu'). Defaults to 'auto'.
    num_workers : int
        Number of subprocesses for data loading. Defaults to 4.
    output_dir : str
        Directory to save checkpoints, logs, and artifacts. Defaults to 'outputs/'.
    """

    modality: Modality
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-4
    optimizer: str = "Adam"
    scheduler: str | None = None
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    output_dir: str = "outputs/"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'.")
