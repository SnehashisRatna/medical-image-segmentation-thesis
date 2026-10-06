"""Tests for ExperimentConfig."""

import pytest
from src.core.config import ExperimentConfig
from src.core.enums import Modality


def test_experiment_config_valid():
    config = ExperimentConfig(modality=Modality.CT, batch_size=4)
    assert config.modality == Modality.CT
    assert config.batch_size == 4
    assert config.epochs == 100


def test_experiment_config_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        ExperimentConfig(modality=Modality.CT, batch_size=0)


def test_experiment_config_invalid_device():
    with pytest.raises(ValueError, match="device must be 'auto', 'cuda', or 'cpu'"):
        ExperimentConfig(modality=Modality.CT, device="tpu")
