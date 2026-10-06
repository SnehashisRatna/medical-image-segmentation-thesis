from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

import scripts.train as train
from src.core.enums import Modality


def test_parse_args_defaults():
    test_args = [
        "train.py",
        "--data-root",
        "/dummy/path",
        "--modality",
        "CT",
        "--sequence",
        "CT",
    ]
    with patch("sys.argv", test_args):
        args = train.parse_args()

    assert args.data_root == Path("/dummy/path")
    assert args.modality == "CT"
    assert args.sequence == "CT"
    assert args.output_dir == "outputs/"
    assert args.batch_size == 16
    assert args.epochs == 100
    assert args.learning_rate == 1e-4
    assert args.seed == 42
    assert args.num_workers == 4
    assert args.device == "auto"

@patch("scripts.train.Trainer")
@patch("scripts.train.UNet")
@patch("scripts.train.DataLoader")
@patch("scripts.train.DatasetSplitter")
@patch("scripts.train.DatasetIndexer")
def test_train_main_orchestration(
    mock_indexer, mock_splitter, mock_dataloader, mock_unet, mock_trainer
):
    # Setup mocks
    mock_indexer_instance = mock_indexer.return_value

    # Create some dummy samples
    sample1 = MagicMock()
    sample1.modality = Modality.CT
    sample1.sequence = "CT"

    sample2 = MagicMock()
    sample2.modality = Modality.CT
    sample2.sequence = "CT"

    mock_indexer_instance.index.return_value = [sample1, sample2]

    mock_splitter_instance = mock_splitter.return_value
    mock_split_result = MagicMock()
    mock_split_result.train = [sample1]
    mock_split_result.validation = [sample2]
    mock_split_result.test = []
    mock_splitter_instance.split.return_value = mock_split_result

    mock_model_instance = mock_unet.return_value
    mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]

    mock_trainer_instance = mock_trainer.return_value
    mock_trainer_instance.best_metric = 0.85

    test_args = [
        "train.py",
        "--data-root",
        "/dummy/path",
        "--modality",
        "CT",
        "--sequence",
        "CT",
        "--epochs",
        "2",
        "--device",
        "cpu"
    ]

    with patch("sys.argv", test_args):
        train.main()

    # Verify indexer was called with data_root
    mock_indexer.assert_called_once_with(Path("/dummy/path"))

    # Verify split was called
    mock_splitter_instance.split.assert_called_once()

    # Verify UNet instantiated
    mock_unet.assert_called_once_with(in_channels=1, num_classes=2)

    # Verify model was moved to device
    mock_model_instance.to.assert_called()

    # Verify Trainer instantiated and run
    mock_trainer.assert_called_once()
    mock_trainer_instance = mock_trainer.return_value
    mock_trainer_instance.run.assert_called_once()
