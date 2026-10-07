from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
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
    assert args.resume is None


def test_parse_args_resume():
    test_args = [
        "train.py",
        "--data-root",
        "/dummy/path",
        "--modality",
        "CT",
        "--sequence",
        "CT",
        "--epochs",
        "100",
        "--resume",
        "/ckpts/latest_model.pt",
    ]
    with patch("sys.argv", test_args):
        args = train.parse_args()

    assert args.resume == Path("/ckpts/latest_model.pt")
    assert args.epochs == 100

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
    # No --resume: fresh training, no checkpoint loaded
    mock_trainer_instance.resume_from_checkpoint.assert_not_called()


def _setup_pipeline_mocks(mock_indexer, mock_splitter, mock_unet):
    sample = MagicMock()
    sample.modality = Modality.CT
    sample.sequence = "CT"
    mock_indexer.return_value.index.return_value = [sample, sample]

    split_result = MagicMock()
    split_result.train = [sample]
    split_result.validation = [sample]
    split_result.test = []
    mock_splitter.return_value.split.return_value = split_result

    mock_unet.return_value.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]


_BASE_ARGS = [
    "train.py",
    "--data-root",
    "/dummy/path",
    "--modality",
    "CT",
    "--sequence",
    "CT",
    "--epochs",
    "100",
    "--device",
    "cpu",
]


@patch("scripts.train.Trainer")
@patch("scripts.train.UNet")
@patch("scripts.train.DataLoader")
@patch("scripts.train.DatasetSplitter")
@patch("scripts.train.DatasetIndexer")
def test_train_main_resume(
    mock_indexer, mock_splitter, mock_dataloader, mock_unet, mock_trainer, tmp_path
):
    _setup_pipeline_mocks(mock_indexer, mock_splitter, mock_unet)
    mock_trainer.return_value.best_metric = 0.85

    ckpt_path = tmp_path / "latest_model.pt"
    ckpt_path.touch()

    with patch("sys.argv", [*_BASE_ARGS, "--resume", str(ckpt_path)]):
        train.main()

    trainer_instance = mock_trainer.return_value
    # --epochs is passed through unchanged as the total target epoch
    assert mock_trainer.call_args.kwargs["config"].epochs == 100
    trainer_instance.resume_from_checkpoint.assert_called_once_with(ckpt_path)

    # Resume must happen before training starts
    call_names = [c[0] for c in trainer_instance.method_calls]
    assert call_names.index("resume_from_checkpoint") < call_names.index("run")


@patch("scripts.train.Trainer")
@patch("scripts.train.DatasetIndexer")
def test_train_main_resume_missing_file_exits(mock_indexer, mock_trainer, tmp_path):
    missing = tmp_path / "does_not_exist.pt"

    with patch("sys.argv", [*_BASE_ARGS, "--resume", str(missing)]):
        with pytest.raises(SystemExit):
            train.main()

    mock_indexer.assert_not_called()
    mock_trainer.assert_not_called()
