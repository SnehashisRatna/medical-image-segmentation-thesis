import os
from pathlib import Path

import pytest

from src.core.enums import Modality
from src.datasets.dataset_indexer import DatasetIndexer


CHAOS_TEST_ROOT = Path("data/raw/CHAOS_Test_Sets/Test_Sets")
CHAOS_TRAIN_ROOT = Path("data/raw/CHAOS_Train_Sets/Train_Sets")


def test_indexes_locally_available_chaos_test_set() -> None:
    """Exercise the frozen layout against real data when it is present locally."""
    if not CHAOS_TEST_ROOT.is_dir():
        pytest.skip("Local CHAOS test dataset is unavailable.")

    samples = DatasetIndexer(CHAOS_TEST_ROOT).index()

    assert samples
    assert {sample.modality for sample in samples} == {Modality.CT, Modality.MRI}
    assert {
        sample.sequence for sample in samples if sample.modality is Modality.MRI
    } == {
        "T1DUAL/InPhase",
        "T1DUAL/OutPhase",
        "T2SPIR",
    }
    assert all(sample.mask_path is None for sample in samples)
    assert samples == DatasetIndexer(CHAOS_TEST_ROOT).index()


def test_indexes_locally_available_chaos_training_t1dual(tmp_path: Path) -> None:
    """Verify real CHAOS training T1DUAL labels apply to InPhase only."""
    train_mr = CHAOS_TRAIN_ROOT / "MR"
    if not train_mr.is_dir():
        pytest.skip("Local CHAOS training dataset is unavailable.")

    root = tmp_path / "Train_Sets"
    root.mkdir()
    try:
        os.symlink(train_mr, root / "MR", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Cannot link the local CHAOS MR directory: {error}")

    samples = DatasetIndexer(root).index()
    in_phase = [sample for sample in samples if sample.sequence == "T1DUAL/InPhase"]
    out_phase = [sample for sample in samples if sample.sequence == "T1DUAL/OutPhase"]

    assert in_phase and out_phase
    assert all(sample.mask_path is not None for sample in in_phase)
    assert all(sample.mask_path is None for sample in out_phase)
