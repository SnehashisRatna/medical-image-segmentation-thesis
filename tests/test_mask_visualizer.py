"""Integration test for rendering a real CHAOS ground truth mask."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.data.ground_truth_reader import GroundTruthReader
from src.visualization.mask_visualizer import MaskVisualizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CHAOS_Train_Sets"
    / "Train_Sets"
    / "CT"
    / "1"
    / "Ground"
    / "liver_GT_045.png"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "visualization"
    / "chaos_liver_mask.png"
)


def test_render_real_chaos_mask() -> None:
    """Load, validate, render, save, and display a CHAOS ground truth mask."""
    if not MASK_PATH.exists():
        pytest.skip(f"Real CHAOS mask is not available at {MASK_PATH}")

    reader = GroundTruthReader()
    mask = reader.read_mask(MASK_PATH)
    statistics = reader.mask_statistics(mask)
    empty_mask = reader.is_empty_mask(mask)

    print(f"Mask path: {MASK_PATH}")
    print(f"Shape: {mask.shape}")
    print(f"Dtype: {mask.dtype}")
    print(f"Unique labels: {reader.unique_labels(mask)}")
    print(f"Foreground pixels: {statistics.foreground_pixels}")
    print(f"Background pixels: {statistics.background_pixels}")
    print(f"Empty mask: {empty_mask}")

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.ndim == 2
    assert mask.shape == (512, 512)
    assert not empty_mask

    figure = MaskVisualizer().render(mask, title="CHAOS Liver Ground Truth Mask")

    assert isinstance(figure, Figure)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH)
    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0

    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(figure)

    print("MaskVisualizer integration test completed successfully.")
