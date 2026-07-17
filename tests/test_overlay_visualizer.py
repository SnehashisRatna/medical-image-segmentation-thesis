"""Integration test for rendering a real CHAOS CT image and liver mask."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.data.dicom_reader import DICOMReader
from src.data.ground_truth_reader import GroundTruthReader
from src.visualization.overlay_visualizer import OverlayVisualizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DICOM_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CHAOS_Train_Sets"
    / "Train_Sets"
    / "CT"
    / "1"
    / "DICOM_anon"
    / "i0045,0000b.dcm"
)
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
OUTPUT_PATH = PROJECT_ROOT / "tests" / "output" / "overlay_visualization.png"


def test_render_real_chaos_overlay() -> None:
    """Load, render, save, display, and close a real CHAOS overlay."""
    if not DICOM_PATH.exists():
        pytest.skip(f"Real CHAOS DICOM image is not available at {DICOM_PATH}")
    if not MASK_PATH.exists():
        pytest.skip(f"Real CHAOS liver mask is not available at {MASK_PATH}")

    image = DICOMReader().read_image(DICOM_PATH)
    mask = GroundTruthReader().read_mask(MASK_PATH)

    print("Overlay Visualization Test")
    print(f"Image Shape: {image.shape}")
    print(f"Image Dtype: {image.dtype}")
    print(f"Mask Shape: {mask.shape}")
    print(f"Mask Dtype: {mask.dtype}")

    assert isinstance(image, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert image.ndim == 2
    assert mask.ndim == 2
    assert image.shape == mask.shape
    assert image.shape == (512, 512)
    assert mask.shape == (512, 512)

    figure = OverlayVisualizer().render(image, mask, mask_alpha=0.4)
    assert isinstance(figure, Figure)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH)

    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0

    print(f"Output File: {OUTPUT_PATH}")
    print(f"File Size: {OUTPUT_PATH.stat().st_size}")

    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(figure)

    print("Test Passed")
