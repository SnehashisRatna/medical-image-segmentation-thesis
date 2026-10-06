"""Tests for segmentation_collate_fn."""

from pathlib import Path

import torch

from src.core.enums import Modality
from src.datasets.sample import Sample
from src.datasets.collate import segmentation_collate_fn


def test_segmentation_collate_fn():
    # 2 samples
    s1 = Sample(image_path=Path("1.dcm"), mask_path=Path("1.png"), patient_id="p1", modality=Modality.CT, sequence="ct", slice_index=1)
    s2 = Sample(image_path=Path("2.dcm"), mask_path=Path("2.png"), patient_id="p2", modality=Modality.CT, sequence="ct", slice_index=2)

    img1 = torch.zeros(1, 256, 256)
    img2 = torch.ones(1, 256, 256)

    mask1 = torch.zeros(256, 256, dtype=torch.long)
    mask2 = torch.ones(256, 256, dtype=torch.long)

    batch = [
        (img1, mask1, s1),
        (img2, mask2, s2),
    ]

    images, masks, samples = segmentation_collate_fn(batch)

    assert images.shape == (2, 1, 256, 256)
    assert masks.shape == (2, 256, 256)
    assert len(samples) == 2
    assert samples[0] == s1
    assert samples[1] == s2
    assert torch.all(images[0] == 0)
    assert torch.all(images[1] == 1)
