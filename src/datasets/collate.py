"""Custom collate functions for medical image segmentation."""

from __future__ import annotations

import torch

from src.datasets.sample import Sample


def segmentation_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, Sample]],
) -> tuple[torch.Tensor, torch.Tensor, list[Sample]]:
    """Collate a batch of samples into tensors and metadata lists.

    Parameters
    ----------
    batch : list[tuple[torch.Tensor, torch.Tensor, Sample]]
        A list of elements returned by SupervisedSegmentationDataset.__getitem__.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, list[Sample]]
        * **images** - Batched image tensor of shape [B, C, H, W]
        * **masks** - Batched mask tensor of shape [B, H, W]
        * **samples** - A list of the corresponding Sample objects.
    """
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    samples = [item[2] for item in batch]

    return torch.stack(images, dim=0), torch.stack(masks, dim=0), samples
