"""Geometric augmentation for supervised binary liver segmentation training.

Frozen augmentation policy (training only)
------------------------------------------
* Random horizontal flip — probability 0.5
* Random rotation in [−10°, +10°] — probability 0.5

Both the image and the mask receive **exactly the same** spatial transform.

* Image interpolation: bilinear (continuous)
* Mask interpolation:  nearest-neighbor (preserves integer class labels)

Validation and test data must NOT be augmented.  ``TrainingAugmentation``
is only instantiated for training splits.

The following augmentations are explicitly excluded from the frozen spec:
vertical flip, 90° rotation, translation, scaling, elastic deformation,
Gaussian noise, intensity shift, contrast augmentation.
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class TrainingAugmentation:
    """Apply random geometric augmentation to an (image, mask) pair.

    Both the image and the mask receive the same geometric transform so
    that spatial alignment is preserved.  Image pixels are interpolated
    with bilinear filtering; mask labels are interpolated with
    nearest-neighbor filtering to prevent blending of integer class values.

    Parameters
    ----------
    flip_prob : float, optional
        Probability of applying a random horizontal flip.  Default ``0.5``.
    rotation_prob : float, optional
        Probability of applying a random in-plane rotation.  Default ``0.5``.
    max_rotation_deg : float, optional
        Maximum absolute rotation angle in degrees.  The actual angle is
        sampled uniformly from ``[−max_rotation_deg, +max_rotation_deg]``.
        Default ``10.0``.
    """

    def __init__(
        self,
        *,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        max_rotation_deg: float = 10.0,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.max_rotation_deg = max_rotation_deg

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to a matched (image, mask) pair.

        Parameters
        ----------
        image : torch.Tensor
            Float32 image tensor of shape ``[1, H, W]``.
        mask : torch.Tensor
            Long mask tensor of shape ``[H, W]``, values in ``{0, 1}``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Augmented ``(image, mask)`` with the same shapes and dtypes
            as the inputs.
        """
        # ------------------------------------------------------------------ #
        # Random horizontal flip — identical transform for image and mask    #
        # ------------------------------------------------------------------ #
        if random.random() < self.flip_prob:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        # ------------------------------------------------------------------ #
        # Random rotation — same angle, different interpolation              #
        # ------------------------------------------------------------------ #
        if random.random() < self.rotation_prob:
            angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)

            # Image: bilinear interpolation (continuous signal)
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
            )

            # Mask: nearest-neighbor interpolation (integer labels must not
            # be blended); convert to float for TF.rotate then back to long
            mask_3d = mask.unsqueeze(0).float()
            mask_3d = TF.rotate(
                mask_3d,
                angle,
                interpolation=InterpolationMode.NEAREST,
            )
            mask = mask_3d.squeeze(0).long()

        return image, mask
