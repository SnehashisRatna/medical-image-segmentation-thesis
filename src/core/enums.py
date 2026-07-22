"""Enumerations used by the framework's core data contracts."""

from enum import Enum


class Modality(Enum):
    """Represent a supported medical imaging modality.

    Attributes
    ----------
    CT
        Computed tomography.
    MRI
        Magnetic resonance imaging.
    PET
        Positron emission tomography.
    """

    CT = "CT"
    MRI = "MRI"
    PET = "PET"
