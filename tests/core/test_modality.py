"""Unit tests for the :class:`Modality` enum."""

from src.core.enums import Modality


########################################################################
# Enum Values
########################################################################


def test_modality_values() -> None:
    """Verify the values of all supported modalities."""
    assert Modality.CT.value == "CT"
    assert Modality.MRI.value == "MRI"
    assert Modality.PET.value == "PET"


def test_modality_has_exactly_three_members() -> None:
    """Verify that exactly three modalities are defined."""
    assert len(Modality) == 3


def test_modality_member_names() -> None:
    """Verify the names of all supported modalities."""
    assert [member.name for member in Modality] == ["CT", "MRI", "PET"]
