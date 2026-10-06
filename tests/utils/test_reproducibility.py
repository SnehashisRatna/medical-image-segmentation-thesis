"""Tests for reproducibility utilities."""

import random
import numpy as np
import torch

from src.utils.reproducibility import set_seed


def test_set_seed_makes_random_reproducible():
    set_seed(42)
    val1 = random.random()
    np1 = np.random.rand()
    t1 = torch.rand(1).item()

    set_seed(42)
    val2 = random.random()
    np2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert val1 == val2
    assert np1 == np2
    assert t1 == t2
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
