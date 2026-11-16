"""Shared fixtures: white noise founders (tomb's origin material)."""

import pytest
import numpy as np


@pytest.fixture
def noise_2s():
    """2 seconds of white noise — tomb's seed material."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(44100 * 2) * 0.5).astype(np.float32)


@pytest.fixture
def noise_bank():
    """A bank of 6 noise blobs of varied length (like a founder population)."""
    rng = np.random.default_rng(7)
    return [(rng.standard_normal(44100 * d) * 0.5).astype(np.float32)
            for d in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)]