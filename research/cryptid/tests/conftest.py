"""Shared fixtures: random genomes and ears."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_genome(rng):
    """A 4-node network with mixed laws."""
    from cryptid.genome import random_genome

    return random_genome(1, rng, n_nodes=4, n_edges=6)


@pytest.fixture
def small_ear(rng):
    """A random ear for a 6-var process."""
    from cryptid.ear import random_ear

    return random_ear(1, rng, 6, k=3)