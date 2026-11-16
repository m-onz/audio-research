"""cryptid — breeding evolved dynamical processes.

The sound IS the state. No oscillators, no notes, no filters, no
envelopes — just laws, coupling, and measurement. The genome is a
growable network of heterogeneous dynamical units (jerk chaos,
excitable membranes, sandpiles, Boolean networks...); the ear is an
evolvable measurement that decides which slice of the state you hear.

MCC pairing: a process is viable iff some ear hears it alive; an ear
is viable iff it hears some process alive. Curation is the only
quality pressure.
"""

__version__ = "0.1.0"

from .engine import Cryptid, CryptidConfig, SR
from .ear import EarGenome, mutate_ear, random_ear
from .genome import ProcessGenome, mutate_genome, random_genome
from .process import LAWS, LAW_NAMES, default_params, n_state_vars

__all__ = [
    "__version__",
    "Cryptid", "CryptidConfig", "SR",
    "ProcessGenome", "random_genome", "mutate_genome",
    "EarGenome", "random_ear", "mutate_ear",
    "LAWS", "LAW_NAMES", "default_params", "n_state_vars",
]