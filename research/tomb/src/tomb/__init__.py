"""tomb — in-place audio novelty search. Only the discriminable survive.

Sounds and ears coevolve under minimal criteria:
    sound MC:  some ear distinguishes it from all living sounds
    ear MC:    discriminates a pair no other ear discriminates

No fitness, no behavior characterization, no archive. Pure drift.
"""

__version__ = "0.1.0"

from .engine import Tomb, TombConfig
from .ears import Ear, EarQueue, features_vector, random_ear
from .ops import FAMILIES, OPS, mutate, random_op_recipe
from .sounds import SoundGenome, SoundQueue

__all__ = [
    "__version__",
    "Tomb", "TombConfig",
    "SoundGenome", "SoundQueue",
    "Ear", "EarQueue", "features_vector", "random_ear",
    "OPS", "FAMILIES", "mutate", "random_op_recipe",
]