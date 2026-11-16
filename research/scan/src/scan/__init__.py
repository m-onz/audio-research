"""scan — the wavetable arena. Frames are born, not designed; the
morph path, the vector mix and the pitch path are genes; echo's
archive decides which children live."""

import os

# The loop's matrices are tiny (169x1025 @ 1025x64 and smaller), and
# multithreaded OpenBLAS busy-spins worker threads at EVERY small
# matmul on Apple silicon — measured on an M4: echo's mel matmul
# 66-92 ms spinning vs 0.9 ms pinned, whole encode 113 ms vs 9.4 ms
# (12x). Pin before numpy loads anywhere in this process; must run
# before the first numpy import to take effect (spawned workers
# re-import scan first, so they inherit it).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from .organism import (WtGenome, render, random_genome, mutate, tonality,
                       viable, write_wav, to_dict, from_dict, ARCHETYPES,
                       SR, SECONDS, MORPH_MODES, PITCH_MODES)
from .champions import CHAMPIONS, champion, blurb, stats

__all__ = ["WtGenome", "render", "random_genome", "mutate", "tonality",
           "viable", "write_wav", "to_dict", "from_dict", "ARCHETYPES",
           "SR", "SECONDS", "MORPH_MODES", "PITCH_MODES", "CHAMPIONS",
           "champion", "blurb", "stats"]

__version__ = "0.1.0"