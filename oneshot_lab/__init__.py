"""oneshot_lab — general-purpose oneshot mutation library.

Turn a folder of sounds into a bank of unique, weird variations.

Command line (batch a folder into ./oneshot_out):

    python -m oneshot_lab ./oneshots -o ./oneshot_out
    python -m oneshot_lab ./oneshots --transforms paulstretch,glitch --random-chains 2
    python -m oneshot_lab --list

Library use:

    import numpy as np
    from oneshot_lab import core, engine, timestretch

    x, sr = core.read_audio("hit.wav")
    rng = np.random.default_rng(0)
    y = timestretch.paulstretch(x, sr, rng, stretch=30.0, window_s=0.4)

    # or a fully random mutation:
    y, chain = engine.mutate(x, sr, rng)

Batch API:

    engine.run_folder("./oneshots", "./oneshot_out", random_chains=2)
"""
from . import core, engine, glitch, harmonics, novel, pitchfx, registry, spectral, timestretch, warp

__all__ = ["core", "engine", "glitch", "harmonics", "novel", "pitchfx", "registry",
           "spectral", "timestretch", "warp", "TRANSFORMS"]
from .registry import TRANSFORMS, FAMILIES  # noqa: E402