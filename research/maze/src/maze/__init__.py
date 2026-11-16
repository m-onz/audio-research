"""maze — the audio maze.

A minimal demonstration that echo's memory changes what evolution
finds. One tiny voice synthesizer, four selection policies, identical
everything else:

    greedy    pick the most 'tonal' parent        -> converges to a sine
    random    pick any parent                     -> diffuses slowly
    cell      pick from grid cells (womb's rule)  -> spreads, collides
    novelty   pick by echo archive novelty         -> goes somewhere

The genome labels its own ground truth (tonal / noisy / rhythmic /
textural), so coverage is measured without circularity: no audio
metric judges the audio — the genome does, and an independent
plain-feature set double-checks it.
"""

from .voice import (VoiceGenome, render, random_genome, mutate,
                    ARCHETYPES, tonality)
from .dx import DxGenome, ARCHETYPES as DX_ARCHETYPES
from .measure import plain_features, PLAIN_NAMES

__all__ = ["VoiceGenome", "DxGenome", "render", "random_genome", "mutate",
           "ARCHETYPES", "DX_ARCHETYPES", "tonality", "plain_features",
           "PLAIN_NAMES", "run_experiment", "report_run", "ARMS", "ORGANISMS"]

__version__ = "0.1.0"


def run_experiment(*a, **k):
    from .engine import run_experiment
    return run_experiment(*a, **k)


ARMS = ("greedy", "random", "cell", "novelty")
ORGANISMS = ("dx", "bumps")


def report_run(*a, **k):
    from .engine import report_run
    return report_run(*a, **k)