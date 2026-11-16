"""prism: the score organism — one CPPN, many phenotypes.

The first phenotype is the score: a growable compositional graph
sampled over a tick grid whose inputs are a SPIKE-ENCODED SPINE
(the scaffold — NEAT Drummer's law: parts as functions of parts,
g(p(t)) over f(t)), whose outputs are voices, and whose phenotype
is an agnostic event score that maps onto any sound material.
"""

from .genome import (FUNCTIONS, INPUTS, INPUT_IDS, Genome, random_genome)
from .scaffold import (SENSES, STREAMS, WEATHER, Scaffold, load_spine,
                       pick_onsets, scaffold_from_array, spike_train)
from .score import (DEFAULT_SENSES, extract_events, levels_matrix,
                    load_score, score_from, score_summary, write_score)
from .champions import CHAMPIONS, build_champion
from .preview import render_preview

__version__ = "0.1.0"

__all__ = ["Genome", "random_genome", "FUNCTIONS", "INPUTS", "INPUT_IDS",
           "Scaffold", "load_spine", "scaffold_from_array", "spike_train",
           "pick_onsets", "STREAMS", "WEATHER", "SENSES",
           "levels_matrix", "extract_events", "score_from", "write_score",
           "load_score", "score_summary", "DEFAULT_SENSES",
           "CHAMPIONS", "build_champion", "render_preview", "__version__"]