"""eden — a little world of singing agents, heard through a wandering
listener.

The genome is a WORLD: agents on a 2D torus, each with a synthesis
technique, motion genes, and reaction genes (what a collision does —
and collisions mutate the agents' own voice genes within the episode).
A listener with heritable behavior (wander + a seek gene that draws it
toward loudness) moves through the same space; the episode's audio is
what the listener hears — proximity gains, stereo pans, agents
entering and leaving earshot as everything moves and collides.

The phenotype is therefore SOUND OVER TIME: scenes, arcs, events,
chases — exactly what a chunked machine-listening vector exists to
hear. Echo's novelty score on the episode drives the evolution; the
complexification (agents, interactions, structure) is measured, not
assumed.
"""

from .world import WorldGenome, AgentSpec, ListenerSpec, render, REACT
from .genetics import mutate, crossover
from .engine import run_experiment, report_run, ARMS

__all__ = ["WorldGenome", "AgentSpec", "ListenerSpec", "render",
           "TECHNIQUES", "REACTIONS", "mutate", "crossover",
           "run_experiment", "report_run", "ARMS"]

__version__ = "0.1.0"