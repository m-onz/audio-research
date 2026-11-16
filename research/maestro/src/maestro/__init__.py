"""maestro: the orchestration organ — a score organism over a spine.

A growable CPPN (the same organism design as prism's genome; genomes
are JSON and travel) sampled over a tick grid whose inputs are a
SPIKE-ENCODED SPINE — the scaffold. Every voice is a function of the
spine's state at each tick (NEAT Drummer's law: discover g(p(t)),
never f(t)), so the score inherits the spine's structure, style, and
global shape. The phenotype is an agnostic event score: preview pings
make it listenable, the mapper renders the piece, the engine breeds
score organisms under echo's rotating-lens novelty (prism's loop,
adopted).
"""

from .genome import (FUNCTIONS, INPUTS, INPUT_IDS, Genome, random_genome)
from .champions import (CHAMPIONS, build_champion, random_score_genome)
from .scaffold import (SENSES, STREAMS, WEATHER, Scaffold, estimate_bpm,
                       load_spine, pick_onsets, scaffold_from_array,
                       spike_train)
from .score import (DEFAULT_SENSES, extract_events, levels_matrix,
                    load_score, score_from, score_summary, write_score)
from .preview import preview_audio, render_preview
from .mapper import load_map, mapped_audio, render_mapped, write_map
from .form import FORMS, Form, build_form, describe_form
from .palette import POOLS, auto_voices, corpus_files
from .post import (apply_post, duck, plan_post, post_process, seg_gate,
                   seg_reverse, seg_speed, seg_stretch, seg_stutter)
from .compose import compose, resolve_genome
from .club import (best_specimen, club_compose, driver_score,
                    quantize_top)
from .measure import (SCORE_NAMES, PLAIN_NAMES, anatomy, atlas_png,
                      complexity, plain_coverage, plain_features,
                      plain_radius, score_coverage, score_features)
from .engine import (ARMS, LENSES, floor_ok, lens_rank, report_run,
                     run_arm, run_experiment, tonality, viability)

__version__ = "0.2.0"

__all__ = ["Genome", "random_genome", "random_score_genome", "FUNCTIONS",
           "INPUTS", "INPUT_IDS",
           "Scaffold", "load_spine", "scaffold_from_array", "spike_train",
           "pick_onsets", "estimate_bpm", "STREAMS", "WEATHER", "SENSES",
           "levels_matrix", "extract_events", "score_from", "write_score",
           "load_score", "score_summary", "DEFAULT_SENSES",
           "CHAMPIONS", "build_champion",
           "preview_audio", "render_preview",
           "load_map", "write_map", "render_mapped", "mapped_audio",
           "FORMS", "Form", "build_form", "describe_form",
           "POOLS", "auto_voices", "corpus_files",
           "plan_post", "apply_post", "post_process", "duck",
           "seg_stretch", "seg_stutter", "seg_speed", "seg_reverse",
           "seg_gate",
           "compose", "resolve_genome",
           "club_compose", "driver_score", "best_specimen", "quantize_top",
           "PLAIN_NAMES", "SCORE_NAMES", "plain_features", "plain_coverage",
           "plain_radius", "score_features", "score_coverage", "complexity",
           "anatomy", "atlas_png",
           "ARMS", "LENSES", "viability", "tonality", "lens_rank",
           "floor_ok", "run_arm", "run_experiment", "report_run",
           "__version__"]