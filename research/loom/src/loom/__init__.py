"""loom — a modular library for algorithmic audio composition.

Point it at a .wav or a folder of .wavs; it chunks, processes, re-processes,
recombines, and renders a finished piece at a defined length.

    import loom
    result = loom.compose(
        sources="corpus2/MOT-everything.wav",
        preset="storm",
        length=180.0,
        seed=7,
    )
    print(result.summary())

Everything is a swappable strategy: chunkers, processors, arrangers all
live in open registries. See README.md for the full guide.
"""

__version__ = "0.1.0"

from .arrangers import ARRANGERS, arranger_help, list_arrangers, make_arranger, register_arranger
from .catalog import Catalog, order_ids
from .chunkers import CHUNKERS, list_chunkers, make_chunker, register_chunker
from .chunks import Chunk, ChunkRef
from .curves import Curve, as_curve
from .features import FEATURE_NAMES, Features, analyze, feature_help
from .io import peak_normalize, read_wav, resample, to_stereo, write_wav
from .ml import CodebookProcessor, fit_codebook, register_processor_instance
from .pipeline import ComposeResult, compose
from .presets import PRESETS, apply_preset, list_presets
from .processors import PROCESSORS, Chain, Processor, build_chain, list_processors, processor_help, register_processor
from .render import render
from .rng import Seeded
from .score import Event, Score
from .sources import Source, Sources, open_sources
from .store import DiskStore, MemoryStore, Store, open_store

__all__ = [
    "__version__",
    # high-level API
    "compose",
    "ComposeResult",
    "PRESETS",
    "apply_preset",
    "list_presets",
    # score & render
    "Score",
    "Event",
    "render",
    # pool
    "Catalog",
    "order_ids",
    "Chunk",
    "ChunkRef",
    "analyze",
    "Features",
    "FEATURE_NAMES",
    "feature_help",
    # sources & io
    "Source",
    "Sources",
    "open_sources",
    "read_wav",
    "write_wav",
    "resample",
    "peak_normalize",
    "to_stereo",
    # strategies + registries
    "CHUNKERS",
    "register_chunker",
    "make_chunker",
    "list_chunkers",
    "PROCESSORS",
    "Processor",
    "register_processor",
    "build_chain",
    "Chain",
    "list_processors",
    "processor_help",
    "ARRANGERS",
    "register_arranger",
    "make_arranger",
    "list_arrangers",
    "arranger_help",
    # plumbing
    "Curve",
    "as_curve",
    "Seeded",
    "Store",
    "MemoryStore",
    "DiskStore",
    "open_store",
    # ml
    "CodebookProcessor",
    "fit_codebook",
    "register_processor_instance",
]