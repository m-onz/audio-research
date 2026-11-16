"""quilt — the seamstress organ.

echo is the memory: every sound ever heard becomes a point in a
556-dimensional feature space. quilt is the voice that speaks *through*
that memory. The corpus folder is cut IN PLACE into a pile of pieces
(`quilt split`: onset, silence, and fixed scissors choose themselves
per file; originals are replaced by their chunks — splitting, never
duplicating). Every piece of the pile is encoded by echo's ear and
pinned into the space it belongs, and then a walker takes a trip
through the territory and sews what it finds into one continuous
piece of audio:

    corpus ──split (in place)──> the pile of chunk files
    pile ──echo.encode──> the bank (feature space, provenance in names)
                 │
                 walker (chain | contrast | roam | attractor | drift | follow)
                 │
                 route ──glue──> the piece (no processing, ever)

The law: ears select, processes energize, and quilt never makes sound
— it only re-arranges sound that already exists. Every output sample
is a sample from the input corpus; the only synthesis is the order.

Lineage: this is corpus-based concatenative synthesis (cataRT,
Diemo Schwarz / IRCAM) rebuilt on echo's fitted metric, with the
latent-space navigation paradigm of RAVE (Caillon & Esling) mapped
onto a discrete space — see README §rave for the research verdict.
"""

import os as _os

# This machine's OpenBLAS multithreads small matmuls into pathology:
# measured (172x1025)@(1025x64) = 590 ms with 4 threads, 5 ms with 1.
# echo's ear lives on small matmuls; one thread restores its
# documented speed. setdefault: an explicit env var always wins.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")
try:                                                # numpy already loaded
    import threadpoolctl as _tp                    # elsewhere? env vars
    _THREAD_LIMIT = _tp.threadpool_limits(limits=1)  # came too late for —
except Exception:                                   # this still helps.
    _THREAD_LIMIT = None                            # optional, not fatal

from .io import read_audio, write_audio
from .naming import CHUNK_RE, is_chunk_file, parse_chunk_name
from .chunking import (chunk, chunk_fixed, chunk_onset, chunk_silence,
                       chunk_whole, STRATEGIES, DEFAULT_STRATEGIES,
                       MIN_DUR, MAX_SEG, FIXED_DUR, WHOLE_MAX)
from .gpu import GpuEar, mps_available, resolve_device
from .split import split_corpus
from .index import build_bank, iter_audio, provenance_path
from .atlas import ChunkAtlas
from .navigate import walk, MODES, Route, Step
from .render import render_route, write_route_json, load_route

__version__ = "0.2.0"

__all__ = [
    "read_audio", "write_audio",
    "chunk", "chunk_fixed", "chunk_onset", "chunk_silence", "chunk_whole",
    "STRATEGIES", "DEFAULT_STRATEGIES", "MIN_DUR", "MAX_SEG", "FIXED_DUR",
    "WHOLE_MAX", "GpuEar", "mps_available", "resolve_device",
    "split_corpus", "CHUNK_RE", "is_chunk_file",
    "parse_chunk_name", "build_bank", "iter_audio", "provenance_path",
    "ChunkAtlas", "walk", "MODES", "Route", "Step",
    "render_route", "write_route_json", "load_route",
    "__version__",
]