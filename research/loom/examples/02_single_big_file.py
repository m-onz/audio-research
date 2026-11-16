"""Compose from a single hour-long .wav without ever loading it.

Sources are lazy: chunking streams in 30 s blocks, chunks are metadata
(audio is decoded per-chunk on demand), rendering seeks. This exact script
turns a 906 MB / 55-minute file into a 3-minute piece in ~12 s and well
under 1 GB RAM.

Run:  python3 examples/02_single_big_file.py big.wav
"""

import sys
from pathlib import Path

import loom

big = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[2] / "corpus2" / "MOT-oct26-everything.wav"

result = loom.compose(
    sources=big,
    preset="drift",  # ambient granular: onset chunks, thinning cloud
    length=180.0,
    seed=3,
    bits=24,  # match the corpus bit depth
    store="disk",  # spill any derived chunks to disk, not RAM
    verbose=True,
)
print(result.summary())