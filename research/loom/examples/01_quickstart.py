"""Quickstart: compose from a folder of .wavs in five lines.

Run:  python3 examples/01_quickstart.py path/to/corpus [more/paths...]
"""

import sys
from pathlib import Path

import loom

sources = sys.argv[1:] or [Path(__file__).parents[2] / "audio_lab" / "clubbed"]

result = loom.compose(
    sources=sources,
    preset="storm",  # maximal-contrast collage
    length=60.0,
    seed=7,
)
print(result.summary())

# Every render writes a manifest next to the wav — the full resolved
# config, seed, feature medians, and the output checksum:
#   loom_out/storm_7.wav
#   loom_out/storm_7.manifest.json