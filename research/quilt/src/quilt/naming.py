"""naming: the pile's provenance, carried in filenames.

`quilt split` cuts `orig.wav` in place into
`orig__onset_0000.wav, orig__onset_0001.wav, ...` — the scissor's
name and the chunk's temporal order survive in the filename. That is
the whole contract between split, index, and the walkers:

  * index parses a name into (strategy, original stem, index)
  * `follow` walks original sequences by index + 1
  * `--strategy` filters walks by scissor

Unmarked files are originals kept whole: ("whole", stem, -1).
"""

from __future__ import annotations

import re

__all__ = ["CHUNK_RE", "is_chunk_file", "parse_chunk_name"]

# the pile's naming convention: original provenance in the filename
CHUNK_RE = re.compile(r"__(onset|silence|fixed)_(\d{4,})\.wav$",
                      re.IGNORECASE)


def is_chunk_file(name: str) -> bool:
    """Does this filename carry quilt's chunk marker?"""
    return CHUNK_RE.search(name) is not None


def parse_chunk_name(rel: str):
    """A pile filename -> (strategy, orig_stem, idx). Unmarked files
    are originals kept whole: ("whole", stem, -1)."""
    m = CHUNK_RE.search(rel)
    if m:
        return m.group(1).lower(), rel[:m.start()], int(m.group(2))
    stem = rel.rsplit(".", 1)[0]
    return "whole", stem, -1