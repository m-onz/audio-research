"""Chunkers: strategies for slicing sources into a pool of chunks.

A chunker receives `Sources` and returns a list of `ChunkRef` — pure
metadata, no audio. The registry is open: register your own with
`@register_chunker`.

Built-ins:
- ``fixed``     uniform windows (seconds, overlap, jitter)
- ``onset``     energy-onset segmentation; *streams* the source in blocks,
                so it is safe on hour-long files
- ``random``    N random windows per source
- ``beats``     fixed windows locked to a tempo grid (bars of a bpm)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .chunks import ChunkRef
from .rng import Seeded
from .sources import Sources

__all__ = ["CHUNKERS", "register_chunker", "make_chunker", "list_chunkers"]

CHUNKERS: dict[str, Callable[..., list[ChunkRef]]] = {}


def register_chunker(fn: Callable[..., list[ChunkRef]]) -> Callable[..., list[ChunkRef]]:
    CHUNKERS[fn.__name__] = fn
    return fn


def make_chunker(spec: str | Callable) -> Callable[..., list[ChunkRef]]:
    """Resolve 'name' or a callable into a chunker function."""
    if callable(spec):
        return spec
    if spec in CHUNKERS:
        return CHUNKERS[spec]
    raise ValueError(f"unknown chunker {spec!r}; have {sorted(CHUNKERS)}")


def list_chunkers() -> list[str]:
    return sorted(CHUNKERS)


# --------------------------------------------------------------------------- built-ins

@register_chunker
def fixed(
    sources: Sources,
    rng: Seeded,
    seconds: float = 1.0,
    overlap: float = 0.0,
    jitter: float = 0.0,
    max_per_source: int | None = None,
) -> list[ChunkRef]:
    """Uniform windows of `seconds`, optionally overlapping and jittered."""
    refs = []
    for si in range(len(sources)):
        src = sources[si]
        dur = src.duration
        step = max(1e-6, seconds * (1.0 - overlap))
        t, count = 0.0, 0
        while t < dur:
            if max_per_source is not None and count >= max_per_source:
                break
            j = rng.float(-jitter, jitter) * seconds if jitter else 0.0
            start = max(0.0, t + j)
            d = min(seconds, dur - start)
            if d > 0.01:
                refs.append(ChunkRef(si, start, d))
            t += step
            count += 1
    return refs


@register_chunker
def beats(
    sources: Sources,
    rng: Seeded,
    bpm: float = 120.0,
    bars: int = 1,
    overlap: float = 0.0,
) -> list[ChunkRef]:
    """Windows locked to a musical grid: `bars` bars of `bpm` 4/4."""
    seconds = 4.0 * 60.0 / bpm * bars
    return fixed(sources, rng, seconds=seconds, overlap=overlap)


@register_chunker
def random(
    sources: Sources,
    rng: Seeded,
    n: int = 64,
    min_seconds: float = 0.25,
    max_seconds: float = 4.0,
) -> list[ChunkRef]:
    """N random windows per source."""
    refs = []
    for si in range(len(sources)):
        dur = sources[si].duration
        for _ in range(n):
            d = rng.float(min_seconds, max_seconds)
            start = rng.float(0.0, max(0.0, dur - d)) if dur > d else 0.0
            refs.append(ChunkRef(si, start, min(d, dur)))
    return refs


@register_chunker
def onset(
    sources: Sources,
    rng: Seeded,
    min_seconds: float = 0.15,
    max_seconds: float = 8.0,
    fallback_seconds: float = 2.0,
    threshold: float = 2.0,
    floor_db: float = -50.0,
) -> list[ChunkRef]:
    """Slice at energy onsets (adaptive local-median threshold).

    Streams each source in 30 s blocks — the reference implementation for
    analyzing hour-long files without loading them. Falls back to fixed
    windows at `fallback_seconds` for sources with no detectable onsets.
    """
    refs = []
    for si in range(len(sources)):
        times = _onset_times(sources[si], threshold=threshold, floor_db=floor_db)
        dur = sources[si].duration
        if not times:  # flat material: fall back to fixed windows
            merged = [
                [t, min(fallback_seconds, dur - t)]
                for t in np.arange(0.0, dur, fallback_seconds)
                if dur - t > 0.02
            ]
        else:
            bounds = [0.0, *times, dur]
            segs = [(a, b - a) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 0.02]
            # absorb tiny segments into neighbours; split overlong ones
            merged = []
            for start, d in segs:
                if merged and d < min_seconds:
                    merged[-1][1] = start + d - merged[-1][0]
                    continue
                while d > max_seconds:
                    merged.append([start, max_seconds])
                    start += max_seconds
                    d -= max_seconds
                if d > 0.02:
                    merged.append([start, d])
        for start, d in merged:
            refs.append(ChunkRef(si, float(start), float(d)))
    return refs


def _onset_times(src, threshold: float = 2.0, floor_db: float = -50.0) -> list[float]:
    """Onset times via frame-RMS rises over a running local median.

    hop = 10 ms, window = 30 ms. An onset fires when the rms exceeds
    `threshold` x the local median of recent history, is a local rise, and
    is at least 60 ms after the previous onset. Streams in 30 s blocks; the
    block overlap carries context so no analysis window straddles a gap.
    """
    hop_s, win_s, hist_n = 0.01, 0.03, 200
    hop_n = max(8, int(hop_s * src.sr))
    win_n = max(16, int(win_s * src.sr))
    floor = 10.0 ** (floor_db / 20.0)
    block_n = int(30.0 * src.sr)
    times: list[float] = []
    env: list[float] = []  # frame rms history for the local median
    last_t = -1.0
    block_start = 0  # absolute sample index of the current block
    for block in src.iter_blocks(block_seconds=30.0, overlap_seconds=win_n / src.sr):
        mono = block.mean(axis=1)
        n = len(block)
        for fs in range(0, block_n, hop_n):
            if fs + win_n > n:
                break
            w = mono[fs : fs + win_n]
            r = float(np.sqrt(np.mean(w * w)))
            t = (block_start + fs + win_n / 2) / src.sr
            env.append(r)
            if len(env) >= 3:
                hist_med = float(np.median(env[-hist_n:]))
                if r > env[-2] and r > threshold * hist_med and r > floor and (t - last_t) > 0.06:
                    times.append(t)
                    last_t = t
        block_start += block_n
    return times