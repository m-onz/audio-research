"""Renderer: a Score + Catalog -> finished .wav.

The renderer is deliberately boring and absolutely reliable:

1. allocate a float64 master buffer of the exact requested length,
2. for each event: materialize the chunk (lazy read or store lookup),
   run its processor chain, varispeed, truncate/fade, pan, and mix in,
3. peak-normalize (or soft-clip), write PCM, return stats.

Events past the end of the piece are clipped, never wrapped. NaN/Inf can
never reach the output (the chain guards). Rendering a score twice with the
same seed produces byte-identical files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .io import resample, to_stereo, write_wav
from .rng import Seeded

if TYPE_CHECKING:  # pragma: no cover
    from .catalog import Catalog
    from .score import Score

__all__ = ["render", "mix_event"]


def render(
    score: "Score",
    catalog: "Catalog",
    out: str | Path,
    bits: int = 16,
    normalize: bool = True,
    headroom_db: float = 0.3,
    soft_clip: bool = False,
    verbose: bool = False,
) -> dict:
    """Mix a score down and write `out`. Returns render stats."""
    sources = catalog.sources
    store = catalog.store
    sr = score.sr
    n_total = score.n_samples
    master = np.zeros((n_total, 2), dtype=np.float64)

    rng_root = Seeded(score.seed)
    order = sorted(range(len(score.events)), key=lambda i: score.events[i].t)
    clipped = 0
    for progress, i in enumerate(order):
        ev = score.events[i]
        if ev.t >= score.length or ev.dur <= 0:
            clipped += 1
            continue
        chunk = catalog.chunks[ev.chunk]
        audio = chunk.materialize(sources, store)
        if audio.shape[0] == 0:
            continue
        audio = to_stereo(audio)
        t_norm = min(1.0, max(0.0, ev.t / max(1e-9, score.length)))
        if ev.chain is not None:
            audio = ev.chain.run(audio, sr, t_norm, rng_root.spawn(f"ev{i}"))
        if abs(ev.speed - 1.0) > 1e-9:
            # play at ev.speed: n / speed samples (varispeed, pitch x speed)
            audio = resample(audio, max(1, int(sr * ev.speed)), sr)
        target = int(round(ev.dur * sr))
        if ev.truncate and audio.shape[0] > target:
            audio = audio[:target]
        mix_event(master, audio, ev, sr)
        if verbose and progress % 50 == 0:
            print(f"  rendered {progress}/{len(order)} events", flush=True)

    if soft_clip:
        master = np.tanh(master)

    stats = write_wav(out, master, sr, bits=bits, normalize=normalize, headroom_db=headroom_db)
    stats["events_rendered"] = len(score.events) - clipped
    stats["events_clipped"] = clipped
    return stats


def mix_event(master: np.ndarray, audio: np.ndarray, ev, sr: int) -> None:
    """Overlap-add one event into the master with equal-power panning."""
    n = audio.shape[0]
    if n == 0:
        return
    start = int(round(ev.t * sr))
    end = start + n
    m_start = max(0, start)
    m_end = min(master.shape[0], end)
    if m_end <= m_start:
        return
    a = audio[m_start - start : m_end - start]

    # fades: cosine ramps, clamped to what's left of the event
    fade_in_n = min(a.shape[0], max(0, int(ev.fade_in * sr)))
    fade_out_n = min(a.shape[0], max(0, int(ev.fade_out * sr)))
    if fade_in_n > 1:
        a[:fade_in_n] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(fade_in_n) / fade_in_n))[:, None]
    if fade_out_n > 1:
        a[-fade_out_n:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(fade_out_n) / fade_out_n))[:, None]

    theta = (min(1.0, max(-1.0, ev.pan)) + 1.0) * np.pi / 4.0
    l_gain = float(ev.gain) * np.cos(theta)
    r_gain = float(ev.gain) * np.sin(theta)
    master[m_start:m_end, 0] += a[:, 0] * l_gain
    master[m_start:m_end, 1] += a[:, 1] * r_gain