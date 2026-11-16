"""score: the traversal — one CPPN pass over the tick grid.

The paper's mechanics, kept faithfully: sample the network at discrete
ticks; every output is a voice; the output LEVEL at a tick is the
velocity of a strike; two consecutive audible ticks are two separate
strikes; a pause must be PRODUCED (output below threshold for the
intervening ticks). No memory, no recurrence — the pattern is a pure
function of the scaffold's state at each tick, so all global structure
is imported from the spine (NEAT Drummer §5: discover g(p(t)), never
f(t)). When the spine diverges, every voice diverges with it — coupled
by construction, the Johnny Cope measure-23 effect.

Two event semantics read the same matrix:

    strike   the paper's drums — every audible tick is an event of
             one tick; velocity carries the nuance (softness is
             free; machine-gun runs are the cost)
    gate     sustained material — a rising edge opens an event held
             while the level stays audible; velocity is the run's
             peak, duration the run's length

The score is JSON and knows nothing about audio: events are (voice,
tick, t, velocity, dur). Mapping voices onto sounds is the mapper's
business (loom renders — phase D). Same score, different mapper: the
same organism, a different piece.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .genome import Genome
from .scaffold import SENSES, Scaffold

__all__ = ["levels_matrix", "extract_events", "score_from", "write_score",
           "load_score", "score_summary", "DEFAULT_SENSES"]

DEFAULT_SENSES = ("spikes", "weather")


def levels_matrix(genome: Genome, feeds: dict) -> dict:
    """One forward pass: every output sampled at every tick. The feeds
    broadcast to a common (n_ticks,) shape — the whole score is a
    single vectorized evaluation of the memoryless graph."""
    out = genome.evaluate(feeds)
    return {k: np.asarray(v, np.float32).reshape(-1) for k, v in out.items()}


def extract_events(levels: dict, tick_hz: float, threshold: float = 0.12,
                   mode: str = "strike") -> list[dict]:
    """Levels -> events. Velocity = level rectified and clipped to
    [0, 1]; below `threshold` is silence the way the paper means it:
    inaudible output between strikes."""
    if mode not in ("strike", "gate"):
        raise ValueError(f"mode must be strike|gate, got {mode!r}")
    ev = []
    for voice in sorted(levels):
        v = np.clip(np.asarray(levels[voice], float), 0.0, 1.0)
        audible = v >= threshold
        if mode == "strike":
            for i in np.where(audible)[0]:
                ev.append({"voice": voice, "tick": int(i),
                           "t": float(i / tick_hz),
                           "dur": float(1.0 / tick_hz),
                           "velocity": float(v[i])})
        else:
            i, n = 0, len(v)
            while i < n:
                if not audible[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and audible[j + 1]:
                    j += 1
                ev.append({"voice": voice, "tick": int(i),
                           "t": float(i / tick_hz),
                           "dur": float((j - i + 1) / tick_hz),
                           "velocity": float(v[i: j + 1].max())})
                i = j + 1
    ev.sort(key=lambda e: (e["t"], e["voice"]))
    return ev


def score_from(genome: Genome, scaffold: Scaffold,
               senses=DEFAULT_SENSES, threshold: float = 0.12,
               mode: str = "strike", note: str | None = None) -> dict:
    """Genome + spine -> the score, with full provenance: the genome,
    the scaffold's identity, the senses, and the raw level matrix (the
    phenotype — analyzable, replayable, and the input for mapper and
    any score-space novelty later)."""
    senses = tuple(senses)
    bad = [s for s in senses if s not in SENSES]
    if bad:
        raise ValueError(f"unknown senses {bad}; valid: {SENSES}")
    feeds = scaffold.feeds(senses=senses)
    levels = levels_matrix(genome, feeds)
    events = extract_events(levels, scaffold.tick_hz, threshold, mode)
    return {
        "prism": "score", "version": 1,
        "grid": {"bpm": scaffold.bpm, "ticks_per_beat": scaffold.ticks_per_beat,
                 "tick_hz": scaffold.tick_hz, "n_ticks": scaffold.n_ticks,
                 "length_s": scaffold.length_s},
        "spine": {"path": scaffold.path, "sha256": scaffold.sha256,
                  "spine_length_s": scaffold.spine_length_s,
                  "onsets": scaffold.onsets},
        "senses": list(senses), "threshold": float(threshold),
        "mode": mode, "note": note,
        "genome": genome.to_dict(),
        "n_events": len(events), "events": events,
        "matrix": {k: [float(x) for x in v] for k, v in levels.items()},
    }


def write_score(score: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # no rounding: JSON round-trips must be bit-exact (loom's law)
    path.write_text(json.dumps(score, indent=1))
    return path


def load_score(path) -> dict:
    return json.loads(Path(path).read_text())


def score_summary(score: dict) -> dict:
    """Per-voice census: events, density (events/s), velocity stats.
    The quick read before listening — 'unappealing patterns' often
    show in the density row first (the paper's visual-cull argument)."""
    length = score["grid"]["length_s"] or 1.0
    per = {}
    for e in score["events"]:
        d = per.setdefault(e["voice"], {"events": 0, "velocity": []})
        d["events"] += 1
        d["velocity"].append(e["velocity"])
    out = {}
    for voice, d in sorted(per.items()):
        vv = np.asarray(d["velocity"], float)
        out[voice] = {"events": d["events"],
                      "density": round(d["events"] / length, 2),
                      "velocity_mean": round(float(vv.mean()), 3),
                      "velocity_max": round(float(vv.max()), 3)}
    return {"length_s": round(score["grid"]["length_s"], 2),
            "tick_hz": score["grid"]["tick_hz"],
            "mode": score["mode"], "threshold": score["threshold"],
            "n_events": score["n_events"], "voices": out}