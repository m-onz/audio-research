"""Arrangers: strategies that turn a chunk pool into a Score.

An arranger is registered with `@register_arranger` and has the signature::

    fn(catalog, ids, length, sr, rng, params, chain) -> list[Event]

- `ids` is the selection it may use (already queried by the pipeline),
- `params` may contain curves (lists) for any time-varying parameter,
- `chain` is the render-time processor chain attached to every event.

Built-ins:
- ``sequence``  chunks end-to-end, with order / gap / speed / gain curves
- ``cloud``     scattered grains with a density curve (granular textures)
- ``euclid``    chunks on a Euclidean rhythm grid (pulse music)
- ``layers``    several sub-arrangers on feature-filtered voices
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .catalog import Catalog, order_ids
from .curves import as_curve
from .processors import Chain, build_chain
from .rng import Seeded
from .score import Event

__all__ = ["ARRANGERS", "register_arranger", "make_arranger", "list_arrangers", "arranger_help"]

ARRANGERS: dict[str, Callable[..., list[Event]]] = {}


def register_arranger(fn: Callable[..., list[Event]]) -> Callable[..., list[Event]]:
    ARRANGERS[fn.__name__] = fn
    return fn


def make_arranger(spec: str | Callable) -> Callable[..., list[Event]]:
    if callable(spec):
        return spec
    if spec in ARRANGERS:
        return ARRANGERS[spec]
    raise ValueError(f"unknown arranger {spec!r}; have {sorted(ARRANGERS)}")


def list_arrangers() -> list[str]:
    return sorted(ARRANGERS)


def arranger_help() -> str:
    lines = [
        "loom arrangers:",
        "  sequence  order|gap|speed|gain|pan as constants or curves; order kinds:",
        "            random | by:FEATURE | markov[:k] | farthest | chase",
        "  cloud     density curve (events/min) scatters grains; overlap freely",
        "  euclid    bpm|steps|pulses|rotate grid; truncates chunks to the step",
        "  layers    voices=[{select, arranger, params, gain}, ...] — each voice",
        "            is its own arranger over a feature-query selection",
    ]
    return "\n".join(lines)


def _clamp_speed(x: float) -> float:
    return float(np.clip(x, 0.06, 24.0))


# --------------------------------------------------------------------------- sequence

@register_arranger
def sequence(
    catalog: Catalog,
    ids: list[int],
    length: float,
    sr: int,
    rng: Seeded,
    params: dict,
    chain: Chain | None,
) -> list[Event]:
    """Chunks end-to-end. gap < 0 overlaps, gap > 0 leaves silence."""
    p = dict(params or {})
    order = str(p.get("order", "random"))
    gap = as_curve(p.get("gap"), 0.0)
    speed = as_curve(p.get("speed"), 1.0)
    gain = as_curve(p.get("gain"), 1.0)
    pan = as_curve(p.get("pan"), 0.0)
    fade = float(p.get("fade", 0.004))

    ordered = order_ids(catalog, ids, kind=order, rng=rng, n=max(1, 4 * len(ids)))
    events: list[Event] = []
    t, k = 0.0, 0
    guard = 0
    while t < length and guard < 100000:
        guard += 1
        u = t / length
        cid = ordered[k % len(ordered)]
        k += 1
        sp = _clamp_speed(speed.at(u))
        dur = max(0.01, catalog.chunks[cid].dur / sp)
        events.append(
            Event(
                chunk=cid,
                t=t,
                dur=dur,
                speed=sp,
                gain=float(gain.at(u)),
                pan=float(np.clip(pan.at(u), -1.0, 1.0)),
                fade_in=fade,
                fade_out=fade,
                chain=chain,
            )
        )
        t += dur + gap.at(u)
    return events


# --------------------------------------------------------------------------- cloud

@register_arranger
def cloud(
    catalog: Catalog,
    ids: list[int],
    length: float,
    sr: int,
    rng: Seeded,
    params: dict,
    chain: Chain | None,
) -> list[Event]:
    """Scatter grains across the timeline at a time-varying density."""
    p = dict(params or {})
    density = as_curve(p.get("density"), 60.0)  # events per minute
    speed = as_curve(p.get("speed"), 1.0)
    gain = as_curve(p.get("gain"), 1.0)
    spread = float(p.get("pan_spread", 0.6))
    fade = float(p.get("fade", 0.01))
    max_dur = float(p.get("max_dur", 4.0))
    order = str(p.get("order", "random"))

    ordered = order_ids(catalog, ids, kind=order, rng=rng, n=max(1, len(ids)))
    events: list[Event] = []
    bucket = 0.5
    t0 = 0.0
    k = 0
    while t0 < length:
        u = t0 / length
        lam = max(0.0, density.at(u)) * bucket / 60.0
        count = int(rng.rng.poisson(lam))
        for _ in range(count):
            cid = ordered[k % len(ordered)]
            k += 1
            sp = _clamp_speed(speed.at(u))
            dur = min(max_dur, max(0.01, catalog.chunks[cid].dur / sp))
            tt = rng.float(t0, min(t0 + bucket, length))
            if tt + dur <= 0:
                continue
            events.append(
                Event(
                    chunk=cid,
                    t=tt,
                    dur=dur,
                    speed=sp,
                    gain=float(gain.at(u)),
                    pan=rng.float(-spread, spread),
                    fade_in=fade,
                    fade_out=fade,
                    chain=chain,
                )
            )
        t0 += bucket
    return events


# --------------------------------------------------------------------------- euclid

@register_arranger
def euclid(
    catalog: Catalog,
    ids: list[int],
    length: float,
    sr: int,
    rng: Seeded,
    params: dict,
    chain: Chain | None,
) -> list[Event]:
    """Place chunks on a Euclidean rhythm grid, bar after bar."""
    p = dict(params or {})
    bpm = float(p.get("bpm", 120.0))
    steps = int(p.get("steps", 16))
    pulses = int(p.get("pulses", 5))
    rotate = int(p.get("rotate", 0))
    gate = float(p.get("gate", 0.9))  # fraction of the step a chunk sounds
    gain = as_curve(p.get("gain"), 1.0)
    accent = float(p.get("accent", 0.7))  # off-step gain
    order = str(p.get("order", "random"))

    pattern = _euclid_pattern(steps, pulses)
    pattern = pattern[rotate % steps:] + pattern[: rotate % steps]
    step_dur = 60.0 / bpm / 4.0  # 16ths at bpm
    ordered = order_ids(catalog, ids, kind=order, rng=rng, n=max(1, len(ids)))
    events: list[Event] = []
    bar_len = step_dur * steps
    k = 0
    n_bars = int(np.ceil(length / bar_len))
    for bar in range(n_bars):
        for s in range(steps):
            if not pattern[s]:
                continue
            t = bar * bar_len + s * step_dur
            if t >= length:
                break
            u = t / length
            cid = ordered[k % len(ordered)]
            k += 1
            dur = step_dur * gate
            events.append(
                Event(
                    chunk=cid,
                    t=t,
                    dur=dur,
                    speed=1.0,
                    gain=float(gain.at(u)) * (1.0 if s % 4 == 0 else accent),
                    pan=0.0,
                    fade_in=0.003,
                    fade_out=min(0.02, dur * 0.3),
                    truncate=True,
                    chain=chain,
                )
            )
    return events


def _euclid_pattern(steps: int, pulses: int) -> list[int]:
    """Bjorklund's algorithm: `pulses` as evenly spaced as possible in `steps`."""
    steps, pulses = int(steps), int(pulses)
    if steps <= 0 or pulses <= 0:
        return [0] * max(0, steps)
    pulses = min(pulses, steps)
    pattern = [0] * steps
    for i in range(pulses):
        pattern[(i * steps) // pulses] = 1
    return pattern


# --------------------------------------------------------------------------- layers

@register_arranger
def layers(
    catalog: Catalog,
    ids: list[int],
    length: float,
    sr: int,
    rng: Seeded,
    params: dict,
    chain: Chain | None,
) -> list[Event]:
    """Run sub-arrangers per voice; each voice is a query + arranger + gain.

    A voice may also carry its own `chain` (a build_chain spec) — otherwise
    the piece-wide render chain applies. Example params::

        voices = [
            {"select": "bass>0.7", "arranger": "euclid", "gain": 1.0},
            {"select": "air>0.6", "arranger": "cloud",
             "chain": ["highpass", "fades"],
             "params": {"density": [12, 90, 12]}},
        ]
    """
    p = dict(params or {})
    voices = p.get("voices") or []
    if not voices:
        raise ValueError("layers needs voices=[{select, arranger, params, gain, chain}, ...]")
    events: list[Event] = []
    for vi, voice in enumerate(voices):
        v = dict(voice)
        cond = str(v.get("select", "all"))
        v_ids = catalog.query(cond)
        if not v_ids:  # a voice that selects nothing stays silent
            continue
        if set(v_ids) != set(ids):
            # keep only chunks from the pipeline selection, intersected
            v_ids = [i for i in v_ids if i in set(ids)] or v_ids
        sub = make_arranger(str(v.get("arranger", "sequence")))
        voice_chain = build_chain(v["chain"]) if v.get("chain") else chain
        sub_events = sub(
            catalog,
            v_ids,
            length,
            sr,
            rng.spawn(f"voice{vi}"),
            dict(v.get("params") or {}),
            voice_chain,
        )
        g = float(v.get("gain", 1.0))
        for e in sub_events:
            e.gain *= g
            e.note = f"voice{vi}"
        events.extend(sub_events)
    events.sort(key=lambda e: e.t)
    return events