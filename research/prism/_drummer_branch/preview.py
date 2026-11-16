"""preview: the artifact — the score as sound, no mapper required.

The ecosystem's law: every artifact is listenable, and selection
needs something to listen to. Phase D maps voices onto real material
through loom's catalog; until then (and after — previews are cheap)
preview synthesizes each voice: a seeded ping per event, frequency
by voice, gain by velocity, pan spread across the field. Mix the
spine underneath (the paper lets the user hear the drum track alone
or with its song — so does prism) and judge the coupling with your
ears. You are the metric.

Deterministic from the score: same score, byte-identical wav.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from echo.probes import SR, decode, write_wav

__all__ = ["render_preview"]

_TAIL_S = 1.0          # ring-out room after the last tick


def _voice_colors(voices):
    """Deterministic voice colors: pitch ladder + equal-power pans."""
    n = max(len(voices), 1)
    freqs = 160.0 * 2.0 ** (np.arange(n) * 0.5)     # 160, 226, 320, 453...
    pans = np.linspace(-0.7, 0.7, n)
    return {v: (float(freqs[i]), float(pans[i])) for i, v in enumerate(voices)}


def render_preview(score: dict, out, spine_gain: float = 0.6) -> str:
    """Score JSON -> stereo wav. Each event is a sine ping (decay
    scaled by the event's duration) plus a seeded noise transient;
    velocity is gain. spine_gain > 0 mixes the spine (from the
    score's provenance) underneath — hear the functional coupling."""
    grid = score["grid"]
    tick_hz, n_ticks = grid["tick_hz"], grid["n_ticks"]
    length = n_ticks / tick_hz + _TAIL_S
    N = int(length * SR)
    L, R = np.zeros(N), np.zeros(N)

    events = score.get("events", [])
    voices = sorted({e["voice"] for e in events}) or \
        sorted(score.get("matrix", {}).keys())
    colors = _voice_colors(voices)
    rng = np.random.default_rng(
        (int(score.get("genome", {}).get("seed", 0)) or 0) & 0x7FFFFFFF)

    for e in events:
        f, pan = colors.get(e["voice"], (440.0, 0.0))
        vel = float(np.clip(e.get("velocity", 1.0), 0.0, 1.0))
        if vel <= 0.0:
            continue
        p0 = int(e["t"] * SR)
        if p0 >= N:
            continue
        dur = max(float(e.get("dur", 1.0 / tick_hz)), 1.0 / tick_hz)
        klen = int(min(max(dur, 0.05), 4.0) * SR)
        tt = np.arange(klen) / SR
        tau = float(np.clip(0.6 * dur, 0.04, 1.5))
        ping = np.sin(2 * np.pi * f * tt) * np.exp(-tt / tau)
        nz = min(int(0.008 * SR), klen)                    # the attack
        ping[:nz] += rng.normal(0, 0.30, nz) * np.exp(-tt[:nz] / 0.002)
        ping *= 0.8 * vel
        end = min(p0 + klen, N)
        seg = ping[: end - p0]
        L[p0: end] += seg * np.cos((pan + 1) * np.pi / 4)
        R[p0: end] += seg * np.sin((pan + 1) * np.pi / 4)

    if spine_gain > 0:
        spine = score.get("spine", {}).get("path", "")
        if spine and Path(spine).exists():
            x = decode(spine, sr=SR).astype(np.float64)
            if len(x) < N:
                x = np.pad(x, (0, N - len(x)))
            L[:len(x)] += x[:N] * spine_gain
            R[:len(x)] += x[:N] * spine_gain

    stereo = np.stack([L, R], axis=1).astype(np.float32)
    peak = float(np.abs(stereo).max())
    if peak > 1e-9:
        stereo = stereo / peak * 0.92
    if not np.isfinite(stereo).all():
        stereo = np.zeros_like(stereo)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_wav(out, stereo, SR)