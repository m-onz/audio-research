"""preview: the artifact — the score as sound, no mapper required.

The ecosystem's law: every artifact is listenable, and selection
needs something to listen to. Phase D maps voices onto real material
(mapper.py); before then (and after — previews are cheap) preview
synthesizes each voice: a seeded ping per event, frequency by voice,
gain by velocity, pan spread across the field. Mix the spine
underneath (the paper lets the user hear the drum track alone or
with its song — so does maestro) and judge the coupling with your
ears. You are the metric.

Mixing is rms-staged, prism's _polish philosophy: the score sits at
a fixed working loudness and the spine rides UNDER it at a dial —
never a global peak-normalize that lets a loud spine bury the score.
Two paths share one builder:

    preview_audio(score, normalize=True)   staged for listening
    preview_audio(score, normalize=False)  raw ping gain — the engine's
                                           phenotype: density is loudness,
                                           so viability gates can work

Deterministic from the score: same score, byte-identical wav.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from echo.probes import SR, decode, write_wav

__all__ = ["preview_audio", "render_preview", "PING_RMS", "SPINE_RMS"]

_TAIL_S = 1.0          # ring-out room after the last tick
PING_RMS = 0.18        # the score's working loudness (staged path)
SPINE_RMS = 0.40       # the spine's loudness at spine_gain = 1.0


def _voice_colors(voices):
    """Deterministic voice colors: pitch ladder + equal-power pans."""
    n = max(len(voices), 1)
    freqs = 160.0 * 2.0 ** (np.arange(n) * 0.5)     # 160, 226, 320, 453...
    pans = np.linspace(-0.7, 0.7, n)
    return {v: (float(freqs[i]), float(pans[i])) for i, v in enumerate(voices)}


def _ping_mix(score: dict) -> tuple[np.ndarray, np.ndarray]:
    """Raw stereo ping bus (L, R), unnormalized — velocity is gain, so
    a sparse score is genuinely quiet. The engine's phenotype."""
    grid = score["grid"]
    tick_hz, n_ticks = grid["tick_hz"], grid["n_ticks"]
    N = int((grid.get("length_s", n_ticks / tick_hz) + _TAIL_S) * SR)
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
    return L, R


def _stage(x: np.ndarray, target_rms: float) -> np.ndarray:
    """rms-stage a bus; silence stays silence, peaks stay under 0.98."""
    x = np.nan_to_num(np.asarray(x, float))
    rms = float(np.sqrt((x * x).mean())) if x.size else 0.0
    if rms < 1e-9:
        return x
    y = x * (target_rms / rms)
    peak = float(np.abs(y).max())
    if peak > 0.98:
        y *= 0.98 / peak
    return y


def preview_audio(score: dict, spine_gain: float = 0.0,
                  normalize: bool = True) -> np.ndarray:
    """Score -> stereo audio in memory. normalize=True stages the ping
    bus to PING_RMS and the spine to SPINE_RMS * spine_gain underneath
    (the listening artifact). normalize=False returns the raw ping bus
    alone — density is loudness, the engine's phenotype for gates."""
    L, R = _ping_mix(score)
    if normalize:
        L = _stage(L, PING_RMS)
        R = _stage(R, PING_RMS)
        if spine_gain > 0:
            spine = score.get("spine", {}).get("path", "")
            if spine and Path(spine).exists():
                x = decode(spine, sr=SR).astype(np.float64)
                N = len(L)
                if len(x) < N:
                    x = np.pad(x, (0, N - len(x)))
                sx = _stage(x[:N], SPINE_RMS * float(spine_gain))
                L += sx
                R += sx
    stereo = np.stack([L, R], axis=1).astype(np.float32)
    peak = float(np.abs(stereo).max())
    if normalize and peak > 0.95:                 # cap, never erase
        stereo = stereo / peak * 0.95
    if not np.isfinite(stereo).all():
        stereo = np.zeros_like(stereo)
    return stereo


def render_preview(score: dict, out, spine_gain: float = 0.6) -> str:
    """Score JSON -> staged stereo wav (the listenable artifact)."""
    stereo = preview_audio(score, spine_gain=spine_gain, normalize=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_wav(out, stereo, SR)