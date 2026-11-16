"""mapper: phase D — voices onto real material, the final piece.

The score is agnostic on purpose; this is where sound happens. A
voice map assigns each score voice a SOUND (any audio file in the
corpus or beyond), with gain, pan, varispeed, and an optional
truncate-to-duration. Every event places its sound at the event's
tick with velocity as gain — the paper's semantics, kept: output
level is strike volume. Overlap-add, cosine fades at every cut, one
master, one wav.

The loom integration (voice -> catalog QUERY -> chunk -> chain) drops
in exactly here: a query resolves to the same file path this map
consumes. Until then the map is the interface — deterministic,
offline, reproducible from the score alone.

    score.json + {"pulse": "kick.wav", "flow": "grain.wav"}  ->  piece.wav
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from echo.probes import SR, decode, write_wav

__all__ = ["render_mapped", "mapped_audio", "load_map", "write_map",
           "VoiceSpec"]

_FADE_S = 0.004          # edge guard at every cut — never a click


class VoiceSpec:
    """One voice's sound: file + gain/pan/varispeed/truncate."""

    def __init__(self, file: str, gain: float = 1.0, pan: float = 0.0,
                 speed: float = 1.0, truncate: bool = False):
        self.file = str(file)
        self.gain = float(gain)
        self.pan = float(np.clip(pan, -1.0, 1.0))
        self.speed = float(np.clip(speed, 0.25, 4.0))
        self.truncate = bool(truncate)

    def to_dict(self) -> dict:
        return {"file": self.file, "gain": self.gain, "pan": self.pan,
                "speed": self.speed, "truncate": self.truncate}

    @classmethod
    def from_dict(cls, d) -> "VoiceSpec":
        return cls(d["file"], d.get("gain", 1.0), d.get("pan", 0.0),
                   d.get("speed", 1.0), d.get("truncate", False))


def load_map(path) -> dict:
    """A voice map json: {"voices": {name: spec}, "spine_gain": 0.0}.
    A spec is a file path or {"file": ..., "gain": ..., ...}."""
    d = json.loads(Path(path).read_text())
    voices = {}
    for name, spec in d.get("voices", {}).items():
        voices[name] = spec if isinstance(spec, str) \
            else VoiceSpec.from_dict(spec).to_dict()
    return {"voices": voices, "spine_gain": float(d.get("spine_gain", 0.0))}


def write_map(voice_map: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(voice_map, indent=1))
    return path


def _varispeed(x: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-6 or x.shape[0] < 2:
        return x
    n = max(2, int(x.shape[0] / speed))
    pos = np.linspace(0.0, x.shape[0] - 1, n)
    base = np.arange(x.shape[0])
    if x.ndim == 1:
        return np.interp(pos, base, x)
    return np.stack([np.interp(pos, base, x[:, c])
                     for c in range(x.shape[1])], axis=1)


def _fade_edges(x: np.ndarray, sr: int) -> np.ndarray:
    nf = min(int(_FADE_S * sr), x.shape[0] // 2)
    if nf > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
        x[:nf] *= ramp[:, None] if x.ndim == 2 else ramp
        x[-nf:] *= (ramp[::-1])[:, None] if x.ndim == 2 else ramp[::-1]
    return x


def mapped_audio(score: dict, voice_map, tail_s: float = 2.0,
                 spine_loop: bool = False) -> np.ndarray:
    """Score + voice map -> the stereo mix in memory (compose's path:
    the piece continues into post). The map may be the load_map format
    ({"voices": {name: spec}, "spine_gain": g}) or flat
    ({name: spec, "spine_gain": g}) — auto_voices returns the flat one.
    Every event places its voice's sound at the event's tick, velocity
    as gain, the voice's pan and varispeed applied; gate events may
    truncate to their duration. spine_gain > 0 mixes the spine (the
    score's provenance) underneath — LOOPED to fill the piece when
    the spine is shorter (compose loops its scaffold; the bed should
    keep up). NOT normalized yet — post owns the master."""
    grid = score["grid"]
    length = grid.get("length_s", grid["n_ticks"] / grid["tick_hz"]) + tail_s
    N = int(length * SR)
    L, R = np.zeros(N), np.zeros(N)
    vm = voice_map if isinstance(voice_map, dict) else load_map(voice_map)
    if "voices" in vm:
        voices = vm["voices"]
    else:                      # flat: every key but spine_gain is a voice
        voices = {k: v for k, v in vm.items() if k != "spine_gain"}
    sg = float(vm.get("spine_gain", 0.0))
    cache: dict[str, np.ndarray] = {}
    for e in score.get("events", []):
        spec = voices.get(e["voice"])
        if spec is None:
            continue
        spec = spec if isinstance(spec, dict) else {"file": spec}
        key = (spec["file"], float(spec.get("speed", 1.0)))
        if key not in cache:
            x = decode(spec["file"], sr=SR, stereo=True).astype(np.float64)
            if x.ndim == 1:
                x = x[:, None] * np.ones((1, 2))
            x = _varispeed(x, float(spec.get("speed", 1.0)))
            x = _fade_edges(x, SR)
            cache[key] = x
        x = cache[key]
        vel = float(np.clip(e.get("velocity", 1.0), 0.0, 1.0))
        g = vel * float(spec.get("gain", 1.0))
        if g <= 0.0:
            continue
        p0 = int(e["t"] * SR)
        if p0 >= N:
            continue
        seg = x
        if spec.get("truncate") and e.get("dur"):
            k = max(64, int(float(e["dur"]) * SR))
            seg = x[: k]
        seg = seg * g
        pan = float(np.clip(spec.get("pan", 0.0), -1.0, 1.0))
        th = (pan + 1.0) * np.pi / 4.0
        # equal-power pan, center-unity: pan 0 changes nothing
        gl = np.cos(th) / np.cos(np.pi / 4)
        gr = np.sin(th) / np.cos(np.pi / 4)
        end = min(p0 + seg.shape[0], N)
        seg = seg[: end - p0]
        if seg.shape[1] == 1:
            seg = seg * np.ones((1, 2))
        L[p0: end] += seg[:, 0] * gl
        R[p0: end] += seg[:, 1] * gr

    # the spine underneath: staged to a working rms, LOOPED to fill
    # the piece when shorter (compose loops its scaffold; the bed
    # keeps up), else zero-padded
    if sg > 0:
        spine = score.get("spine", {}).get("path", "")
        if spine and Path(spine).exists():
            xs = decode(spine, sr=SR, stereo=True).astype(np.float64)
            if xs.ndim == 1:
                xs = xs[:, None] * np.ones((1, 2))
            if spine_loop and 0 < len(xs) < N:
                reps = int(np.ceil(N / len(xs)))
                xs = np.tile(xs, (reps, 1))
            if len(xs) < N:
                xs = np.pad(xs, ((0, N - len(xs)), (0, 0)))
            rms = float(np.sqrt((xs[:N] ** 2).mean()))
            if rms > 1e-9:
                xs = xs * (0.40 * sg / rms)
            L[:N] += xs[:N, 0]
            R[:N] += xs[:N, 1]

    stereo = np.stack([L, R], axis=1).astype(np.float32)
    if not np.isfinite(stereo).all():
        stereo = np.nan_to_num(stereo, copy=False)
    return stereo


def render_mapped(score: dict, voice_map, out, tail_s: float = 2.0) -> str:
    """Score + voice map -> the piece, one normalized wav."""
    stereo = mapped_audio(score, voice_map, tail_s=tail_s)
    peak = float(np.abs(stereo).max())
    if peak > 1e-9:
        stereo = stereo / peak * 0.92
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_wav(out, stereo, SR)