"""render: the glue — a route becomes audio.

The law of the house: NO PROCESSING. Every sample in the output is a
sample from the corpus, in the walker's order, butt-spliced. The
optional crossfade is off by default and exists only because hard
splices click at discontinuities — use it if the clicks offend, or
keep the law and let them be part of the material.

Pieces are read from the pile on demand (libsndfile fast path on the
pile's native wav, ffmpeg for anything unusual) and cached: a route
that roams one original reuses the read. A piece is always the whole
file it names — the pile was pre-cut by `quilt split`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from echo.probes import write_wav, SR
from .io import read_audio

__all__ = ["render_route", "write_route_json", "load_route"]

_CACHE_BYTES = 1_500_000_000        # ~1.5 GB of decoded audio in RAM


def load_route(path) -> dict:
    """A route JSON -> dict (the render contract)."""
    d = json.loads(Path(path).read_text())
    if d.get("quilt") != "route":
        raise ValueError(f"{path} is not a quilt route")
    return d


def write_route_json(route_json: dict, path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(route_json, indent=1))
    return str(p)


class _DecodeCache:
    """path -> stereo float32, oldest evicted past the byte budget."""

    def __init__(self, budget: int = _CACHE_BYTES):
        self.budget, self.cache, self.order, self.bytes = budget, {}, [], 0

    def get(self, path, sr: int):
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        audio = read_audio(key, sr)
        self.cache[key] = audio
        self.order.append(key)
        self.bytes += audio.nbytes
        while self.bytes > self.budget and len(self.order) > 1:
            old = self.order.pop(0)
            self.bytes -= self.cache[old].nbytes
            del self.cache[old]
        return audio


def _crossfade(pieces, xf_samples: int) -> np.ndarray:
    """Linear crossfade between consecutive pieces; a piece shorter
    than the fade fades over all it has."""
    if not pieces:
        raise ValueError("no pieces to stitch")
    out = np.asarray(pieces[0], np.float64).copy()
    for p in pieces[1:]:
        p = np.asarray(p, np.float64)
        L = int(min(xf_samples, len(out), len(p)))
        if L <= 0:
            out = np.concatenate([out, p])
            continue
        w = np.linspace(0.0, 1.0, L).reshape(-1, 1)   # (L, 1) stereo-safe
        out = np.concatenate([out[:-L], out[-L:] * (1.0 - w) + p[:L] * w,
                              p[L:]])
    return out


def render_route(route, out_path, crossfade_ms: float = 0.0,
                 exact: bool = True, corpus_root=None, sr: int = SR) -> dict:
    """Route JSON (from walk/to_json or load_route) -> wav. Returns
    the render report."""
    steps = route.get("steps", [])
    if not steps:
        raise ValueError("route has no steps")
    root = Path(corpus_root or route.get("corpus_root") or ".")
    xf = int(round(crossfade_ms * sr / 1000.0))
    cache = _DecodeCache()

    pieces = []
    for st in steps:
        path = root / st["file"]
        if not path.exists():
            raise FileNotFoundError(
                f"corpus file missing: {path} — re-index or pass "
                f"--corpus-root")
        audio = cache.get(path, sr)
        a = int(round(st["t0"] * sr))
        b = int(round(st["t1"] * sr))
        piece = audio[a:b]
        if len(piece) == 0:
            raise ValueError(f"empty piece {st['file']} "
                             f"[{st['t0']:.3f}, {st['t1']:.3f}]")
        pieces.append(piece)

    if xf > 0:
        out = _crossfade(pieces, xf)
    else:
        out = np.concatenate(pieces, axis=0)          # the law: butt-splice

    if exact:
        target = int(round(float(route.get("seconds", 0.0)) * sr))
        if target > 0 and len(out) > target:
            out = out[:target]                        # trim, never pad

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(str(out_path), out.astype(np.float32), sr)
    return {"out": str(out_path), "samples": int(len(out)),
            "duration_s": round(len(out) / sr, 3),
            "pieces": len(pieces), "crossfade_ms": crossfade_ms,
            "exact": bool(exact)}