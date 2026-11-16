"""measure: the independent judge — nothing here uses echo.

Two instruments, both outside echo's metric:

    plain_features(audio)   8 plain-numpy descriptors (~[0,1] by design:
                            different windows, no mel, no chunks) —
                            coverage, radius, and the 24×24 discovery
                            grid the engine counts cells on
    complexity(genome)      the brant_gecco19 metrics, read straight
                            off the graph: nodes, connections, and the
                            OUTPUT ALPHABET — complexity that can only
                            rise if the encoding lets it

plus the atlas: a contact sheet of spectrograms, one tile per
specimen, so a run's territory can be *seen* before it is heard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["plain_features", "PLAIN_NAMES", "plain_coverage",
           "plain_radius", "complexity", "atlas_png"]

SR = 44100
PLAIN_NAMES = ("rms", "centroid", "flatness", "zcr", "crest",
               "flux", "onsets", "modrate")


def plain_features(audio, sr: int = SR) -> np.ndarray:
    """8-D independent descriptor. Deliberately NOT echo's pipeline:
    93 ms windows (echo: 46 ms), no mel, no temporal chunks, plain
    magnitudes — an alien instrument measuring the same sounds."""
    x = np.asarray(audio, np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.nan_to_num(x)
    n = len(x)
    if n < 2048:
        return np.zeros(8)
    win, hop = 4096, 2048
    T = max(2, 1 + (n - win) // hop)
    idx = np.arange(T) * hop
    rms = np.array([np.sqrt((x[i:i + win] ** 2).mean() + 1e-12)
                    for i in idx])
    peak = np.array([np.abs(x[i:i + win]).max() for i in idx])
    S = np.abs(np.fft.rfft(
        np.stack([x[i:i + win] * np.hanning(win) for i in idx]), axis=1)) + 1e-9
    fr = np.fft.rfftfreq(win, 1 / sr)
    cent = (S * fr).sum(axis=1) / S.sum(axis=1)
    flat = np.exp(np.log(S).mean(axis=1)) / S.mean(axis=1)
    zcr = np.array([np.abs(np.diff(np.signbit(x[i:i + win]))).mean()
                    for i in idx])
    logS = np.log(S)
    flux = np.concatenate([[0.0], np.maximum(
        logS[1:] - logS[:-1], 0.0).mean(axis=1)])
    med = np.median(rms) + 1e-9
    above = rms > 1.6 * med
    onsets = np.count_nonzero(above[1:] & ~above[:-1]) / (n / sr)
    e = rms - rms.mean()
    ES = np.abs(np.fft.rfft(e * np.hanning(len(e)))) + 1e-9
    efr = np.fft.rfftfreq(len(e), hop / sr)
    sel = (efr > 0.5) & (efr < 16)
    modrate = float((efr[sel] * ES[sel]).sum() / ES[sel].sum()) if sel.any() else 0.0
    return np.array([
        float(rms.mean()),
        float(np.clip(cent.mean() / 8000, 0, 1)),
        float(np.clip(flat.mean(), 0, 1)),
        float(np.clip(zcr.mean() * 2, 0, 1)),
        float(np.clip((peak / (rms + 1e-9)).mean() / 24, 0, 1)),
        float(np.clip(flux.mean() * 20, 0, 1)),
        float(np.clip(onsets / 10, 0, 1)),
        float(np.clip(modrate / 8, 0, 1)),
    ])


def plain_coverage(feats: np.ndarray, cells: int = 8) -> dict:
    """Spread of a population in the independent space: mean pairwise
    distance + unique cells on the three listener axes (brightness,
    noisiness, event rate)."""
    X = np.atleast_2d(feats)
    if len(X) < 2:
        return {"pairwise": 0.0, "cells": 0, "n": len(X)}
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    pairwise = float(d.sum() / (len(X) * (len(X) - 1)))
    axes = (1, 2, 6)
    keys = set()
    for x in X:
        keys.add(tuple(int(np.clip(x[a] * cells, 0, cells - 1)) for a in axes))
    return {"pairwise": pairwise, "cells": len(keys), "n": len(X)}


def plain_radius(feats: np.ndarray, founder: np.ndarray) -> float:
    """How far a population traveled from its origin, in the
    independent [0,1] feature space."""
    X = np.atleast_2d(feats)
    return float(np.sqrt(((X - np.asarray(founder)) ** 2)
                         .sum(axis=1)).mean())


# ------------------------------------------------------------------ genome


def complexity(genome) -> dict:
    """The brant_gecco19 question, read off the graph: can this
    lineage's representational capacity still grow, and has it?"""
    nodes = sum(1 for n in genome.nodes.values() if n.kind != "in")
    conns = sum(1 for c in genome.conns if c.on)
    fns: dict[str, int] = {}
    for n in genome.nodes.values():
        if n.kind != "in":
            fns[n.fn] = fns.get(n.fn, 0) + 1
    return {"nodes": nodes, "conns": conns,
            "outputs": len(genome.outputs), "functions": fns}


# ------------------------------------------------------------------ atlas


def _spectro_tile(audio, sr: int = SR, rows: int = 48, cols: int = 32
                  ) -> np.ndarray:
    """One small log-magnitude spectrogram, rows = log-spaced bins.
    RAW — normalization is the atlas's business (one world, one
    contrast: per-tile percentiles would stretch every voice to the
    same dynamic range and lie about the quiet ones)."""
    x = np.asarray(audio, np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    win, hop = 1024, 512
    T = max(2, 1 + (len(x) - win) // hop)
    S = np.abs(np.fft.rfft(
        np.stack([x[i:i + win] * np.hanning(win)
                  for i in np.arange(T) * hop]), axis=1)) + 1e-9
    f = np.fft.rfftfreq(win, 1 / sr)
    band = f > 30.0
    S = S[:, band]
    f = f[band]
    logf = np.linspace(np.log(f[0]), np.log(f[-1]), rows)
    img = np.stack([np.interp(logf, np.log(f), S[t]) for t in range(T)])
    img = img.T                                   # (rows, T)
    c = np.linspace(0, img.shape[1] - 1, cols).astype(int)
    return img[:, c]


def atlas_png(path, audios, sr: int = SR, cols: int | None = None) -> str:
    """Contact sheet of spectrograms — the run's territory, visible.
    All tiles share ONE percentile range: loud reads loud, quiet
    reads quiet, and between-voice contrast survives."""
    n = len(audios)
    if n == 0:
        return str(path)
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    tiles = [_spectro_tile(a, sr) for a in audios]
    stack = np.concatenate([t.ravel() for t in tiles])
    lo, hi = np.percentile(stack, [5, 99])
    if hi - lo < 1e-9:
        lo, hi = 0.0, 1.0
    th, tw = tiles[0].shape
    sheet = np.zeros((rows * (th + 2), cols * (tw + 2)), np.float32)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (th + 2), c * (tw + 2)
        sheet[y:y + th, x:x + tw] = np.clip((tile - lo) / (hi - lo), 0, 1)
    img = (np.flipud(sheet) * 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(path)
    return str(path)