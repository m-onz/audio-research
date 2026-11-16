"""measure: the independent judge — nothing here uses echo.

Two coverages, both outside echo's metric:

    plain_features(audio)  8 plain-numpy descriptors (different window
                           sizes, no mel, no chunking, no z-norm) —
                           coverage/radius in an alien feature space
    archetype(genome)      ground truth read straight off the genome —
                           the non-circular census of 'kinds found'

The echo diagnostic (parent-kNN recall) lives in engine.py where the
bank is; it measures echo against true lineage, not echo against
itself.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plain_features", "PLAIN_NAMES", "plain_coverage",
           "plain_radius", "ascii_map"]

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


# ------------------------------------------------------------------ coverage

def plain_coverage(feats: np.ndarray, cells: int = 8) -> dict:
    """Spread of a population in the independent space: mean pairwise
    distance + unique cells on the three listener axes (brightness,
    noisiness, event rate). The features are already ~[0, 1] by design —
    no z-scoring, so tight populations can't explode."""
    X = np.atleast_2d(feats)
    if len(X) < 2:
        return {"pairwise": 0.0, "cells": 0, "n": len(X)}
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    pairwise = float(d.sum() / (len(X) * (len(X) - 1)))
    axes = (1, 2, 6)                       # centroid, flatness, onsets
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


def ascii_map(feats_list: dict[str, np.ndarray], w: int = 34, h: int = 12,
             founder=None) -> str:
    """ASCII territory map: each arm's final population scattered on
    (centroid, flatness) — the independent axes a listener would name:
    'brightness' vs 'noisiness'. One glyph per arm."""
    glyph = {"greedy": "G", "random": "R", "cell": "C", "novelty": "N"}
    allX = np.concatenate([np.atleast_2d(v) for v in feats_list.values()])
    ax_x, ax_y = 1, 2                         # centroid, flatness
    lo = np.percentile(allX[:, ax_x], 2), np.percentile(allX[:, ax_y], 2)
    hi = np.percentile(allX[:, ax_x], 98), np.percentile(allX[:, ax_y], 98)
    grid = [[" "] * w for _ in range(h)]
    for arm, X in feats_list.items():
        X = np.atleast_2d(X)
        for x in X:
            px = int(np.clip((x[ax_x] - lo[0]) / (hi[0] - lo[0] + 1e-9), 0, 1) * (w - 1))
            py = int(np.clip((x[ax_y] - lo[1]) / (hi[1] - lo[1] + 1e-9), 0, 1) * (h - 1))
            grid[h - 1 - py][px] = glyph.get(arm, "?")
    rows = ["+" + "-" * w + "+"]
    for r in grid:
        rows.append("|" + "".join(r) + "|")
    rows.append("+" + "-" * w + "+")
    rows.append("x: brightness (centroid)  y: noisiness (flatness)  "
                + "  ".join(f"{v}={k}" for k, v in glyph.items()))
    return "\n".join(rows)