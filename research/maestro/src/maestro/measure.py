"""measure: the independent judge — nothing here uses echo.

Adopted from prism's measure.py (the maze discipline: measure
discovery with an alien instrument, never the selector's own
metric), plus the score-space instruments maestro needs:

    plain_features(audio)   8 plain-numpy descriptors — coverage,
                           radius, the 24x24 discovery grid
    score_features(score)   8 arrangement descriptors — density,
                           velocity, inter-onset life, voice spread;
                           the score grid counts arrangement cells
    complexity(genome)      the brant_gecco19 metrics off the graph
    anatomy(genome)         ground truth off the genome — including
                           WHICH senses are wired (purity audit)
    atlas_png(scores)       a contact sheet of piano rolls — a run's
                           territory, visible before it is heard
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["plain_features", "PLAIN_NAMES", "plain_coverage", "plain_radius",
           "score_features", "SCORE_NAMES", "score_coverage",
           "complexity", "anatomy", "atlas_png", "SR"]

SR = 44100
PLAIN_NAMES = ("rms", "centroid", "flatness", "zcr", "crest",
               "flux", "onsets", "modrate")
SCORE_NAMES = ("voices", "density", "vel_mean", "vel_std",
               "ioi_mean", "ioi_cv", "poly", "sync")


# ------------------------------------------------------------------ audio


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


# ------------------------------------------------------------------ score


def score_features(score: dict) -> np.ndarray:
    """8-D arrangement descriptor in [0, 1], read off the events and
    the level matrix — never the audio. The axes:

    voices    the size of the ensemble (the alphabet in use)
    density   events per tick
    vel_*     the velocity curve's center and spread (the paper's nuance)
    ioi_*     inter-onset life: how events pace themselves
    poly      mean simultaneous audible voices per tick
    sync      the fraction of event ticks where >=2 voices strike together
    """
    matrix = score.get("matrix", {})
    n_ticks = max(score["grid"]["n_ticks"], 1)
    events = score.get("events", [])
    voices = sorted(matrix.keys())
    n_voices = len(voices)
    vels = np.array([e["velocity"] for e in events], float)
    iois = []
    for v in voices:
        tv = np.array([e["t"] for e in events if e["voice"] == v])
        if len(tv) > 1:
            iois.extend(np.diff(np.sort(tv)).tolist())
    iois = np.array(iois, float) if iois else np.array([0.0])
    if n_voices:
        audible = np.stack([np.clip(np.asarray(matrix[v], float), 0, 1)
                            >= score.get("threshold", 0.12) for v in voices])
        poly = float(audible.mean(axis=0).mean())
        ticks_hit = audible.any(axis=0)
        together = (audible.sum(axis=0) >= 2) & ticks_hit
        sync = float(together.sum() / max(ticks_hit.sum(), 1))
    else:
        poly, sync = 0.0, 0.0
    ioi_mean = float(np.clip(iois.mean() / 2.0, 0, 1))
    ioi_cv = float(np.clip(iois.std() / (iois.mean() + 1e-9) / 4.0, 0, 1))
    return np.array([
        np.clip(n_voices / 8.0, 0, 1),
        np.clip(len(events) / n_ticks, 0, 1),
        float(vels.mean()) if len(vels) else 0.0,
        float(vels.std()) if len(vels) else 0.0,
        ioi_mean, ioi_cv, poly, sync,
    ])


def score_coverage(feats: np.ndarray, cells: int = 8) -> dict:
    """Arrangement-territory spread: pairwise distance + unique cells on
    the arrangement axes (density, pacing)."""
    X = np.atleast_2d(feats)
    if len(X) < 2:
        return {"pairwise": 0.0, "cells": 0, "n": len(X)}
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    pairwise = float(d.sum() / (len(X) * (len(X) - 1)))
    axes = (1, 4)
    keys = set()
    for x in X:
        keys.add(tuple(int(np.clip(x[a] * cells, 0, cells - 1)) for a in axes))
    return {"pairwise": pairwise, "cells": len(keys), "n": len(X)}


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


def anatomy(genome) -> dict:
    """Ground truth off the genome — the maze discipline, plus the
    purity audit: WHICH senses are wired. A score genome wired only
    to s*/live_* is pure (all structure inherited from the spine);
    x/t wiring is impure and shows here."""
    from .genome import INPUT_IDS
    fns: dict[str, int] = {}
    for n in genome.nodes.values():
        if n.kind != "in":
            fns[n.fn] = fns.get(n.fn, 0) + 1
    used = sorted({c.src for c in genome.conns if c.on})
    senses = [name for name, nid in INPUT_IDS.items() if nid in used]
    return {
        "nodes": sum(1 for n in genome.nodes.values() if n.kind != "in"),
        "conns": sum(1 for c in genome.conns if c.on),
        "outputs": list(genome.outputs),
        "functions": fns,
        "senses": senses,
        "seed": genome.seed,
    }


# ------------------------------------------------------------------ atlas


def _roll_tile(score: dict, rows: int = 48, cols: int = 64) -> np.ndarray:
    """One piano-roll tile: a voice per row-band, ticks as columns,
    velocity as brightness. The score seen from above."""
    matrix = score.get("matrix", {})
    voices = sorted(matrix.keys()) or ["-"]
    n_v = max(len(voices), 1)
    n_ticks = max(score["grid"]["n_ticks"], 1)
    img = np.zeros((rows, cols), np.float32)
    band = max(rows // n_v, 1)
    for vi, v in enumerate(voices):
        lv = np.clip(np.asarray(matrix.get(v, 0.0), float), 0, 1)
        if lv.size == 0:
            continue
        c = np.linspace(0, lv.size - 1, cols).astype(int)
        stripe = lv[c]
        y0 = rows - (vi + 1) * band            # low voice at the bottom
        for j in range(cols):
            img[y0: y0 + band, j] = max(img[y0: y0 + band, j].max(),
                                        float(stripe[j]))
    lo, hi = np.percentile(img, [50, 99.5])
    if hi - lo < 1e-9:
        img = np.zeros_like(img)
    else:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    return img


def atlas_png(path, scores, cols: int | None = None) -> str:
    """Contact sheet of piano rolls — the run's arrangement territory,
    visible before it is heard."""
    n = len(scores)
    if n == 0:
        return str(path)
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    tiles = [_roll_tile(s) for s in scores]
    th, tw = tiles[0].shape
    sheet = np.zeros((rows * (th + 2), cols * (tw + 2)), np.float32)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (th + 2), c * (tw + 2)
        sheet[y:y + th, x:x + tw] = tile
    img = (np.flipud(sheet) * 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(path)
    return str(path)