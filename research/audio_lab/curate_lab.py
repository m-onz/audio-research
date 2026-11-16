#!/usr/bin/env python3
"""
curate_lab.py — the spread-set curator (port of alien-python's curation layer).

Featurizes every generated wav in audio_lab with the 84-d machine-listening
set (16 log-band means + stds, dominant-pitch/salience, 16-point envelope,
flux stats, 32-point coarse spectrum), robust-whitens, then runs FARTHEST-
POINT SELECTION: start from the most unusual artifact, then greedily add
whatever is FURTHEST from everything kept. Near-duplicates vanish by
construction. The result is a maximally-spread curated corpus — the source
pool that makes every downstream stage hear diverse material.

Usage:
    python3 curate_lab.py                 # -> audio_lab/curated_corpus/
    python3 curate_lab.py --target 96
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
N_FFT = 1024
HOP = 256


def views(y, sr, max_s=60.0):
    """The 84-d machine-listening vector (after alien-python program.py)."""
    if y.ndim > 1:
        y = y.mean(axis=1)
    cap = int(max_s * sr)
    y = y[:cap]
    if len(y) < N_FFT:
        return None
    nfr = 1 + (len(y) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(nfr)[:, None]
    frames = y[idx]
    win = np.hanning(N_FFT)
    mags = np.log1p(np.abs(np.fft.rfft(frames * win, axis=1)))
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    mean_mag = mags.mean(axis=0)
    peak_bin = int(np.argmax(mean_mag))
    edges = np.logspace(np.log10(60), np.log10(sr / 2), 17)
    band = np.clip(np.digitize(freqs, edges) - 1, 0, 15)
    Bm = np.stack([mags[:, band == i].mean(axis=1) for i in range(16)], axis=1)
    Bm = np.nan_to_num(Bm, nan=0.0)          # some log-bands hold no bins
    dom = float(freqs[peak_bin])
    sal = float(mean_mag[peak_bin] / (mean_mag.mean() + 1e-9))
    A = np.concatenate([Bm.mean(axis=0), Bm.std(axis=0),
                        [np.log10(dom + 1), np.log10(sal + 1)]])
    rms = np.sqrt((mags ** 2).mean(axis=1))
    env16 = np.interp(np.linspace(0, len(rms) - 1, 16), np.arange(len(rms)), rms)
    flux = np.maximum(mags[1:] - mags[:-1], 0).sum(axis=1)
    Bv = np.concatenate([env16, [flux.mean(), flux.std()]])
    C = np.interp(np.linspace(0, mags.shape[1] - 1, 32),
                  np.arange(mags.shape[1]), mean_mag)
    return np.nan_to_num(np.concatenate([A, Bv, C]),
                         nan=0.0).astype(np.float32)


def fit_whiten(X):
    med = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    return med, iqr


def farthest_point(X, target, eps):
    """Seed with the most-unusual artifact, then greedily maximise spread."""
    seed = int(np.argmax(np.linalg.norm(X - np.median(X, axis=0), axis=1)))
    kept = [seed]
    mindist = np.linalg.norm(X - X[seed], axis=1)
    mindist[seed] = -np.inf
    while len(kept) < target:
        j = int(np.argmax(mindist))
        if mindist[j] < eps:
            break
        kept.append(j)
        d = np.linalg.norm(X - X[j], axis=1)
        mindist = np.minimum(mindist, d)
        mindist[kept] = -np.inf
    return kept


def main():
    ap = argparse.ArgumentParser(
        description="Curate a maximally-spread corpus from all generated audio.")
    formatter_class = None
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="audio_lab")
    ap.add_argument("-o", "--out", default="audio_lab/curated_corpus")
    ap.add_argument("--target", type=int, default=96)
    ap.add_argument("--eps", type=float, default=1.5)
    ap.add_argument("--max-s", type=float, default=60.0)
    args = ap.parse_args()

    import montage_lab as mg
    files = []
    for f in mg.discover(args.base, "curated_corpus"):
        if Path(f).parent.name in ("final_pieces", "final_review", "sounds"):
            continue
        files.append(Path(f))
    if len(files) < 8:
        ap.error("not enough audio found")
    print(f"featurizing {len(files)} pieces (84-d, {args.max_s:.0f}s cap) ...")
    X, kept_files = [], []
    for i, f in enumerate(files):
        try:
            y, sr = sf.read(str(f), always_2d=True)
        except Exception:
            continue
        v = views(y, sr, args.max_s)
        if v is None:
            continue
        X.append(v)
        kept_files.append(f)
    X = np.stack(X)
    med, iqr = fit_whiten(X)
    Xw = (X - med) / iqr
    kept = farthest_point(Xw, min(args.target, len(X)), args.eps)
    print(f"curated: {len(kept)}/{len(X)} kept (eps {args.eps})")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, j in enumerate(kept):
        src = kept_files[j]
        dest = outdir / f"cur_{rank:03d}__{src.stem[:48]}.wav"
        y, sr = sf.read(str(src), always_2d=True)
        sf.write(str(dest), y, sr, subtype="PCM_16")
        min_d = float(np.min(np.linalg.norm(Xw[kept] - Xw[j], axis=1))) \
            if len(kept) > 1 else 0.0
        rows.append(dict(rank=rank, file=dest.name, source=str(src),
                         min_dist=round(min_d, 3)))
    with open(outdir / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.save(outdir / "features.npy", X[kept])
    with open(outdir / "features_files.json", "w") as fh:
        json.dump([r["file"] for r in rows], fh)
    print(f"done -> {outdir} ({len(rows)} spread sounds + features.npy)")


if __name__ == "__main__":
    sys.exit(main())