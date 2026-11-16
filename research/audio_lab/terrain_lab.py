#!/usr/bin/env python3
"""
terrain_lab.py — walk the archive as a landscape (port of alien-python 09).

Embeds the curated corpus with the 84-d features, projects to 2D with t-SNE,
and composes PATHS across the map: Lissajous figures, spirals, noise-walks.
The corpus file nearest each path point plays in sequence — the archive
becomes a landscape you can walk arrangements across. Walk speed, direction
changes and dwell times make each walk a different piece.

Usage:
    python3 terrain_lab.py -o audio_lab/terrain_out --pieces 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.manifold import TSNE

import granular_lab as gl

SR = 44100
EPS = 1e-10


def load_map(corpus_dir, rng):
    X = np.load(Path(corpus_dir) / "features.npy")
    names = json.load(open(Path(corpus_dir) / "features_files.json"))
    emb = TSNE(n_components=2, perplexity=min(12.0, len(X) - 1),
               init="pca", random_state=int(rng.integers(1 << 30)),
               n_iter_without_progress=400).fit_transform(X)
    return emb, names


def path_points(kind, n, rng):
    t = np.linspace(0, 1, n)
    if kind == "lissajous":
        a, b = int(rng.integers(1, 4)), int(rng.integers(2, 5))
        ph = rng.uniform(0, np.pi)
        x = np.sin(a * t * 2 * np.pi + ph)
        y = np.sin(b * t * 2 * np.pi)
    elif kind == "spiral":
        r = 1 - 0.85 * t
        x = r * np.cos(t * 2 * np.pi * rng.uniform(1.5, 4))
        y = r * np.sin(t * 2 * np.pi * rng.uniform(1.5, 4))
    else:  # noise walk with momentum
        steps = rng.standard_normal((n, 2))
        v = np.cumsum(steps * 0.02, axis=0)
        v -= v.min(axis=0)
        v /= (v.max(axis=0) + EPS)
        x, y = 2 * v[:, 0] - 1, 2 * v[:, 1] - 1
    return np.stack([x, y], axis=1)


def render_walk(emb, files, kind, target_s, rng, out_path,
                dwell=(0.8, 3.5)):
    n = int(target_s / 2.2) + 8
    pts = path_points(kind, n, rng)
    lo, hi = emb.min(axis=0), emb.max(axis=0)
    pts = pts * (0.5 * (hi - lo)) + 0.5 * (hi + lo)
    out = []
    t = 0.0
    last = -1
    for i in range(n):
        d = np.linalg.norm(emb - pts[i], axis=1)
        j = int(np.argmin(d))
        # avoid immediate repeats unless the path genuinely dwells
        if j == last and rng.random() < 0.6:
            order = np.argsort(d)
            j = int(order[min(2, len(order) - 1)]) if len(order) > 2 else j
        last = j
        try:
            y, sr = sf.read(str(files[j]), always_2d=True)
        except Exception:
            continue
        if sr != SR:
            n2 = int(len(y) * SR / sr)
            y = np.stack([np.interp(np.linspace(0, len(y) - 1, n2),
                                    np.arange(len(y)), y[:, c])
                          for c in range(y.shape[1])], axis=1)
        take = min(len(y), int(rng.uniform(*dwell) * SR))
        if take < len(y):
            st = int(rng.integers(0, len(y) - take))
            seg = y[st:st + take].copy()
        else:
            seg = y.copy()
        f = min(len(seg) // 4, int(0.05 * SR))
        seg[:f] *= np.linspace(0, 1, f)[:, None]
        seg[-f:] *= np.linspace(1, 0, f)[:, None]
        amp = float(rng.uniform(0.5, 0.95)) * min(6.0, 0.35 / (np.std(seg) + EPS))
        out.append((int(t * SR), (seg * amp).astype(np.float32)))
        t += len(seg) / SR * float(rng.uniform(0.75, 0.95))
        if t >= target_s:
            break
    if not out:
        return None
    L = max(o + len(y2) for o, y2 in out)
    audio = np.zeros((L + SR, 2), np.float32)
    for o, y2 in out:
        e = min(len(audio), o + len(y2))
        audio[o:e] += y2[: e - o]
    audio = audio[: L]
    pk = np.abs(audio).max()
    if pk > 1e-4:
        audio *= 0.97 / pk
    if out_path is not None:
        gl.write_wav(out_path, SR, audio)
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="audio_lab/curated_corpus")
    ap.add_argument("-o", "--out", default="audio_lab/terrain_out")
    ap.add_argument("--pieces", type=int, default=3)
    ap.add_argument("--len", type=float, default=185.0)
    args = ap.parse_args()

    rng = np.random.default_rng(31)
    emb, names = load_map(args.corpus, rng)
    files = [Path(args.corpus) / n for n in names]
    print(f"map: {len(files)} sounds embedded in 2D")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    kinds = ["lissajous", "spiral", "walk"]
    rows = []
    for i in range(args.pieces):
        kind = kinds[i % len(kinds)]
        out = outdir / f"walk{i:02d}_{kind}.wav"
        audio = render_walk(emb, files, kind, args.len, rng, out)
        if audio is not None:
            print(f"  {out.name} ({len(audio)/SR/60:.1f} min, {kind})")
            rows.append(dict(file=out.name, kind=kind))
    with open(outdir / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "kind"])
        w.writeheader()
        w.writerows(rows)
    # save the map for Pd / plotting
    np.save(outdir / "map_coords.npy", emb)
    print(f"done -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())