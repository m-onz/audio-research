#!/usr/bin/env python3
"""
structure_lab.py — self-similarity structure cloning (port of alien-python 25).

Computes the SELF-SIMILARITY MATRIX of a reference track (which parts sound
like which), extracts its STRUCTURE via a Foote novelty curve (where similarity
to the recent past breaks down = section boundaries), then CLONES the
structure: a new piece with the same relative section lengths, where each
section stays internally coherent by drawing material from one corpus
'region'. The arrangement is cloned; the content is regenerated.

Usage (standalone clone):
    python3 structure_lab.py audio_lab/edm_pieces/edm04_dnb_31771.wav -o out.wav
Importable: extract_structure(path) + render_clone(...)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import granular_lab as gl
import rhythm_lab as rl

SR = 44100
HOP_MS = 46.0
EPS = 1e-10


def band_features(y, sr):
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y[: int(240 * sr)]
    n_fft, hop = 2048, int(sr * HOP_MS / 1000)
    if len(y) < n_fft * 2:
        return np.zeros((1, 16), np.float32)
    nfr = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nfr)[:, None]
    mags = np.log1p(np.abs(np.fft.rfft(y[idx] * np.hanning(n_fft), axis=1)))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.logspace(np.log10(60), np.log10(sr / 2), 17)
    band = np.clip(np.digitize(freqs, edges) - 1, 0, 15)
    F = np.stack([mags[:, band == i].mean(axis=1) for i in range(16)], axis=1)
    return F.astype(np.float32)


def extract_structure(path, n_sections=None):
    """Reference track -> section plan: [(frac, energy, chaos), ...]."""
    y, sr = sf.read(str(path), always_2d=True)
    F = band_features(y, sr)
    if len(F) < 32:
        return [dict(frac=1.0, energy=0.5, chaos=0.3)]
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + EPS)
    n = len(Fn)
    S = np.zeros((n, n), np.float32)
    B = 512
    for i in range(0, n, B):
        S[i:i + B] = Fn[i:i + B] @ Fn.T          # chunked cosine SSM
    k = min(32, max(4, n // 16))
    nov = np.zeros(n, np.float32)
    for i in range(k, n - k):
        before = S[i - k:i, i - k:i].mean()
        after = S[i:i + k, i:i + k].mean()
        cross = S[i - k:i, i:i + k].mean()
        nov[i] = cross - 0.5 * (before + after)
    nov = np.convolve(nov, np.hanning(9), "same")
    n_sec = n_sections or int(np.clip(n // 300, 4, 8))
    sep = max(8, n // n_sec)
    peaks = []
    for i in np.argsort(nov)[::-1]:
        j = int(i)
        if j < sep or j > n - sep:
            continue
        if all(abs(j - p) > sep for p in peaks):
            peaks.append(j)
        if len(peaks) >= n_sec - 1:
            break
    bounds = sorted([0] + peaks + [n])
    lo, hi = float(F.min()), float(F.max())
    secs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 4:
            continue
        seg = F[a:b]
        secs.append(dict(frac=(b - a) / n,
                        energy=float(np.clip((seg.mean() - lo) /
                                             (hi - lo + EPS), 0.08, 1.0)),
                        chaos=float(np.clip(np.std(nov[a:b]) * 8, 0.0, 1.0))))
    if not secs:
        secs = [dict(frac=1.0, energy=0.5, chaos=0.3)]
    return secs


def build_units(corpus_dir, rng, per_file=6, cap=900):
    """Onset units from the corpus with compact (centroid, rms, flat) features."""
    files = sorted(Path(corpus_dir).glob("*.wav"))
    _, chunks = rl.extract_pool(files, rng, pool_cap=cap, per_file=per_file)
    units = []
    for c in chunks:
        if len(c.data) < 4096:
            continue
        units.append(dict(data=c.data, rms=c.rms,
                          centroid=c.centroid, flat=c.flat,
                          f=np.array([c.centroid, np.log10(c.rms + EPS),
                                       c.flat])))
    return units


def render_clone(structure, units, target_s, rng, out_path=None,
                 extra_pool=None, dream_weight=0.0):
    """Clone the structure: sections keep their relative lengths; each section
    draws material from one corpus region (units near a section exemplar)."""
    F = np.stack([u["f"] for u in units])
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + EPS)
    out = []
    t = 0.0
    for sec in structure:
        dur = sec["frac"] * target_s
        ex = int(rng.integers(len(units)))
        d = np.linalg.norm(F - F[ex], axis=1)
        near = np.argsort(d)[: max(8, len(units) // 3)]
        region = [units[i] for i in near]
        if extra_pool and rng.random() < dream_weight:
            region = region + extra_pool
        pos = 0.0
        while pos < dur - 0.05:
            u = region[int(rng.integers(len(region)))] \
                if isinstance(region, list) else region
            if not isinstance(u, dict) or "data" not in u:
                u = dict(data=u, rms=max(EPS, float(np.std(u))))
            y = u["data"]
            if len(y) < 4096:
                continue
            take = min(len(y), int(rng.uniform(1.5, 5.0) * SR))
            if take < len(y) and rng.random() < 0.6:
                st = int(rng.integers(0, len(y) - take))
                y = y[st:st + take].copy()
            else:
                y = y[:take].copy()
            f = min(len(y) // 4, int(0.04 * SR))
            y[:f] *= np.linspace(0, 1, f)[:, None]
            y[-f:] *= np.linspace(1, 0, f)[:, None]
            g = (0.25 + 0.75 * sec["energy"]) * min(8.0, 0.2 / (u.get("rms", 0.1) + EPS)) \
                * 0.12
            out.append((int(t * SR), (y * g).astype(np.float32)))
            t += len(y) / SR * float(rng.uniform(0.7, 0.95))
            pos += len(y) / SR
    if not out:
        return None
    L = max(o + len(y) for o, y in out)
    audio = np.zeros((L + SR, 2), np.float32)
    for o, y in out:
        e = min(len(audio), o + len(y))
        audio[o:e] += y[: e - o]
    audio = audio[: L]
    pk = np.abs(audio).max()
    if pk > 1e-4:
        audio *= 0.97 / pk
    if out_path:
        gl.write_wav(out_path, SR, audio)
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="wav whose structure to clone")
    ap.add_argument("--corpus", default="audio_lab/curated_corpus")
    ap.add_argument("-o", "--out", default="audio_lab/structure_out/clone.wav")
    ap.add_argument("--len", type=float, default=180.0)
    args = ap.parse_args()

    rng = np.random.default_rng(11)
    secs = extract_structure(args.reference)
    print(f"structure: {len(secs)} sections: "
          + " ".join(f"{s['frac']:.2f}/{s['energy']:.2f}" for s in secs))
    units = build_units(args.corpus, rng)
    print(f"{len(units)} corpus units")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    audio = render_clone(secs, units, args.len, rng, args.out)
    print(f"clone -> {args.out} ({len(audio)/SR:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())