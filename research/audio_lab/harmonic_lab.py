#!/usr/bin/env python3
"""
harmonic_lab.py — mine the lab's most harmonically/timbrally interesting
chunks, then play them through spectral effects that read their content.

STAGE 1 — MINE. Every wav under audio_lab (trimmed, experiments, rhythms,
alien, flux, storm, edm, binary, curated, clubbed, fusion, final_review,
grammar/terrain/structure, even the sounds/ one-shot bank — ~750 files)
is scanned with sliding windows. Each window is scored:
   pitch confidence   (autocorrelation peak strength — how pitched it is)
   harmonic ratio     (energy at k*f0, k=1..10, vs total — series structure)
   tonal body         (1 - spectral flatness — how far from noise it is)
   uniqueness         (distance from the global chunk median in feature
                       space — rarity across the whole lab)
The best windows become the harmonic corpus: saved as wavs to
audio_lab/harmonic_corpus/ with full provenance (source, time, f0, scores).

STAGE 2 — EFFECTS. Effects that play on the harmonic/spectral content of
each chunk:
   reich        2-4 copies of a chunk looped at speed ratios like
                1.000 / 1.006 / 0.994 — classic gradual phasing
   pv_stretch   a real PHASE VOCODER time stretch (2-10x) — duration
                stretches, pitch holds, harmonics bloom
   resonator    the chunk's own f0 and harmonic amplitudes are measured
                and re-rung as decaying partials — the sound rings its
                own pitch series back
   shimmer      delay taps with TAPE-PITCH-SHIFTED feedback at interval
                ratios (octave/fifth/fourth) — a harmonizer cascade that
                plays the chunk's harmonic space

STAGE 3 — PIECES. Five recipes x 2: drift suites, cathedral stretches,
bell progressions (chunks walked in f0 order), shimmer towers, and full
effect chains.

Usage:
    python3 harmonic_lab.py                    # mine + 10 pieces
    python3 harmonic_lab.py --pieces 10 --seed 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import granular_lab as gl

SR = 44100
EPS = 1e-10
EXCLUDE_DIRS = {"harmonic_pieces", "harmonic_corpus", "__pycache__"}


# ------------------------------------------------------------------ analysis

def harmonic_analysis(y, sr):
    """Pitched-ness, harmonic-series structure, tonal body of a window."""
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = len(y)
    if n < 2048:
        return None
    # autocorrelation pitch (via FFT), 50-800 Hz
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(np.abs(X) ** 2, nfft)[: n // 2]
    if ac[0] <= EPS:
        return None
    ac /= ac[0]
    lagmin, lagmax = max(2, sr // 800), min(sr // 50, n // 2 - 1)
    seg = ac[lagmin:lagmax]
    k = lagmin + int(np.argmax(seg))
    conf = float(seg.max())
    if conf < 0.25 or k < 3:
        return None
    d = 0.0
    if 0 < k < len(ac) - 1:
        den = ac[k - 1] - 2 * ac[k] + ac[k + 1]
        d = 0.5 * (ac[k - 1] - ac[k + 1]) / den if abs(den) > EPS else 0.0
    f0 = sr / (k + d)
    if f0 < 45 or f0 > 900:
        return None
    # harmonic ratio: energy at k*f0 vs total
    S = np.abs(np.fft.rfft(y * np.hanning(n)))
    fr = np.fft.rfftfreq(n, 1 / sr)
    total = float(S.sum() + EPS)
    harm = 0.0
    for h in range(1, 11):
        c = h * f0
        if c > sr / 2 - 100:
            break
        band = (fr > c * 0.98) & (fr < c * 1.02)
        harm += float(S[band].sum())
    harm_ratio = harm / total
    flatness = float(np.exp(np.mean(np.log(S + EPS))) / (np.mean(S) + EPS))
    centroid = float((fr * S).sum() / (total + EPS))
    rms = float(np.sqrt(np.mean(y ** 2)))
    return dict(f0=f0, conf=conf, harm=harm_ratio, flat=flatness,
                centroid=centroid, rms=rms)


def harvest(base, out_dir, rng, cap_per_file=3, cap_total=400,
            cap_per_folder=70):
    """Scan everything; score windows; keep the most interesting chunks."""
    files = []
    for p in sorted(Path(base).rglob("*.wav")):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        files.append(p)
    print(f"mining {len(files)} files ...")
    cands = []
    for fi, f in enumerate(files):
        try:
            info = sf.info(str(f))
            dur = info.frames / info.samplerate
            cap = min(dur, 100.0)
            y, sr = sf.read(str(f), frames=int(cap * info.samplerate),
                            always_2d=True)
        except Exception:
            continue
        if y.ndim > 1 and y.shape[1] > 2:
            y = y[:, :2]
        if len(y) < 2048:
            continue
        win, hop = 2.6, 1.3
        local = []
        if dur <= 3.2:                      # one-shot: whole file
            a = harmonic_analysis(y, sr)
            if a and a["conf"] >= 0.3 and a["rms"] > 5e-3:
                local.append((0.0, min(dur, 4.0), a))
        else:
            w_n = int(win * sr)
            h_n = int(hop * sr)
            for st in range(0, max(1, len(y) - w_n), h_n):
                seg = y[st: st + w_n]
                a = harmonic_analysis(seg, sr)
                if a is None or a["conf"] < 0.3 or a["rms"] < 5e-3:
                    continue
                local.append((st / sr, win, a))
        local.sort(key=lambda c: -(0.5 * c[2]["conf"] +
                                   0.3 * c[2]["harm"] +
                                   0.2 * (1 - c[2]["flat"])))
        for st, wl, a in local[:cap_per_file]:
            cands.append(dict(file=str(f), t=st, dur_s=wl, folder=f.parent.name,
                              **a))
        if (fi + 1) % 150 == 0:
            print(f"  scanned {fi + 1}/{len(files)} -> {len(cands)} candidates")
    if not cands:
        return []
    # uniqueness: distance from the global median in feature space
    feats = np.array([[np.log2(c["f0"]), np.log10(c["centroid"] + EPS),
                       c["flat"], c["conf"], c["harm"]] for c in cands])
    med = np.median(feats, axis=0)
    iqr = (np.percentile(feats, 75, axis=0) -
           np.percentile(feats, 25, axis=0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    uniq = np.linalg.norm((feats - med) / iqr, axis=1)
    for c, u in zip(cands, uniq):
        c["unique"] = float(u)
        c["score"] = round(0.30 * c["conf"] + 0.22 * c["harm"] +
                           0.18 * (1 - c["flat"]) + 0.30 * min(u, 6.0), 4)
    cands.sort(key=lambda c: -c["score"])
    kept, per_folder = [], {}
    for c in cands:
        if len(kept) >= cap_total:
            break
        if per_folder.get(c["folder"], 0) >= cap_per_folder:
            continue
        per_folder[c["folder"]] = per_folder.get(c["folder"], 0) + 1
        kept.append(c)
    # save corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(kept):
        y, sr = sf.read(c["file"], start=int(c["t"] * sr),
                        frames=int(c["dur_s"] * sr), always_2d=True)
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)
        elif y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        if sr != SR:
            n = int(len(y) * SR / sr)
            y = np.stack([np.interp(np.linspace(0, len(y) - 1, n),
                                    np.arange(len(y)), y[:, ch])
                          for ch in range(y.shape[1])], axis=1)
        rms = float(np.sqrt(np.mean(y ** 2) + EPS))
        y = np.ascontiguousarray(y * min(10.0, 0.2 / max(rms, EPS)), np.float32)
        name = f"hc_{i:03d}_f{int(c['f0'])}_{Path(c['file']).stem[:28]}.wav"
        sf.write(str(out_dir / name), y, SR, subtype="PCM_16")
        c["chunk"] = str(out_dir / name)
        c["data"] = y
    with open(out_dir / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in kept[0] if k != "data"])
        w.writeheader()
        for c in kept:
            w.writerow({k: v for k, v in c.items() if k != "data"})
    return kept


# ------------------------------------------------------------------ effects

def pv_stretch(y, factor, n_fft=2048):
    """Phase vocoder: duration stretches, pitch holds, harmonics bloom."""
    if factor <= 1.05 or len(y) < n_fft * 2:
        return y
    if y.ndim == 2:
        return np.stack([pv_stretch(y[:, c], factor, n_fft)
                         for c in range(y.shape[1])], axis=1)
    hop = n_fft // 4
    win = np.hanning(n_fft)
    pos = np.arange(0, len(y) - n_fft, hop)
    S = np.stack([np.fft.rfft(y[p:p + n_fft] * win) for p in pos])
    if len(S) < 4:
        return y
    mag = np.abs(S)
    ph = np.angle(S)
    fr = np.fft.rfftfreq(n_fft, 1 / SR)
    exp_adv = 2 * np.pi * hop * fr / SR
    dph = np.diff(ph, axis=0)
    dph = dph - np.round((dph - exp_adv[None, :]) / (2 * np.pi)) * 2 * np.pi
    n_out = int(len(S) * factor)
    out = np.zeros((n_out - 1) * hop + n_fft)
    syn_ph = ph[0].copy()
    for i in range(n_out):
        j = min(int(i / factor), len(S) - 2)
        if i > 0:
            syn_ph = syn_ph + dph[j]
            syn_ph = (syn_ph + np.pi) % (2 * np.pi) - np.pi
        frame = mag[j] * np.exp(1j * syn_ph)
        p_out = i * hop
        out[p_out:p_out + n_fft] += np.fft.irfft(frame, n_fft) * win
    out = out / 1.5
    return out.astype(np.float32)


def reich(y, dur, rates=(1.0, 1.006, 0.994), rng=None, pan_spread=0.6):
    """Classic gradual phasing: copies of the chunk looped at slightly
    different speeds, drifting in and out of sync."""
    rng = rng or np.random.default_rng(0)
    loop = np.asarray(y, np.float32).copy()
    f = min(len(loop) // 8, int(0.05 * SR))
    loop[:f] *= np.linspace(0, 1, f)[:, None] if loop.ndim == 2 else np.linspace(0, 1, f)
    loop[-f:] *= np.linspace(1, 0, f)[:, None] if loop.ndim == 2 else np.linspace(1, 0, f)
    out = []
    for vi, r in enumerate(rates):
        need = int(dur * SR * r) + len(loop)
        tiles = np.concatenate([loop] * int(np.ceil(need / len(loop))))[:need]
        voice = gl.resample_grain(tiles, r, "hermite")[: int(dur * SR)]
        if voice.ndim == 1:
            voice = np.stack([voice, voice], axis=1)
        pan = ((vi / max(1, len(rates) - 1)) * 2 - 1) * pan_spread
        th = (pan + 1) * np.pi / 4
        voice = voice * np.array([np.cos(th), np.sin(th)], np.float32)
        out.append(voice)
    L = min(len(v) for v in out)
    return np.sum([v[:L] for v in out], axis=0) / len(out)


def _harm_amplitudes(y, f0, n_partials=8):
    if y.ndim > 1:
        y = y.mean(axis=1)
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    amps, phases = [], []
    for h in range(1, n_partials + 1):
        c = h * f0
        if c > SR / 2 - 200:
            break
        band = (fr > c * 0.97) & (fr < c * 1.03)
        if not band.any():
            amps.append(0.0)
            phases.append(0.0)
            continue
        i0 = np.flatnonzero(band)[int(np.argmax(S[band]))]
        i1 = int(np.flatnonzero(band)[0])
        Z = np.fft.rfft(y * np.hanning(len(y)))
        amps.append(float(np.abs(Z[i0])) / len(y) * 2)
        phases.append(float(np.angle(Z[i0])))
    mx = max(amps) + EPS
    return np.array(amps) / mx, np.array(phases)


def resonator(y, f0, dur, n_partials=8, decay=3.0, detune=0.0012):
    """The chunk's own harmonic series, measured and re-rung as decaying
    partials — the sound rings its own pitch back."""
    amps, phases = _harm_amplitudes(y, f0, n_partials)
    n = int(dur * SR)
    t = np.arange(n) / SR
    ring = np.zeros(n)
    for h, (a, p) in enumerate(zip(amps, phases), start=1):
        if a < 0.01:
            continue
        det = 1 + detune * (1 if h % 2 else -1)
        ring += a * np.sin(2 * np.pi * h * f0 * det * t + p) / h ** 0.3
    ring *= np.exp(-t * decay)
    env = np.minimum(1, t / 0.01)
    ring = ring * env
    stereo = np.stack([ring, np.roll(ring, int(0.002 * SR))], axis=1)
    mx = np.abs(stereo).max() + EPS
    return (stereo / mx * 0.5).astype(np.float32)


def shimmer(y, dur, taps=((0.31, 2.0, 0.5), (0.47, 1.5, 0.42), (0.19, 0.5, 0.5)),
            fb=0.8, iters=12):
    """Delay taps with tape-pitch-shifted feedback at interval ratios —
    the harmonizer cascade plays the chunk's harmonic space."""
    y = np.asarray(y, np.float32)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    n = int(dur * SR)
    buf = np.zeros((n, 2), np.float32)
    m = min(len(y), n)
    buf[:m] = y[:m] * 0.7
    acc = buf.copy()
    for _ in range(iters):
        nxt = np.zeros_like(buf)
        for d, ratio, g in taps:
            off = int(d * SR)
            shifted = gl.resample_grain(buf, ratio, "hermite")
            tail = shifted[:n - off] if len(shifted) >= n - off else \
                np.pad(shifted, ((0, n - off - len(shifted)), (0, 0)))
            nxt[off:] += tail * g
        buf = nxt * fb
        acc += buf
        if np.abs(buf).max() < 1e-4:
            break
    acc = acc[:n]
    pk = np.abs(acc).max()
    if pk > EPS:
        acc = acc / pk * 0.6
    return acc.astype(np.float32)


def _fade(y, s=0.05):
    f = min(len(y) // 4, int(s * SR))
    if f > 1:
        y[:f] *= np.linspace(0, 1, f)[:, None] if y.ndim == 2 else np.linspace(0, 1, f)
        y[-f:] *= np.linspace(1, 0, f)[:, None] if y.ndim == 2 else np.linspace(1, 0, f)
    return y


# ------------------------------------------------------------------ pieces

def add(events, y, t, gain=1.0):
    if y is None or len(y) == 0:
        return
    events.append((max(0, int(t * SR)),
                   np.ascontiguousarray(y * gain, np.float32)))


def assemble(events, target_s):
    L = int(target_s * SR) + SR
    out = np.zeros((L, 2), np.float32)
    for t, y in events:
        e = min(L, t + len(y))
        out[t:e] += y[: e - t]
    out = out[: max(int(target_s * SR), 1)]
    f = int(0.05 * SR)
    out[:f] *= np.linspace(0, 1, f)[:, None]
    out[-f:] *= np.linspace(1, 0, f)[:, None]
    pk = np.abs(out).max()
    if pk > 1e-4:
        out *= 0.97 / pk
    return out


def r_drift_suite(chunks, rng, target_s):
    events = []
    picks = [c for c in chunks if c["conf"] > 0.4]
    rng.shuffle(picks)
    picks = picks[:3] or chunks[:2]
    for vi, c in enumerate(picks):
        rates = (1.0, 1.0 + rng.uniform(0.003, 0.012) * (1 if vi % 2 else -1))
        if rng.random() < 0.4:
            rates = rates + (1.0 + rng.uniform(-0.02, 0.02),)
        y = reich(c["data"], target_s * rng.uniform(0.8, 1.0), rates, rng)
        add(events, y, rng.uniform(0, target_s * 0.1), 0.45)
    if picks:
        pad = pv_stretch(picks[0]["data"], float(rng.uniform(3, 6)))
        add(events, pad, 0.0, 0.3)
    return assemble(events, target_s), \
        f"drift_suite: {len(picks)} phased chunks at rates {[round(r,4) for r in rates]}"


def r_cathedral(chunks, rng, target_s):
    events = []
    t = 0.0
    while t < target_s - 5:
        c = chunks[int(rng.integers(len(chunks)))]
        fac = float(rng.uniform(3, 9))
        y = pv_stretch(c["data"], fac)[: int(rng.uniform(18, 50) * SR)]
        y = _fade(y, 1.5)
        add(events, y, t, 0.55)
        if rng.random() < 0.35:
            sh = shimmer(c["data"], min(20.0, target_s - t),
                         taps=((0.29, 2.0, 0.5), (0.41, 1.5, 0.4),
                               (0.23, 0.5, 0.45)))
            add(events, sh, t + 2, 0.25)
        t += len(y) / SR * rng.uniform(0.3, 0.6)
    return assemble(events, target_s), \
        f"cathedral: phase-vocoder stretches 3-9x, shimmer accents"


def r_bells(chunks, rng, target_s):
    events = []
    ranked = sorted([c for c in chunks if c["conf"] > 0.35],
                    key=lambda c: c["f0"])
    if len(ranked) < 4:
        ranked = sorted(chunks, key=lambda c: c["f0"])
    idx = np.linspace(0, len(ranked) - 1, int(rng.integers(7, 13))).astype(int)
    walk = [ranked[i] for i in idx]
    t = 0.0
    for c in walk:
        dur = float(rng.uniform(6, 16))
        ring = resonator(c["data"], c["f0"], dur,
                         n_partials=int(rng.integers(5, 10)),
                         decay=float(rng.uniform(1.5, 4.0)))
        ring[:min(len(c["data"]), len(ring))] += c["data"][:len(ring)] * 0.4
        add(events, _fade(ring, 0.1), t, 0.6)
        t += dur * rng.uniform(0.4, 0.75)
        if t > target_s - 4:
            break
    return assemble(events, target_s), \
        f"bells: {len(walk)} chunks rung in f0 order ({walk[0]['f0']:.0f}-{walk[-1]['f0']:.0f} Hz)"


def r_shimmer_tower(chunks, rng, target_s):
    events = []
    picks = rng.permutation(len(chunks))[:3]
    for c_i in picks:
        c = chunks[c_i]
        dur = float(rng.uniform(30, 70))
        taps = tuple((float(rng.uniform(0.15, 0.6)),
                      float(rng.choice(np.array([2.0, 1.5, 0.5, 3.0]))),
                      float(rng.uniform(0.35, 0.55)))
                     for _ in range(int(rng.integers(2, 4))))
        sh = shimmer(c["data"], min(dur, target_s * 0.8), taps=taps,
                     fb=float(rng.uniform(0.72, 0.86)))
        add(events, sh, rng.uniform(0, target_s * 0.2), 0.5)
    bed = chunks[int(rng.integers(len(chunks)))]
    ring = resonator(bed["data"], bed["f0"], target_s, decay=0.6)
    add(events, ring, 0.0, 0.25)
    return assemble(events, target_s), \
        f"shimmer_tower: 3 interval-ratio cascades over an f0={bed['f0']:.0f}Hz resonator bed"


def r_chain(chunks, rng, target_s):
    events = []
    t = 0.0
    while t < target_s - 6:
        c = chunks[int(rng.integers(len(chunks)))]
        y = pv_stretch(c["data"], float(rng.uniform(2, 4)))
        y = shimmer(y, min(14.0, target_s - t),
                    taps=((0.33, 2.0, 0.5), (0.21, 1.5, 0.4)),
                    iters=8)
        tail = resonator(c["data"], c["f0"], float(rng.uniform(5, 10)),
                        decay=2.5)
        y = np.concatenate([y[: len(tail)] * 0 + y[:len(tail)],
                            np.zeros((0, 2))], axis=0) if False else y
        add(events, _fade(y, 0.4), t, 0.55)
        add(events, _fade(tail, 0.1), t + min(len(y), int(10 * SR)) / SR, 0.4)
        t += (len(y) / SR) * rng.uniform(0.5, 0.8)
    return assemble(events, target_s), \
        "chain: stretch -> shimmer -> resonator tail, per chunk"


RECIPES = [
    ("drift_suite", r_drift_suite),
    ("cathedral", r_cathedral),
    ("bells", r_bells),
    ("shimmer_tower", r_shimmer_tower),
    ("chain", r_chain),
]


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="audio_lab")
    ap.add_argument("--corpus-out", default="audio_lab/harmonic_corpus")
    ap.add_argument("-o", "--out", default="audio_lab/harmonic_pieces")
    ap.add_argument("--pieces", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--remined", action="store_true",
                    help="re-mine even if a corpus exists")
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    chunks = None
    cpath = Path(args.corpus_out) / "manifest.csv"
    if cpath.exists() and not args.remined:
        import csv as _csv
        rows = list(_csv.DictReader(open(cpath)))
        for r in rows:
            try:
                y, sr = sf.read(r["chunk"], always_2d=True)
                if sr == SR:
                    r["data"] = np.ascontiguousarray(y, np.float32)
                    for kk in ("t", "f0", "conf", "harm", "flat", "centroid",
                               "rms", "unique", "score", "dur_s"):
                        r[kk] = float(r[kk])
                    chunks = rows
            except Exception:
                continue
        if chunks:
            print(f"reusing mined corpus: {len(chunks)} chunks")
    if not chunks:
        rng = np.random.default_rng(args.seed)
        chunks = harvest(args.base, Path(args.corpus_out), rng)
        chunks = [c for c in chunks if "data" in c]
    if not chunks:
        ap.error("no harmonic chunks mined")
    f0s = np.array([c["f0"] for c in chunks])
    print(f"corpus: {len(chunks)} chunks | f0 {np.median(f0s):.0f}Hz median "
          f"({f0s.min():.0f}-{f0s.max():.0f}) | conf median "
          f"{np.median([c['conf'] for c in chunks]):.2f}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    mpath = outdir / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.load(open(mpath))
        except Exception:
            manifest = {}
    for i in range(args.pieces):
        name, fn = RECIPES[i % len(RECIPES)]
        pseed = args.seed + 7919 * i + 13
        prng = np.random.default_rng(pseed)
        pick_n = min(len(chunks), int(rng_seed_count := int(prng.integers(12, 30))))
        pool = [chunks[j] for j in prng.choice(len(chunks), size=pick_n,
                                               replace=False)]
        audio, desc = fn(pool, prng, float(np.random.default_rng(pseed).uniform(200, 235)))
        wav = outdir / f"harmonic{i:02d}_{name}_{pseed}.wav"
        gl.write_wav(wav, SR, audio)
        print(f"  harmonic{i:02d} {name} ({len(audio)/SR/60:.1f} min) -> {wav.name}")
        print(f"     {desc}")
        manifest[str(wav)] = dict(name=name, seed=pseed, description=desc,
                                  chunks=[Path(c["chunk"]).name for c in pool])
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done -> {outdir} (corpus: {args.corpus_out})")


if __name__ == "__main__":
    sys.exit(main())