"""squelch_lab — WET, squelchy, disgusting, intricate dubstep/dub bass built
from PHASING / PHASOR effects and harmonic/spectral DIGITAL transforms.

Phase-family modules (plugged into the shared patcher, so they also combine
with every earlier module — modal, Duffing, wtscan, cheby, xenmap, waveset…):

  SOURCE  pdistort    modulated PHASE-DISTORTION (Casio-CZ style): a phasor
                      warped so a resonant formant sweeps the harmonics, the
                      sweep driven by an LFO/envelope -> vocal squelch.

  SHAPES  phaser      modulated feedback ALLPASS cascade (4-10 notches swept by
                      LFO + optional envelope-follow + resonant feedback) -> the
                      classic wet squelchy phaser, pushed to self-oscillation.
          tzflange    through-zero flanger: modulated fractional delay with
                      feedback -> jet/whoosh wetness.
          dispersion  cascade of fixed allpass filters -> frequency-dependent
                      delay, rubbery metallic "wet boing".
          specphase   STFT PHASE surgery (magnitude kept): randomize / lock /
                      diffuse / rotate / shuffle the phase -> smeared, robotic,
                      gaseous digital textures.
          phasorsmear per-bin phase DRIFT in the phase vocoder -> wet detuned
                      shimmer.

Each patch is forced to use one phase module (rotating), gets a rich pitched
body + a "gross" nonlinear shape (wavefold / drive / crush), a tempo-synced
wobble, and a wet finalize (subtle chorus + steady sub + limiter). A large
candidate POOL is rendered, then CURATED by machine-listening (farthest-point
over a spectral feature space, biased toward movement + complexity) down to the
most diverse, interesting variations.

    python3 squelch_lab.py                     # pool -> curate ~150, 140 BPM
    python3 squelch_lab.py --bpm 150 --force phaser --keep 120
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # oneshot_lab + echo live at the repo root

from oneshot_lab import core, novel                       # noqa: E402
import bass_lab as bl                                      # noqa: E402
import bass_lab2 as bl2                                    # noqa: E402
import hifi                                                 # noqa: E402
import bass_lab3 as bl3                                    # noqa: E402  registers cheby/xenmap/…
import dub_lab as dl                                       # noqa: E402  registers wtscan/waveset/… + movement

SR = 44100
OUT = HERE / "m-onz_squelch"
BPM = 140.0
PHASE = ["pdistort", "phaser", "tzflange", "dispersion", "specphase", "phasorsmear"]
_norm = bl2._norm


def _lfo(shape, rate, n, sr, rng):
    t = np.arange(n) / sr
    ph = (rate * t) % 1.0
    if shape == "sine":
        return 0.5 - 0.5 * np.cos(2 * np.pi * ph)
    if shape == "tri":
        return 1 - np.abs(2 * ph - 1)
    if shape == "saw":
        return ph
    if shape == "ramp":
        return 1 - ph
    cyc = np.floor(rate * t).astype(int)                   # sample & hold
    vals = rng.random(int(cyc.max()) + 2)
    return vals[cyc]


# ================================================================= SOURCE: pdistort

def src_pdistort(donor, base_f, sr, dur, rng):
    """Phase-distortion / CZ-resonant oscillator: a sync window times a cosine
    at k·f0, with k (the resonant formant) swept by an LFO -> vocal squelch."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    vib = 1.0 + 0.003 * np.sin(2 * np.pi * rng.uniform(3, 6) * t)
    ph = (np.cumsum(base_f * vib) / sr) % 1.0
    beat = 60.0 / BPM
    rate = float(rng.choice([0.5, 1, 2, 3])) / beat
    lfo = _lfo(str(rng.choice(["sine", "tri", "saw"])), rate, n, sr, rng)
    if rng.random() < 0.5:
        e = core.rms_env(np.asarray(donor, np.float32), sr, 0.02)
        e = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(e)), e)
        lfo = 0.5 * lfo + 0.5 * (e / (e.max() + 1e-9))
    kmax = rng.uniform(6, 16)
    k = 1.0 + (kmax - 1.0) * lfo                            # resonant ratio sweep
    window = 0.5 - 0.5 * np.cos(2 * np.pi * ph)             # CZ sync window
    y = window * np.cos(2 * np.pi * k * ph)
    y += 0.5 * np.sin(2 * np.pi * np.cumsum(base_f * vib) / sr)   # sub weight
    return _norm(y.astype(np.float32))


# ================================================================= SHAPE: phaser

def shape_phaser(x, donor, base_f, sr, rng):
    y = np.asarray(x, np.float64)
    n = len(y)
    stages = int(rng.integers(4, 11))
    fb = float(rng.uniform(0.3, 0.85))
    mix = float(rng.uniform(0.4, 0.6))
    beat = 60.0 / BPM
    rate = float(rng.choice([0.5, 1, 1.5, 2, 3])) / beat
    lfo = _lfo(str(rng.choice(["sine", "tri", "saw", "sh"])), rate, n, sr, rng)
    if rng.random() < 0.5:
        e = core.rms_env(y.astype(np.float32), sr, 0.02)
        lfo = 0.6 * lfo + 0.4 * (e / (e.max() + 1e-9))
    f_lo = base_f * rng.uniform(1.5, 4.0)
    f_hi = base_f * rng.uniform(16.0, 48.0)
    fc = np.clip(f_lo * (f_hi / f_lo) ** lfo, 20.0, sr * 0.45)
    tanf = np.tan(np.pi * fc / sr)
    c = ((1.0 - tanf) / (1.0 + tanf))                      # allpass coeff / sample
    out = np.empty(n)
    xs = np.zeros(stages)
    ys = np.zeros(stages)
    zfb = 0.0
    for i in range(n):
        s = y[i] + fb * zfb
        ci = c[i]
        for k in range(stages):
            ap = -ci * s + xs[k] + ci * ys[k]
            xs[k] = s
            ys[k] = ap
            s = ap
        zfb = s
        out[i] = (1.0 - mix) * y[i] + mix * s
    return _norm(out.astype(np.float32))


# ================================================================= SHAPE: tzflange

def shape_tzflange(x, donor, base_f, sr, rng):
    y = np.asarray(x, np.float64)
    n = len(y)
    maxd = max(4, int(rng.uniform(0.002, 0.007) * sr))
    buf = np.zeros(maxd + 4)
    L = len(buf)
    fb = float(rng.uniform(0.3, 0.85))
    mix = float(rng.uniform(0.45, 0.6))
    beat = 60.0 / BPM
    rate = float(rng.choice([0.25, 0.5, 1, 2])) / beat
    lfo = _lfo(str(rng.choice(["sine", "tri"])), rate, n, sr, rng)
    delay = (0.05 + 0.9 * lfo) * (maxd - 2)
    out = np.empty(n)
    wp = 0
    for i in range(n):
        d = delay[i]
        rp = (wp - d) % L
        i0 = int(np.floor(rp))
        fr = rp - i0
        rd = buf[i0] * (1 - fr) + buf[(i0 + 1) % L] * fr
        v = y[i] + fb * rd
        buf[wp] = v
        out[i] = (1 - mix) * y[i] + mix * rd
        wp = (wp + 1) % L
    return _norm(out.astype(np.float32))


# ================================================================= SHAPE: dispersion

def shape_dispersion(x, donor, base_f, sr, rng):
    y = np.asarray(x, np.float32)
    stages = int(rng.integers(4, 12))
    for _ in range(stages):
        c = float(rng.uniform(0.5, 0.9)) * float(rng.choice([1, -1]))
        y = lfilter([c, 1.0], [1.0, c], y).astype(np.float32)   # 1st-order allpass
    return _norm(y)


# ================================================================= SHAPE: specphase

def shape_specphase(x, donor, base_f, sr, rng):
    try:
        y = np.asarray(x, np.float32)
        Z = novel._stft(y, sr)
        mag, ph = np.abs(Z), np.angle(Z)
        mode = str(rng.choice(["randomize", "lock", "diffuse", "rotate", "shuffle"]))
        nb, nf = mag.shape
        if mode == "randomize":
            ph2 = rng.uniform(-np.pi, np.pi, mag.shape)
        elif mode == "lock":
            ph2 = np.zeros_like(ph)
        elif mode == "diffuse":
            step = rng.uniform(0.2, 1.2)
            ph2 = ph + np.cumsum(rng.standard_normal(mag.shape) * step, axis=1)
        elif mode == "rotate":                              # bin-dependent ramp
            ramp = np.linspace(0, rng.uniform(4, 20) * np.pi, nf)
            ph2 = ph + (np.arange(nb)[:, None] / nb) * ramp[None, :]
        else:                                               # shuffle phase frames
            ph2 = ph[:, rng.permutation(nf)]
        out = novel._istft(mag * np.exp(1j * ph2), sr)
        out = out[: len(y)] if len(out) >= len(y) else np.pad(out, (0, len(y) - len(out)))
        return _norm(out.astype(np.float32))
    except Exception:
        return np.asarray(x, np.float32)


# ================================================================= SHAPE: phasorsmear

def shape_phasorsmear(x, donor, base_f, sr, rng):
    try:
        y = np.asarray(x, np.float32)
        Z = novel._stft(y, sr)
        mag, ph = np.abs(Z), np.angle(Z)
        nb, nf = mag.shape
        drift = rng.uniform(0.05, 0.5) * rng.standard_normal(nb)[:, None]
        ph2 = ph + np.cumsum(np.repeat(drift, nf, axis=1), axis=1)  # slow per-bin drift
        out = novel._istft(mag * np.exp(1j * ph2), sr)
        out = out[: len(y)] if len(out) >= len(y) else np.pad(out, (0, len(y) - len(out)))
        return _norm(out.astype(np.float32))
    except Exception:
        return np.asarray(x, np.float32)


bl2.SOURCES["pdistort"] = src_pdistort
bl2.SHAPES.update({"phaser": shape_phaser, "tzflange": shape_tzflange,
                   "dispersion": shape_dispersion, "specphase": shape_specphase,
                   "phasorsmear": shape_phasorsmear})


# ================================================================= wet finalize

def _chorus(y, sr, rng):
    n = len(y)
    maxd = int(0.012 * sr)
    buf = np.zeros(maxd + 4)
    L = len(buf)
    lfo = _lfo("sine", rng.uniform(0.3, 1.2), n, sr, rng)
    delay = (0.3 + 0.6 * lfo) * (maxd - 2)
    out = np.empty(n)
    wp = 0
    for i in range(n):
        rp = (wp - delay[i]) % L
        i0 = int(np.floor(rp))
        fr = rp - i0
        rd = buf[i0] * (1 - fr) + buf[(i0 + 1) % L] * fr
        buf[wp] = y[i]
        out[i] = 0.7 * y[i] + 0.5 * rd
        wp = (wp + 1) % L
    return out.astype(np.float32)


def wet_finalize(y, base_f, sr, rng):
    y = np.nan_to_num(np.asarray(y, np.float32))
    if y.ndim > 1:
        y = core.mono(y)
    if rng.random() < 0.7:
        y = _norm(_chorus(y, sr, rng))
    return dl.dub_finalize(y, base_f, sr, rng)


# ================================================================= PATCH + RENDER

def squelch_patch(rng, force):
    body = ["pdistort", "wtscan", "fbfm", "cheby", "sub_osc", "grain", "psola"]
    phase_shapes = ["phaser", "tzflange", "dispersion", "specphase", "phasorsmear"]
    gross = ["wavefold", "drive", "crushmod", "waveset", "ringlattice",
             "vowelseq", "bode", "chordsnap"]
    src = force if force in body else str(rng.choice(body))
    shapes = []
    if force in phase_shapes:
        shapes.append(force)
    # always guarantee at least one phase shape for wetness
    if not any(s in phase_shapes for s in shapes):
        shapes.append(str(rng.choice(phase_shapes)))
    if rng.random() < 0.8:                                 # a gross nonlinearity
        shapes.append(str(rng.choice(gross)))
    if rng.random() < 0.4:                                 # extra texture
        shapes.append(str(rng.choice(phase_shapes + gross)))
    coren = str(rng.choice(["none", "none", "comb", "modal"]))
    return dict(source=src, core=coren, shapes=shapes,
                move=rng.random() < 0.85)


def _clip_len(y, sr, max_s):
    """Keep outputs ~bar-length and bound the per-sample loops (waveset repeat
    can otherwise balloon the signal)."""
    n = int(max_s * sr)
    y = np.asarray(y, np.float32)
    if len(y) > n:
        y = y[:n].copy()
        f = min(len(y) // 8, int(0.02 * sr))
        if f > 2:
            y[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    return y


def render(patch, donor, base_f, sr, dur, rng):
    max_s = 2.0 * 4 * (60.0 / BPM)                          # <= 2 bars
    y = _clip_len(bl2.SOURCES[patch["source"]](donor, base_f, sr, dur, rng), sr, max_s)
    y = _clip_len(bl2.CORES[patch["core"]](y, base_f, sr, rng), sr, max_s)
    for s in patch["shapes"]:
        try:
            y = _clip_len(bl2.SHAPES[s](y, donor, base_f, sr, rng), sr, max_s)
        except Exception:
            pass
    if patch["move"]:
        y = dl.movement_wobble(y, base_f, sr, rng)
    return wet_finalize(y, base_f, sr, rng)


def make(donor_path, idx, seed, force, uid=""):
    rng = np.random.default_rng(seed)
    x, sr = sf.read(str(donor_path), always_2d=True, dtype="float32")
    mod = x.mean(axis=1)
    if sr != SR:
        mod = core.resample(mod, sr / float(SR)); sr = SR
    f0, conf = core.detect_pitch(mod, sr)
    base_f = bl.pick_root(rng, f0, conf, lo=36.0, hi=150.0)
    beat = 60.0 / BPM
    dur = float(rng.choice([1.0, 1.0, 1.5])) * 4 * beat    # ~1-1.5 bars (bound loops)
    patch = squelch_patch(rng, force)
    y = render(patch, mod, base_f, sr, dur, rng)
    tag = f"{patch['source']}-{patch['core']}-" + "+".join(patch["shapes"])
    name = f"{uid}{Path(donor_path).stem[:30]}__{tag}.wav"
    OUT.mkdir(parents=True, exist_ok=True)
    hifi.write(OUT / name, y, sr)
    return dict(out=str(OUT / name), donor=str(donor_path), forced=force,
                base_f=round(base_f, 2), bpm=BPM, dur_s=round(len(y) / sr, 2), **patch)


# ================================================================= curation

def _feats(path):
    try:
        x, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception:
        return None
    m = x.mean(axis=1)
    if not np.all(np.isfinite(m)) or np.max(np.abs(m)) < 1e-4:
        return None
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m)))) + 1e-9
    fr = np.fft.rfftfreq(len(m), 1.0 / sr)
    tot = spec.sum()
    sub = spec[fr < 80].sum() / tot
    low = spec[fr < 300].sum() / tot
    cent = (fr * spec).sum() / tot
    flat = np.exp(np.mean(np.log(spec))) / (spec.mean() + 1e-9)
    peaky = np.sort(spec)[::-1][:20].sum() / tot
    rms = np.sqrt(np.mean(m * m) + 1e-12)
    crest = np.max(np.abs(m)) / (rms + 1e-9)
    # spectral flux = movement
    nn, hop = 1024, 512
    F = [np.abs(np.fft.rfft(m[i:i + nn] * np.hanning(nn)))
         for i in range(0, len(m) - nn, hop)]
    flux = float(np.mean(np.abs(np.diff(np.array(F), axis=0))) /
                 (np.array(F).mean() + 1e-9)) if len(F) > 1 else 0.0
    vec = np.array([sub, low, np.log10(cent + 1), flat, peaky,
                    np.clip(crest / 10, 0, 1), flux], np.float32)
    interesting = flux + 0.5 * peaky + 0.3 * flat          # movement+complexity
    return vec, float(interesting)


def curate(recs, keep, rng):
    vecs, scores, paths = [], [], []
    for r in recs:
        f = _feats(r["out"])
        if f is None:
            os.remove(r["out"])
            continue
        vecs.append(f[0]); scores.append(f[1]); paths.append(r["out"])
    if not vecs:
        return []
    X = np.array(vecs)
    med = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, 0) - np.percentile(X, 25, 0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    Xw = (X - med) / iqr
    scores = np.array(scores)
    m = len(Xw)
    if m <= keep:
        chosen = list(range(m))
    else:
        seed = int(np.argmax(scores))                      # start from most interesting
        chosen = [seed]
        mind = np.linalg.norm(Xw - Xw[seed], axis=1)
        while len(chosen) < keep:
            adj = mind * (0.6 + 0.4 * (scores / (scores.max() + 1e-9)))
            adj[chosen] = -np.inf
            j = int(np.argmax(adj))
            chosen.append(j)
            mind = np.minimum(mind, np.linalg.norm(Xw - Xw[j], axis=1))
    keepset = set(paths[i] for i in chosen)
    kept = []
    for r in recs:
        if r["out"] in keepset:
            kept.append(r)
        elif os.path.exists(r["out"]):
            os.remove(r["out"])                            # drop the less-diverse ones
    return kept


# ================================================================= main

def main(argv=None):
    global BPM
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="both", choices=["raw", "processed", "both"])
    ap.add_argument("--pick", type=int, default=44, help="donors to draw from")
    ap.add_argument("--pool-per-donor", type=int, default=5, help="candidate patches/donor")
    ap.add_argument("--keep", type=int, default=150, help="curated survivors")
    ap.add_argument("--max-scan", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--bpm", type=float, default=140.0)
    ap.add_argument("--force", default=None, choices=PHASE)
    args = ap.parse_args(argv)
    BPM = args.bpm

    srcs = ["raw", "processed"] if args.sources == "both" else [args.sources]
    picks = bl.listen_and_pick(srcs, args.pick, args.max_scan, args.seed)
    pool = len(picks) * args.pool_per_donor
    print(f"\n{len(picks)} donors; rendering a pool of {pool} wet/squelchy "
          f"candidates @ {args.bpm:g} BPM ...")

    OUT.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    for pi, r in enumerate(picks):
        for v in range(args.pool_per_donor):
            force = args.force or PHASE[(pi * args.pool_per_donor + v) % len(PHASE)]
            try:
                recs.append(make(r["path"], v, args.seed * 100003 + pi * 37 + v,
                                 force, uid=f"{pi:03d}{v}_"))
            except Exception as e:
                print(f"  ! {Path(r['path']).name} v{v}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
        done = (pi + 1) * args.pool_per_donor
        if done % 50 == 0 or pi == len(picks) - 1:
            print(f"  rendered {done}/{pool} ({time.time() - t0:.0f}s)", flush=True)

    print(f"curating {len(recs)} -> {args.keep} most diverse + interesting ...")
    kept = curate(recs, args.keep, np.random.default_rng(args.seed))
    from collections import Counter
    print("forced phase-module distribution:",
          dict(Counter(r["forced"] for r in kept)))
    with open(OUT / "_manifest.json", "w") as fh:
        json.dump(kept, fh, indent=1)
    print(f"\ndone: {len(kept)} wet squelchy bass one-shots (from {len(recs)} "
          f"candidates) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
