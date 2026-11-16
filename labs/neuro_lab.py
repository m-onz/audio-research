"""neuro_lab — COMPOSED, evolving bass GESTURES (not static textures).

The earlier labs stack effects onto one sustained sound, which plateaus at
"a wobble". Interesting dubstep/neuro bass is *structured*: a rhythmic,
pitched, morphing articulation. So each one-shot here is a short arrangement:

  • RHYTHM   euclidean onset pattern + accents + ratchets + rests + ties + a
             fill at the bar end (a groove, not a drone).
  • PITCH    a riff drawn from a scale (root / octave / 5th / b7 / upper
             "talk" notes) — the bass MOVES.
  • TIMBRE   every step is re-synthesised (wavetable-scan / phase-distortion /
             FM / Chebyshev …) and passed through a resonant filter whose
             cutoff follows a smooth per-bar AUTOMATION curve -> it "talks".
  • DETAIL   per-step neuro FX (reverse / crush / waveset / phaser / vowel) on
             a fraction of hits for intricate micro-variation.
  • WEIGHT   a steady clean SUB layer under the riff.
  • BUS      a macro resonant filter sweep over the whole bar (automation + a
             tempo LFO = modulation-of-modulation) + a wet phaser.

A large candidate POOL is rendered, then CURATED by machine-listening
(farthest-point over a spectral feature space, biased toward movement +
complexity) down to the most diverse, sophisticated variations.

    python3 neuro_lab.py                        # pool -> curate ~130, 140 BPM
    python3 neuro_lab.py --bpm 150 --keep 100
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # oneshot_lab + echo live at the repo root

from oneshot_lab import core                               # noqa: E402
import bass_lab as bl                                       # noqa: E402
import bass_lab2 as bl2                                     # noqa: E402
import hifi                                                  # noqa: E402
import squelch_lab as sq                                    # noqa: E402  (chains bl3/dub; registers every module + curate)

SR = 44100
OUT = HERE / "m-onz_neuro"
BPM = 140.0

ENGINES = ["wtscan", "pdistort", "cheby", "fbfm", "sub_osc"]
STEP_FX = ["reverse", "crushmod", "waveset", "bode", "vowelseq", "ringlattice"]
SCALES = {"minpent": [0, 3, 5, 7, 10], "phrygian": [0, 1, 3, 5, 7, 8, 10],
          "minor": [0, 2, 3, 5, 7, 8, 10], "dorian": [0, 2, 3, 5, 7, 9, 10]}
_norm = bl2._norm


# ================================================================= composition

def euclid(k, n):
    pat, bucket = [], 0
    for _ in range(n):
        bucket += k
        if bucket >= n:
            bucket -= n
            pat.append(1)
        else:
            pat.append(0)
    return pat


def gen_rhythm(nsteps, rng):
    k = int(rng.choice([5, 7, 9, 11, 13]))
    pat = euclid(k, nsteps)
    rot = int(rng.integers(0, nsteps))
    pat = pat[rot:] + pat[:rot]
    ev = []
    for on in pat:
        if on and rng.random() < 0.10:
            on = 0
        rat = int(rng.choice([2, 3, 4])) if (on and rng.random() < 0.20) else 1
        acc = 1.5 if (on and rng.random() < 0.30) else 1.0
        tie = bool(on and rng.random() < 0.12)
        ev.append(dict(on=bool(on), rat=rat, acc=acc, tie=tie))
    if rng.random() < 0.6:                                 # end-of-bar fill
        ev[-1] = dict(on=True, rat=int(rng.choice([3, 4, 6, 8])), acc=1.6, tie=False)
    return ev


def gen_pitch(nsteps, root, rng):
    scale = SCALES[str(rng.choice(list(SCALES)))]
    motif_len = int(rng.choice([3, 4, 6, 8]))
    weights = np.array([0.4 if d == 0 else 0.15 for d in scale])
    weights /= weights.sum()
    motif = []
    for _ in range(motif_len):
        semi = int(rng.choice(scale, p=weights))
        octv = int(rng.choice([0, 0, 0, 12, -12, 12]))      # mostly low, some jumps
        motif.append(semi + octv)
    freqs = []
    for i in range(nsteps):
        semi = motif[i % motif_len]
        f = root * 2.0 ** (semi / 12.0)
        freqs.append(float(np.clip(f, 30.0, 360.0)))
    return freqs


def automation(k, rng):
    v = np.cumsum(rng.standard_normal(k))
    v -= v.min()
    return v / (v.max() + 1e-9)


def artic_env(note, hold_s, sr, tie):
    n = len(note)
    if n < 8:
        return note
    env = np.ones(n, np.float32)
    a = min(max(2, int(0.004 * sr)), n // 4)
    env[:a] *= np.linspace(0, 1, a)
    hold = min(n, int(hold_s * sr * (2.4 if tie else 1.0)))
    if hold < n:
        env[hold:] *= np.linspace(1, 0, n - hold) ** 1.5
    return (note * env).astype(np.float32)


def build_phrase(donor, root, sr, rng):
    beat = 60.0 / BPM
    steps = int(rng.choice([16, 16, 12]))                  # straight or triplet
    bars = int(rng.choice([1, 1, 2]))
    nsteps = steps * bars
    step_s = beat / (steps / 4.0)
    ev = gen_rhythm(nsteps, rng)
    freqs = gen_pitch(nsteps, root, rng)
    auto = automation(nsteps, rng)
    engA = str(rng.choice(ENGINES))
    engB = str(rng.choice(ENGINES))
    N = int(nsteps * step_s * sr) + int(0.4 * sr)
    buf = np.zeros(N, np.float32)
    for i, e in enumerate(ev):
        if not e["on"]:
            continue
        rat = e["rat"]
        subdur = step_s / rat
        for j in range(rat):
            eng = engB if (e["acc"] > 1.2 and rng.random() < 0.5) else engA
            f = freqs[i]
            note = bl2.SOURCES[eng](donor, f, sr, max(subdur * 1.7, 0.06), rng)
            cut = 180.0 * (28.0 ** auto[i])                # per-step brightness
            b, a = bl._rbj_lowpass(cut, float(rng.uniform(3, 9)), sr)
            note = lfilter(b, a, note).astype(np.float32)
            if rng.random() < 0.28:                        # per-step neuro detail
                fx = str(rng.choice(STEP_FX))
                try:
                    note = note[::-1].copy() if fx == "reverse" \
                        else bl2.SHAPES[fx](note, donor, f, sr, rng)
                except Exception:
                    pass
            note = artic_env(_norm(note), subdur, sr, e["tie"]) * e["acc"]
            pos = int(i * step_s * sr) + int(j * subdur * sr)
            end = min(N, pos + len(note))
            buf[pos:end] += note[: end - pos]
    # steady sub weight
    t = np.arange(N) / sr
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * (1.0 / (nsteps * step_s)) * t)
    sub = np.sin(2 * np.pi * root * t).astype(np.float32) * swell.astype(np.float32)
    return _norm(buf) * 0.9 + 0.5 * sub


# ================================================================= bus + finalize

def bus(y, root, sr, rng):
    n = len(y)
    t = np.arange(n) / sr
    k = max(2, int(n / (0.12 * sr)))
    curve = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, k), automation(k, rng))
    beat = 60.0 / BPM
    lfo = 0.5 - 0.5 * np.cos(2 * np.pi * (float(rng.choice([0.5, 1, 2])) / beat) * t)
    mod = np.clip(0.6 * curve + 0.4 * lfo, 0, 1)            # automation + LFO
    lo = root * rng.uniform(2, 5)
    hi = root * rng.uniform(20, 60)
    cut = (lo * (hi / lo) ** mod).astype(np.float32)
    y = bl.resonant_sweep(np.asarray(y, np.float32), sr, cut, float(rng.uniform(4, 11)))
    if rng.random() < 0.6:                                 # wet phaser
        try:
            y = sq.shape_phaser(y, y, root, sr, rng)
        except Exception:
            pass
    y = hifi.dc_block(_norm(y), sr)
    d = int(rng.uniform(0.003, 0.009) * sr)
    R = np.concatenate([np.zeros(d, np.float32), y])[:n]
    stereo = bl._declick(np.stack([y, R * 0.95], axis=1), sr)
    return hifi.master(stereo, sr, drive=float(rng.uniform(1.4, 2.2)))


def make(donor_path, idx, seed, uid=""):
    rng = np.random.default_rng(seed)
    x, sr = sf.read(str(donor_path), always_2d=True, dtype="float32")
    mod = x.mean(axis=1)
    if sr != SR:
        mod = core.resample(mod, sr / float(SR)); sr = SR
    f0, conf = core.detect_pitch(mod, sr)
    root = bl.pick_root(rng, f0, conf, lo=36.0, hi=130.0)
    y = bus(build_phrase(mod, root, sr, rng), root, sr, rng)
    name = f"{uid}{Path(donor_path).stem[:30]}__neuro_{BPM:g}.wav"
    OUT.mkdir(parents=True, exist_ok=True)
    hifi.write(OUT / name, y, sr)
    return dict(out=str(OUT / name), donor=str(donor_path), root=round(root, 2),
                bpm=BPM, dur_s=round(len(y) / sr, 2))


# ================================================================= main

def main(argv=None):
    global BPM
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="both", choices=["raw", "processed", "both"])
    ap.add_argument("--pick", type=int, default=40)
    ap.add_argument("--pool-per-donor", type=int, default=4)
    ap.add_argument("--keep", type=int, default=130)
    ap.add_argument("--max-scan", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--bpm", type=float, default=140.0)
    args = ap.parse_args(argv)
    BPM = args.bpm

    srcs = ["raw", "processed"] if args.sources == "both" else [args.sources]
    picks = bl.listen_and_pick(srcs, args.pick, args.max_scan, args.seed)
    pool = len(picks) * args.pool_per_donor
    print(f"\n{len(picks)} donors; composing a pool of {pool} bass gestures "
          f"@ {args.bpm:g} BPM ...")

    OUT.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    for pi, r in enumerate(picks):
        for v in range(args.pool_per_donor):
            try:
                recs.append(make(r["path"], v, args.seed * 100003 + pi * 37 + v,
                                 uid=f"{pi:03d}{v}_"))
            except Exception as e:
                print(f"  ! {Path(r['path']).name} v{v}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
        done = (pi + 1) * args.pool_per_donor
        if done % 40 == 0 or pi == len(picks) - 1:
            print(f"  composed {done}/{pool} ({time.time() - t0:.0f}s)", flush=True)

    print(f"curating {len(recs)} -> {args.keep} most diverse + sophisticated ...")
    kept = sq.curate(recs, args.keep, np.random.default_rng(args.seed))
    with open(OUT / "_manifest.json", "w") as fh:
        json.dump(kept, fh, indent=1)
    print(f"\ndone: {len(kept)} composed bass gestures (from {len(recs)}) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
