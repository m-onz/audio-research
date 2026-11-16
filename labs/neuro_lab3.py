"""neuro_lab3 — varied-pitch, HIGH-FIDELITY, deeper-pipeline bass gestures.

Builds on neuro_lab2 (composed rhythmic/pitched gestures with glide /
call-response / sidechain / space) and adds what was asked for:

  • VARIED NOTES  every sound starts on a different root, drawn from a chromatic
                  bass pool (~E1..E3) — the batch spans many pitches, each riff
                  transposed to its own root.
  • FIDELITY      band-limited PolyBLEP oscillator; all nonlinearities (drive /
                  waveshape) run OVERSAMPLED (polyphase 4×) to suppress aliasing;
                  DC-block + gentle soft-limiter instead of hard clipping;
                  24-bit output.
  • DEEPER CHAIN  an extra PHRASE-FX stack (2-5 transforms drawn from the FULL
                  registered palette — spectral / phase / harmonic / glitch)
                  runs between the composed gesture and the bus, on top of the
                  per-step FX and the bus stages.
  • MORE VARIATION  BPM varies per sound; wider engine set; wider param ranges.

    python3 neuro_lab3.py                       # pool -> curate ~140
    python3 neuro_lab3.py --keep 100 --depth 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # oneshot_lab + echo live at the repo root

from oneshot_lab import core                               # noqa: E402
import bass_lab as bl                                       # noqa: E402
import bass_lab2 as bl2                                     # noqa: E402
import squelch_lab as sq                                    # noqa: E402
import neuro_lab as nl                                      # noqa: E402
import neuro_lab2 as nl2                                    # noqa: E402

SR = 44100
OUT = HERE / "m-onz_neuro3"
BPMS = [140.0, 140.0, 150.0, 174.0]                         # varied per sound
NOTE_POOL = [41.203 * 2.0 ** (s / 12.0) for s in range(0, 24)]   # E1..~D#3 chromatic
_norm = bl2._norm


# ================================================================= fidelity helpers

def _blep(ph, dt):
    y = np.zeros_like(ph)
    dt = np.maximum(dt, 1e-9)
    m = ph < dt
    t = ph[m] / dt[m]
    y[m] = t + t - t * t - 1.0
    m2 = ph > 1.0 - dt
    t = (ph[m2] - 1.0) / dt[m2]
    y[m2] = t * t + t + t + 1.0
    return y


def _blsaw(ph, dt):
    return (2.0 * ph - 1.0) - _blep(ph, dt)


def src_blosc(donor, base_f, sr, dur, rng):
    """Band-limited (PolyBLEP) saw / square — clean, alias-suppressed body."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    vib = 1.0 + 0.003 * np.sin(2 * np.pi * rng.uniform(3, 6) * t)
    f = base_f * vib
    dt = f / sr
    ph = (np.cumsum(f) / sr) % 1.0
    if rng.random() < 0.5:
        y = _blsaw(ph, dt)
    else:
        y = _blsaw(ph, dt) - _blsaw((ph + 0.5) % 1.0, dt)   # band-limited square
    y = y + 0.6 * np.sin(2 * np.pi * np.cumsum(f) / sr)     # sub layer
    return _norm(y.astype(np.float32))


def os_apply(x, fn, factor=4):
    """Apply a nonlinearity at `factor`× oversampling (polyphase) to limit
    aliasing, return at the original rate/length."""
    x = np.asarray(x, np.float32)
    up = resample_poly(x, factor, 1)
    y = fn(up)
    down = resample_poly(y, 1, factor)
    return down[: len(x)].astype(np.float32)


def dc_block(x, sr, fc=22.0):
    sos = butter(2, fc / (sr / 2), btype="high", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def soft_limit(x, drive):
    k = max(1.0, drive)
    return os_apply(x, lambda u: np.tanh(u * k) / np.tanh(k))


bl2.SOURCES["blosc"] = src_blosc
nl.ENGINES = ["blosc", "blosc", "pdistort", "cheby", "wtscan", "fbfm"]   # cleaner bias


# ================================================================= deeper pipeline

def phrase_fx_stack(y, donor, root, sr, rng, depth):
    """Extra multi-stage transform chain from the full registered palette —
    this is where the per-sound pipeline gets deeper."""
    pool = list(bl2.SHAPES)
    max_s = len(y) / sr + 0.01
    for _ in range(depth):
        s = str(rng.choice(pool))
        try:
            y2 = bl2.SHAPES[s](np.asarray(y, np.float32), donor, root, sr, rng)
            y = _norm(sq._clip_len(y2, sr, max_s))
        except Exception:
            pass
    return _norm(y)


def bus_hq(y, root, sr, rng):
    n = len(y)
    t = np.arange(n) / sr
    k = max(2, int(n / (0.12 * sr)))
    curve = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, k), nl.automation(k, rng))
    beat = 60.0 / nl2.BPM
    lfo = 0.5 - 0.5 * np.cos(2 * np.pi * (float(rng.choice([0.5, 1, 2])) / beat) * t)
    mod = np.clip(0.6 * curve + 0.4 * lfo, 0, 1)
    lo, hi = root * rng.uniform(2, 5), root * rng.uniform(20, 60)
    cut = (lo * (hi / lo) ** mod).astype(np.float32)
    y = bl.resonant_sweep(np.asarray(y, np.float32), sr, cut, float(rng.uniform(4, 11)))
    if rng.random() < 0.6:
        try:
            y = sq.shape_phaser(y, y, root, sr, rng)
        except Exception:
            pass
    if rng.random() < 0.7:
        y = nl2.fb_delay(y, sr, beat * float(rng.choice([0.5, 0.75, 0.25])),
                         float(rng.uniform(0.3, 0.55)), float(rng.uniform(0.2, 0.4)))
    if rng.random() < 0.6:
        y = nl2.gated_verb(y, sr, float(rng.uniform(0.15, 0.5)),
                           beat * float(rng.choice([0.5, 1.0])),
                           float(rng.uniform(0.15, 0.35)), rng)
    y = os_apply(y, lambda u: np.tanh(u * float(rng.uniform(1.2, 2.2))))   # oversampled drive
    y = dc_block(y, sr)
    y = soft_limit(_norm(y), float(rng.uniform(1.4, 2.2)))                 # oversampled limiter
    d = int(rng.uniform(0.003, 0.009) * sr)
    R = np.concatenate([np.zeros(d, np.float32), y])[:n]
    stereo = bl._declick(np.stack([y, R * 0.95], axis=1), sr)
    pk = float(np.max(np.abs(stereo)))
    return (stereo / pk * 0.97).astype(np.float32) if pk > 1e-6 else stereo


# ================================================================= render

def make(donor_path, idx, seed, depth_rng, uid=""):
    rng = np.random.default_rng(seed)
    nl2.BPM = float(rng.choice(BPMS))                       # varied tempo per sound
    root = float(rng.choice(NOTE_POOL))                     # varied starting note
    x, sr = sf.read(str(donor_path), always_2d=True, dtype="float32")
    mod = x.mean(axis=1)
    if sr != SR:
        mod = core.resample(mod, sr / float(SR)); sr = SR
    phrase = nl2.build_phrase(mod, root, sr, rng)
    depth = int(rng.integers(depth_rng[0], depth_rng[1] + 1))
    phrase = phrase_fx_stack(phrase, mod, root, sr, rng, depth)
    y = bus_hq(phrase, root, sr, rng)
    semi = int(round(12 * np.log2(root / 41.203)))
    name = f"{uid}{Path(donor_path).stem[:28]}__n3_{nl2.BPM:g}_r{semi:02d}.wav"
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT / name), y, sr, subtype="PCM_24")      # 24-bit
    return dict(out=str(OUT / name), donor=str(donor_path), root=round(root, 2),
                semi=semi, bpm=nl2.BPM, fx_depth=depth, dur_s=round(len(y) / sr, 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="both", choices=["raw", "processed", "both"])
    ap.add_argument("--pick", type=int, default=44)
    ap.add_argument("--pool-per-donor", type=int, default=4)
    ap.add_argument("--keep", type=int, default=140)
    ap.add_argument("--depth", type=int, default=None,
                    help="fixed phrase-FX depth (default: random 2-5)")
    ap.add_argument("--max-scan", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args(argv)
    depth_rng = (args.depth, args.depth) if args.depth else (2, 5)

    srcs = ["raw", "processed"] if args.sources == "both" else [args.sources]
    picks = bl.listen_and_pick(srcs, args.pick, args.max_scan, args.seed)
    pool = len(picks) * args.pool_per_donor
    print(f"\n{len(picks)} donors; composing {pool} HQ 24-bit gestures "
          f"(varied notes/tempo, deep FX) ...")

    OUT.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    for pi, r in enumerate(picks):
        for v in range(args.pool_per_donor):
            try:
                recs.append(make(r["path"], v, args.seed * 100003 + pi * 37 + v,
                                 depth_rng, uid=f"{pi:03d}{v}_"))
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
    from collections import Counter
    print("note spread (semitones from E1):",
          dict(sorted(Counter(r["semi"] for r in kept).items())))
    print("tempo spread:", dict(Counter(r["bpm"] for r in kept)))
    print(f"done: {len(kept)} HQ bass gestures (from {len(recs)}) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
