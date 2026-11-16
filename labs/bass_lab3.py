"""bass_lab3 — harmonically STRANGE bass: modules that rebuild/re-tune the
harmonic structure, registered into the bass_lab2 generative patcher so they
also combine with its resonators, chaos and shapers.

New modules (all operate at the chosen bass note):

  SOURCE  cheby        Chebyshev target-harmonic waveshaper: a sine driven
                       through F(x)=Σ hₖTₖ(x) with a DESIGNED strange recipe
                       (odd / prime / Fibonacci / notched) that MORPHS between
                       two recipes over time -> precise, moving harmonics.

  SHAPES  xenmap       Xenharmonic partial remap: detect the partials and
                       re-tune each onto a strange grid — Bohlen-Pierce
                       (13-ED-3, no octave), stretched/compressed octave,
                       golden-ratio, or prime-harmonic — then resynthesize.
          chordsnap    Quantize the whole spectrum onto a chord/scale grid —
                       turns train noise into a sustained chord.
          ringlattice  Cascade of ring-modulators at (in)harmonic ratios ->
                       dense clangorous bell/gong sideband lattices.

Every patch is FORCED to contain one of these four (rotating) so the output is
reliably "harmonically different", while the rest of the chain is still drawn
from the full bass_lab2 palette.

    python3 bass_lab3.py                       # 40 donors, 4 patches each
    python3 bass_lab3.py --force xenmap        # every patch uses xenmap
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # oneshot_lab + echo live at the repo root

from oneshot_lab import core, novel                      # noqa: E402
import bass_lab as bl                                     # noqa: E402
import bass_lab2 as bl2                                   # noqa: E402
import hifi                                               # noqa: E402

SR = 44100
OUT = HERE / "m-onz_bass3"
HARMONIC = ["cheby", "xenmap", "chordsnap", "ringlattice"]

_PRIMES = {2, 3, 5, 7, 11, 13}
_FIB = {1, 2, 3, 5, 8, 13}
PHI = 1.6180339887


# ================================================================= SOURCE: cheby

def _recipe(K, rng):
    """A strange target harmonic recipe hₖ (k=1..K)."""
    mode = rng.choice(["odd", "prime", "fib", "notched"])
    h = np.zeros(K + 1)
    for k in range(1, K + 1):
        if mode == "odd":
            h[k] = 1.0 if k % 2 == 1 else 0.0
        elif mode == "prime":
            h[k] = 1.0 if k in _PRIMES else 0.0
        elif mode == "fib":
            h[k] = 1.0 if k in _FIB else 0.0
        else:
            h[k] = 1.0
    if mode == "notched":
        for _ in range(int(rng.integers(2, 5))):
            h[int(rng.integers(1, K + 1))] = 0.0
    h *= (1.0 / np.arange(1, K + 2)) ** rng.uniform(0.3, 0.9)   # spectral tilt
    h[0] = 0.0
    s = h.sum()
    return h / s if s > 1e-9 else h


def src_cheby(donor, base_f, sr, dur, rng):
    n = int(dur * sr)
    t = np.arange(n) / sr
    vib = 1.0 + 0.003 * np.sin(2 * np.pi * rng.uniform(3, 6) * t)
    x = np.sin(2 * np.pi * np.cumsum(base_f * vib) / sr).astype(np.float64)
    K = 12
    T = np.empty((K + 1, n))
    T[0] = 1.0
    T[1] = x
    for k in range(1, K):
        T[k + 1] = 2.0 * x * T[k] - T[k - 1]       # Chebyshev recurrence
    h0, h1 = _recipe(K, rng), _recipe(K, rng)
    if rng.random() < 0.5:                          # ping-pong morph
        w = 0.5 - 0.5 * np.cos(2 * np.pi * rng.uniform(0.5, 2.0) * t)
    else:
        w = np.linspace(0, 1, n)                     # one-way recipe drift
    c = (1 - w)[None, :] * h0[:, None] + w[None, :] * h1[:, None]
    y = np.sum(c * T, axis=0)
    return bl2._norm(y.astype(np.float32))


# ================================================================= SHAPE: xenmap

def _remap_freqs(freqs, base_f, mode, p, sr):
    """Re-tune partial frequencies onto a strange grid."""
    f = np.asarray(freqs, np.float64)
    r = np.clip(f / max(base_f, 1e-6), 1e-3, 1e6)
    L = np.log2(r)                                   # interval in octaves
    if mode == "stretch":
        L2 = L * p                                   # p<1 compress, p>1 stretch
    elif mode == "bp":
        step = np.log2(3.0) / 13.0                   # Bohlen-Pierce step
        L2 = np.round(L / step) * step
    elif mode == "golden":
        step = np.log2(PHI)
        L2 = np.round(L / step) * step
    else:  # prime-harmonic: snap partial index to nearest prime
        plist = np.array([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], float)
        idx = np.clip(np.round(r), 1, plist.max())
        snapped = plist[np.argmin(np.abs(plist[None, :] - idx[:, None]), axis=1)]
        L2 = np.log2(snapped)
    newf = base_f * (2.0 ** L2)
    return np.clip(newf, 20.0, sr * 0.45)


def shape_xenmap(x, donor, base_f, sr, rng):
    try:
        y = np.asarray(x, np.float32)
        Z = novel._stft(y, sr)
        mag, ph = np.abs(Z), np.angle(Z)
        if mag.shape[1] < 2:
            return y
        peaks = novel._all_peaks(mag, 14, sr)
        mode = str(rng.choice(["bp", "stretch", "golden", "prime"]))
        p = float(rng.uniform(1.12, 1.4)) if rng.random() < 0.5 \
            else float(rng.uniform(0.7, 0.9))
        child = np.zeros_like(mag)
        for j in range(mag.shape[1]):
            pk = peaks[min(j, len(peaks) - 1)]
            res = mag[:, j].copy()
            if len(pk):
                novel._add_gauss(res, pk[:, 0], -pk[:, 1], sr)     # strip partials
                res = np.maximum(res, 0.0) * 0.3                   # keep some body
                nf = _remap_freqs(pk[:, 0], base_f, mode, p, sr)
                novel._add_gauss(res, nf, pk[:, 1], sr)            # re-tuned partials
            child[:, j] = res
        out = novel._istft(child * np.exp(1j * ph), sr)
        out = out[: len(y)] if len(out) >= len(y) else np.pad(out, (0, len(y) - len(out)))
        return bl2._norm(out.astype(np.float32))
    except Exception:
        return np.asarray(x, np.float32)


# ================================================================= SHAPE: chordsnap

_CHORDS = {
    "min": [0, 3, 7], "maj": [0, 4, 7], "quartal": [0, 5, 10],
    "sus": [0, 5, 7], "cluster": [0, 1, 2], "min7": [0, 3, 7, 10],
    "oct5": [0, 7, 12],
}


def shape_chordsnap(x, donor, base_f, sr, rng):
    try:
        y = np.asarray(x, np.float32)
        Z = novel._stft(y, sr)
        mag, ph = np.abs(Z), np.angle(Z)
        binf = np.fft.rfftfreq(novel._NP, 1.0 / sr)          # == mag rows
        chord = _CHORDS[str(rng.choice(list(_CHORDS)))]
        allowed = []
        for octv in range(0, 6):
            for s in chord:
                fchord = base_f * 2.0 ** (octv + s / 12.0)
                if 20.0 < fchord < sr * 0.45:
                    allowed.append(fchord)
        allowed = np.array(sorted(set(allowed)))
        if len(allowed) < 2:
            return y
        # each FFT bin -> nearest allowed chord freq -> nearest FFT row
        nearest = allowed[np.clip(np.searchsorted(allowed, binf), 0, len(allowed) - 1)]
        prev = allowed[np.clip(np.searchsorted(allowed, binf) - 1, 0, len(allowed) - 1)]
        pick = np.where(np.abs(binf - prev) < np.abs(binf - nearest), prev, nearest)
        target_row = np.clip(np.round(pick * novel._NP / sr).astype(int),
                             0, mag.shape[0] - 1)
        newmag = np.zeros_like(mag)
        np.add.at(newmag, target_row, mag)                   # scatter energy to chord
        out = novel._istft(newmag * np.exp(1j * ph), sr)
        out = out[: len(y)] if len(out) >= len(y) else np.pad(out, (0, len(y) - len(out)))
        return bl2._norm(out.astype(np.float32))
    except Exception:
        return np.asarray(x, np.float32)


# ================================================================= SHAPE: ringlattice

def shape_ringlattice(x, donor, base_f, sr, rng):
    y = np.asarray(x, np.float32)
    n = len(y)
    t = np.arange(n) / sr
    N = int(rng.integers(2, 5))
    mix = float(rng.uniform(0.6, 0.95))
    z = y.copy()
    for _ in range(N):
        if rng.random() < 0.5:
            ratio = float(rng.choice([1.5, 2.0, 2.5, 3.0, 4.0]))   # harmonic-ish
        else:
            ratio = float(rng.uniform(1.3, 4.7))                   # inharmonic
        z = bl2._norm(z * np.sin(2 * np.pi * base_f * ratio * t).astype(np.float32))
    return bl2._norm(mix * z + (1.0 - mix) * y)


# ================================================================= register

bl2.SOURCES["cheby"] = src_cheby
bl2.SHAPES["xenmap"] = shape_xenmap
bl2.SHAPES["chordsnap"] = shape_chordsnap
bl2.SHAPES["ringlattice"] = shape_ringlattice


# ================================================================= render

def make3(donor_path, variant, seed, force, uid=""):
    rng = np.random.default_rng(seed)
    x, sr = sf.read(str(donor_path), always_2d=True, dtype="float32")
    mod = x.mean(axis=1)
    if sr != SR:
        mod = core.resample(mod, sr / float(SR)); sr = SR
    f0, conf = core.detect_pitch(mod, sr)
    base_f = bl.pick_root(rng, f0, conf, lo=36.0, hi=120.0)
    dur = float(np.clip(len(mod) / sr, 0.9, 2.2))
    patch = bl2.build_patch(rng, force=force)
    y = bl2.render_patch(patch, mod, base_f, sr, dur, rng)
    tag = f"{patch['source']}-{patch['core']}"
    if patch["shapes"]:
        tag += "-" + "+".join(patch["shapes"])
    name = f"{uid}{Path(donor_path).stem[:34]}__{tag}__v{variant}.wav"
    OUT.mkdir(parents=True, exist_ok=True)
    hifi.write(OUT / name, y, sr)
    return dict(out=str(OUT / name), donor=str(donor_path), forced=force,
                base_f=round(base_f, 2), dur_s=round(len(y) / sr, 2), **patch)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="both", choices=["raw", "processed", "both"])
    ap.add_argument("--pick", type=int, default=40)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--max-scan", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--force", default=None, choices=HARMONIC,
                    help="use this harmonic module in every patch "
                         "(default: rotate through all four)")
    args = ap.parse_args(argv)

    srcs = ["raw", "processed"] if args.sources == "both" else [args.sources]
    picks = bl.listen_and_pick(srcs, args.pick, args.max_scan, args.seed)
    print(f"\n{len(picks)} donors; rendering {len(picks) * args.variants} "
          f"harmonically-strange bass patches ...")

    OUT.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    total = len(picks) * args.variants
    for pi, r in enumerate(picks):
        for v in range(args.variants):
            force = args.force or HARMONIC[(pi * args.variants + v) % len(HARMONIC)]
            try:
                recs.append(make3(r["path"], v, args.seed * 100003 + pi * 37 + v,
                                  force, uid=f"{pi:03d}_"))
            except Exception as e:
                print(f"  ! {Path(r['path']).name} v{v}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
        done = (pi + 1) * args.variants
        if done % 40 == 0 or pi == len(picks) - 1:
            print(f"  {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    from collections import Counter
    print("\nforced-module distribution:",
          dict(Counter(r["forced"] for r in recs)))
    print("full patches (source/core):",
          dict(Counter(f"{r['source']}/{r['core']}" for r in recs)))
    with open(OUT / "_manifest.json", "w") as fh:
        json.dump(recs, fh, indent=1)
    print(f"\ndone: {len(recs)} harmonically-strange bass one-shots -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
