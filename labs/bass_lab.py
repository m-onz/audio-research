"""bass_lab — weird, novel, squelchy BASS one-shots from the m-onz one-shots.

Two stages:

  1. MACHINE LISTENING — analyse every one-shot (raw + processed) and score its
     "bassiness" from a spectral feature vector: sub-band (<60 Hz) and low-band
     (<150 Hz) energy fractions, spectral centroid, rolloff, and a detected
     low f0. The bassiest sources are chosen as timbre donors.

  2. TIMBRE STAMPING — synthesize sub-bass carriers (a harmonically-rich osc:
     saw / supersaw / FM, layered with a pure sub sine) and SPECTRAL-STAMP each
     chosen one-shot onto them via cross-synthesis (novel.formant_walk /
     env_splice / spectral_dna): the carrier keeps its fine partial structure
     but wears the train one-shot's moving spectral envelope. An acid resonant
     low-pass sweep, a touch of drive, and a pluck/declick envelope turn it
     squelchy.

    python3 bass_lab.py                       # scan raw+processed, pick 40
    python3 bass_lab.py --sources processed --pick 24 --variants 3
    python3 bass_lab.py --list-picks          # just print the bassy ranking

Output: m-onz_bass/<source>__<carrier>_<stamp>_<sweep>__vN.wav  (+ _picks.json)
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
sys.path.insert(0, str(HERE.parent))          # oneshot_lab + echo live at the repo root
sys.path.insert(0, str(HERE))

from oneshot_lab import core, novel             # noqa: E402
import hifi                                      # noqa: E402

SR = 44100
OUT = HERE / "m-onz_bass"
SRC_DIRS = {"raw": HERE / "m-onz_oneshots" / "raw",
            "processed": HERE / "m-onz_oneshots" / "processed"}

BASS_ROOTS = [36.71, 41.20, 43.65, 49.00, 55.00, 61.74, 65.41]   # D1..C2-ish
# wider chromatic pool so every rendered sound can start on a DIFFERENT note
NOTE_POOL = [41.203 * 2.0 ** (s / 12.0) for s in range(0, 25)]    # E1..E3


def pick_root(rng, f0=0.0, conf=0.0, lo=36.0, hi=130.0):
    """A varied starting note: mostly a random pitch from NOTE_POOL (so a batch
    spans many notes), occasionally locked to the donor's own detected pitch."""
    if conf >= 0.3 and 30.0 <= f0 <= 200.0 and rng.random() < 0.25:
        r = f0 / 2.0 if f0 > 100.0 else f0
    else:
        r = float(rng.choice(NOTE_POOL))
    return float(np.clip(r, lo, hi))


# ============================================================ machine listening

def bassiness(path, cap_s=6.0):
    """Spectral feature vector + a bass score for one file. None if unusable."""
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    if info.frames < int(0.2 * info.samplerate):
        return None
    try:
        x, sr = sf.read(str(path), frames=int(min(cap_s, info.duration) * info.samplerate),
                        always_2d=True, dtype="float32")
    except Exception:
        return None
    m = x.mean(axis=1)
    if sr != SR:
        m = core.resample(m, sr / float(SR)); sr = SR
    n = len(m)
    rms = float(np.sqrt(np.mean(m * m) + 1e-12))
    if rms < 0.015 or n < 1024:
        return None
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(m * w)) + 1e-9
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    tot = float(spec.sum())
    sub_frac = float(spec[fr < 60].sum() / tot)
    low_frac = float(spec[fr < 150].sum() / tot)
    mid_frac = float(spec[(fr >= 150) & (fr < 500)].sum() / tot)
    hi_frac = float(spec[fr >= 2000].sum() / tot)
    centroid = float((fr * spec).sum() / tot)
    cdf = np.cumsum(spec) / tot
    rolloff = float(fr[np.searchsorted(cdf, 0.85)])
    f0, conf = core.detect_pitch(m, sr)
    pitched_bass = conf if (30.0 <= f0 <= 180.0 and conf >= 0.3) else 0.0
    # score: reward sub + low dominance and low-leaning spectrum, punish hiss
    score = (3.0 * sub_frac + 1.5 * low_frac - 1.0 * hi_frac
             + 0.6 * np.clip(1.0 - centroid / 1500.0, 0, 1)
             + 0.5 * pitched_bass)
    return dict(path=str(path), score=float(score), sub_frac=sub_frac,
                low_frac=low_frac, mid_frac=mid_frac, hi_frac=hi_frac,
                centroid=centroid, rolloff=rolloff, f0=float(f0),
                conf=float(conf), rms=rms, dur=float(info.duration))


def listen_and_pick(sources, pick, max_scan, seed, verbose=True):
    rng = np.random.default_rng(seed)
    files = []
    for key in sources:
        d = SRC_DIRS[key]
        if d.is_dir():
            fs = sorted(d.rglob("*.wav"))
            if max_scan and len(fs) > max_scan:
                idx = np.sort(rng.permutation(len(fs))[:max_scan])
                fs = [fs[i] for i in idx]
            files += fs
    if not files:
        raise SystemExit(f"no one-shots found in {[str(SRC_DIRS[k]) for k in sources]}"
                         " — run process_m_onz.py first")
    if verbose:
        print(f"machine-listening over {len(files)} one-shots ...", flush=True)
    t0 = time.time()
    recs = []
    for i, f in enumerate(files):
        r = bassiness(f)
        if r is not None:
            recs.append(r)
        if verbose and (i + 1) % 1000 == 0:
            print(f"  analysed {i + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    # gate to genuinely bass-dominant material, then rank
    gated = [r for r in recs if r["low_frac"] >= 0.45 or r["sub_frac"] >= 0.20]
    gated.sort(key=lambda r: r["score"], reverse=True)
    picks = gated[:pick]
    if verbose:
        print(f"  {len(recs)} usable, {len(gated)} bass-dominant, "
              f"picking top {len(picks)} ({time.time() - t0:.0f}s)")
    return picks


# ============================================================ carrier synthesis

def _declick(y, sr, s=0.004):
    f = min(len(y) // 8, int(s * sr))
    if f > 2:
        r = np.linspace(0, 1, f, dtype=np.float32)
        y[:f] *= r[:, None] if y.ndim > 1 else r
        y[-f:] *= r[::-1, None] if y.ndim > 1 else r[::-1]
    return y


def make_carrier(dur, sr, base_f, kind, rng):
    """Harmonically-rich sub carrier, (n,2) stereo with a touch of detune/vibrato
    and a pure sub-sine layer underneath for weight."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    vib = 1.0 + 0.004 * np.sin(2 * np.pi * rng.uniform(3.5, 6.5) * t)   # slow life

    def osc(f_mul, detune):
        f = base_f * f_mul * detune * vib
        ph = 2 * np.pi * np.cumsum(f) / sr
        if kind == "saw":
            return 2.0 * (ph / (2 * np.pi) % 1.0) - 1.0
        if kind == "square":
            duty = rng.uniform(0.35, 0.5)
            return np.where((ph / (2 * np.pi) % 1.0) < duty, 1.0, -1.0)
        if kind == "fm":
            ratio = float(rng.choice([1.0, 2.0, 3.0, 1.5]))
            idx = rng.uniform(2.0, 7.0) * np.exp(-3.0 * t / max(dur, 1e-3))  # growl decay
            return np.sin(ph + idx * np.sin(ratio * ph))
        # supersaw: a few detuned saws
        acc = np.zeros(n)
        for dt in (-0.013, -0.006, 0.0, 0.007, 0.014):
            fo = base_f * f_mul * (1 + dt) * vib
            pho = 2 * np.pi * np.cumsum(fo) / sr
            acc += 2.0 * (pho / (2 * np.pi) % 1.0) - 1.0
        return acc / 5.0

    left = 0.9 * osc(1.0, 1.0)
    right = 0.9 * osc(1.0, 1.0 + rng.uniform(0.003, 0.009))        # stereo width
    sub = np.sin(2 * np.pi * np.cumsum(base_f * vib) / sr)          # pure sub sine
    y = np.stack([left, right], axis=1).astype(np.float32)
    y += 0.6 * sub[:, None].astype(np.float32)
    return y / (np.max(np.abs(y)) + 1e-9)


# ============================================================ squelch filter

def _rbj_lowpass(fc, Q, sr):
    fc = float(np.clip(fc, 25.0, sr * 0.45))
    w0 = 2 * np.pi * fc / sr
    cw, swv = np.cos(w0), np.sin(w0)
    alpha = swv / (2.0 * max(0.3, Q))
    b0, b1, b2 = (1 - cw) / 2, 1 - cw, (1 - cw) / 2
    a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    return (np.array([b0, b1, b2]) / a0,
            np.array([1.0, a1 / a0, a2 / a0]))


def resonant_sweep(y, sr, cut_env, Q, block=256):
    """Time-varying resonant low-pass (RBJ biquad, piecewise-constant per block
    with carried state) — the acid squelch."""
    y = np.asarray(y, np.float32)
    chans = y if y.ndim > 1 else y[:, None]
    out = np.zeros_like(chans)
    for c in range(chans.shape[1]):
        sig = chans[:, c]
        zi = np.zeros(2)
        for s in range(0, len(sig), block):
            e = min(len(sig), s + block)
            fc = float(np.mean(cut_env[s:e]))
            b, a = _rbj_lowpass(fc, Q, sr)
            seg, zi = lfilter(b, a, sig[s:e], zi=zi)
            out[s:e, c] = seg
    out = np.nan_to_num(out, copy=False)
    return out if y.ndim > 1 else out[:, 0]


def cutoff_envelope(mode, n, sr, base_f, mod_env, rng):
    """Cutoff contour (Hz) across n samples for the squelch sweep."""
    t = np.linspace(0, 1, n)
    lo = base_f * rng.uniform(1.5, 3.0)
    hi = base_f * rng.uniform(14.0, 36.0)
    if mode == "down":                                   # pluck: bright -> dark
        env = hi * np.exp(-rng.uniform(3.0, 7.0) * t) + lo
    elif mode == "up":
        env = lo + (hi - lo) * t ** rng.uniform(1.0, 2.5)
    elif mode == "wobble":                               # dubstep wobble
        rate = rng.uniform(2.5, 7.0)
        lfo = 0.5 - 0.5 * np.cos(2 * np.pi * rate * t)
        env = lo + (hi - lo) * lfo
    else:                                                # follow the donor's shape
        me = np.interp(t, np.linspace(0, 1, len(mod_env)), mod_env)
        me = me / (me.max() + 1e-9)
        env = lo + (hi - lo) * me
    return np.clip(env, 25.0, sr * 0.45).astype(np.float32)


def amp_envelope(mode, n, sr, rng):
    t = np.linspace(0, 1, n)
    atk = int(np.clip(rng.uniform(0.004, 0.02) * sr, 2, n // 4))
    if mode == "pluck":
        env = np.exp(-rng.uniform(2.5, 5.5) * t)
    else:                                                # sustain (for wobbles)
        env = np.ones(n)
    env[:atk] *= np.linspace(0, 1, atk)
    rel = min(n // 6, int(0.05 * sr))
    if rel > 2:
        env[-rel:] *= np.linspace(1, 0, rel)
    return env.astype(np.float32)


def _drive(y, amt):
    return np.tanh(y * (1.0 + 4.0 * amt)) / np.tanh(1.0 + 4.0 * amt)


# ============================================================ one bass one-shot

STAMPS = ("formant", "env", "dna", "formant_env")
CARRIERS = ("saw", "supersaw", "fm", "square")
SWEEPS = ("down", "wobble", "follow", "up")


def make_bass(donor_path, variant, seed):
    rng = np.random.default_rng(seed)
    x, sr = sf.read(str(donor_path), always_2d=True, dtype="float32")
    mod = x.mean(axis=1)
    if sr != SR:
        mod = core.resample(mod, sr / float(SR)); sr = SR
    f0, conf = core.detect_pitch(mod, sr)
    base_f = pick_root(rng, f0, conf, lo=36.0, hi=120.0)
    dur = float(np.clip(len(mod) / sr, 0.8, 2.6))

    kind = CARRIERS[variant % len(CARRIERS)]
    stamp = str(rng.choice(STAMPS))
    sweep = str(rng.choice(SWEEPS))

    carrier = make_carrier(dur, sr, base_f, kind, rng)

    # ---- spectral timbre-stamp the donor onto the carrier
    if stamp == "formant":
        y = novel.formant_walk(carrier, sr, rng, partner=mod,
                               one_way=bool(rng.random() < 0.5))
    elif stamp == "env":
        y = novel.env_splice(carrier, sr, rng, partner=mod, graft=0.0)
    elif stamp == "dna":
        y = novel.spectral_dna(carrier, sr, rng, partner=mod,
                               cross=float(rng.uniform(0.6, 0.9)))
    else:  # formant then wear the donor's amplitude motion
        y = novel.formant_walk(carrier, sr, rng, partner=mod, one_way=True)
        y = novel.env_splice(y, sr, rng, partner=mod, graft=0.0)
    y = np.nan_to_num(np.asarray(y, np.float32))
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    n = len(y)

    # ---- acid resonant squelch
    mod_env = core.rms_env(mod, sr, 0.02)
    cut = cutoff_envelope(sweep, n, sr, base_f, mod_env, rng)
    Q = float(rng.uniform(3.5, 10.0))
    y = resonant_sweep(y, sr, cut, Q)

    # ---- drive, amp env, guaranteed sub layer, declick
    y = _drive(y, float(rng.uniform(0.2, 0.6)))
    aenv = amp_envelope("sustain" if sweep == "wobble" else "pluck", n, sr, rng)
    y *= aenv[:, None]
    t = np.arange(n) / sr
    sub = np.sin(2 * np.pi * base_f * t).astype(np.float32) * aenv
    y = y + 0.45 * sub[:, None]
    y = _declick(np.nan_to_num(y), sr)
    y = hifi.master(y, sr, drive=1.4)
    stem = Path(donor_path).stem[:40]
    name = f"{stem}__{kind}_{stamp}_{sweep}__v{variant}.wav"
    outp = OUT / name
    OUT.mkdir(parents=True, exist_ok=True)
    hifi.write(outp, y, sr)
    return dict(out=str(outp), donor=str(donor_path), base_f=round(base_f, 2),
                carrier=kind, stamp=stamp, sweep=sweep, Q=round(Q, 1),
                dur_s=round(n / sr, 2))


# ============================================================ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="both",
                    choices=["raw", "processed", "both"])
    ap.add_argument("--pick", type=int, default=40, help="bassiest sources to use")
    ap.add_argument("--variants", type=int, default=4, help="bass one-shots per source")
    ap.add_argument("--max-scan", type=int, default=6000,
                    help="cap files scanned per folder (seeded sample)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--list-picks", action="store_true",
                    help="only print the bass ranking, synthesize nothing")
    args = ap.parse_args(argv)

    sources = ["raw", "processed"] if args.sources == "both" else [args.sources]
    picks = listen_and_pick(sources, args.pick, args.max_scan, args.seed)

    print("\nbassiest picks:")
    for r in picks[:min(len(picks), 25)]:
        print(f"  {r['score']:+.2f}  sub={r['sub_frac']:.2f} low={r['low_frac']:.2f} "
              f"cent={r['centroid']:6.0f}Hz f0={r['f0']:5.0f}  "
              f"{Path(r['path']).name}")
    if len(picks) > 25:
        print(f"  ... (+{len(picks) - 25} more)")

    if args.list_picks:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    t0 = time.time()
    total = len(picks) * args.variants
    for pi, r in enumerate(picks):
        for v in range(args.variants):
            try:
                rec = make_bass(r["path"], v,
                                seed=args.seed * 100003 + pi * 37 + v)
                rec["donor_score"] = r["score"]
                recs.append(rec)
            except Exception as e:
                print(f"  ! {Path(r['path']).name} v{v}: {type(e).__name__}: {e}",
                      file=sys.stderr)
        done = (pi + 1) * args.variants
        if done % 40 == 0 or pi == len(picks) - 1:
            print(f"  {done}/{total} bass one-shots ({time.time() - t0:.0f}s)",
                  flush=True)
    with open(OUT / "_picks.json", "w") as fh:
        json.dump({"picks": picks, "outputs": recs}, fh, indent=1)
    print(f"\ndone: {len(recs)} squelchy bass one-shots -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
