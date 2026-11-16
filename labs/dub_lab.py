"""dub_lab — glitchy, novel-digital DUBSTEP bass one-shots.

New synthesis / transformation modules (plugged into the bass_lab2 patcher, so
they also combine with every earlier module — modal, Duffing, cheby, xenmap …):

  SOURCES  wtscan    the train turned into a WAVETABLE: single-cycle frames
                     lifted from the recording, scanned by an LFO at the bass
                     note (the core neuro/dubstep "moving-midrange" engine).
           fbfm      feedback-FM: a self-modulating operator with an index LFO
                     -> aggressive evolving growl.

  SHAPES   waveset   Trevor Wishart WAVESET DISTORTION — operate on the signal
                     between zero-crossings (repeat / reverse / sine-replace /
                     average / omit) -> unpredictable digital glitch & sub-octaves.
           vowelseq  formant/vowel SEQUENCER (resonant A-E-I-O-U banks stepped
                     rhythmically) -> a "talking" growl.
           crushmod  bitcrush + sample-rate reduction with LFO-modulated depth
                     -> evolving aliasing / digital grit.
           bufferglitch  tempo-grid BEAT-REPEAT: cells repeated / reversed /
                     gated / re-pitched -> rhythmic glitch edits.

Every patch is FORCED to use one of these six (rotating) for a novel body, then
a mandatory tempo-synced WOBBLE/GROWL movement stage (LFO-modulated resonant
filter + optional amp gate) and a dub FINALIZE (steady sub + limiter) make it a
ready-to-drop dubstep one-shot (~1-2 bars at the chosen BPM).

    python3 dub_lab.py                      # 40 donors, 4 patches each, 140 BPM
    python3 dub_lab.py --bpm 150 --force waveset
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

from oneshot_lab import core                              # noqa: E402
import bass_lab as bl                                      # noqa: E402
import bass_lab2 as bl2                                    # noqa: E402
import bass_lab3 as bl3                                    # noqa: E402  (registers cheby/xenmap/… into bl2)
import hifi                                                 # noqa: E402

SR = 44100
OUT = HERE / "m-onz_dub"
BPM = 140.0                                                # set by --bpm at runtime
NOVEL = ["wtscan", "fbfm", "waveset", "vowelseq", "crushmod", "bufferglitch"]
_norm = bl2._norm


# ================================================================= SOURCES

def src_wtscan(donor, base_f, sr, dur, rng):
    m = np.asarray(donor, np.float32)
    n = int(dur * sr)
    L, M = 1024, 8
    table = np.zeros((M, L), np.float32)
    for k in range(M):
        pos = int((k / max(1, M - 1)) * max(0, len(m) - L - 1))
        fr = m[pos: pos + L]
        if len(fr) < L:
            fr = np.pad(fr, (0, L - len(fr)))
        fr = fr - fr.mean()
        mx = float(np.max(np.abs(fr)))
        table[k] = fr / mx if mx > 1e-6 else fr
    t = np.arange(n)
    ph = (np.cumsum(np.full(n, base_f)) / sr) % 1.0
    idx = (ph * L).astype(int) % L
    scan = rng.uniform(0.5, 4.0)
    posf = (0.5 + 0.5 * np.sin(2 * np.pi * scan * t / sr)) * (M - 1)
    i0 = np.clip(np.floor(posf).astype(int), 0, M - 1)
    i1 = np.clip(i0 + 1, 0, M - 1)
    frac = (posf - np.floor(posf)).astype(np.float32)
    y = (1 - frac) * table[i0, idx] + frac * table[i1, idx]
    return _norm(y.astype(np.float32))


def src_fbfm(donor, base_f, sr, dur, rng):
    n = int(dur * sr)
    t = np.arange(n) / sr
    ratio = float(rng.choice([1.0, 2.0, 3.0, 0.5, 1.5]))
    fb = float(rng.uniform(0.3, 0.9))
    ilfo = rng.uniform(1, 6) * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 5) * t))
    ph = 2 * np.pi * np.cumsum(np.full(n, base_f)) / sr
    mph = 2 * np.pi * np.cumsum(np.full(n, base_f * ratio)) / sr
    y = np.zeros(n)
    yp = 0.0
    for i in range(n):
        mod = ilfo[i] * np.sin(mph[i] + fb * yp)
        yp = np.sin(ph[i] + mod)
        y[i] = yp
    return _norm(y.astype(np.float32))


# ================================================================= SHAPES

def shape_waveset(x, donor, base_f, sr, rng):
    y = np.asarray(x, np.float32)
    s = np.sign(y)
    s[s == 0] = 1
    zc = np.where(np.diff(s) != 0)[0] + 1
    if len(zc) < 6:
        return y
    segs = [y[zc[i]: zc[i + 1]] for i in range(len(zc) - 1)]
    op = str(rng.choice(["repeat", "reverse", "sine", "average", "omit"]))
    out = []
    if op == "repeat":
        k = int(rng.integers(2, 4))
        for seg in segs:
            out += [seg] * k
    elif op == "reverse":
        out = [seg[::-1] for seg in segs]
    elif op == "sine":
        for seg in segs:
            Lk = len(seg)
            if Lk < 1:
                continue
            sign = 1.0 if seg.sum() >= 0 else -1.0
            pk = float(np.max(np.abs(seg)))
            out.append((sign * pk * np.sin(np.linspace(0, np.pi, Lk))).astype(np.float32))
    elif op == "omit":
        nn = int(rng.integers(2, 4))
        out = [seg for i, seg in enumerate(segs) if i % nn != 0]
    else:  # average groups of N wavesets
        N = int(rng.integers(2, 5))
        for i in range(0, len(segs), N):
            grp = segs[i: i + N]
            Lk = max(1, int(np.mean([len(g) for g in grp])))
            acc = np.zeros(Lk)
            for g in grp:
                acc += np.interp(np.linspace(0, 1, Lk),
                                 np.linspace(0, 1, len(g)), g) if len(g) > 1 else 0
            out += [(acc / len(grp)).astype(np.float32)] * len(grp)
    z = np.concatenate(out) if out else y
    return _norm(z)


_VOWELS = {"a": [800, 1150, 2900], "e": [400, 1600, 2700], "i": [350, 2000, 2800],
           "o": [450, 800, 2830], "u": [325, 700, 2700]}


def shape_vowelseq(x, donor, base_f, sr, rng):
    y = np.asarray(x, np.float32)
    n = len(y)
    seq = [str(rng.choice(list(_VOWELS))) for _ in range(int(rng.integers(3, 7)))]
    seglen = max(1, n // len(seq))
    out = np.zeros(n, np.float32)
    for si, v in enumerate(seq):
        a = si * seglen
        b = min(n, a + seglen)
        chunk = y[a:b]
        if len(chunk) < 8:
            continue
        acc = np.zeros_like(chunk)
        for f in _VOWELS[v]:
            bb, aa = bl2._rbj_bandpass(f, rng.uniform(5, 12), sr)
            acc += lfilter(bb, aa, chunk).astype(np.float32)
        out[a:b] = acc
    return _norm(out)


def shape_crushmod(x, donor, base_f, sr, rng):
    y = np.asarray(x, np.float32)
    n = len(y)
    t = np.linspace(0, 1, n)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 8) * t)
    bits = 3.0 + 9.0 * (1.0 - lfo)                         # 3..12 bits
    levels = np.maximum(2.0, 2.0 ** bits)
    yq = np.round(y * levels) / levels
    D = (1 + lfo * rng.uniform(4, 20)).astype(int)         # S&H downsample
    out = yq.copy()
    i = 0
    while i < n:
        d = max(1, int(D[i]))
        out[i: i + d] = yq[i]
        i += d
    return _norm(out)


def shape_bufferglitch(x, donor, base_f, sr, rng):
    y = np.asarray(x, np.float32)
    n = len(y)
    beat = 60.0 / BPM
    cell = max(256, int(beat / float(rng.choice([2, 4, 4, 8])) * sr))
    nc = n // cell
    if nc < 2:
        return y
    out = np.zeros(n, np.float32)
    prev = None
    for c in range(nc):
        a = c * cell
        src = y[a: a + cell].copy()
        r = rng.random()
        if r < 0.20 and prev is not None:
            cv = prev
        elif r < 0.40:
            cv = src[::-1]
        elif r < 0.50:
            cv = np.zeros_like(src)
        elif r < 0.65:
            rs = core.resample(src, float(rng.choice([0.5, 2.0])))
            cv = np.pad(rs, (0, cell))[:cell]
        else:
            cv = src
        f = min(64, cell // 8)
        if f > 2:
            cv = cv.copy()
            cv[:f] *= np.linspace(0, 1, f)
            cv[-f:] *= np.linspace(1, 0, f)
        out[a: a + cell] = cv[:cell]
        prev = cv
    return _norm(out)


bl2.SOURCES.update({"wtscan": src_wtscan, "fbfm": src_fbfm})
bl2.SHAPES.update({"waveset": shape_waveset, "vowelseq": shape_vowelseq,
                   "crushmod": shape_crushmod, "bufferglitch": shape_bufferglitch})


# ================================================================= MOVEMENT + FINALIZE

def movement_wobble(y, base_f, sr, rng):
    y = np.asarray(y, np.float32)
    n = len(y)
    t = np.arange(n) / sr
    beat = 60.0 / BPM
    rate = float(rng.choice([0.5, 1, 1.5, 2, 3, 4])) / beat   # cycles / sec
    shape = str(rng.choice(["sine", "tri", "saw", "ramp", "sqr", "sh"]))
    phase = (rate * t) % 1.0
    if shape == "sine":
        lfo = 0.5 - 0.5 * np.cos(2 * np.pi * phase)
    elif shape == "tri":
        lfo = 1 - np.abs(2 * phase - 1)
    elif shape == "saw":
        lfo = phase
    elif shape == "ramp":
        lfo = 1 - phase
    elif shape == "sqr":
        lfo = (phase < rng.uniform(0.3, 0.6)).astype(float)
    else:
        cyc = np.floor(rate * t).astype(int)
        vals = rng.random(int(cyc.max()) + 2)
        lfo = vals[cyc]
    lo = base_f * rng.uniform(1.5, 3.0)
    hi = base_f * rng.uniform(18.0, 55.0)
    cut = (lo * (hi / lo) ** lfo).astype(np.float32)
    w = bl.resonant_sweep(y, sr, cut, float(rng.uniform(4, 12)))
    if rng.random() < 0.4:                                   # amp growl-gate
        w = w * (0.25 + 0.75 * lfo).astype(np.float32)
    return _norm(w)


def dub_finalize(y, base_f, sr, rng):
    y = np.nan_to_num(np.asarray(y, np.float32))
    if y.ndim > 1:
        y = core.mono(y)
    n = len(y)
    t = np.arange(n) / sr
    env = np.ones(n, np.float32)
    a, r = min(int(0.01 * sr), n // 4), min(int(0.06 * sr), n // 4)
    if a > 1: env[:a] *= np.linspace(0, 1, a)
    if r > 1: env[-r:] *= np.linspace(1, 0, r)
    sub = np.sin(2 * np.pi * base_f * t).astype(np.float32) * env     # steady sub
    mix = _norm(y) * env * 0.85 + 0.55 * sub
    mix = hifi.dc_block(mix.astype(np.float32), sr)
    d = int(rng.uniform(0.003, 0.009) * sr)
    R = np.concatenate([np.zeros(d, np.float32), mix])[:n]
    stereo = bl._declick(np.stack([mix, R * 0.95], axis=1), sr)
    return hifi.master(stereo, sr, drive=1.6)


# ================================================================= PATCH + RENDER

def dub_patch(rng, force):
    body = ["wtscan", "fbfm", "cheby", "sub_osc", "grain", "psola"]
    glitch = ["waveset", "vowelseq", "crushmod", "bufferglitch"]
    src = force if force in body else str(rng.choice(body))
    shapes = [force] if force in glitch else []
    if rng.random() < 0.6:                                   # one extra texture
        e = str(rng.choice(glitch + ["ringlattice", "drive", "bode"]))
        if e not in shapes:
            shapes.append(e)
    coren = str(rng.choice(["none", "none", "comb", "modal", "none"]))
    return dict(source=src, core=coren, shapes=shapes)


def make_dub(donor_path, variant, seed, force, uid=""):
    rng = np.random.default_rng(seed)
    x, sr = sf.read(str(donor_path), always_2d=True, dtype="float32")
    mod = x.mean(axis=1)
    if sr != SR:
        mod = core.resample(mod, sr / float(SR)); sr = SR
    f0, conf = core.detect_pitch(mod, sr)
    base_f = bl.pick_root(rng, f0, conf, lo=36.0, hi=130.0)
    beat = 60.0 / BPM
    dur = float(rng.choice([1, 2])) * 4 * beat               # 1-2 bars
    patch = dub_patch(rng, force)
    y = bl2.SOURCES[patch["source"]](mod, base_f, sr, dur, rng)
    y = bl2.CORES[patch["core"]](y, base_f, sr, rng)
    for s in patch["shapes"]:
        y = bl2.SHAPES[s](y, mod, base_f, sr, rng)
    y = movement_wobble(y, base_f, sr, rng)
    y = dub_finalize(y, base_f, sr, rng)
    tag = f"{patch['source']}-{patch['core']}"
    if patch["shapes"]:
        tag += "-" + "+".join(patch["shapes"])
    name = f"{uid}{Path(donor_path).stem[:32]}__{tag}__v{variant}.wav"
    OUT.mkdir(parents=True, exist_ok=True)
    hifi.write(OUT / name, y, sr)
    return dict(out=str(OUT / name), donor=str(donor_path), forced=force,
                base_f=round(base_f, 2), bpm=BPM, dur_s=round(len(y) / sr, 2), **patch)


def main(argv=None):
    global BPM
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="both", choices=["raw", "processed", "both"])
    ap.add_argument("--pick", type=int, default=40)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--max-scan", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--bpm", type=float, default=140.0)
    ap.add_argument("--force", default=None, choices=NOVEL,
                    help="use this module in every patch (default: rotate)")
    args = ap.parse_args(argv)
    BPM = args.bpm

    srcs = ["raw", "processed"] if args.sources == "both" else [args.sources]
    picks = bl.listen_and_pick(srcs, args.pick, args.max_scan, args.seed)
    print(f"\n{len(picks)} donors; rendering {len(picks) * args.variants} "
          f"dubstep bass one-shots @ {BPM:g} BPM ...")

    OUT.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    total = len(picks) * args.variants
    for pi, r in enumerate(picks):
        for v in range(args.variants):
            force = args.force or NOVEL[(pi * args.variants + v) % len(NOVEL)]
            try:
                recs.append(make_dub(r["path"], v, args.seed * 100003 + pi * 37 + v,
                                     force, uid=f"{pi:03d}_"))
            except Exception as e:
                print(f"  ! {Path(r['path']).name} v{v}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
        done = (pi + 1) * args.variants
        if done % 40 == 0 or pi == len(picks) - 1:
            print(f"  {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    from collections import Counter
    print("\nforced-module distribution:", dict(Counter(r["forced"] for r in recs)))
    with open(OUT / "_manifest.json", "w") as fh:
        json.dump(recs, fh, indent=1)
    print(f"\ndone: {len(recs)} dubstep bass one-shots @ {BPM:g} BPM -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
