"""neuro_lab2 — composed bass GESTURES with performance + space.

Extends neuro_lab with the four sophistication layers:

  • GLIDES        portamento between steps (tape-glissando) — the signature
                  neuro "rubber-band" pitch slides.
  • CALL/RESPONSE two contrasting timbres alternate across the bar (by half /
                  by beat / by motif-repeat), the "response" voice brighter and
                  often an octave up — question-and-answer phrasing.
  • SIDECHAIN     the sub (and, lightly, the mid) duck to a beat kick grid ->
                  pumping, clean low end.
  • SPACE         tempo-synced feedback DELAY throws + GATED REVERB tails.

Everything else (euclidean rhythm, scale riffs, per-step resynthesis + filter
automation, per-step neuro FX, machine-listening curation) is inherited from
neuro_lab.

    python3 neuro_lab2.py                       # pool -> curate ~130, 140 BPM
    python3 neuro_lab2.py --bpm 150 --keep 100
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # oneshot_lab + echo live at the repo root

from oneshot_lab import core                               # noqa: E402
import bass_lab as bl                                       # noqa: E402
import bass_lab2 as bl2                                     # noqa: E402
import squelch_lab as sq                                    # noqa: E402
import neuro_lab as nl                                      # noqa: E402
import hifi                                                  # noqa: E402

SR = 44100
OUT = HERE / "m-onz_neuro2"
BPM = 140.0
_norm = bl2._norm


# ================================================================= composition

def build_phrase(donor, root, sr, rng):
    beat = 60.0 / BPM
    steps = int(rng.choice([16, 16, 12]))
    bars = int(rng.choice([1, 1, 2]))
    nsteps = steps * bars
    step_s = beat / (steps / 4.0)
    spb = steps / 4.0                                      # steps per beat
    ev = nl.gen_rhythm(nsteps, rng)
    freqs = nl.gen_pitch(nsteps, root, rng)
    auto = nl.automation(nsteps, rng)
    engA, engB = str(rng.choice(nl.ENGINES)), str(rng.choice(nl.ENGINES))
    cr = str(rng.choice(["half", "beat", "motif"]))
    N = int(nsteps * step_s * sr) + int(0.5 * sr)
    buf = np.zeros(N, np.float32)
    prev_f = None
    for i, e in enumerate(ev):
        if not e["on"]:
            continue
        beat_idx = int(i // spb)
        if cr == "half":
            resp = i >= nsteps / 2
        elif cr == "beat":
            resp = (beat_idx % 2 == 1)
        else:
            resp = ((i // 4) % 2 == 1)
        eng = engB if resp else engA
        f = freqs[i]
        if resp and rng.random() < 0.5:
            f = min(f * 2.0, 360.0)                        # response answers higher
        rat = e["rat"]
        subdur = step_s / rat
        slide = rng.random() < 0.35
        for j in range(rat):
            note = bl2.SOURCES[eng](donor, f, sr, max(subdur * 1.7, 0.06), rng)
            cut = 180.0 * (28.0 ** auto[i]) * (1.8 if resp else 1.0)
            b, a = bl._rbj_lowpass(min(cut, sr * 0.45), float(rng.uniform(3, 9)), sr)
            note = lfilter(b, a, note).astype(np.float32)
            if j == 0 and slide and prev_f and abs(np.log2(prev_f / f)) > 0.02:
                ratio = float(np.clip(prev_f / f, 0.5, 2.0))
                note = core.glissando(note, ratio, 1.0).astype(np.float32)   # portamento
            if rng.random() < 0.28:
                fx = str(rng.choice(nl.STEP_FX))
                try:
                    note = note[::-1].copy() if fx == "reverse" \
                        else bl2.SHAPES[fx](note, donor, f, sr, rng)
                except Exception:
                    pass
            note = nl.artic_env(_norm(note), subdur, sr, e["tie"]) * e["acc"]
            pos = int(i * step_s * sr) + int(j * subdur * sr)
            end = min(N, pos + len(note))
            buf[pos:end] += note[: end - pos]
        prev_f = f

    # sidechain duck to a beat kick grid
    duck = np.ones(N, np.float32)
    depth = float(rng.uniform(0.45, 0.8))
    tau = beat * float(rng.uniform(0.25, 0.5))
    nbeats = int(np.ceil(nsteps / spb))
    for bi in range(nbeats):
        tk = int(bi * beat * sr)
        if tk >= N:
            break
        tt = np.arange(N - tk) / sr
        duck[tk:] = np.minimum(duck[tk:], 1.0 - depth * np.exp(-tt / tau))
    t = np.arange(N) / sr
    sub = np.sin(2 * np.pi * root * t).astype(np.float32) * duck
    mid = _norm(buf) * (0.6 + 0.4 * duck)                  # light pump on the mid too
    return _norm(mid) * 0.9 + 0.5 * sub


# ================================================================= space fx

def fb_delay(y, sr, time_s, fb, mix):
    d = max(1, int(time_s * sr))
    out = np.asarray(y, np.float64).copy()
    for i in range(d, len(out)):
        out[i] += fb * out[i - d]
    return ((1 - mix) * y + mix * out).astype(np.float32)


def gated_verb(y, sr, decay, gate_s, mix, rng):
    L = int(min(0.6, decay * 3) * sr)
    if L < 16:
        return y
    ir = (rng.standard_normal(L) * np.exp(-np.arange(L) / (decay * sr))).astype(np.float32)
    wet = _norm(fftconvolve(y, ir)[: len(y)].astype(np.float32))
    gl = int(gate_s * sr)
    g = np.ones(len(y), np.float32)
    if gl < len(y):
        fe = min(256, max(1, gl // 4))
        g[gl:] = 0.0
        g[gl - fe: gl] = np.linspace(1, 0, fe)
    return ((1 - mix) * y + mix * (wet * g)).astype(np.float32)


def bus(y, root, sr, rng):
    n = len(y)
    t = np.arange(n) / sr
    k = max(2, int(n / (0.12 * sr)))
    curve = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, k), nl.automation(k, rng))
    beat = 60.0 / BPM
    lfo = 0.5 - 0.5 * np.cos(2 * np.pi * (float(rng.choice([0.5, 1, 2])) / beat) * t)
    mod = np.clip(0.6 * curve + 0.4 * lfo, 0, 1)
    lo, hi = root * rng.uniform(2, 5), root * rng.uniform(20, 60)
    cut = (lo * (hi / lo) ** mod).astype(np.float32)
    y = bl.resonant_sweep(np.asarray(y, np.float32), sr, cut, float(rng.uniform(4, 11)))
    if rng.random() < 0.55:
        try:
            y = sq.shape_phaser(y, y, root, sr, rng)
        except Exception:
            pass
    if rng.random() < 0.7:                                 # tempo-synced delay throw
        y = fb_delay(y, sr, beat * float(rng.choice([0.5, 0.75, 0.25])),
                     float(rng.uniform(0.3, 0.55)), float(rng.uniform(0.2, 0.4)))
    if rng.random() < 0.6:                                 # gated reverb tail
        y = gated_verb(y, sr, float(rng.uniform(0.15, 0.5)),
                       beat * float(rng.choice([0.5, 1.0])),
                       float(rng.uniform(0.15, 0.35)), rng)
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
    name = f"{uid}{Path(donor_path).stem[:30]}__neuro2_{BPM:g}.wav"
    OUT.mkdir(parents=True, exist_ok=True)
    hifi.write(OUT / name, y, sr)
    return dict(out=str(OUT / name), donor=str(donor_path), root=round(root, 2),
                bpm=BPM, dur_s=round(len(y) / sr, 2))


def main(argv=None):
    global BPM
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="both", choices=["raw", "processed", "both"])
    ap.add_argument("--pick", type=int, default=40)
    ap.add_argument("--pool-per-donor", type=int, default=4)
    ap.add_argument("--keep", type=int, default=130)
    ap.add_argument("--max-scan", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--bpm", type=float, default=140.0)
    args = ap.parse_args(argv)
    BPM = args.bpm

    srcs = ["raw", "processed"] if args.sources == "both" else [args.sources]
    picks = bl.listen_and_pick(srcs, args.pick, args.max_scan, args.seed)
    pool = len(picks) * args.pool_per_donor
    print(f"\n{len(picks)} donors; composing {pool} gestures (glide/call-response/"
          f"sidechain/space) @ {args.bpm:g} BPM ...")

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
    print(f"\ndone: {len(kept)} performed bass gestures (from {len(recs)}) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
