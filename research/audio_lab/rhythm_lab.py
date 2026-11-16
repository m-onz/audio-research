#!/usr/bin/env python3
"""
rhythm_lab.py — turn a folder of experimental audio into drum-machine pieces.

Second-generation instrument: takes a corpus of wavs (e.g. audio_lab/experiments/),
slices it into percussive one-shot chunks, classifies every chunk by spectral
character into drum roles — darkest = kick, low-mid = tom, noisy mids = snare,
snappy noisy = clap, brightest = hat, tonal mids = stab — then sequences the
roles with genre-flavoured patterns:

    four_floor  house/techno, kick every beat, offbeat hats, claps on 2 & 4
    dnb         two-step around 170 bpm, ghost snares, dense skippy hats
    grime       ~140 halftime, big snare on beat 3, swung skippy hats
    dubstep     ~140 halftime, kick 1 & 2.75, wobble stabs
    broken      Euclidean everything, mutations, experimental

Every bar:
  - Euclidean rhythms E(pulses, steps) carry the syncopation
  - complexity (0..1) controls density, ghost notes, ratchets, mutations
  - polymetric voices can cycle at 12/14/20/24 steps against the 16-step bar,
    phasing in and out of sync (real polyrhythm)
  - tempo glides per section, plus a piece-wide drift (accel/decel)
  - sections change genre, complexity and tempo completely

Per-hit chop & layer: random chunk choice within a role (rms-normalised for
consistent velocity), micro-timing jitter, swing, velocity accents, ratchet
rolls (accelerating fills), occasional repitch / reverse, sub-kick layers,
clap-on-snare layers, alternating hat pans.

Usage:
    python3 rhythm_lab.py audio_lab/experiments -o audio_lab/rhythms
    python3 rhythm_lab.py audio_lab/experiments -o out --pieces 8 --seed 3
    python3 rhythm_lab.py audio_lab/experiments -o out --genre dnb --bpm 174 --pieces 4
    python3 rhythm_lab.py audio_lab/experiments -o out --export-bank   # one-shot sample bank
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import granular_lab as gl

EPS = 1e-10
STEP = 16  # 16th-note steps per bar

ROLES = ("kick", "tom", "snare", "clap", "hat", "stab")
ROLE_CFG = {
    #       target loudness, max one-shot length, default pan
    "kick":  dict(target_rms=0.30, max_dur=0.50, pan=0.00),
    "tom":   dict(target_rms=0.22, max_dur=0.40, pan=0.20),
    "snare": dict(target_rms=0.24, max_dur=0.35, pan=-0.12),
    "clap":  dict(target_rms=0.20, max_dur=0.28, pan=0.22),
    "hat":   dict(target_rms=0.12, max_dur=0.18, pan=0.00),
    "stab":  dict(target_rms=0.16, max_dur=0.60, pan=0.00),
}
GENRE_BPM = {"four_floor": (124, 132), "dnb": (168, 176), "grime": (136, 144),
             "dubstep": (138, 144), "broken": (96, 160)}
GENRE_SWING = {"four_floor": 0.06, "dnb": 0.02, "grime": 0.20,
               "dubstep": 0.10, "broken": 0.12}
GENRE_POOL = (["four_floor"] * 5 + ["dnb"] * 5 + ["grime"] * 4 +
              ["dubstep"] * 4 + ["broken"] * 2)


# --------------------------------------------------------------------------- sample pool

@dataclass
class Chunk:
    data: np.ndarray          # (n, 2) float32
    rms: float
    dur: float
    centroid: float
    flat: float
    trans: float
    src: str
    start: float
    bright: float = 0.0       # percentile rank of centroid (0=darkest)
    noisy: float = 0.0        # percentile rank of spectral flatness
    trans_rank: float = 0.0
    role: str = "stab"


def _resample_linear(x, sr_from, sr_to):
    if sr_from == sr_to or len(x) == 0:
        return x
    n = int(round(len(x) * sr_to / sr_from))
    pos = np.linspace(0, len(x) - 1, n)
    base = np.arange(len(x), dtype=np.float64)
    return np.stack([np.interp(pos, base, x[:, c]) for c in range(x.shape[1])],
                    axis=1).astype(np.float32)


def extract_pool(paths, rng, pool_cap=9000, per_file=260, min_rms=0.008):
    """Onset-slice the corpus into one-shot chunks; return (sr, chunks)."""
    chunks, sr0 = [], None
    for path in paths:
        sr, a = gl.read_wav(path)
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        elif a.shape[1] > 2:
            a = a[:, :2]
        a = np.ascontiguousarray(a, np.float32)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            a = _resample_linear(a, sr, sr0)
        segs = gl._onset_segments(a, sr0, dict(min_grain=0.07, max_grain=0.5))
        if len(segs) < 16:                      # dense texture: fall back to a grid
            sz = max(256, int(0.35 * sr0))
            hop = max(128, int(0.22 * sr0))
            segs = [(st, sz) for st in range(0, max(1, len(a) - sz), hop)]
        if len(segs) > per_file:
            segs = segs[:: int(np.ceil(len(segs) / per_file))]
        for st, sz in segs:
            d = a[st: st + sz]
            if d.shape[0] < 256:
                continue
            f = gl.compute_features(d, sr0)
            if f["rms"] < min_rms:
                continue
            chunks.append(Chunk(d, f["rms"], d.shape[0] / sr0, f["centroid"],
                                f["flatness"], f["transient"], path.stem, st / sr0))
    if len(chunks) > pool_cap:
        chunks = chunks[:: int(np.ceil(len(chunks) / pool_cap))]
    # spectral ranks across the whole pool (corpus-agnostic classification)
    for attr, key in (("centroid", "bright"), ("flat", "noisy"), ("trans", "trans_rank")):
        vals = np.array([getattr(c, attr) for c in chunks], float)
        order = np.argsort(vals, kind="stable")
        r = np.empty(len(vals))
        r[order] = np.arange(len(vals)) / max(1, len(vals) - 1)
        for c, v in zip(chunks, r):
            setattr(c, key, float(v))
    return sr0, chunks


def assign_roles(chunks):
    """Frequency-sorted classification: who plays kick, who plays hihat..."""
    for c in chunks:
        if c.bright < 0.22:
            c.role = "kick"        # darkest fifth of the pool
        elif c.bright < 0.38:
            c.role = "tom"         # low-mid
        elif c.bright > 0.85:
            c.role = "hat"         # brightest
        elif c.noisy > 0.55 and c.trans_rank > 0.5:
            c.role = "clap"        # snappy & broadband
        elif c.noisy > 0.32:
            c.role = "snare"       # noisy mids
        else:
            c.role = "stab"        # tonal mids (wobbles, layers)
    pools = {r: [c for c in chunks if c.role == r] for r in ROLES}
    borrow = {"kick": "tom", "tom": "kick", "snare": "clap", "clap": "snare",
              "hat": "stab", "stab": "snare"}
    for r in ROLES:                # never leave a role empty
        if not pools[r]:
            src = borrow[r] if pools.get(borrow[r]) else max(pools, key=lambda k: len(pools[k]))
            pools[r] = list(pools[src])
    return pools


# --------------------------------------------------------------------------- patterns

def _rot(bits, r):
    r %= len(bits)
    return bits[r:] + bits[:r]


def build_bar(genre, c, rng, last_bar):
    """One bar of hits per voice: {voice: [(step, velocity, mods)]}."""
    hits = {v: [] for v in ROLES}

    def add(v, s, vel=1.0, **m):
        hits[v].append((int(s), float(vel), m))

    def eu(v, p, vel=0.7, rot=0, acc=True):
        for i, b in enumerate(gl._euclid_bits(int(p), STEP)):
            if b:
                add(v, i, 1.0 if (acc and i % 4 == 0) else vel)

    g = rng
    if genre == "four_floor":
        for s in (0, 4, 8, 12):
            add("kick", s, 1.0)
            if g.random() < 0.25 + 0.3 * c:
                add("kick", s, 0.5, speed=0.5, dur=0.7)          # sub layer
        for s in (4, 12):
            add("clap", s, 0.85)
            if g.random() < c * 0.5:
                add("snare", s, 0.5)
        for s in (2, 6, 10, 14):
            add("hat", s, 0.75)
        if c > 0.25:
            eu("hat", 4 + int(c * 8), vel=0.4, rot=int(g.integers(STEP)))
        if g.random() < 0.2 + 0.5 * c:
            eu("stab", 1 + int(g.integers(0, 2)), vel=0.5)
        if g.random() < c * 0.5:
            add("tom", 14, 0.5, ratchet=2)
    elif genre == "dnb":
        add("kick", 0, 1.0)
        add("kick", 10, 0.95)
        if g.random() < c:
            add("kick", 7, 0.6, speed=1.2)
        add("snare", 4, 1.0)
        add("snare", 12, 1.0)
        if g.random() < 0.35 + 0.5 * c:
            add("clap", 12, 0.6)
        if g.random() < c * 0.6:
            add("snare", int(g.integers(6, 9)), 0.3)             # ghost
        if g.random() < c * 0.6:
            add("snare", int(g.integers(13, 16)), 0.3)
        eu("hat", 7 + int(c * 6), vel=0.5, rot=int(g.integers(STEP)))
        if g.random() < c * 0.4:
            add("hat", int(g.integers(0, STEP)), 0.6, speed=0.6, dur=0.3)   # open hat
        if g.random() < c * 0.5:
            eu("tom", int(2 + c * 3), vel=0.5)
    elif genre == "grime":
        add("kick", 0, 1.0)
        if g.random() < 0.8:
            add("kick", 6, 0.9)
        if g.random() < c * 0.7:
            add("kick", 10, 0.7)
        add("snare", 8, 1.0)
        if g.random() < c * 0.6:
            add("clap", 8, 0.7)
        if g.random() < c * 0.5:
            add("snare", 14, 0.35)                              # eski skip
        eu("hat", 4 + int(c * 7), vel=0.5, rot=int(g.integers(STEP)))
        if g.random() < 0.25 + 0.6 * c:
            eu("stab", 1 + int(g.integers(0, 2)), vel=0.55)
        if g.random() < c * 0.6:
            add("stab", int(g.integers(0, 8)), 0.6, ratchet=2)
    elif genre == "dubstep":
        add("kick", 0, 1.0)
        add("kick", 10, 0.95)
        add("snare", 8, 1.0)
        if g.random() < 0.6:
            add("clap", 8, 0.6)
        if g.random() < c * 0.6:
            add("kick", 4, 0.65)
        if g.random() < c * 0.5:
            add("kick", 14, 0.6, speed=1.15)
        eu("hat", 2 + int(c * 5), vel=0.5)
        eu("stab", 2 + int(c * 3), vel=0.6, rot=int(g.integers(STEP)))       # wobbles
        if g.random() < c * 0.5:
            add("stab", int(g.integers(0, STEP)), 0.7, speed=0.5, dur=0.8)  # bass drop
    else:  # broken — Euclidean everything, heavy mutation
        eu("kick", 3 + int(g.integers(0, 2 + int(c * 3))), vel=0.9)
        eu("snare", 2 + int(g.integers(0, 2 + int(c * 2))), vel=0.8, rot=int(g.integers(STEP)))
        eu("hat", 5 + int(c * 9), vel=0.5, rot=int(g.integers(STEP)))
        if g.random() < 0.6:
            eu("tom", int(1 + c * 5), vel=0.55, rot=int(g.integers(STEP)))
        if g.random() < 0.5:
            eu("stab", int(1 + c * 4), vel=0.5, rot=int(g.integers(STEP)))
        if g.random() < c * 0.7:
            add("snare", int(g.integers(0, STEP)), 0.4, ratchet=int(2 + g.integers(0, 3)))
    if last_bar:                                                 # fill
        add("snare", 12, 0.9, ratchet=int(2 + rng.integers(0, 3)), accel=True)
        add("tom", 8, 0.6)
        add("tom", 10, 0.6)
        add("tom", 14, 0.7, ratchet=2)
    return hits


# --------------------------------------------------------------------------- arrangement

@dataclass
class Section:
    genre: str
    bars: int
    bpm0: float
    bpm1: float
    c0: float                  # complexity at section start
    c1: float                  # ... and end
    swing: float
    poly: dict = field(default_factory=dict)   # voice -> cycle length (12/14/20/24)


def random_arrangement(rng, pin_genre=None, pin_bpm=None, pin_bars=None):
    secs, prev = [], None
    for i in range(int(rng.integers(3, 6))):
        genre = pin_genre or str(rng.choice(GENRE_POOL))
        if genre == prev:
            genre = pin_genre or str(rng.choice(GENRE_POOL))
        bars = pin_bars or int(rng.choice([4, 4, 6, 8]))
        lo, hi = GENRE_BPM[genre]
        if pin_bpm:
            b0 = b1 = float(pin_bpm)
        else:
            b0 = float(rng.uniform(lo, hi))
            b1 = float(rng.uniform(lo, hi))
        c0 = float(rng.uniform(0.15, 0.6)) if i == 0 else float(rng.uniform(0.2, 0.95))
        c1 = float(rng.uniform(0.25, 0.95))
        swing = max(0.0, GENRE_SWING[genre] + float(rng.uniform(-0.03, 0.03)))
        poly = {}
        if (genre in ("broken", "grime") and rng.random() < 0.55) or rng.random() < 0.30:
            for v in rng.choice(np.array(["hat", "tom", "stab"]),
                               size=int(1 + rng.integers(0, 2)), replace=False):
                poly[str(v)] = int(rng.choice(np.array([12, 14, 20, 24])))
        secs.append(Section(genre, bars, b0, b1, c0, c1, swing, poly))
        prev = genre
    return secs


# --------------------------------------------------------------------------- rendering

def _emit(events, pool, role, t, vel, mods, rng, sr, stepdur, state):
    count = max(1, int(mods.get("ratchet", 1)))
    speed = float(mods.get("speed", 1.0))
    dur = float(mods.get("dur", ROLE_CFG[role]["max_dur"]))
    ch = pool[int(rng.integers(len(pool)))]
    gain = min(12.0, ROLE_CFG[role]["target_rms"] / max(1e-4, ch.rms))
    sp = stepdur / count
    for k in range(count):
        tt = t + k * sp * (0.72 ** k if mods.get("accel") else 1.0)
        v = vel * (1.0 - 0.18 * k)
        y = gl.resample_grain(ch.data, speed, "hermite")
        m = int(dur * sr)
        if len(y) > m:
            y = y[:m]
        env = gl.make_env("const", len(y))
        amp = gain * v * 10.0 ** (rng.uniform(-1.5, 1.5) / 20.0)
        y = y * env[:, None] * amp
        if mods.get("rev"):
            y = y[::-1]
        if role == "hat":                       # alternating hat pans
            state["hatpan"] = -state.get("hatpan", 1.0)
            pan = 0.55 * state["hatpan"] + rng.uniform(-0.1, 0.1)
        else:
            pan = ROLE_CFG[role]["pan"] + rng.uniform(-0.12, 0.12)
        th = (pan + 1.0) * (np.pi / 4.0)
        y[:, 0] *= np.cos(th)
        y[:, 1] *= np.sin(th)
        events.append((max(0, int(tt * sr)), y))


def render_piece(sections, pools, rng, sr, drift=(1.0, 1.0)):
    events, t0 = [], 0.0
    total_bars = sum(s.bars for s in sections)
    done_bars = 0
    state = {"hatpan": 1.0}
    for sec in sections:
        c_mid = 0.5 * (sec.c0 + sec.c1)
        poly_bits = {}
        for v, n in sec.poly.items():            # polymetric cycles vs the 16-grid
            p = max(2, int(round(n * (0.12 + 0.35 * c_mid))))
            poly_bits[v] = (n, _rot(gl._euclid_bits(p, n), int(rng.integers(n))))
        for b in range(sec.bars):
            prog = (done_bars + b) / max(1, total_bars - 1)
            scale = drift[0] + (drift[1] - drift[0]) * prog
            bpm = (sec.bpm0 + (sec.bpm1 - sec.bpm0) * b / max(1, sec.bars - 1)) * scale
            stepdur = 60.0 / bpm / 4.0
            c = sec.c0 + (sec.c1 - sec.c0) * b / max(1, sec.bars - 1)
            hits = build_bar(sec.genre, c, rng, b == sec.bars - 1)
            for v, (n, bits) in poly_bits.items():
                if pools.get(v):
                    hits[v] = [(s, 0.6, {}) for s in range(STEP)
                               if bits[(b * STEP + s) % n]]
            bar_t = t0
            t0 += STEP * stepdur
            for v, hl in hits.items():
                pool = pools.get(v) or []
                if not pool:
                    continue
                for step, vel, mods in hl:
                    t = bar_t + step * stepdur
                    if step % 2 == 1:
                        t += sec.swing * stepdur
                    t += rng.uniform(-1.0, 1.0) * 0.004
                    _emit(events, pool, v, t, vel, mods, rng, sr, stepdur, state)
        done_bars += sec.bars
    return events


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(
        description="Corpus-fed drum machine: classify chunks by spectral role, "
                    "sequence with Euclidean / polymetric / genre patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="corpus wav files or directories")
    ap.add_argument("-o", "--out", default="rhythms", help="output directory")
    ap.add_argument("--pieces", type=int, default=8, help="number of pieces to render")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--genre", choices=list(GENRE_BPM), help="pin every section to one genre")
    ap.add_argument("--bpm", type=float, help="pin tempo (drift still applies within piece)")
    ap.add_argument("--bars", type=int, help="pin bars per section")
    ap.add_argument("--export-bank", action="store_true",
                    help="also write a one-shot sample bank to OUT/bank/")
    ap.add_argument("--pool-cap", type=int, default=9000, help="max chunks in the pool")
    args = ap.parse_args()

    files = gl.gather_inputs(args.inputs)
    if not files:
        ap.error("no .wav files found in the given inputs")
    print(f"slicing {len(files)} corpus file(s) into one-shots...")
    rng = np.random.default_rng(args.seed)
    sr, chunks = extract_pool(files, rng, args.pool_cap)
    if not chunks:
        ap.error("no chunks extracted - is the corpus silent?")
    pools = assign_roles(chunks)
    print(f"pool: {len(chunks)} chunks @ {sr} Hz, classified by spectral centroid:")
    for r in ROLES:
        ps = pools[r]
        med = float(np.median([c.centroid for c in ps])) if ps else 0.0
        print(f"  {r:6s} {len(ps):5d}  | median centroid {med:6.0f} Hz")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    mpath = outdir / "manifest.json"
    manifest = {}
    if mpath.exists():
        try:
            with open(mpath) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    if args.export_bank:
        bank = outdir / "bank"
        bank.mkdir(exist_ok=True)
        for r in ROLES:
            ps = sorted(pools[r], key=lambda c: -c.rms)[:32]
            for i, c in enumerate(ps):
                y = c.data[: int(ROLE_CFG[r]["max_dur"] * sr)]
                amp = min(12.0, ROLE_CFG[r]["target_rms"] / max(1e-4, c.rms))
                gl.write_wav(bank / f"{r}_{i:03d}.wav", sr,
                             y * gl.make_env("const", len(y))[:, None] * amp)
        print(f"sample bank -> {bank}/")

    DRIFTS = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.06), (1.0, 1.12),
              (1.10, 0.95), (0.95, 1.08)]
    for i in range(args.pieces):
        pseed = args.seed + 7919 * i + 13
        prng = np.random.default_rng(pseed)
        secs = random_arrangement(prng, args.genre, args.bpm, args.bars)
        drift = DRIFTS[int(prng.integers(len(DRIFTS)))]
        events = render_piece(secs, pools, prng, sr, drift)
        audio = gl.finalize(events, sr, 0.97)
        tag = "-".join(dict.fromkeys(s.genre for s in secs))
        path = outdir / f"rhythm{i:02d}_{tag}_{pseed}.wav"
        gl.write_wav(path, sr, audio)
        dur = audio.shape[0] / sr
        desc = " | ".join(
            f"{s.genre} {s.bars}b {s.bpm0:.0f}->{s.bpm1:.0f}bpm c{s.c0:.2f}->{s.c1:.2f}"
            for s in secs)
        print(f"  rhythm{i:02d} ({dur:5.1f}s, drift {drift[0]:.2f}->{drift[1]:.2f}): {desc}")
        manifest[str(path)] = dict(
            seed=pseed, drift=list(drift), sr=sr,
            sections=[dict(genre=s.genre, bars=s.bars, bpm0=s.bpm0, bpm1=s.bpm1,
                           c0=s.c0, c1=s.c1, swing=s.swing, poly=s.poly)
                      for s in secs],
            pool={r: len(pools[r]) for r in ROLES},
        )

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"done -> {outdir}/ (manifest.json records every arrangement)")


if __name__ == "__main__":
    sys.exit(main())