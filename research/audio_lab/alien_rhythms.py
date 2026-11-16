#!/usr/bin/env python3
"""
alien_rhythms.py — real drum one-shots driven by a Lisp pattern language.

Third-generation instrument, built on three ideas:

1. REAL DRUMS. The sample bank comes from a folder of one-shot drum sounds
   (kick/, snare/, hihat/, rimshot/, clap/, perc/, ...). Every hit is stochastically
   *processed*: pitch shifts, reversal, tanh saturation, sample-hold decimation,
   one-pole filtering, micro-timing — so the same sample never plays the same way
   twice. A synthesized sub-bass thump layers under kick hits for real low end.

2. A LISP PATTERN LANGUAGE (a Python port of the `alien` language for Pure Data):
   (euclid 3 8 0 1) (mask (ramp 0.5 1 8) (euclid 4 16)) (prob (drunk 16 2 0.7) 60)
   (speed (mirror (euclid 5 12)) 2) ...  Expressions are evaluated EVERY BAR, so
   stochastic operators (prob/drunk/rand/choose/shuffle) make patterns breathe.

3. GENERATED SCORES. Instead of hardcoded genre rhythms, each piece *composes its
   own pattern expressions* from a grammar, then mutates them across sections
   (rotate / stretch / densify / regenerate). Any pattern length other than 16
   automatically becomes a polymeter phasing against the bar.

The full lispy score of every piece is written next to the wav as
`alienNN_score.txt` — paste the patterns straight into [alien] in Pd.

Usage:
    python3 alien_rhythms.py audio_lab/sounds -o audio_lab/alien_rhythms
    python3 alien_rhythms.py audio_lab/sounds -o out --pieces 12 --seed 7
    python3 alien_rhythms.py audio_lab/sounds -o out --bpm 170 --bars 6
    python3 alien_rhythms.py audio_lab/sounds -o out --bed audio_lab/rhythms \\
        --bed-gain 0.55                       # layer drums over the rhythm pieces
    python3 alien_rhythms.py audio_lab/sounds -o out --voice 'kick=(euclid 4 16)'
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import granular_lab as gl

try:
    from scipy.signal import lfilter
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

REST = None
STEP = 16

FOLDER_ROLE = {
    "kick": "kick", "snare": "snare", "hihat": "hat", "openhh": "ohat",
    "openhihats": "ohat", "clap": "clap", "rimshot": "rim", "perc": "perc",
    "rides": "ride", "shaker": "shaker", "crash": "crash", "woodblock": "block",
    "toms": "tom", "vox": "vox", "mbass": "bass", "angry-bass": "bass",
    "mbreathy-bass": "bass", "mblip": "blip", "mhit": "perc", "mhightonal": "tonal",
}
ROLE_CFG = {
    #          rms target, max one-shot length
    "kick":  dict(target=0.30, max_dur=0.9),
    "snare": dict(target=0.22, max_dur=0.6),
    "hat":   dict(target=0.10, max_dur=0.30),
    "ohat":  dict(target=0.11, max_dur=0.70),
    "clap":  dict(target=0.18, max_dur=0.50),
    "rim":   dict(target=0.15, max_dur=0.30),
    "perc":  dict(target=0.16, max_dur=0.70),
    "tom":   dict(target=0.20, max_dur=0.70),
    "ride":  dict(target=0.10, max_dur=1.40),
    "shaker": dict(target=0.09, max_dur=0.40),
    "crash": dict(target=0.12, max_dur=2.50),
    "block": dict(target=0.16, max_dur=0.30),
    "vox":   dict(target=0.14, max_dur=1.20),
    "bass":  dict(target=0.24, max_dur=0.70),
    "blip":  dict(target=0.14, max_dur=0.60),
    "tonal": dict(target=0.12, max_dur=1.00),
    "tex":   dict(target=0.09, max_dur=1.80),
    "air":   dict(target=0.06, max_dur=1.40),
}
VOICE_PROFILE = {
    # k = euclid hit-count range, val = velocity bias, poly = polymeter chance
    "kick":   dict(k=(3, 7), val="hi", steady=0.22),
    "snare":  dict(k=(1, 4), val="mid"),
    "hat":    dict(k=(5, 13), val="lo", poly=0.35),
    "ohat":   dict(k=(1, 3), val="lo"),
    "clap":   dict(k=(1, 3), val="mid"),
    "rim":    dict(k=(2, 6), val="lo", poly=0.2),
    "perc":   dict(k=(3, 9), val="mid", poly=0.3),
    "tom":    dict(k=(1, 4), val="mid"),
    "ride":   dict(k=(3, 8), val="lo", poly=0.2),
    "shaker": dict(k=(4, 10), val="lo"),
    "block":  dict(k=(1, 4), val="mid"),
    "blip":   dict(k=(2, 6), val="hi", poly=0.3),
    "tonal":  dict(k=(1, 4), val="mid", poly=0.25),
    "vox":    dict(k=(1, 3), val="mid"),
    "crash":  dict(k=(1, 2), val="mid", steps=32),
    "bass":   dict(k=(2, 5), val="hi", poly=0.15),
    "tex":    dict(k=(1, 3), val="mid", poly=0.25),
    "air":    dict(k=(1, 3), val="lo", poly=0.25),
}
BPM_ZONES = [  # weighted pick: house / grime-dubstep / dnb
    (128.0, 134.0), (128.0, 134.0), (136.0, 145.0), (136.0, 145.0), (168.0, 176.0),
]
DRIFTS = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.06), (1.0, 1.12), (1.10, 0.95), (0.95, 1.08)]


# =========================================================== the pattern language

def _atom(t):
    if t in ("-", ".", "_"):
        return REST
    try:
        return float(t) if ("." in t or "e" in t or "E" in t) else int(t)
    except ValueError:
        return t


def parse_pattern(src):
    toks = src.replace("(", " ( ").replace(")", " ) ").split()

    def read():
        if not toks:
            raise ValueError("unexpected end of pattern")
        t = toks.pop(0)
        if t == "(":
            out = []
            while toks and toks[0] != ")":
                out.append(read())
            if not toks:
                raise ValueError("missing )")
            toks.pop(0)
            return out
        if t == ")":
            raise ValueError("unexpected )")
        return _atom(t)

    ast = read()
    if toks:
        raise ValueError("trailing tokens after pattern")
    return ast


def to_src(a):
    if a is None:
        return "-"
    if isinstance(a, list):
        return "(" + " ".join(to_src(x) for x in a) + ")"
    if isinstance(a, float):
        return f"{a:g}"
    return str(a)


def _as_seq(x):
    return x if isinstance(x, list) else [x]


def _mapv(s, f):
    return [None if v is None else f(v) for v in _as_seq(s)]


def _op_seq(a, rng):
    out = []
    for x in a:
        out.extend(_as_seq(x))
    return out


def _op_euclid(a, rng):
    hits, steps = a[0], int(a[1])
    rot = int(a[2]) if len(a) > 2 else 0
    val = a[3] if len(a) > 3 else 1.0
    if steps <= 0:
        return []
    hits_seq = _as_seq(hits) if isinstance(hits, list) else None
    k = len(hits_seq) if hits_seq else int(hits)
    if k <= 0:
        return [REST] * steps
    pat = [0] * steps
    if k >= steps:
        pat = [1] * steps
    else:                                   # alien's bucket algorithm
        bucket = 0
        for i in range(steps):
            bucket += k
            if bucket >= steps:
                bucket -= steps
                pat[i] = 1
    out, j = [], 0
    for i in range(steps):
        if pat[(i + rot) % steps]:
            out.append(hits_seq[j % len(hits_seq)] if hits_seq else val)
            j += 1
        else:
            out.append(REST)
    return out


OPS = {
    "seq": _op_seq,
    "rep": lambda a, rng: _as_seq(a[0]) * max(0, int(a[1])),
    "add": lambda a, rng: _mapv(a[0], lambda v: v + a[1]),
    "sub": lambda a, rng: _mapv(a[0], lambda v: v - a[1]),
    "mul": lambda a, rng: _mapv(a[0], lambda v: v * a[1]),
    "mod": lambda a, rng: _mapv(a[0], lambda v: v % a[1] if a[1] else v),
    "scale": lambda a, rng: _mapv(a[0], lambda v: a[3] + (v - a[1]) *
                                  (a[4] - a[3]) / max(1e-9, a[2] - a[1])),
    "clamp": lambda a, rng: _mapv(a[0], lambda v: min(max(v, a[1]), a[2])),
    "wrap": lambda a, rng: _mapv(a[0], lambda v: a[1] + (v - a[1]) %
                                 (a[2] - a[1]) if a[2] > a[1] else v),
    "fold": lambda a, rng: _mapv(a[0], lambda v: _fold1(v, a[1], a[2])),
    "euclid": _op_euclid,
    "subdiv": lambda a, rng: [v for v in _as_seq(a[0]) for _ in range(max(1, int(a[1])))],
    "reverse": lambda a, rng: list(reversed(_as_seq(a[0]))),
    "rotate": lambda a, rng: _rotseq(_as_seq(a[0]), int(a[1])),
    "interleave": lambda a, rng: _interleave(_as_seq(a[0]), _as_seq(a[1])),
    "shuffle": lambda a, rng: list(rng.permutation(np.asarray(_as_seq(a[0]), object))),
    "mirror": lambda a, rng: _as_seq(a[0]) + list(reversed(_as_seq(a[0])[:-1])),
    "take": lambda a, rng: _as_seq(a[0])[: max(0, int(a[1]))],
    "drop": lambda a, rng: _as_seq(a[0])[max(0, int(a[1])):],
    "slice": lambda a, rng: _as_seq(a[0])[int(a[1]): int(a[2])],
    "every": lambda a, rng: _as_seq(a[0])[:: max(1, int(a[1]))],
    "filter": lambda a, rng: [v for v in _as_seq(a[0]) if v is not None],
    "range": lambda a, rng: list(range(int(a[0]), int(a[1]) + 1,
                                       int(a[2]) if len(a) > 2 else 1)),
    "ramp": lambda a, rng: [a[0] + (a[1] - a[0]) * i / max(1, int(a[2]) - 1)
                           for i in range(max(1, int(a[2])))],
    "choose": lambda a, rng: _as_seq(a[int(rng.integers(len(a)))]),
    "rand": lambda a, rng: [int(rng.integers(int(a[1] if len(a) > 1 else 0),
                                            int(a[2] if len(a) > 2 else 128) + 1))
                            for _ in range(max(0, int(a[0])))],
    "prob": lambda a, rng: [v if (v is not None and rng.random() * 100 < a[1])
                            else REST for v in _as_seq(a[0])],
    "drunk": lambda a, rng: _drunk(rng, a),
    "quantize": lambda a, rng: _mapv(a[0], lambda v: _snap(v, _as_seq(a[1]))),
    "arp": lambda a, rng: _arp(_as_seq(a[0]), int(a[1]), int(a[2])),
    "cycle": lambda a, rng: (_as_seq(a[0]) * (int(a[1]) // max(1, len(_as_seq(a[0]))) + 1))[: int(a[1])],
    "grow": lambda a, rng: _grow(_as_seq(a[0])),
    "gate": lambda a, rng: [v if i % max(1, int(a[1])) == 0 else REST
                            for i, v in enumerate(_as_seq(a[0]))],
    "speed": lambda a, rng: [x for v in _as_seq(a[0])
                             for x in [v] + [REST] * (max(1, int(a[1])) - 1)],
    "mask": lambda a, rng: _mask(_as_seq(a[0]), _as_seq(a[1])),
    "delay": lambda a, rng: [REST] * max(0, int(a[1])) + _as_seq(a[0]),
}


def _fold1(v, lo, hi):
    if hi <= lo:
        return v
    span = hi - lo
    v = (v - lo) % (2 * span)
    return lo + (v if v <= span else 2 * span - v)


def _rotseq(s, n):
    if not s:
        return s
    n %= len(s)
    return s[-n:] + s[:-n]


def _interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def _drunk(rng, a):
    count, step, v = int(a[0]), abs(float(a[1])), float(a[2])
    lo = float(a[3]) if len(a) > 3 else None
    hi = float(a[4]) if len(a) > 4 else None
    out = []
    for _ in range(max(0, count)):
        v += rng.uniform(-step, step)
        if lo is not None and hi is not None and hi > lo:
            v = _fold1(v, lo, hi)
        out.append(v)
    return out


def _snap(v, pcs):
    pcs = [int(p) % 12 for p in pcs if p is not None]
    if not pcs:
        return v
    octv, pc = int(v) // 12, int(v) % 12
    best = min(pcs, key=lambda p: min((pc - p) % 12, (p - pc) % 12))
    return octv * 12 + best


def _arp(notes, direction, length):
    notes = [n for n in notes if n is not None]
    if not notes or length <= 0:
        return []
    if direction == 1:
        seq = list(reversed(notes))
    elif direction == 2:
        seq = notes + list(reversed(notes[1:-1])) if len(notes) > 2 else notes
    else:
        seq = notes
    return [seq[i % len(seq)] for i in range(length)]


def _grow(s):
    out = []
    for i in range(1, len(s) + 1):
        out += s[:i] + [REST] * (len(s) - i)
    return out


def _mask(src, gate):
    out, j = [], 0
    for g in gate:
        if g is None:
            out.append(REST)
        else:
            out.append(src[j % len(src)] if src else REST)
            j += 1
    return out


def ev(node, rng):
    """Evaluate a pattern AST -> list of values (floats) and rests (None)."""
    if not isinstance(node, list):
        return node
    op = node[0]
    if not isinstance(op, str):
        raise ValueError(f"bad operator {op!r}")
    f = OPS.get(op)
    if f is None:
        raise ValueError(f"unknown operator {op!r}")
    args = [ev(a, rng) for a in node[1:]]
    return f(args, rng)


def normalize_velocities(pat):
    """Scale MIDI-ish values to 0..1 velocities; clip; keep rests."""
    pat = _as_seq(pat)
    vals = [v for v in pat if v is not None]
    if not vals:
        return pat
    hi = max(abs(v) for v in vals)
    if hi > 1.5:                                    # looks like 0..127 MIDI values
        pat = [None if v is None else v / 127.0 for v in pat]
    return [None if v is None else float(min(max(v, 0.0), 1.0)) for v in pat]


# =========================================================== expression generator

def gen_value_expr(rng, bias):
    lo, hi = {"hi": (0.7, 1.0), "mid": (0.45, 0.85), "lo": (0.25, 0.6)}[bias]
    r = rng.random()
    if r < 0.4:
        return ["ramp", round(float(rng.uniform(lo, max(lo, hi - 0.25))), 2),
                round(float(rng.uniform(max(lo, hi - 0.2), hi)), 2),
                int(rng.integers(2, 8))]
    if r < 0.62:
        return ["drunk", int(rng.integers(3, 9)), 0.15,
                round(float(rng.uniform(lo, hi)), 2), round(lo, 2), round(hi, 2)]
    if r < 0.85:
        return ["seq"] + [round(float(rng.uniform(lo, hi)), 2)
                          for _ in range(int(rng.integers(2, 5)))]
    return ["rand", int(rng.integers(2, 7)), round(lo, 2), round(hi, 2)]


def gen_gate(rng, prof):
    steps = int(prof.get("steps", 16))
    if rng.random() < prof.get("poly", 0.0):
        steps = int(rng.choice(np.array([12, 14, 20, 24])))
    if prof.get("steady") and rng.random() < prof["steady"]:
        return ["euclid", 4, 16, int(rng.integers(16))]     # four-ish floor
    k = int(rng.integers(prof["k"][0], prof["k"][1] + 1))
    rot = int(rng.integers(0, steps))
    if rng.random() < 0.35:                                  # velocity ramp on hits
        return ["euclid", gen_value_expr(rng, prof["val"]), steps, rot]
    return ["euclid", k, steps, rot]


def gen_pattern(rng, prof):
    pat = gen_gate(rng, prof)
    if rng.random() < 0.55:
        pat = ["mask", gen_value_expr(rng, prof["val"]), pat]
    for _ in range(int(rng.integers(0, 3))):
        pat = _wrap(rng, pat, prof)
    return pat


def _wrap(rng, pat, prof):
    w = rng.random()
    if w < 0.18:
        return ["rotate", pat, int(rng.integers(1, 8))]
    if w < 0.36:
        return ["prob", pat, int(rng.integers(35, 95))]
    if w < 0.50:
        return ["speed", pat, int(rng.integers(1, 3))]
    if w < 0.62:
        return ["gate", pat, int(rng.integers(2, 4))]
    if w < 0.74:
        return ["subdiv", pat, 2] if rng.random() < 0.5 else ["mirror", pat]
    if w < 0.88:
        return ["interleave", pat, gen_gate(rng, prof)]
    return ["cycle", pat, 16]


def mutate(ast, rng, prof):
    ast = copy.deepcopy(ast)
    r = rng.random()
    if r < 0.25:
        return ["rotate", ast, int(rng.integers(1, 6))]
    if r < 0.45:
        return ["prob", ast, int(rng.integers(40, 90))]
    if r < 0.62:
        return ["subdiv", ast, 2] if rng.random() < 0.5 else ["speed", ast, 2]
    if r < 0.85:
        return _tweak_number(ast, rng)
    return gen_pattern(rng, prof)


def _tweak_number(ast, rng):
    positions = []

    def walk(n, path):
        if isinstance(n, list):
            for i, c in enumerate(n):
                walk(c, path + [i])
        elif isinstance(n, (int, float)) and not isinstance(n, bool):
            positions.append(path)

    walk(ast, [])
    if not positions:
        return ast
    path = positions[int(rng.integers(len(positions)))]
    cur = ast
    for i in path[:-1]:
        cur = cur[i]
    old = cur[path[-1]]
    if isinstance(old, int):
        cur[path[-1]] = max(0, old + int(rng.integers(-2, 3)))
    else:
        cur[path[-1]] = float(np.clip(old * rng.uniform(0.6, 1.5), 0.02, 1.2))
    return ast


# =========================================================== sample bank

@dataclass
class Sample:
    data: np.ndarray
    rms: float
    role: str
    name: str


def _stereo(a):
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    elif a.shape[1] > 2:
        a = a[:, :2]
    return np.ascontiguousarray(a, np.float32)


def _resample_linear(x, sr_from, sr_to):
    if sr_from == sr_to or len(x) == 0:
        return x
    n = int(round(len(x) * sr_to / sr_from))
    pos = np.linspace(0, len(x) - 1, n)
    base = np.arange(len(x), dtype=np.float64)
    return np.stack([np.interp(pos, base, x[:, c]) for c in range(x.shape[1])],
                    axis=1).astype(np.float32)


def _highpass(x, sr, fc, poles=2):
    """Cascade of one-pole highpasses - clears sub space for kick/bass."""
    if not HAVE_SCIPY or len(x) == 0:
        return x
    al = float(np.exp(-2 * np.pi * fc / sr))
    y = x
    for _ in range(poles):
        y = np.stack([lfilter([al, -al], [1, -al], y[:, c]) for c in range(y.shape[1])],
                     axis=1).astype(np.float32)
    return y


def _prep_sample(a, sr, role, name):
    a = _stereo(a)
    env = np.abs(a).max(axis=1)
    thr = float(env.max()) * 0.04
    idx = np.flatnonzero(env > thr)
    if len(idx):
        a = a[max(0, int(idx[0]) - int(0.003 * sr)):]
    cfg = ROLE_CFG[role]
    a = a[: int(cfg["max_dur"] * sr)]
    if len(a) < 64:
        return None
    f = min(len(a) // 4, int(0.003 * sr))
    if f > 1:
        a[:f] *= np.linspace(0, 1, f)[:, None]
        a[-f:] *= np.linspace(1, 0, f)[:, None]
    rms = float(np.sqrt(np.mean(a * a) + 1e-12))
    a = a * min(40.0, cfg["target"] / max(rms, 1e-4))
    return Sample(a, float(np.sqrt(np.mean(a * a))), role, name)


def load_bank(sounds_dir, rng, per_role=28):
    bank, sr0 = {}, 44100
    for sub in sorted(Path(sounds_dir).iterdir()):
        if not sub.is_dir():
            continue
        role = FOLDER_ROLE.get(sub.name.lower())
        if not role:
            continue
        files = sorted(q for q in sub.iterdir()
                       if q.suffix.lower() in (".wav", ".wave", ".aif", ".aiff", ".flac"))
        if len(files) > per_role:
            files = [files[i] for i in sorted(rng.choice(len(files), size=per_role,
                                                         replace=False))]
        for f in files:
            try:
                sr, a = gl.read_wav(f)
            except Exception as e:
                print(f"  warning: cannot read {f.name}: {e}")
                continue
            if sr != sr0:
                a = _resample_linear(_stereo(a), sr, sr0)
            s = _prep_sample(a, sr0, role, f.name)
            if s is not None:
                bank.setdefault(role, []).append(s)
    return bank


def load_texture(corpus_files, rng, cap=900, per_file=30):
    """'tex' (mid) and 'air' (bright) one-shots from the experimental corpus —
    the granular 'layer of sound' that sits under the drums."""
    pools, chunks = {"tex": [], "air": []}, []
    for f in corpus_files:
        try:
            sr, a = gl.read_wav(f)
        except Exception:
            continue
        a = _stereo(a)
        segs = gl._onset_segments(a, sr, dict(min_grain=0.15, max_grain=1.4))
        if len(segs) > per_file:
            segs = segs[:: int(np.ceil(len(segs) / per_file))]
        for st, sz in segs:
            d = a[st: st + sz]
            if d.shape[0] < 2048:
                continue
            ft = gl.compute_features(d, sr)
            if ft["rms"] < 0.006:
                continue
            s = Sample(d, ft["rms"], "tex", f.stem)
            s.bright = ft["centroid"]
            chunks.append(s)
            if len(chunks) >= cap * 3:
                break
    if len(chunks) > cap:
        chunks = chunks[:: int(np.ceil(len(chunks) / cap))]
    cents = np.array([c.bright for c in chunks])
    rank = np.empty(len(cents))
    rank[np.argsort(cents, kind="stable")] = np.arange(len(cents)) / max(1, len(cents) - 1)
    for c, r in zip(chunks, rank):
        role = "air" if r > 0.65 else "tex"
        cfg = ROLE_CFG[role]
        c.role = role
        c.data = c.data[: int(cfg["max_dur"] * 44100)]
        c.data = _highpass(c.data, 44100, 110.0)      # stay out of the kick's lane
        c.data = c.data * min(30.0, cfg["target"] / max(c.rms, 1e-4))
        pools[role].append(c)
    return {k: v for k, v in pools.items() if v}


# =========================================================== hit processing

def piece_character(rng):
    return dict(
        pitch_p=float(rng.uniform(0.0, 0.5)),
        rev_p=float(rng.uniform(0.02, 0.12)),
        drive_p=float(rng.uniform(0.05, 0.30)),
        crush_p=float(rng.uniform(0.0, 0.15)),
        tilt_p=float(rng.uniform(0.05, 0.30)),
        jit_ms=float(rng.uniform(2.0, 7.0)),
    )


def process_hit(y, sr, rng, ch, role):
    speed = 1.0
    if role == "bass" and rng.random() < 0.7:
        speed = float(rng.uniform(0.5, 0.78))          # pitch down = weight
    elif rng.random() < ch["pitch_p"]:
        speed = float(rng.choice(np.array([0.5, 0.62, 0.75, 1.5, 2.0])))
    if speed != 1.0:
        y = gl.resample_grain(y, speed, "hermite")
        y = y[: int(ROLE_CFG[role]["max_dur"] * sr)]
    if rng.random() < ch["rev_p"] and len(y) > 400:
        y = y[::-1]
    if rng.random() < ch["drive_p"]:
        y = np.tanh(y * 2.2) / np.tanh(2.2)
    if rng.random() < ch["crush_p"]:
        hold = int(rng.integers(2, 6))
        y = np.repeat(y[::hold], hold, axis=0)[: len(y)]
    if rng.random() < ch["tilt_p"] and HAVE_SCIPY:
        fc = float(rng.uniform(900.0, 4000.0))
        al = float(np.exp(-2 * np.pi * fc / sr))
        y = np.stack([lfilter([1 - al], [1, -al], y[:, c]) for c in range(2)],
                     axis=1).astype(np.float32)
    return np.ascontiguousarray(y, np.float32)


def sub_thump(sr, vel, rng):
    """Synthesized 808-ish sub-bass drop — the 'real bass' layer."""
    f0 = float(rng.uniform(65, 95))
    dur = 0.16 + float(rng.uniform(0, 0.12))
    n = max(64, int(dur * sr))
    t = np.arange(n) / sr
    f = f0 * 0.5 + (f0 * 0.5) * np.exp(-t * 18)
    y = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 12) * vel * 0.65
    y = np.tanh(y * 1.5)
    out = np.stack([y, y], axis=1).astype(np.float32)
    f5 = min(64, n // 4)
    out[-f5:] *= np.linspace(1, 0, f5)[:, None]
    return out


# =========================================================== arrangement & render

@dataclass
class Section:
    bars: int
    bpm0: float
    bpm1: float
    swing: float
    sub_p: float
    exprs: dict = field(default_factory=dict)


def build_piece(rng, voices, pin, args):
    zone = (args.bpm, args.bpm) if args.bpm else BPM_ZONES[int(rng.integers(len(BPM_ZONES)))]
    n_sec = int(rng.integers(3, 6))
    sections, prev_exprs = [], {}
    for si in range(n_sec):
        bars = args.bars or int(rng.choice([4, 4, 6, 8]))
        if args.bpm:
            b0 = b1 = float(args.bpm)
        else:
            b0 = float(rng.uniform(zone[0], zone[1]))
            b1 = float(rng.uniform(zone[0], zone[1]))
        exprs = {}
        for v in voices:
            if v in pin:
                exprs[v] = pin[v]
                continue
            prof = VOICE_PROFILE[v]
            prev = prev_exprs.get(v)
            r = rng.random()
            if si == 0 or prev is None or r >= 0.95:
                exprs[v] = gen_pattern(rng, prof)
            elif r < 0.35:
                exprs[v] = prev                       # continuity
            elif r < 0.75:
                exprs[v] = mutate(prev, rng, prof)     # evolution
            else:
                exprs[v] = gen_pattern(rng, prof)      # fresh idea
        prev_exprs = exprs
        sections.append(Section(bars, b0, b1,
                                float(rng.uniform(0.0, 0.16)),
                                float(rng.uniform(0.25, 0.65)), exprs))
    return sections


def render_piece(sections, voices, bank, rng, sr, ch, drift):
    events, t0 = [], 0.0
    total_bars = sum(s.bars for s in sections)
    done, state = 0, {"hatpan": 1.0}
    for sec in sections:
        for b in range(sec.bars):
            prog = (done + b) / max(1, total_bars - 1)
            scale = drift[0] + (drift[1] - drift[0]) * prog
            bpm = (sec.bpm0 + (sec.bpm1 - sec.bpm0) * b / max(1, sec.bars - 1)) * scale
            stepdur = 60.0 / bpm / 4.0
            fill = (b == sec.bars - 1) and (done + b < total_bars - 1)
            for v in voices:
                pool = bank.get(v)
                if not pool:
                    continue
                try:
                    pat = normalize_velocities(ev(sec.exprs[v], rng))
                except Exception as e:
                    print(f"  warning: pattern for {v} failed ({e}), voice skipped")
                    sec.exprs[v] = ["euclid", 4, 16, 0]
                    continue
                L = len(pat)
                if L == 0:
                    continue
                hits = [(s, float(pat[(b * STEP + s) % L]))
                        for s in range(STEP)
                        if pat[(b * STEP + s) % L] is not None
                        and pat[(b * STEP + s) % L] > 0.02]
                if fill and v in ("snare", "clap", "tom") and rng.random() < 0.8:
                    hits += [(s, 0.45 + 0.35 * rng.random())
                             for s in range(8, STEP) if rng.random() < 0.45]
                if fill and v == "kick" and rng.random() < 0.5:
                    hits = [(s, val) for s, val in hits if s < 12]
                if b == 0 and v == "crash" and rng.random() < 0.5 and not any(
                        s == 0 for s, _ in hits):
                    hits.append((0, 0.8))
                for s, val in hits:
                    t = t0 + s * stepdur
                    if s % 2 == 1:
                        t += sec.swing * stepdur
                    t += rng.uniform(-1, 1) * ch["jit_ms"] / 1000.0
                    sm = pool[int(rng.integers(len(pool)))]
                    y = process_hit(sm.data, sr, rng, ch, v)
                    amp = val * 10.0 ** (rng.uniform(-1.2, 1.2) / 20.0)
                    y = y * gl.make_env("const", len(y))[:, None] * amp
                    if v == "hat":
                        state["hatpan"] = -state["hatpan"]
                        pan = 0.55 * state["hatpan"] + rng.uniform(-0.1, 0.1)
                    elif v in ("tex", "air", "ride", "shaker"):
                        pan = rng.uniform(-0.8, 0.8)
                    else:
                        pan = rng.uniform(-0.15, 0.15)
                    th = (pan + 1.0) * (np.pi / 4.0)
                    y[:, 0] *= np.cos(th)
                    y[:, 1] *= np.sin(th)
                    events.append((max(0, int(t * sr)), y))
                    if v == "kick" and rng.random() < sec.sub_p:
                        events.append((max(0, int(t * sr)), sub_thump(sr, val, rng)))
            t0 += STEP * stepdur
        done += sec.bars
    return events, t0


def make_bed(path, dur_s, sr, gain, rng):
    """Loop-tile an existing piece (e.g. a rhythms render) under the drums."""
    try:
        sr_b, x = gl.read_wav(path)
    except Exception:
        return None
    x = _resample_linear(_stereo(x), sr_b, sr)
    if len(x) < sr // 2:
        return None
    x = _highpass(x, sr, 100.0)                     # let the new kick own the sub
    need = int(dur_s * sr) + sr
    out = np.zeros((need, 2), np.float32)
    f = max(2, min(int(0.04 * sr), len(x) // 4))
    x = x.copy()
    x[:f] *= np.linspace(0, 1, f)[:, None]
    x[-f:] *= np.linspace(1, 0, f)[:, None]
    pos = 0
    step = max(1, len(x) - f)
    while pos < need:
        end = min(need, pos + len(x))
        out[pos:end] += x[: end - pos]
        pos += step
    return out * float(gain)


# =========================================================== CLI

def main():
    ap = argparse.ArgumentParser(
        description="Real drum one-shots driven by generated alien-lisp patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("sounds", help="folder of one-shot drum sample folders")
    ap.add_argument("-o", "--out", default="alien_rhythms")
    ap.add_argument("--corpus", default=None,
                    help="experimental corpus for tex/air texture voices (dir)")
    ap.add_argument("--bed", default=None,
                    help="wav/folder mixed underneath as a looped bed layer")
    ap.add_argument("--bed-gain", type=float, default=0.55)
    ap.add_argument("--pieces", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bpm", type=float, help="pin tempo")
    ap.add_argument("--bars", type=int, help="pin bars per section")
    ap.add_argument("--voice", dest="voices", action="append", default=[],
                    metavar="NAME=(expr)", help="pin a voice pattern (repeatable)")
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    rng = np.random.default_rng(args.seed)
    print(f"loading sample bank from {args.sounds} ...")
    bank = load_bank(args.sounds, rng)
    if not bank:
        ap.error("no usable samples found")
    print("bank: " + ", ".join(f"{r}x{len(v)}" for r, v in sorted(bank.items())))

    if args.corpus:
        cfiles = gl.gather_inputs([args.corpus])
        print(f"loading texture layer from {args.corpus} ({len(cfiles)} files)...")
        bank.update(load_texture(cfiles, rng))

    pin = {}
    for spec in args.voices:
        name, _, src = spec.partition("=")
        name = name.strip().lower()
        try:
            pin[name] = parse_pattern(src.strip())
        except ValueError as e:
            ap.error(f"bad --voice {spec!r}: {e}")

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

    core = [v for v in ("kick", "snare", "hat", "bass") if v in bank]
    extras = [v for v in ("clap", "rim", "ohat", "perc", "tom", "ride", "shaker",
                          "block", "blip", "vox", "tonal", "crash") if v in bank]
    if not core:
        ap.error("bank is missing kick/snare/hat/bass folders entirely")

    bed_files = []
    if args.bed:
        p = Path(args.bed)
        bed_files = sorted(p.glob("*.wav")) if p.is_dir() else [p]

    for i in range(args.pieces):
        pseed = args.seed + 7919 * i + 17
        prng = np.random.default_rng(pseed)
        n_extra = min(len(extras), int(2 + prng.integers(0, 4)))
        voices = core + [str(v) for v in prng.choice(
            np.array(extras, dtype=object), size=n_extra, replace=False)] \
            + [v for v in ("tex", "air") if v in bank]
        voices = list(dict.fromkeys(voices + [v for v in pin if v in bank]))
        ch = piece_character(prng)
        sections = build_piece(prng, voices, pin, args)
        drift = DRIFTS[int(prng.integers(len(DRIFTS)))] if not args.bpm else (1.0, 1.0)
        events, dur_s = render_piece(sections, voices, bank, prng, 44100, ch, drift)
        if not events:
            print(f"  alien{i:02d}: no events, skipped")
            continue
        if bed_files:
            bed = make_bed(bed_files[int(prng.integers(len(bed_files)))],
                           dur_s, 44100, args.bed_gain, prng)
            if bed is not None:
                events.append((0, bed))
        audio = gl.finalize(events, 44100, 0.97)
        wav = outdir / f"alien{i:02d}_{pseed}.wav"
        gl.write_wav(wav, 44100, audio)
        score = outdir / f"alien{i:02d}_{pseed}_score.txt"
        with open(score, "w") as f:
            f.write(f";; alien{i:02d} — seed {pseed} — {len(sections)} sections, "
                   f"drift {drift[0]:.2f}->{drift[1]:.2f}\n"
                   f";; voices: {' '.join(voices)}\n"
                   ";; paste any pattern into [alien] in Pure Data\n\n")
            for si, sec in enumerate(sections):
                f.write(f"[section {si + 1}] bars={sec.bars} "
                        f"bpm={sec.bpm0:.0f}->{sec.bpm1:.0f} swing={sec.swing:.2f}\n")
                for v in voices:
                    f.write(f"{v}: {to_src(sec.exprs[v])}\n")
                f.write("\n")
        print(f"  alien{i:02d} ({audio.shape[0] / 44100:5.1f}s) voices: {' '.join(voices)} "
              f"| {len(sections)} sections -> {wav.name} (+ score)")
        manifest[str(wav)] = dict(
            seed=pseed, drift=list(drift), character=ch, voices=voices,
            sections=[dict(bars=s.bars, bpm0=s.bpm0, bpm1=s.bpm1, swing=s.swing,
                           sub_p=s.sub_p,
                           exprs={v: to_src(s.exprs[v]) for v in voices})
                      for s in sections],
            score_file=str(score))

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"done -> {outdir}/ (each piece ships with a pasteable _score.txt)")


if __name__ == "__main__":
    sys.exit(main())