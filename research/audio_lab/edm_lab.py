#!/usr/bin/env python3
"""
edm_lab.py — dance-music final stage: genre skeletons + EDM arrangement.

The brief: four-to-the-floor that sometimes goes skatty and chaotic — but
ALWAYS in time, never freeform. Layered percussion from ALL existing sound
sources re-chunked. Pieces in house, grime, dub, dubstep and drum & bass
formats, with real EDM structure: intro -> build -> DROP -> break -> build
-> DROP -> outro.

How it works:
- VOICES. A real drum kit (kick/snare/clap/hat/ohat/rim/shaker/crash from
  audio_lab/sounds) carries the skeleton. On top: the whole corpus
  (trimmed, experiments, rhythms, alien_rhythms, flux_rhythms, storm) is
  onset-chunked and spectrally classified into percussion layers — dark
  pulses, mids, claps, bright air, tonal stabs — the "re-chunked" layer.
- ARRANGEMENT. Sections carry an energy curve; each voice has an energy
  threshold, so elements enter and leave musically. Builds ramp the filter
  open, fire a synthesized riser + reverse-air, and end in accelerating
  snare rolls + a beat of silence before the drop. Drops open with an
  impact hit. Drops 1 and 2 use pattern variants.
- SKATTY BARS. Inside drops, every few bars one bar mutates: re-rolled
  Euclidean hats/perc, 32nd ratchets on accents, pitch-ladder stab bursts,
  one stutter-gated beat — everything quantised to the grid.
- SUB BASS. A synthesized bassline per genre: house offbeat 8ths, grime
  stabs, dub long slides, dubstep wobble (tremolo), dnb notes under the
  kick. Root note chosen per piece.
- MIX GLUE. The texture/percussion bus is sidechain-ducked by the kick
  (the pumping) and rides a filter automation curve (breaks are dark,
  builds sweep open, drops are wide open).

Usage:
    python3 edm_lab.py -o audio_lab/edm_pieces                # 10 pieces, 2 per genre
    python3 edm_lab.py -o out --pieces 5 --seed 3              # one per genre
    python3 edm_lab.py -o out --genre dubstep --pieces 4 --bpm 140
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from scipy.signal import lfilter
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

import granular_lab as gl
import rhythm_lab as rl
import alien_rhythms as ar
import montage_lab as mg

SR = 44100
STEP = 16
GENRES = ("house", "grime", "dub", "dubstep", "dnb")
GENRE_BPM = {"house": (124, 128), "grime": (138, 142), "dub": (68, 74),
             "dubstep": (138, 142), "dnb": (170, 175)}
GENRE_SWING = {"house": 0.08, "grime": 0.20, "dub": 0.12, "dubstep": 0.10, "dnb": 0.03}

# voice -> (source, pool key, gain, pan) ; "corpus" voices ride the texture bus
VOICE = {
    "kick":   ("drum", "kick",   1.00, 0.00),
    "snare":  ("drum", "snare",  0.90, -0.05),
    "clap":   ("drum", "clap",   0.80, 0.10),
    "hat":    ("drum", "hat",    0.50, 0.00),
    "ohat":   ("drum", "ohat",   0.55, 0.00),
    "rim":    ("drum", "rim",    0.60, 0.15),
    "shaker": ("drum", "shaker", 0.45, -0.20),
    "crash":  ("drum", "crash",  0.70, 0.00),
    "csub":   ("corpus", "kick",  0.60, 0.00),
    "ctom":   ("corpus", "tom",   0.50, 0.20),
    "cperc":  ("corpus", "snare", 0.55, 0.10),
    "cclap":  ("corpus", "clap",  0.50, -0.10),
    "cair":   ("pad",   "bright", 0.40, 0.00),
    "cstab":  ("corpus", "stab",  0.60, 0.00),
    "ctex":   ("pad",   "dark",   0.55, 0.00),
}
#                min energy          sections the voice may play in
VOICE_ON = {
    "kick":   (0.30, {"intro", "build", "drop", "outro"}),
    "snare":  (0.45, {"build", "drop"}),
    "clap":   (0.45, {"build", "drop", "outro"}),
    "ohat":   (0.35, {"intro", "build", "drop", "outro"}),
    "hat":    (0.50, {"build", "drop"}),
    "rim":    (0.60, {"drop"}),
    "shaker": (0.60, {"drop"}),
    "csub":   (0.80, {"drop"}),
    "ctom":   (0.65, {"drop"}),
    "cperc":  (0.50, {"build", "drop", "break"}),
    "cclap":  (0.70, {"drop"}),
    "cair":   (0.30, {"intro", "build", "drop", "break", "outro"}),
    "cstab":  (0.40, {"intro", "build", "drop", "break", "outro"}),
    "ctex":   (0.15, {"intro", "build", "break", "outro"}),
}
CORPUS_ROLE_TARGET = {"kick": 0.22, "tom": 0.16, "snare": 0.14,
                      "clap": 0.13, "hat": 0.08, "stab": 0.12}


# --------------------------------------------------------------------------- synths

def load_pads(files, rng, cap=260, per_file=3):
    """Long sustained windows from the corpus, split dark/bright — the pad bed."""
    pads = []
    for f in files:
        info = sf.info(str(f))
        dur = info.frames / info.samplerate
        if dur < 3.0:
            continue
        for _ in range(per_file):
            d = float(rng.uniform(1.2, 2.5))
            start = 0.0 if dur <= d else float(rng.uniform(0, dur - d))
            with sf.SoundFile(str(f)) as fh:
                fh.seek(int(start * info.samplerate))
                y = fh.read(int(min(d, dur) * info.samplerate),
                            dtype="float32", always_2d=True)
            if len(y) < 44100:
                continue
            y = ar._stereo(y)
            if info.samplerate != SR:
                y = ar._resample_linear(y, info.samplerate, SR)
            ft = gl.compute_features(y, SR)
            if ft["rms"] < 0.004:
                continue
            s = ar.Sample(y, ft["rms"], "pad", Path(f).stem)
            s.bright = ft["centroid"]
            pads.append(s)
    if len(pads) > cap:
        pads = pads[:: int(np.ceil(len(pads) / cap))]
    order = np.argsort([p.bright for p in pads])
    cut = int(len(order) * 0.55)
    pools = dict(dark=[pads[i] for i in order[:cut]] or pads,
                bright=[pads[i] for i in order[cut:]] or pads)
    for pool in pools.values():
        for p in pool:
            g = min(20.0, 0.07 / max(1e-4, p.rms))
            p.data = np.ascontiguousarray(p.data * g, np.float32)
    return pools


def synth_sub(sr, f, dur, vel=1.0, wobble_hz=0.0, wobble_depth=0.0, slide_from=None):
    n = max(64, int(dur * sr))
    t = np.arange(n) / sr
    if slide_from:
        f_t = f + (slide_from - f) * np.exp(-t * 6)
    else:
        f_t = np.full(n, f)
    y = np.sin(2 * np.pi * np.cumsum(f_t) / sr)
    if wobble_hz > 0:
        y *= 1.0 - wobble_depth * (0.5 + 0.5 * np.sin(2 * np.pi * wobble_hz * t))
    env = np.minimum(1.0, t / 0.010) * np.exp(-np.maximum(0.0, t - dur * 0.75) * 8)
    y = np.tanh(y * 1.3) * env * vel * 0.8
    out = np.stack([y, y], axis=1).astype(np.float32)
    f5 = min(64, n // 4)
    out[-f5:] *= np.linspace(1, 0, f5)[:, None]
    return out


def synth_riser(sr, dur, rng):
    n = max(1024, int(dur * sr))
    t = np.arange(n) / sr
    env = (t / t[-1]) ** 2.4
    f0, f1 = 200.0, float(rng.uniform(1400, 2600))
    f_t = f0 * (f1 / f0) ** (t / t[-1])
    tone = 0.35 * np.sin(2 * np.pi * np.cumsum(f_t) / sr)
    noise = 0.22 * rng.standard_normal(n)
    c = 0.4 + 0.6 * (t / t[-1])
    y = (tone * c + noise) * env * 0.55
    out = np.stack([y, y * 0.9], axis=1).astype(np.float32)
    out[-int(0.01 * sr):] *= np.linspace(1, 0, int(0.01 * sr))[:, None]
    return out


def synth_impact(sr, rng):
    n = int(1.2 * sr)
    t = np.arange(n) / sr
    f_t = 70 * np.exp(-t * 9) + 28
    y = np.sin(2 * np.pi * np.cumsum(f_t) / sr) * np.exp(-t * 3.5)
    y += 0.25 * rng.standard_normal(n) * np.exp(-t * 22)
    out = np.stack([y, y], axis=1).astype(np.float32)
    return np.tanh(out * 1.6) * 0.9


# --------------------------------------------------------------------------- arrangement

@dataclass
class Section:
    type: str
    bars: int
    e0: float
    e1: float


def arrangement(rng, total_bars):
    """The classic EDM arc, scaled to the piece length."""
    props = [("intro", 0.10, 0.18, 0.32), ("build", 0.08, 0.35, 0.72),
             ("drop", 0.30, 0.92, 1.00), ("break", 0.14, 0.34, 0.50),
             ("build", 0.08, 0.40, 0.78), ("drop", 0.22, 0.94, 1.00),
             ("outro", 0.08, 0.38, 0.12)]
    secs = []
    for name, p, e0, e1 in props:
        bars = max(4, 4 * int(round(total_bars * p / 4)))
        secs.append(Section(name, bars, e0, e1))
    secs[-1].bars = max(4, total_bars - sum(s.bars for s in secs[:-1]))
    return secs


# --------------------------------------------------------------------------- patterns

def _eu(k, n, rot, vel, acc=True):
    bits = gl._euclid_bits(int(k), int(n))
    bits = bits[rot % n:] + bits[:rot % n]
    return [(i, 1.0 if (acc and i % 4 == 0) else vel, {}) for i, b in enumerate(bits) if b]


def bar_hits(genre, sec, e, chaos, roll, variant, rng):
    """One bar of hits per voice. Everything is grid-quantised 16ths."""
    hits = {v: [] for v in VOICE}
    rot = int(rng.integers(16)) * (2 if variant == 2 else 1)

    def add(v, s, vel=1.0, **m):
        hits[v].append((int(s), float(vel), m))

    # ---- kick skeleton (the law: four to the floor, in time)
    if genre == "house":
        for s in (0, 4, 8, 12):
            add("kick", s, 1.0)
        if chaos:
            add("kick", int(rng.choice([6, 10, 14])), 0.35)     # in-time ghost
    elif genre == "grime":
        add("kick", 0, 1.0)
        add("kick", 6, 0.9)
        if e > 0.7 or chaos:
            add("kick", 10, 0.7)
    elif genre == "dub":
        add("kick", 0, 1.0)
        if e > 0.7:
            add("kick", 10, 0.8)
    elif genre == "dubstep":
        add("kick", 0, 1.0)
        add("kick", 10, 0.95)
        if chaos and rng.random() < 0.5:
            add("kick", 4, 0.6)
    else:  # dnb
        add("kick", 0, 1.0)
        add("kick", 10, 0.95)
        if chaos:
            add("kick", int(rng.choice([6, 7, 13])), 0.5)

    # ---- backbeat / hats / percussion per genre
    if genre == "house":
        if e >= 0.45:
            for s in (4, 12):
                add("clap", s, 0.85)
                if e >= 0.7:
                    add("cclap", s, 0.5)                          # corpus thickens
        for s in (2, 6, 10, 14):
            add("ohat", s, 0.7)
        if e >= 0.5:
            hits["hat"] += _eu(9 + int(3 * e), 16, rot, 0.35)
        if e >= 0.6:
            hits["shaker"] += _eu(7 + int(3 * e), 16, rot + 2, 0.3)
        if e >= 0.55:
            hits["cperc"] += _eu(2 + int(4 * e), 16, rot + 4, 0.5)
        if e >= 0.4:
            add("cstab", 0, 0.55)                                # house stab
            if e >= 0.8:
                add("cstab", 8, 0.45)
        if e >= 0.8:
            hits["csub"] += _eu(3, 16, rot + 6, 0.5)
    elif genre == "grime":
        if e >= 0.45:
            add("snare", 8, 1.0)
            if e >= 0.7:
                add("clap", 8, 0.6)
        if e >= 0.5:
            hits["hat"] += _eu(5 + int(6 * e), 16, rot, 0.4)
        if e >= 0.6:
            hits["rim"] += _eu(2 + int(4 * e), 16, rot + 3, 0.5)
        if e >= 0.5:
            hits["cperc"] += _eu(3 + int(4 * e), 16, rot + 5, 0.5)
        add("cstab", 0, 0.6)
        if chaos:
            add("cstab", int(rng.choice([3, 11, 14])), 0.5,
                speed=float(rng.uniform(0.8, 1.4)))
        if e >= 0.8:
            hits["csub"] += _eu(2, 16, rot, 0.45)
    elif genre == "dub":
        if e >= 0.45:
            add("rim", 8, 0.8)                                   # cross-stick
        for s in (2, 6, 10, 14):
            add("cstab", s, 0.5, speed=float(rng.uniform(0.85, 1.15)))  # skank
        if e >= 0.5:
            hits["cperc"] += _eu(2 + int(3 * e), 16, rot, 0.45)
        if e >= 0.35:
            add("ohat", 4, 0.5)
            add("ohat", 12, 0.5)
        if e >= 0.8:
            hits["csub"] += _eu(2, 16, rot + 8, 0.5)
    elif genre == "dubstep":
        if e >= 0.45:
            add("snare", 8, 1.0)
        for s in (4, 12):
            add("ohat", s, 0.6)
        if e >= 0.5:
            hits["hat"] += _eu(3 + int(4 * e), 16, rot, 0.4)
        for s in (0, 4, 8, 12):                                  # wobble-synced stabs
            if e >= 0.5:
                add("cstab", s, 0.55, speed=1.5 ** (s // 4 if variant == 2 else 0))
        if e >= 0.55:
            hits["cperc"] += _eu(2 + int(3 * e), 16, rot + 6, 0.5)
        if e >= 0.8:
            hits["csub"] += _eu(3, 16, rot, 0.5)
    else:  # dnb
        if e >= 0.45:
            add("snare", 4, 1.0)
            add("snare", 12, 1.0)
            if chaos and rng.random() < 0.7:
                add("snare", int(rng.choice([7, 14, 15])), 0.3)  # ghost
        if e >= 0.5:
            hits["hat"] += _eu(11 + int(4 * e), 16, rot, 0.3)
        if e >= 0.6:
            hits["shaker"] += _eu(9 + int(4 * e), 16, rot + 2, 0.3)
        if e >= 0.55:
            hits["cperc"] += _eu(4 + int(5 * e), 16, rot + 5, 0.45)
        if e >= 0.7:
            add("cstab", 0, 0.5)
        if e >= 0.8:
            hits["csub"] += _eu(4, 16, rot + 7, 0.5)

    # ---- pad bed (dark corpus chunks, stretched) for the quieter sections
    if sec in ("intro", "build", "break", "outro"):
        add("ctex", 0, 0.8, dur=float(rng.uniform(0.9, 1.5)),
            speed=float(rng.uniform(0.35, 0.6)))
        if rng.random() < 0.5:                     # second pad for coverage
            add("ctex", 8, 0.7, dur=float(rng.uniform(0.9, 1.5)),
                speed=float(rng.uniform(0.35, 0.6)))
        if rng.random() < 0.3:                     # occasional bright wash
            add("cair", 8, 0.5, dur=float(rng.uniform(0.8, 1.2)),
                speed=float(rng.uniform(0.5, 0.8)))

    # ---- skatty bar: in-time chaos on the layered percussion
    if chaos:
        hits["hat"] = hits.get("hat", []) + _eu(int(rng.integers(6, 14)), 16,
                                                int(rng.integers(16)), 0.4)
        for v in ("cperc", "rim"):
            hits[v] = hits.get(v, []) + _eu(int(rng.integers(2, 7)), 16,
                                            int(rng.integers(16)), 0.5)
        add("snare" if e >= 0.5 else "cperc", 12, 0.8, ratchet=int(rng.integers(2, 5)))
        stab_burst = [(8 + 2 * j, 0.6, dict(speed=1.18 ** j)) for j in range(4)]
        hits["cstab"] = hits.get("cstab", []) + stab_burst
        if rng.random() < 0.6:                                   # stutter-gated beat
            s0 = int(rng.choice([4, 8]))
            hits["cperc"] += [(s0 + j, 0.55, {}) for j in range(4)]

    # ---- build rolls: 8ths then 16ths then 32nds, quantised, rising
    if roll == 1:
        hits["snare"] = [(s, 0.4 + 0.4 * s / 16, {}) for s in range(0, 16, 2)]
    elif roll >= 2:
        hits["snare"] = [(s, 0.45 + 0.45 * s / 16, {}) for s in range(0, 16)]
        hits["snare"] += [(s + 0.5, 0.5 + 0.5 * s / 16, {}) for s in range(0, 16)]
    return hits


def sub_line(genre, sec_type, e, chaos, rng, root, bar_i):
    """Synth bass notes: (step, dur_beats, semitone_offset, wobble_hz, slide)."""
    if sec_type in ("intro", "outro") and e < 0.5:
        return []
    if sec_type == "break" and genre == "house":
        return [(2, 2.0, 0, 0.0, None)]              # long filtered bass bed
    wob = float(rng.uniform(1.2, 2.4)) if genre == "dubstep" else 0.0
    if genre == "house":
        return [(s, 0.4, 0, 0.0, None) for s in (2, 6, 10, 14)]
    if genre == "grime":
        notes = [(0, 1.5, 0, 0.0, None)] if bar_i % 2 == 0 else [(8, 1.0, 3, 0.0, None)]
        return notes
    if genre == "dub":
        return [(0, 3.5, 0 if bar_i % 4 < 2 else -2, 0.0,
                 root if chaos and rng.random() < 0.3 else None)]
    if genre == "dubstep":
        off = [0, 0, -2, 3][bar_i % 4]
        return [(0, 3.9, off, wob, None), (8, 3.9, off, wob, None)]
    # dnb: long note under the kick, second under the 2.75 kick
    return [(0, 1.8, 0, 0.0, None), (10, 1.2, -2 if bar_i % 2 else 0, 0.0, None)]


# --------------------------------------------------------------------------- render

def _fade(y, s=0.004):
    f = min(len(y) // 4, int(s * SR))
    if f > 1:
        y[:f] *= np.linspace(0, 1, f)[:, None]
        y[-f:] *= np.linspace(1, 0, f)[:, None]
    return y


def render_piece(rng, genre, bpm, secs, bank, corpus, pads, chaos_dial):
    sixteenth = 60.0 / bpm / 4.0
    swing = GENRE_SWING[genre]
    root = float(rng.uniform(37.0, 49.0))
    drum_bus, tex_bus = [], []               # (t, y) event lists
    kick_times, meta = [], dict(chaos_bars=0, rolls=0)
    bar_i_global = 0
    chaos_every = max(2, int(round(8 - 5 * chaos_dial)))

    for si, sec in enumerate(secs):
        chaos_phase = int(rng.integers(chaos_every))
        for b in range(sec.bars):
            e = sec.e0 + (sec.e1 - sec.e0) * b / max(1, sec.bars - 1)
            chaos = (sec.type == "drop" and (b - chaos_phase) % chaos_every == 0
                     and rng.random() < 0.35 + 0.4 * chaos_dial)
            roll = 0
            if sec.type == "build":
                if b == sec.bars - 2:
                    roll = 1
                elif b == sec.bars - 1:
                    roll = 2
            variant = 2 if (sec.type == "drop" and si >= 4) else 1
            bar_t = bar_i_global * 16 * sixteenth
            hits = bar_hits(genre, sec.type, e, chaos, roll, variant, rng)
            if chaos:
                meta["chaos_bars"] += 1
            if roll:
                meta["rolls"] += 1
            for v, hl in hits.items():
                src, key, gain_v, pan_v = VOICE[v]
                if src == "drum":
                    pool = bank.get(key) or []
                    on_e, on_secs = VOICE_ON.get(v, (0.0, {"drop"}))
                elif src == "corpus":
                    pool = corpus.get(key) or []
                    on_e, on_secs = VOICE_ON.get(v, (0.0, {"drop"}))
                else:
                    pool = pads.get(key) or []
                    on_e, on_secs = VOICE_ON.get(v, (0.0, {"drop"}))
                if not pool or e < on_e or sec.type not in on_secs:
                    continue
                for s, vel, mods in hl:
                    if roll == 2 and s >= 12 and sec.type == "build" \
                            and b == sec.bars - 1:
                        continue
                    t = bar_t + s * sixteenth
                    if s % 2 == 1:
                        t += swing * sixteenth
                    t += rng.uniform(-1, 1) * 0.003
                    sm = pool[int(rng.integers(len(pool)))]
                    if src == "drum":
                        y = sm.data
                        if mods.get("speed", 1.0) != 1.0:
                            y = gl.resample_grain(y, mods["speed"], "hermite")
                        amp = gain_v * vel
                    elif src == "pad":
                        max_d = float(mods.get("dur", 1.5))
                        y = sm.data[: int(max_d * SR)]
                        if mods.get("speed", 1.0) != 1.0:
                            y = gl.resample_grain(y, mods["speed"], "hermite") \
                                [: int(max_d * SR)]
                        amp = gain_v * vel
                    else:
                        max_d = float(mods.get("dur", rl.ROLE_CFG[key]["max_dur"]))
                        y = sm.data[: int(max_d * SR)]
                        if mods.get("speed", 1.0) != 1.0:
                            y = gl.resample_grain(y, mods["speed"], "hermite") \
                                [: int(max_d * SR)]
                        amp = min(12.0, CORPUS_ROLE_TARGET[key] /
                                  max(1e-4, sm.rms)) * gain_v * vel
                    y = _fade(y.copy()) * amp
                    th = ((pan_v + rng.uniform(-0.1, 0.1)) + 1) * np.pi / 4
                    y[:, 0] *= np.cos(th)
                    y[:, 1] *= np.sin(th)
                    rat = int(mods.get("ratchet", 1))
                    for j in range(rat):
                        tt = t + j * sixteenth / 2 / max(1, rat)
                        (tex_bus if src != "drum" else drum_bus).append(
                            (max(0, int(tt * SR)), y * (1.0 - 0.2 * j)))
                    if v == "kick" and vel >= 0.5:
                        kick_times.append(t)
            # ---- synth sub bass (gated by the pre-drop silence like drums)
            for s, dur_b, semi, wob, slide in sub_line(genre, sec.type, e, chaos,
                                                       rng, root, b):
                if roll == 2 and s >= 12:
                    continue
                f = root * 2 ** (semi / 12)
                y = synth_sub(SR, f, dur_b * 4 * sixteenth, vel=0.9,
                              wobble_hz=wob, wobble_depth=0.7 if wob else 0.0,
                              slide_from=(slide if slide else None))
                tex_bus.append((max(0, int((bar_t + s * sixteenth) * SR)), y))
            bar_i_global += 1
        # ---- section-level FX
        if sec.type == "build":
            dur = sec.bars * 16 * sixteenth
            t0 = (bar_i_global - sec.bars) * 16 * sixteenth
            riser = synth_riser(SR, dur - sixteenth * 4, rng)   # ends 1 beat early
            drum_bus.append((int(t0 * SR), riser))
        if sec.type == "drop":
            t0 = (bar_i_global - sec.bars) * 16 * sixteenth
            drum_bus.append((int(t0 * SR), synth_impact(SR, rng)))
            if "crash" in bank:
                pool = bank["crash"]
                sm = pool[int(rng.integers(len(pool)))]
                drum_bus.append((int(t0 * SR), _fade(sm.data.copy()[: int(1.6 * SR)]) * 0.8))
    meta["root_hz"] = root
    return drum_bus, tex_bus, kick_times, meta


# --------------------------------------------------------------------------- mix glue

def assemble(events, total_s):
    L = int(total_s * SR) + SR * 2
    out = np.zeros((L, 2), np.float32)
    for t, y in events:
        i0 = int(t)
        if i0 >= L:
            continue
        end = min(L, i0 + len(y))
        out[i0:end] += y[: end - i0]
    return out


def duck(tex, kick_times, depth=0.6, tau=0.16):
    """Sidechain-style ducking of the texture bus under the kick."""
    n = len(tex)
    env = np.ones(n, np.float32)
    t = np.arange(n) / SR
    for tk in kick_times:
        i0 = int(tk * SR)
        i1 = min(n, i0 + int(0.4 * SR))
        if i0 >= n:
            continue
        seg_t = t[i0:i1] - tk
        d = np.exp(-seg_t / tau)
        env[i0:i1] = np.minimum(env[i0:i1], 1.0 - depth * d)
    return tex * env[:, None]


def filter_curve(tex, cutoffs):
    """Per-bar 2-pole lowpass with equal-power crossfaded chunks."""
    if not HAVE_SCIPY or not cutoffs or len(tex) == 0:
        return tex
    n = len(tex)
    edges = np.linspace(0, n, len(cutoffs) + 1).astype(int)
    xf = int(0.008 * SR)
    out = np.zeros_like(tex)
    for i, cut in enumerate(cutoffs):
        i0, i1 = edges[i], edges[i + 1]
        if i1 <= i0:
            continue
        lo, hi = max(0, i0 - xf), min(n, i1 + xf)
        seg = tex[lo:hi]
        if len(seg) == 0:
            continue
        al = float(np.exp(-2 * np.pi * min(cut, 18000.0) / SR))
        y = seg
        for _ in range(2):
            y = np.stack([lfilter([1 - al], [1, -al], y[:, c]) for c in range(2)],
                         axis=1).astype(np.float32)
        w = np.ones(len(seg), np.float32)
        fin = int(min(xf, i0 - lo))
        fout = int(min(xf, hi - i1))
        if fin > 1:
            w[:fin] = np.linspace(0, 1, fin)
        if fout > 1:
            w[-fout:] = np.linspace(1, 0, fout)
        out[lo:hi] += y * w[:, None]
    return out


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(
        description="EDM final stage: genre skeletons + arrangement engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", default="audio_lab")
    ap.add_argument("--sounds", default="audio_lab/sounds")
    ap.add_argument("-o", "--out", default="edm_pieces")
    ap.add_argument("--pieces", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--genre", choices=GENRES, help="pin the genre")
    ap.add_argument("--bpm", type=float, help="pin tempo")
    ap.add_argument("--chaos", type=float, default=0.6,
                    help="skatty-bar frequency/intensity 0..1")
    ap.add_argument("--min-len", type=float, default=190.0)
    ap.add_argument("--max-len", type=float, default=235.0)
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    rng = np.random.default_rng(args.seed)
    print("loading drum bank ...")
    bank = ar.load_bank(args.sounds, rng)
    print("bank: " + ", ".join(f"{r}x{len(v)}" for r, v in sorted(bank.items())))

    files = mg.discover(args.base, Path(args.out).name)
    files = [Path(f) for f in files if Path(f).parent.name not in
             ("final_pieces", "sounds", "edm_pieces")]
    print(f"re-chunking {len(files)} corpus pieces ...")
    sr, chunks = rl.extract_pool(files, rng, pool_cap=7000)
    pools = rl.assign_roles(chunks)
    corpus = {k: pools[k] for k in ("kick", "tom", "snare", "clap", "hat", "stab")
              if pools.get(k)}
    print("corpus layers: " + ", ".join(f"{k}x{len(v)}" for k, v in corpus.items()))
    print("loading pad bed ...")
    pads = load_pads(files, rng)

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

    for i in range(args.pieces):
        pseed = args.seed + 7919 * i + 53
        prng = np.random.default_rng(pseed)
        genre = args.genre or GENRES[i % len(GENRES)]
        bpm = args.bpm or float(prng.uniform(*GENRE_BPM[genre]))
        bar = 16 * 60.0 / bpm / 4.0
        total_bars = int(prng.uniform(args.min_len, args.max_len) / bar)
        secs = arrangement(prng, total_bars)
        drum_bus, tex_bus, kick_times, meta = render_piece(
            prng, genre, bpm, secs, bank, corpus, pads, args.chaos)
        total_s = total_bars * 16 * 60.0 / bpm / 4.0
        drums = assemble(drum_bus, total_s)
        tex = assemble(tex_bus, total_s)
        tex = duck(tex, kick_times)
        cutoffs = []
        for s in secs:                                   # filter automation per bar
            for b in range(s.bars):
                e = s.e0 + (s.e1 - s.e0) * b / max(1, s.bars - 1)
                if s.type in ("drop",):
                    cutoffs.append(18000.0)
                elif s.type == "build":
                    cutoffs.append(900.0 + 16000.0 * (b / max(1, s.bars - 1)))
                elif s.type == "break":
                    cutoffs.append(1600.0 + 2400.0 * e)
                else:
                    cutoffs.append(700.0 + 2600.0 * e)
        tex = filter_curve(tex, cutoffs)
        audio = drums + tex
        f = int(0.02 * SR)
        audio[:f] *= np.linspace(0, 1, f)[:, None]
        audio[-f:] *= np.linspace(1, 0, f)[:, None]
        pk = float(np.abs(audio).max())
        if pk > 1e-4:
            audio *= 0.97 / pk
        wav = outdir / f"edm{i:02d}_{genre}_{pseed}.wav"
        gl.write_wav(wav, SR, audio)
        print(f"  edm{i:02d} {genre:8s} {bpm:5.0f}bpm ({audio.shape[0] / SR / 60:.1f} min, "
              f"{total_bars} bars, root {meta['root_hz']:.0f}Hz, "
              f"{meta['chaos_bars']} skatty bars, {meta['rolls']} rolls) -> {wav.name}")
        manifest[str(wav)] = dict(
            seed=pseed, genre=genre, bpm=bpm, bars=total_bars,
            root_hz=meta["root_hz"], chaos_dial=args.chaos,
            sections=[dict(type=s.type, bars=s.bars, e0=s.e0, e1=s.e1) for s in secs])

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"done -> {outdir}/")


if __name__ == "__main__":
    sys.exit(main())