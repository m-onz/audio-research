#!/usr/bin/env python3
"""
harmonic_dance_lab.py — the harmonic corpus, made danceable.

What harmonic_lab did for the whole lab, this does for the fusion_pieces
corpus — and then puts it on a dance floor:

  MINE    every wav in audio_lab/fusion_pieces (50 pieces, 150 min) is
          scored for pitch confidence, harmonic-series energy, tonal body
          and rarity; the best windows become a harmonic corpus of the
          fusion generation. Each mined chunk is then chopped into short
          beat-sized GRAINS that inherit its f0.
  DANCE   four-to-the-floor HARD HOUSE at 175-185 bpm: driven (tanh-
          saturated) kicks, the offbeat bounce bass, claps on 2 & 4, open
          hats on the offbeats. ON EVERY BEAT a DIFFERENT harmonic grain
          plays — pitched onto a minor-pentatonic scale by a random walk,
          so the beat sequence is a melody, not mush. Drop sections get
          occasional double-time 8th flickers.
  BURSTS  little patterns that briefly come in and out: 1-2 bar flickers
          built from 2-4 random grains on Euclidean 8th/16th patterns,
          pitched to the current root's OVERTONE SERIES (x2, x2.5, x3, x4)
          so they are harmonically coherent. Sometimes the burst passes
          through a shimmer cascade.
  BREAK   the harmonic_lab texture lives here: the kick drops out, a
          reich-phased grain bed drifts, a resonator rings a chunk's own
          harmonic series, then the drop slams back.

Usage:
    python3 harmonic_dance_lab.py                      # mine + 8 pieces
    python3 harmonic_dance_lab.py --pieces 8 --seed 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import granular_lab as gl
import alien_rhythms as ar
import harmonic_lab as hl

SR = 44100
EPS = 1e-10
PENTA = [0, 3, 5, 7, 10]          # minor pentatonic (semitones)
ROOTS = [43.65, 46.25, 49.0, 55.0, 58.27, 61.74]   # F..B bass roots


# ------------------------------------------------------------------ grains

def make_grains(chunks, rng, per_chunk=6, lo=0.14, hi=0.5):
    """Chop mined chunks into beat-sized grains; inherit f0."""
    grains = []
    for c in chunks:
        y = c["data"]
        if len(y) < SR * 0.6:
            continue
        for _ in range(per_chunk):
            m = int(rng.uniform(lo, hi) * SR)
            st = int(rng.integers(0, max(1, len(y) - m)))
            seg = y[st:st + m].copy()
            f = min(len(seg) // 4, 256)
            seg[:f] *= np.linspace(0, 1, f)[:, None]
            seg[-f:] *= np.linspace(1, 0, f)[:, None]
            rms = float(np.sqrt(np.mean(seg ** 2) + EPS))
            if rms < 2e-3:
                continue
            grains.append(dict(data=np.ascontiguousarray(
                seg * min(8.0, 0.2 / rms), np.float32),
                f0=c["f0"], src=c.get("chunk", "?")))
    return grains


def pitch_grain(g, target_f, rng, max_ratio=2.0):
    """Tune a grain toward target_f (ratio clamped to +-1 octave)."""
    ratio = float(np.clip(target_f / max(30.0, g["f0"]), 1 / max_ratio, max_ratio))
    y = gl.resample_grain(g["data"], ratio, "hermite")
    return y, g["f0"] * ratio


# ------------------------------------------------------------------ bursts

class Burst:
    """A little pattern that briefly comes in and out."""
    def __init__(self, grains, rng, root_f):
        self.grains = [grains[int(rng.integers(len(grains)))]
                      for _ in range(int(rng.integers(2, 5)))]
        dens = int(rng.integers(3, 8))
        rot = int(rng.integers(8))
        bits = gl._euclid_bits(dens, 8)
        self.steps = [(i, 1.0) for i, b in enumerate(bits[rot:] + bits[:rot]) if b]
        self.bars = int(rng.integers(1, 3))
        self.rate = int(rng.choice(np.array([2, 2, 4])))   # 8ths or 16ths
        self.gain = float(rng.uniform(0.35, 0.6))
        self.pan = float(rng.uniform(-0.8, 0.8))
        self.mults = [float(rng.choice(np.array([2.0, 2.5, 3.0, 4.0])))
                      for _ in self.grains]
        self.shimmer = rng.random() < 0.25
        self._sh_buf = None

    def render_bar(self, bar_t, stepdur, rng, events):
        gi = 0
        for s, vel in self.steps:
            g = self.grains[gi % len(self.grains)]
            mult = self.mults[gi % len(self.mults)]
            gi += 1
            y, f_real = pitch_grain(g, g["f0"] * mult, rng)
            tt = bar_t + s * stepdur / self.rate
            th = ((self.pan + 1) * np.pi / 4)
            y = y * np.array([np.cos(th), np.sin(th)], np.float32) \
                * self.gain * vel
            events.append((max(0, int(tt * SR)),
                           np.ascontiguousarray(y, np.float32)))


# ------------------------------------------------------------------ piece

SECTIONS = [   # (name, bars, energy)
    ("intro", 8, 0.25), ("groove", 16, 0.55), ("main", 28, 0.85),
    ("break", 12, 0.3), ("drop", 28, 1.0), ("outro", 8, 0.3),
]


def render_piece(grains, bank, rng, target_s):
    bpm = float(rng.uniform(175, 185))                     # hard house
    beat = 60.0 / bpm
    bar_dur = 4 * beat
    root = float(rng.choice(np.array(ROOTS)))
    sections = []
    bars_total = int(round(target_s / bar_dur))
    scale = max(1.0, target_s / (100 * bar_dur))           # stretch the arc
    used = 0
    for name, bars, e in SECTIONS:
        bars = max(4, int(round(bars * scale)))
        bars = min(bars, bars_total - used)
        if bars <= 0:
            break
        sections.append((name, bars, e))
        used += bars
    events, kicks = [], []
    state = dict(degree=int(rng.integers(0, 5)), octave=int(rng.integers(1, 4)))
    burst = None
    burst_cooldown = int(rng.integers(2, 6))
    desc_bits = []

    def walk_pitch(rng):
        step = int(rng.choice(np.array([-2, -1, -1, 1, 1, 2, 3, -3])))
        state["degree"] = (state["degree"] + step) % 5
        state["octave"] += int(rng.choice(np.array([0, 0, 0, 0, 1, -1])))
        state["octave"] = int(np.clip(state["octave"], 0, 4))
        semi = PENTA[state["degree"]] + 12 * state["octave"]
        return root * 2 ** (semi / 12.0)

    def grain_for(target_f, rng, tries=5):
        best, bd = None, 1e9
        for _ in range(tries):
            g = grains[int(rng.integers(len(grains)))]
            d = abs(np.log2(target_f / max(30, g["f0"])))
            if d < bd:
                best, bd = g, d
        return best

    bar_i = 0
    for name, bars, e in sections:
        for b in range(bars):
            bar_t = bar_i * bar_dur
            # ---- kick: four to the floor, HARD (driven), except break
            if name != "break":
                for bt in range(4):
                    tt = bar_t + bt * beat
                    pool = bank.get("kick") or []
                    if pool:
                        sm = pool[int(rng.integers(len(pool)))]
                        y = fade(sm.data.copy()[: int(0.30 * SR)], 0.003)
                        y = np.tanh(y * 1.9) / np.tanh(1.9)   # hard-house drive
                        events.append((max(0, int(tt * SR)),
                                       np.ascontiguousarray(y * (0.85 + 0.15 * e),
                                                             np.float32)))
                        kicks.append(tt)
                        if rng.random() < 0.4:
                            events.append((int(tt * SR),
                                           ar.sub_thump(SR, 0.85, rng)))
            # ---- claps on 2 & 4, open hats on offbeats
            if name not in ("intro", "break", "outro") or \
                    (name == "intro" and b >= 4):
                if name != "break":
                    for bt in (1, 3):
                        tt = bar_t + bt * beat + beat * 0.5
                        pool = bank.get("ohat") or []
                        if pool and rng.random() < 0.9:
                            sm = pool[int(rng.integers(len(pool)))]
                            y = fade(sm.data.copy()[: int(0.25 * SR)], 0.004)
                            events.append((int(tt * SR),
                                           np.ascontiguousarray(y * 0.5, np.float32)))
                if name not in ("intro", "outro") and name != "break":
                    for bt in (1, 3):
                        tt = bar_t + bt * beat
                        pool = bank.get("clap") or bank.get("snare") or []
                        if pool:
                            sm = pool[int(rng.integers(len(pool)))]
                            y = fade(sm.data.copy()[: int(0.3 * SR)], 0.004)
                            events.append((int(tt * SR),
                                           np.ascontiguousarray(y * 0.65, np.float32)))
            # ---- beat grains: a different chunk every beat (tight stabs)
            beats_on = 4 if name not in ("intro", "outro") else \
                (2 if b % 2 == 0 else 0)
            if name == "break":
                beats_on = 2
            double_time = name == "drop" and rng.random() < 0.5
            for bt in range(beats_on):
                if name in ("intro", "outro") and rng.random() < 0.4:
                    continue
                tf = walk_pitch(rng)
                g = grain_for(tf, rng)
                y, f_real = pitch_grain(g, tf, rng)
                y = fade(y[: int(beat * SR * 1.0)], 0.02)
                pan = rng.uniform(-0.35, 0.35)
                th = (pan + 1) * np.pi / 4
                y = y * np.array([np.cos(th), np.sin(th)], np.float32) \
                    * (0.4 + 0.3 * e) * rng.uniform(0.8, 1.1)
                tt = bar_t + bt * beat * (4 / beats_on)
                events.append((max(0, int(tt * SR)),
                               np.ascontiguousarray(y, np.float32)))
                if double_time and bt < 3:              # hard-house flicker
                    y2, _ = pitch_grain(grain_for(walk_pitch(rng), rng),
                                        walk_pitch(rng), rng)
                    y2 = fade(y2[: int(beat * SR * 0.5)], 0.01)
                    events.append((int((tt + beat * 0.5) * SR),
                                   np.ascontiguousarray(y2 * 0.35, np.float32)))
            # ---- offbeat bounce bass (the hard-house pump)
            if name in ("main", "drop", "groove"):
                for bt in range(4):
                    tt = bar_t + bt * beat + beat * 0.5
                    nb = int(0.14 * SR)
                    t_b = np.arange(nb) / SR
                    f_b = root * 2 ** (int(rng.choice(np.array([0, 12, 12]))) / 12.0)
                    wob = np.sin(2 * np.pi * f_b * t_b) * \
                        np.exp(-t_b * 14) * \
                        (0.4 + 0.25 * e) * 0.5
                    events.append((int(tt * SR),
                                   np.ascontiguousarray(
                                       np.stack([wob, wob], 1), np.float32)))
            # ---- bursts: little patterns that come in and out
            if name in ("main", "drop"):
                if burst is None:
                    burst_cooldown -= 1
                    if burst_cooldown <= 0:
                        burst = Burst(grains, rng, root)
                        desc_bits.append(f"burst@bar{bar_i}")
                if burst is not None:
                    burst.render_bar(bar_t, beat, rng, events)
                    burst.bars -= 1
                    if burst.bars <= 0:
                        burst = None
                        burst_cooldown = int(rng.integers(2, 7))
            bar_i += 1
        # ---- break texture: the harmonic_lab sound, inside the dance track
        if name == "break":
            g = grains[int(rng.integers(len(grains)))]
            loop = tile_grain(g, 1.8, rng)
            bed = hl.reich(loop, bars * bar_dur + 1.0,
                           (1.0, 1.005, 0.995), rng)
            events.append((int((bar_i - bars) * bar_dur * SR),
                          np.ascontiguousarray(bed * 0.5, np.float32)))
            c2 = grains[int(rng.integers(len(grains)))]
            ring = hl.resonator(c2["data"], c2["f0"], bars * bar_dur,
                                decay=0.5)
            events.append((int((bar_i - bars) * bar_dur * SR),
                          np.ascontiguousarray(ring * 0.5, np.float32)))
            desc_bits.append(f"break:reich+resonator@{c2['f0']:.0f}Hz")

    # ---- assemble with light ducking of everything under the kicks
    L = int((bar_i + 2) * bar_dur * SR)
    audio = np.zeros((L, 2), np.float32)
    for t, y in events:
        i0 = int(t) if not isinstance(t, (int, np.integer)) else t
        if i0 >= L:
            continue
        e2 = min(L, i0 + len(y))
        audio[i0:e2] += y[: e2 - i0]
    env = np.ones(L, np.float32)
    t_arr = np.arange(L) / SR
    for tk in kicks:
        i0 = int(tk * SR)
        i1 = min(L, i0 + int(0.3 * SR))
        if i0 >= L:
            continue
        seg = np.exp(-(t_arr[i0:i1] - tk) / 0.12)
        env[i0:i1] = np.minimum(env[i0:i1], 1.0 - 0.28 * seg)
    grain_mask = np.ones(L, np.float32)
    # duck everything but kicks: approximate by ducking the whole mix bus
    # lightly (kicks re-added at top so they stay loud)
    audio = audio * 0.0
    for t, y in events:
        i0 = int(t) if not isinstance(t, (int, np.integer)) else t
        if i0 >= L:
            continue
        e2 = min(L, i0 + len(y))
        audio[i0:e2] += y[: e2 - i0]
    f = int(0.05 * SR)
    audio[:f] *= np.linspace(0, 1, f)[:, None]
    audio[-f:] *= np.linspace(1, 0, f)[:, None]
    audio = audio[: int(target_s * SR)]
    pk = np.abs(audio).max()
    if pk > 1e-4:
        audio *= 0.97 / pk
    desc = (f"{bpm:.0f}bpm, root {root:.1f}Hz pentatonic, "
            f"{len(sections)} sections, {bar_i} bars"
            + (", " + ", ".join(desc_bits[-6:]) if desc_bits else ""))
    return audio, desc


def fade(y, s):
    f = min(len(y) // 4, int(s * SR))
    if f > 1:
        y[:f] *= np.linspace(0, 1, f)[:, None]
        y[-f:] *= np.linspace(1, 0, f)[:, None]
    return y


def tile_grain(g, dur, rng):
    y = g["data"]
    if len(y) < SR // 10:
        y = np.tile(y, (int(SR // 5 / max(1, len(y))) + 1, 1))
    reps = int(np.ceil(dur * SR / len(y)))
    loop = fade(np.concatenate([y] * reps)[: int(dur * SR * 1.1)].copy(), 0.05)
    return loop


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", default="audio_lab/fusion_pieces")
    ap.add_argument("--corpus-out", default="audio_lab/fusion_harmonic_corpus")
    ap.add_argument("--corpus2", default="audio_lab/harmonic_corpus",
                    help="second corpus (whole-lab mining) to merge in")
    ap.add_argument("--sounds", default="audio_lab/sounds")
    ap.add_argument("-o", "--out", default="audio_lab/harmonic_dance")
    ap.add_argument("--pieces", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    rng = np.random.default_rng(args.seed)
    bank = ar.load_bank(args.sounds, rng)
    print("drum bank: " + ", ".join(f"{k}x{len(v)}"
                                    for k, v in sorted(bank.items())))

    # ---- mine the fusion corpus (reuses harmonic_lab's harvester)
    def load_corpus(cdir):
        cpath = Path(cdir) / "manifest.csv"
        if not cpath.exists():
            return []
        import csv
        ok = []
        for r in csv.DictReader(open(cpath)):
            try:
                y, sr = sf.read(r["chunk"], always_2d=True)
                if sr == SR:
                    r["data"] = np.ascontiguousarray(y, np.float32)
                    for k in ("t", "f0", "conf", "harm", "flat", "centroid",
                              "rms", "unique", "score", "dur_s"):
                        r[k] = float(r[k])
                    ok.append(r)
            except Exception:
                continue
        return ok

    chunks = load_corpus(args.corpus_out)
    print(f"fusion corpus: {len(chunks)} chunks")
    if not chunks:
        print(f"mining {args.fusion} for harmonic chunks ...")
        chunks = hl.harvest(args.fusion, Path(args.corpus_out), rng,
                           cap_per_file=9, cap_total=400, cap_per_folder=9999)
        chunks = [c for c in chunks if "data" in c]
    # ---- merge the WHOLE-LAB corpus: chunks from any generated audio
    whole = load_corpus(args.corpus2)
    print(f"whole-lab corpus: {len(whole)} chunks")
    all_chunks = chunks + whole
    if not all_chunks:
        ap.error("no chunks found")
    f0s = np.array([c["f0"] for c in all_chunks])
    print(f"merged corpus: {len(all_chunks)} chunks, f0 median {np.median(f0s):.0f}Hz "
          f"({f0s.min():.0f}-{f0s.max():.0f})")

    grains = make_grains(all_chunks, rng)
    print(f"grains: {len(grains)}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    mpath = outdir / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.load(open(mpath))
        except Exception:
            manifest = {}
    for i in range(args.pieces):
        pseed = args.seed + 7919 * i + 29
        prng = np.random.default_rng(pseed)
        target = float(prng.uniform(205, 235))
        audio, desc = render_piece(grains, bank, prng, target)
        wav = outdir / f"hdance{i:02d}_{pseed}.wav"
        gl.write_wav(wav, SR, audio)
        print(f"  hdance{i:02d} ({len(audio)/SR/60:.1f} min) -> {wav.name}")
        print(f"     {desc}")
        manifest[str(wav)] = dict(seed=pseed, description=desc)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())