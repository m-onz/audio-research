#!/usr/bin/env python3
"""
montage_lab.py — final assembly: long-form pieces from everything so far.

Takes the whole audio_lab output (trimmed recordings, experiments, rhythms,
alien_rhythms, flux_rhythms ...) as its palette and builds 3-4 minute pieces:
passages are lifted from random moments of random pieces, played at different
speeds (tape-style, pitch+time), and interleaved with creative structure.

Each piece is a sequence of MOVEMENTS (30-90s), each with its own strategy:

  flow          long passages (12-40s) chained with wide crossfades; speeds
                follow a piece-wide arc (fast collage -> slow melt, etc.)
  ab_weave      two sources alternate in a call-and-response weave, one slow
                and dark, the other fast and bright
  bursts        one long background passage with short high-speed interjections
                cut on top, placed on a generated Euclidean grid (alien lisp)
  stutter_echo  each passage repeats 2-4 times as an accelerating speed ladder
                with decaying gain - echo-takes of itself
  drift_layer   a half-speed whole-piece loop drifting under a foreground flow

Segments are chosen by energy (lively moments preferred) and chained by a
per-piece transition policy: 'similar' brightness, 'contrast', or random.

Usage:
    python3 montage_lab.py -o audio_lab/final_pieces
    python3 montage_lab.py -o out --pieces 8 --seed 3
    python3 montage_lab.py -o out --min-len 180 --max-len 240
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

import granular_lab as gl
import alien_rhythms as ar

try:
    from scipy.signal import lfilter
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

SR = 44100
EXCLUDE_DIRS = {"sounds", "bank", "__pycache__"}
STRATEGIES = ["flow", "ab_weave", "bursts", "stutter_echo", "drift_layer"]


# --------------------------------------------------------------------------- corpus

def discover(base, outname):
    files = []
    for sub in sorted(Path(base).iterdir()):
        if not sub.is_dir() or sub.name in EXCLUDE_DIRS or sub.name == outname:
            continue
        files += sorted(str(q) for q in sub.glob("*.wav"))
    return files


class Slicer:
    """LRU-cached, seek-based segment picker with an energy bias."""

    def __init__(self, rng, cap_bytes=700e6):
        self.rng = rng
        self.cap = cap_bytes
        self.cache = OrderedDict()          # path -> (data, sr)

    def _load(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        sr, a = gl.read_wav(path)
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        elif a.shape[1] > 2:
            a = a[:, :2]
        self.cache[path] = (np.ascontiguousarray(a, np.float32), int(sr))
        # trim old entries, but never evict the one we just loaded
        while len(self.cache) > 1 and sum(v[0].nbytes for v in self.cache.values()) > self.cap:
            self.cache.popitem(last=False)
        return self.cache[path]

    def pick(self, path, seg_s, prev_cent=None, policy="random"):
        """Return (y, centroid). Prefers energetic windows; policy orders by
        spectral similarity/contrast to the previous passage."""
        rng = self.rng
        data, sr = self._load(path)
        dur = len(data) / sr
        want = int(seg_s * sr)
        cands = []
        if dur <= seg_s * 1.3:
            cands.append((data, 0))
        else:
            k = 4 if dur > 240 else 3
            starts = sorted(rng.integers(0, max(1, len(data) - want), size=k))
            for st in starts:
                cands.append((data[int(st): int(st) + want], float(st) / sr))
        scored = []
        for y, st in cands:
            if len(y) < 1024 or len(y) < int(seg_s * sr * 0.5):
                continue
            f = gl.compute_features(y, sr)
            scored.append((y, f["rms"], f["centroid"]))
        if not scored:
            return None, None
        if policy == "similar" and prev_cent is not None:
            scored.sort(key=lambda s: abs(s[2] - prev_cent))
        elif policy == "contrast" and prev_cent is not None:
            scored.sort(key=lambda s: -abs(s[2] - prev_cent))
        else:
            w = np.array([s[1] ** 2 for s in scored])
            if w.sum() <= 0:
                pass
            else:
                w /= w.sum()
                idx = int(rng.choice(len(scored), p=w))
                scored = [scored[idx]] + [s for i, s in enumerate(scored) if i != idx]
        y, _, cent = scored[0]
        return np.ascontiguousarray(y, np.float32), cent


# --------------------------------------------------------------------------- helpers

def seg_process(y, rng, ch):
    """Light per-passage processing: occasional reverse / drive / tilt."""
    if rng.random() < ch["rev_p"] and len(y) > SR // 2:
        y = y[::-1].copy()
    if rng.random() < ch["drive_p"]:
        y = np.tanh(y * 1.8) / np.tanh(1.8)
    if rng.random() < ch["tilt_p"] and HAVE_SCIPY and len(y):
        fc = float(rng.uniform(1500.0, 6000.0))
        al = float(np.exp(-2 * np.pi * fc / SR))
        y = np.stack([lfilter([1 - al], [1, -al], y[:, c]) for c in range(2)],
                     axis=1).astype(np.float32)
    return y


def montage_character(rng):
    return dict(rev_p=float(rng.uniform(0.03, 0.10)),
                drive_p=float(rng.uniform(0.05, 0.20)),
                tilt_p=float(rng.uniform(0.05, 0.25)))


def _fade_env(n, xf):
    """Raised-cosine fades of xf samples at both ends."""
    e = np.ones(n, np.float32)
    f = int(min(xf, n // 2))
    if f > 1:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(f) / f)
        e[:f] = ramp
        e[-f:] = ramp[::-1]
    return e


class Piece:
    """Unified overlap-add event stream: chain events advance a cursor and
    crossfade; overlay events land at absolute times."""

    def __init__(self):
        self.events = []            # (t_samples, y, fade_samples)
        self.cursor = 0.0
        self.n_chain = 0

    def chain(self, y, xf_s):
        xf = int(xf_s * SR)
        t = 0 if self.n_chain == 0 else max(0, self.cursor - xf_s)
        self.events.append((int(t * SR), y, xf))
        self.cursor = t + len(y) / SR
        self.n_chain += 1

    def overlay(self, t_s, y, xf_s):
        self.events.append((int(t_s * SR), y, int(xf_s * SR)))

    def tiles(self, t0_s, y, step_s, until_s, gain, xf_s=2.5):
        t = t0_s
        while t < until_s:
            self.overlay(t, y * gain, xf_s)
            t += step_s

    def render(self, target_s):
        L = max(t + len(y) for t, y, _ in self.events)
        out = np.zeros((L, 2), np.float32)
        first_t = min(t for t, _, _ in self.events)
        last_end = L
        for t, y, xf in self.events:
            e = _fade_env(len(y), xf)
            if t <= first_t:                      # piece opening: no fade-in
                e[: min(xf, len(y) // 2)] = 1.0
            if t + len(y) >= last_end:            # piece ending: no fade-out
                e[-min(xf, len(y) // 2):] = 1.0
            out[t: t + len(y)] += y * e[:, None]
        return out


# --------------------------------------------------------------------------- movements

def build_flow(rng, piece, slicer, pool, arc, policy, ch, dur_s, prev_cent):
    t0 = piece.cursor
    while piece.cursor - t0 < dur_s - 2.0:
        prog = (piece.cursor - t0) / max(1e-6, dur_s)
        speed = float(np.clip(arc(prog) * rng.uniform(0.85, 1.2), 0.4, 2.4))
        remaining = dur_s - (piece.cursor - t0)
        want = min(rng.uniform(12.0, 38.0), max(4.0, remaining * speed))
        y, cent = slicer.pick(pool[int(rng.integers(len(pool)))], want,
                              prev_cent, policy)
        if y is None:
            continue
        prev_cent = cent
        y = gl.resample_grain(seg_process(y, rng, ch), speed, "hermite")
        piece.chain(y, float(rng.uniform(2.0, 5.0)))
    return prev_cent


def build_ab_weave(rng, piece, slicer, pool, arc, policy, ch, dur_s, prev_cent):
    a, b = pool[int(rng.integers(len(pool)))], pool[int(rng.integers(len(pool)))]
    lo_sp = float(np.clip(arc(0.5) * 0.65, 0.45, 0.95))
    hi_sp = float(np.clip(arc(0.5) * 1.5, 1.15, 2.0))
    t0 = piece.cursor
    flip = False
    while piece.cursor - t0 < dur_s - 2.0:
        remaining = dur_s - (piece.cursor - t0)
        want = min(rng.uniform(6.0, 16.0), max(3.0, remaining * (lo_sp if flip else hi_sp)))
        y, cent = slicer.pick(b if flip else a, want, prev_cent, policy)
        if y is None:
            flip = not flip
            continue
        prev_cent = cent
        y = gl.resample_grain(seg_process(y, rng, ch), lo_sp if flip else hi_sp,
                              "hermite")
        piece.chain(y, float(rng.uniform(1.0, 2.5)))
        if rng.random() < 0.18 and len(pool) > 2:            # interjection
            c = pool[int(rng.integers(len(pool)))]
            z, _ = slicer.pick(c, rng.uniform(1.0, 3.0), None, "random")
            if z is not None:
                z = gl.resample_grain(z, float(rng.uniform(1.5, 2.6)), "hermite")
                pan = rng.uniform(-0.7, 0.7)
                th = (pan + 1) * np.pi / 4
                z[:, 0] *= np.cos(th) * 0.7
                z[:, 1] *= np.sin(th) * 0.7
                piece.overlay(piece.cursor - len(z) / SR * 0.5, z, 0.06)
        flip = not flip
    return prev_cent


def build_bursts(rng, piece, slicer, pool, arc, policy, ch, dur_s, prev_cent):
    bg = pool[int(rng.integers(len(pool)))]
    speed = float(np.clip(arc(0.5) * rng.uniform(0.55, 0.95), 0.4, 1.0))
    y, prev_cent = slicer.pick(bg, min(45.0, dur_s * 0.6), prev_cent, policy)
    if y is None:
        return build_flow(rng, piece, slicer, pool, arc, policy, ch, dur_s, prev_cent)
    y = gl.resample_grain(seg_process(y, rng, ch), speed, "hermite")
    t0 = piece.cursor
    piece.tiles(t0, y, max(4.0, len(y) / SR - 2.5), dur_s, 1.0)
    # Euclidean interjection grid (alien lisp, re-rolled every cycle)
    slot = float(rng.uniform(1.5, 3.5))
    gate = ar.gen_pattern(rng, dict(k=(4, 11), val="mid", poly=0.3))
    tslot, cyc = t0, 0
    while tslot < t0 + dur_s:
        pat = ar.normalize_velocities(ar.ev(gate, rng))
        for s in range(len(pat)):
            v = pat[s % len(pat)]
            if v is None or v <= 0.05:
                continue
            src = pool[int(rng.integers(len(pool)))]
            z, _ = slicer.pick(src, rng.uniform(0.8, 3.0), None, "random")
            if z is None:
                continue
            z = gl.resample_grain(z, float(rng.uniform(1.4, 2.6)), "hermite")
            pan = rng.uniform(-0.75, 0.75)
            th = (pan + 1) * np.pi / 4
            z[:, 0] *= np.cos(th) * (0.3 + 0.5 * v)
            z[:, 1] *= np.sin(th) * (0.3 + 0.5 * v)
            piece.overlay(tslot + s * slot, z, 0.05)
        tslot += len(pat) * slot
        cyc += 1
        if cyc % 2 == 0 and rng.random() < 0.5:             # living grid
            gate = ar.gen_pattern(rng, dict(k=(4, 11), val="mid", poly=0.3))
    piece.chain(np.zeros((int(1.5 * SR), 2), np.float32), 1.0)   # keep the chain moving
    piece.cursor = t0 + dur_s
    return prev_cent


def build_stutter_echo(rng, piece, slicer, pool, arc, policy, ch, dur_s, prev_cent):
    t0 = piece.cursor
    while piece.cursor - t0 < dur_s - 2.0:
        remaining = dur_s - (piece.cursor - t0)
        reps = int(rng.integers(2, 5))
        base = float(np.clip(arc(0.5) * rng.uniform(0.6, 1.1), 0.45, 1.6))
        ratio = float(rng.choice(np.array([1.28, 1.45, 1.6, 0.72, 0.65])))
        want = min(rng.uniform(9.0, 22.0), max(3.0, remaining * base))
        y, cent = slicer.pick(pool[int(rng.integers(len(pool)))], want,
                              prev_cent, policy)
        if y is None:
            continue
        prev_cent = cent
        y = seg_process(y, rng, ch)
        for r in range(reps):
            sp = base * ratio ** r
            sp = float(np.clip(sp, 0.35, 2.4))
            y2 = gl.resample_grain(y, sp, "hermite")
            g = 1.0 / (1.0 + 0.55 * r)
            piece.chain(y2 * g, float(rng.uniform(0.6, 1.5)))
            if piece.cursor - t0 >= dur_s - 1.0:
                break
    return prev_cent


def build_drift_layer(rng, piece, slicer, pool, arc, policy, ch, dur_s, prev_cent):
    bg = pool[int(rng.integers(len(pool)))]
    y, prev_cent = slicer.pick(bg, min(60.0, dur_s), prev_cent, policy)
    if y is None:
        return build_flow(rng, piece, slicer, pool, arc, policy, ch, dur_s, prev_cent)
    y = gl.resample_grain(seg_process(y, rng, ch),
                          float(rng.uniform(0.45, 0.68)), "hermite")
    t0 = piece.cursor
    piece.tiles(t0, y, max(5.0, len(y) / SR - 3.0), dur_s,
                float(rng.uniform(0.35, 0.55)))
    # foreground flow with shorter passages
    fg_pool = [p for p in pool if p != bg] or pool
    while piece.cursor < t0 + dur_s - 2.0:
        remaining = t0 + dur_s - piece.cursor
        speed = float(np.clip(arc((piece.cursor - t0) / dur_s) *
                              rng.uniform(0.9, 1.25), 0.5, 2.0))
        want = min(rng.uniform(8.0, 18.0), max(3.0, remaining * speed))
        z, cent = slicer.pick(fg_pool[int(rng.integers(len(fg_pool)))], want,
                              prev_cent, policy)
        if z is None:
            continue
        prev_cent = cent
        z = gl.resample_grain(seg_process(z, rng, ch), speed, "hermite")
        piece.chain(z, float(rng.uniform(1.5, 3.0)))
    if piece.n_chain == 0:                                   # silence fallback
        piece.chain(np.zeros((int(dur_s * SR), 2), np.float32), 1.0)
    piece.cursor = t0 + dur_s
    return prev_cent


BUILDERS = {"flow": build_flow, "ab_weave": build_ab_weave, "bursts": build_bursts,
            "stutter_echo": build_stutter_echo, "drift_layer": build_drift_layer}


# --------------------------------------------------------------------------- piece

def build_piece(rng, files, args):
    target = float(rng.uniform(args.min_len, args.max_len))
    n_mov = int(rng.integers(3, 6))
    weights = rng.uniform(0.6, 1.4, size=n_mov)
    weights *= (target - 8.0 * n_mov) / weights.sum()
    durs = [max(22.0, w) for w in weights]

    arc_list = [[1.0, 1.0], [1.5, 0.7], [0.7, 1.5], [1.2, 0.8, 1.4], [0.9, 1.6], [1.6, 0.9]]
    arcs = arc_list[int(rng.integers(len(arc_list)))]
    policy = str(rng.choice(np.array(["random", "similar", "contrast"])))
    ch = montage_character(rng)
    slicer = Slicer(rng)
    piece = Piece()
    prev_cent, prev_strat, plan = None, None, []
    mi = 0
    while piece.cursor < target - 24.0 and mi < 12:
        remaining = target - piece.cursor
        dur = durs[mi % len(durs)]
        strat = str(rng.choice(np.array(STRATEGIES)))
        if strat == prev_strat:
            strat = str(rng.choice(np.array(STRATEGIES)))
        pool = [files[i] for i in rng.choice(len(files),
                                             size=min(len(files), int(2 + rng.integers(0, 3))),
                                             replace=False)]

        def arc(p, a=arcs):
            return float(np.interp(p, np.linspace(0, 1, len(a)), a))

        before = piece.cursor
        prev_cent = BUILDERS[strat](rng, piece, slicer, pool, arc, policy, ch,
                                    min(dur, remaining - 2.0), prev_cent)
        prev_strat = strat
        plan.append(dict(strategy=strat, dur_s=round(piece.cursor - before, 1),
                         pool=[Path(p).stem for p in pool[:3]]))
        mi += 1
    return piece, target, dict(arc=arcs, policy=policy, character=ch, movements=plan)


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(
        description="Final assembly: 3-4 minute montages from everything in audio_lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", default="audio_lab", help="lab folder to harvest")
    ap.add_argument("-o", "--out", default="final_pieces")
    ap.add_argument("--pieces", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=float, default=185.0)
    ap.add_argument("--max-len", type=float, default=235.0)
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    files = discover(args.base, Path(args.out).name)
    if not files:
        ap.error(f"no wavs found under {args.base}")
    total = sum(sf.info(f).duration for f in files)
    print(f"corpus: {len(files)} pieces, {total / 60:.0f} min "
          f"({' '.join(sorted({Path(f).parent.name for f in files}))})")

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
        pseed = args.seed + 7919 * i + 31
        prng = np.random.default_rng(pseed)
        piece, target, meta = build_piece(prng, files, args)
        audio = piece.render(target)
        if audio.shape[0] < SR * 10:
            print(f"  final{i:02d}: too short, skipped")
            continue
        f = int(0.02 * SR)
        audio[:f] *= np.linspace(0, 1, f)[:, None]
        audio[-f:] *= np.linspace(1, 0, f)[:, None]
        pk = float(np.abs(audio).max())
        if pk > 1e-4:
            audio *= 0.97 / pk
        wav = outdir / f"final{i:02d}_{pseed}.wav"
        gl.write_wav(wav, SR, audio)
        print(f"  final{i:02d} ({audio.shape[0] / SR / 60:.1f} min, "
              f"{len(piece.events)} events) -> {wav.name}")
        manifest[str(wav)] = dict(seed=pseed, target_s=target, **meta)

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"done -> {outdir}/ (manifest.json records every montage plan)")


if __name__ == "__main__":
    sys.exit(main())