#!/usr/bin/env python3
"""
storm_lab.py — the extreme final stage: beat-locked spine + rate-cycling collage.

Where montage_lab crossfades polite passages, storm_lab goes for the drama:

- BEAT SPINE. Drum passages from the rhythm generations (rhythms/, alien_rhythms/,
  flux_rhythms/) are time-stretched onto ONE global tempo grid, so cuts land on
  the beat. The spine drops in and out via probability - when it's out, the
  collage is naked. Half-time ratios (octave down) give weight.
- RATE-CYCLING COLLAGE. A piece is a sequence of weather phases:
      slow    long passages (1-4 bars), wide crossfades
      med     1/8 - 1/4-note slicing
      fast    16th-note slicing - stutter collage
      accel   slot size ramps 1 bar -> 1/32 note (the frenzy)
      decel   the reverse - grinding to a halt
      air     sparse, silence-heavy breathing room
  accel/decel phases are the fast->slow cycling: the cut rate itself is musical.
- SILENCE IS MATERIAL. Slots are pattern-gated (alien-lisp euclid gates), air
  phases are mostly rests, and 1.5-3.5s FULL-STOP gaps hit at phase boundaries.
- EXTREME SLICING. Per-slice speed up to 4x / down to 0.25x (widening with the
  piece's 'wild' arc), reversal, stutter-echoes with pitch ladders, and five
  interleave modes: lane_alternate (dark/bright), src_alternate (A/B battle),
  src_scan (zip through a whole source file), pitch_ladder, mirrored.

Usage:
    python3 storm_lab.py -o audio_lab/storm_pieces
    python3 storm_lab.py -o out --pieces 8 --seed 3 --bpm 140
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

import granular_lab as gl
import alien_rhythms as ar
import montage_lab as mg

SR = 44100
DRUM_DIRS = ("rhythms", "alien_rhythms", "flux_rhythms")
MODES = ("lane_alternate", "src_alternate", "src_scan", "pitch_ladder", "mirrored")
PROFILES = ("slow", "med", "fast", "accel", "decel", "air")
LAST_RENDER = {}                 # debug/analysis: the raw event schedule


# --------------------------------------------------------------------------- corpus

def load_bpm_map(base):
    bpms = {}
    for d in DRUM_DIRS:
        mpath = Path(base) / d / "manifest.json"
        if not mpath.exists():
            continue
        try:
            with open(mpath) as f:
                m = json.load(f)
        except Exception:
            continue
        for key, e in m.items():
            if "bpm" in e:
                bpms[key] = float(e["bpm"])
            elif "sections" in e and e["sections"]:
                bpms[key] = float(np.mean([s.get("bpm0", 140) for s in e["sections"]]))
    return bpms


def skim_brightness(files, rng, probes=2):
    """Rough per-file brightness from short skims -> class buckets."""
    cents = {}
    for f in files:
        info = sf.info(f)
        vals = []
        for _ in range(probes):
            start = int(rng.integers(0, max(1, info.frames - SR * 2)))
            with sf.SoundFile(f) as fh:
                fh.seek(min(start, max(0, info.frames - SR * 2)))
                y = fh.read(SR * 2, dtype="float32", always_2d=True)
            if len(y) < SR // 2:
                continue
            vals.append(gl.compute_features(y, info.samplerate)["centroid"])
        cents[f] = float(np.mean(vals)) if vals else 1000.0
    order = sorted(files, key=lambda f: cents[f])
    n = len(order)
    return dict(dark=order[: max(1, n // 3)],
                mid=order[n // 3: max(2, 2 * n // 3)],
                bright=order[max(2, 2 * n // 3):],
                cents=cents)


# --------------------------------------------------------------------------- phases

@dataclass
class Phase:
    profile: str
    bars: int
    mode: str
    silence_p: float
    keep_spine_p: float
    spine_gain: float
    density: float              # euclid gate density for collage slots


def phase_sequence(rng, total_bars, wild=0.5):
    if wild >= 0.85:                       # frenzy-biased composition
        templates = [
            ["med", "accel", "fast", "fast", "accel", "fast", "decel", "air",
             "fast", "accel"],
            ["fast", "accel", "fast", "decel", "fast", "air", "accel", "fast"],
            ["slow", "accel", "fast", "fast", "fast", "decel", "med", "accel", "fast"],
        ]
    else:
        templates = [
            ["slow", "accel", "fast", "air", "decel", "slow"],
            ["med", "accel", "fast", "fast", "decel", "air", "slow"],
            ["slow", "med", "accel", "fast", "air", "slow", "med"],
            ["fast", "decel", "slow", "accel", "fast", "air"],
        ]
    seq = list(templates[int(rng.integers(len(templates)))])
    phases, left = [], total_bars
    i = 0
    while left > 6:
        prof = seq[i % len(seq)]
        i += 1
        bars = int(min(left - 4, rng.integers(6, 17)))
        if bars < 4:
            break
        left -= bars
        silence = {"slow": 0.05, "med": 0.12, "fast": 0.15,
                   "accel": 0.05, "decel": 0.15, "air": 0.55}[prof]
        phases.append(Phase(prof, bars, str(rng.choice(np.array(MODES))),
                            silence,
                            {"slow": 0.85, "med": 0.75, "fast": 0.45,
                             "accel": 0.5, "decel": 0.7, "air": 0.3}[prof],
                            {"slow": 1.0, "med": 0.95, "fast": 0.55,
                             "accel": 0.5, "decel": 0.9, "air": 0.6}[prof],
                            {"slow": 0.4, "med": 0.55, "fast": 0.8,
                             "accel": 0.85, "decel": 0.5, "air": 0.25}[prof]))
    return phases


def slot_lengths(phase, rng):
    """Slot sizes in 32nd-note units (1 = 1/32)."""
    n_slots_hint = phase.bars * 8
    if phase.profile == "slow":
        return [int(rng.choice(np.array([32, 48, 64]))) for _ in range(n_slots_hint // 16)]
    if phase.profile == "med":
        return [int(rng.choice(np.array([4, 4, 8, 16]))) for _ in range(n_slots_hint // 2)]
    if phase.profile == "fast":
        return [2 if rng.random() > 0.12 else 1 for _ in range(phase.bars * 16)]
    if phase.profile == "air":
        return [int(rng.choice(np.array([8, 16, 32]))) for _ in range(n_slots_hint // 3)]
    # accel / decel: geometric ramp between 1-bar passages and 1/32 slices
    n = max(6, int(phase.bars * 2.5))
    ramp = np.geomspace(32.0, 1.0, n)
    if phase.profile == "decel":
        ramp = ramp[::-1]
    return [max(1, int(round(v * rng.uniform(0.7, 1.4)))) for v in ramp]


# --------------------------------------------------------------------------- rendering

def _tiny_fades(y, s):
    f = min(len(y) // 4, int(s * SR))
    if f > 1:
        y[:f] *= np.linspace(0, 1, f)[:, None]
        y[-f:] *= np.linspace(1, 0, f)[:, None]
    return y


def draw_speed(rng, wild):
    mx = 1.8 + 2.2 * wild                       # up to 4x at full wild
    return float(np.exp(rng.uniform(-np.log(mx), np.log(mx))))


def render_piece(rng, piece_bars, bpm, files, drum_files, buckets, bpms, wild_arc):
    sixteenth = 60.0 / bpm / 4.0
    u32 = sixteenth / 2.0
    slicer = mg.Slicer(rng)
    phases = phase_sequence(rng, piece_bars, wild_arc[0])
    events, gaps = [], []
    wild = 0.3

    # ---- collage: walk the phase slots on the global grid
    t, step32 = 0.0, 0
    bar_pat = ar.normalize_velocities(ar.ev(
        ar.gen_pattern(rng, dict(k=(6, 13), val="mid")), rng))
    sticky = None            # (y, remaining, pitch, ladder)
    scan = dict(path=None, pos=0.0)
    alt_pair = [rng.choice(files), rng.choice(files)]
    lane = 0
    pitch_lad = 1.0
    mirror = False

    for pi, ph in enumerate(phases):
        mode_local = ph.mode
        if rng.random() < 0.3:                              # mode drifts a bit
            mode_local = str(rng.choice(np.array(MODES)))
        if mode_local == "src_scan" and (scan["path"] is None or rng.random() < 0.5):
            scan["path"] = files[int(rng.integers(len(files)))]
            scan["pos"] = 0.0
        if mode_local == "src_alternate" and rng.random() < 0.4:
            alt_pair = [rng.choice(files), rng.choice(files)]
        slots = slot_lengths(ph, rng)
        for n32 in slots:
            slot_len = n32 * u32
            wild = float(np.interp(t / (piece_bars * 16 * sixteenth),
                                   np.linspace(0, 1, len(wild_arc)), wild_arc))
            # per-bar gate (16 steps), re-rolled each bar
            if step32 % 32 == 0:
                bar_pat = ar.normalize_velocities(ar.ev(
                    ar.gen_pattern(rng, dict(k=(3 + int(ph.density * 10), 13),
                                             val="mid")), rng))
            bit = bar_pat[(step32 // 2) % len(bar_pat)] if len(bar_pat) else 0
            gate = bit is not None and bit > 0.25 * ph.density + rng.random() * 0.1
            step32 += n32
            t += slot_len
            if not gate or rng.random() < ph.silence_p:
                sticky = None
                continue
            speed = draw_speed(rng, wild)
            hard = slot_len < 0.25                     # hard-chop territory
            if hard and mode_local != "pitch_ladder":
                hi = 1.5 + 2.5 * wild                 # adjacent slices always
                if rng.random() < 0.5:               # pitch-contrast: hi/lo
                    speed = float(np.exp(rng.uniform(np.log(1.2), np.log(hi))))
                else:
                    speed = float(np.exp(rng.uniform(-np.log(hi), -np.log(1.2))))
            if sticky is not None:                            # stutter echo
                y, rem, pitch = sticky
                rem -= 1
                sticky = (y, rem, pitch) if rem > 0 else None
                y2 = gl.resample_grain(y, pitch, "hermite")
                pitch *= 1.18
                events.append((t - slot_len, _tiny_fades(y2, 0.003)))
                continue
            # ---- choose source + slice
            if mode_local == "lane_alternate":
                lane ^= 1
                pool = buckets["dark" if lane == 0 else "bright"]
                path = pool[int(rng.integers(len(pool)))]
            elif mode_local == "src_alternate":
                path = alt_pair[int((step32 // max(1, n32)) % 2)]
            elif mode_local == "src_scan":
                path = scan["path"]
            else:
                path = files[int(rng.integers(len(files)))]
            seg_s = slot_len * speed
            if mode_local == "src_scan":
                data, srr = slicer._load(path)
                st = int(scan["pos"] * srr)
                y = data[st: st + max(64, int(seg_s * srr))]
                scan["pos"] += seg_s * rng.uniform(20.0, 160.0)   # violent zip
                if rng.random() < 0.2:                             # scratch back
                    scan["pos"] = max(0.0, scan["pos"] - rng.uniform(5.0, 30.0))
                if scan["pos"] * srr >= len(data) - SR:
                    scan["path"] = files[int(rng.integers(len(files)))]
                    scan["pos"] = rng.uniform(0.0, 0.15) * len(data) / srr
                if len(y) < 64:
                    continue
            else:
                y, _ = slicer.pick(path, max(0.06, seg_s), None, "random")
                if y is None:
                    continue
            if mode_local == "mirrored":
                mirror = not mirror
                if mirror:
                    y = y[::-1]
            if mode_local == "pitch_ladder":
                speed = float(np.clip(pitch_lad, 0.25, 4.0))
                pitch_lad *= float(rng.choice(np.array([1.6, 1.9, 0.55, 0.42])))
                if pitch_lad > 4 or pitch_lad < 0.25:
                    pitch_lad = float(rng.choice(np.array([1.0, 2.0, 0.5])))
            y2 = gl.resample_grain(y, speed, "hermite")
            need = int(slot_len * SR)
            y2 = y2[:need] if len(y2) >= need else np.pad(y2, ((0, need - len(y2)), (0, 0)))
            rms = float(np.sqrt(np.mean(y2 ** 2) + 1e-12))
            gain = min(12.0, 0.12 * 10 ** (rng.uniform(-0.3, 0.3) / 20) / max(rms, 1e-4))
            if rng.random() < 0.05 + 0.12 * wild:            # reverse
                y2 = y2[::-1]
            pan = rng.uniform(-0.8, 0.8) if mode_local != "lane_alternate" else \
                rng.uniform(-0.3, 0.3) if lane == 0 else rng.uniform(0.3, 0.8)
            th = (pan + 1.0) * np.pi / 4
            y2[:, 0] *= np.cos(th) * gain
            y2[:, 1] *= np.sin(th) * gain
            events.append((t - slot_len, _tiny_fades(y2, 0.003 if hard else 0.008)))
            if rng.random() < (0.05 + 0.3 * wild) * (0.4 if hard else 1.0):
                sticky = (y2.copy(), int(2 + rng.integers(0, 5)), 1.15)
        if rng.random() < 0.4 and pi < len(phases) - 1:       # full-stop gap
            gaps.append((t + u32 * 2, t + u32 * 2 + rng.uniform(1.5, 3.5)))

    # ---- drum spine: grid-locked passages
    spine_until_bar, spine_events = 0, []
    bar = 0
    while bar < piece_bars:
        ph = phases[int(min(len(phases) - 1, bar * len(phases) // piece_bars))]
        if bar >= spine_until_bar:
            if rng.random() < ph.keep_spine_p:
                p_bars = int(rng.integers(2, 7))
                if ph.profile in ("fast", "accel"):
                    p_bars = int(rng.integers(1, 3))          # spine stutters too
                path = drum_files[int(rng.integers(len(drum_files)))]
                src_bpm = bpms.get(path, 140.0)
                ratio = float(rng.choice(np.array([1.0, 1.0, 0.5])))
                speed = (bpm / src_bpm) * ratio
                want_out = p_bars * 16 * sixteenth
                y, _ = slicer.pick(path, max(2.0, want_out * speed), None, "random")
                if y is not None:
                    y = gl.resample_grain(y, speed, "hermite")[: int(want_out * SR)]
                    if len(y) > SR:
                        g = ph.spine_gain * rng.uniform(0.8, 1.1)
                        y = _tiny_fades(y, 0.2) * g
                        spine_events.append((bar * 16 * sixteenth, y))
                spine_until_bar = bar + p_bars
            else:
                spine_until_bar = bar + int(rng.integers(1, 4))
        bar += 1
    events += spine_events

    # ---- orthogonal glitch replication (before assembly so it sums in)
    piece_dur = piece_bars * 16 * sixteenth
    events, fx = glitch_replicate(events, rng, wild_arc[-1], gaps, piece_dur)

    # ---- assemble
    L = int(piece_bars * 16 * sixteenth * SR) + SR
    out = np.zeros((L, 2), np.float32)
    for t, y in events:
        i0 = int(t * SR)
        if i0 >= L:
            continue
        end = min(L, i0 + len(y))
        out[i0:end] += y[: end - i0]
    for g0, g1 in gaps:                                       # the full-stop moments
        i0, i1 = int(g0 * SR), min(L, int(g1 * SR))
        f = max(1, int(0.03 * SR))
        out[i0 + f: i1 - f] = 0.0
        out[i0: i0 + f] *= np.linspace(1, 0, f)[:, None]
        out[i1 - f: i1] *= np.linspace(0, 1, f)[:, None]
    # ---- global phasor over the ENTIRE output
    out, warps = vari_speed(out, SR, rng, wild_arc[-1], piece_dur)
    LAST_RENDER.clear()
    LAST_RENDER["events"] = [(t, len(y)) for t, y in events]
    LAST_RENDER["gaps"] = list(gaps)
    LAST_RENDER["piece_dur"] = piece_dur
    fx["warps"] = warps
    return out, phases, fx


# --------------------------------------------------------------------------- glitch + phasor

def _in_gap(t, gaps, pad=0.15):
    return any(g0 - pad <= t <= g1 + pad for g0, g1 in gaps)


def glitch_replicate(events, rng, wild, gaps, piece_dur):
    """Repetition that spawns ORTHOGONALLY: echo loops at rates unrelated to
    the beat grid (irrational multiples of the 16th), plus machine-gun
    loop-locks. Their periods drift against the grid -> glitch."""
    out = list(events)
    n_echo = n_lock = 0
    for t, y in events:
        if rng.random() > 0.04 + 0.10 * wild:
            continue
        period = float(rng.uniform(0.03, 0.35)) * \
            float(rng.choice(np.array([0.37, 0.61, 1.0, 1.41, 2.33])))
        k = int(rng.integers(2, 7))
        decay = float(rng.uniform(0.55, 0.88))
        pitch = float(rng.choice(np.array([1.0, 1.0, 1.059, 0.944])))
        gain = float(rng.uniform(0.35, 0.75))
        yb = y
        for j in range(1, k + 1):
            tt = t + j * period
            if tt > piece_dur or _in_gap(tt, gaps):
                break
            if pitch != 1.0:
                yb = gl.resample_grain(yb, pitch, "hermite")
            out.append((tt, yb * (gain * decay ** j)))
            n_echo += 1
    # loop-locks: short slices machine-gunned at their own fast rate
    for _ in range(int(rng.integers(2, 6))):
        t0 = float(rng.uniform(2.0, max(3.0, piece_dur - 8.0)))
        if _in_gap(t0, gaps, 1.0):
            continue
        src = events[int(rng.integers(len(events)))]
        y0 = src[1]
        if len(y0) < 512:
            continue
        y0 = _tiny_fades(y0[: int(rng.uniform(0.05, 0.30) * SR)], 0.004)
        period = float(rng.uniform(0.028, 0.13))
        dur = float(rng.uniform(1.5, 6.0))
        g = float(rng.uniform(0.28, 0.55))
        pitch_lad = 1.0
        for j in range(int(dur / period)):
            tt = t0 + j * period
            if tt > piece_dur or _in_gap(tt, gaps, 0.3):
                break
            yj = y0
            if j % 4 == 3 and rng.random() < 0.5:           # occasional pitch step
                pitch_lad *= float(rng.choice(np.array([1.059, 0.944, 1.5])))
                yj = gl.resample_grain(y0, pitch_lad, "hermite")
            out.append((tt, yj * g))
            n_lock += 1
    return out, dict(echo_events=n_echo, lock_events=n_lock)


def _interp_hermite(x, pos):
    n = len(x)
    pos = np.clip(pos, 0.0, n - 1)
    i = np.floor(pos).astype(np.int64)
    f = (pos - i).astype(np.float32)
    x0 = x[i]
    xm1 = x[np.maximum(i - 1, 0)]
    x1 = x[np.minimum(i + 1, n - 1)]
    x2 = x[np.minimum(i + 2, n - 1)]
    c1 = 0.5 * (x1 - xm1)
    c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
    c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
    return ((c3 * f + c2) * f + c1) * f + x0


def vari_speed(y, sr, rng, wild, piece_dur):
    """Global PHASOR over the entire output: a slow breathing rate-wobble
    always, plus extreme warp events (accel sweeps / brake sags / wobble
    storms) at random points. Time AND pitch bend together - tape-style."""
    n = len(y)
    ts = np.arange(n, dtype=np.float64) / sr
    rate = np.ones(n, np.float64)
    # breathing: an LFO whose own frequency drifts (a real phasor)
    f0 = rng.uniform(0.05, 0.25)
    drift = 0.3 * np.sin(2 * np.pi * 0.011 * ts + 1.7)
    freq = f0 * (1.0 + drift)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    rate += rng.uniform(0.004, 0.02) * np.sin(phase)
    warps = []
    for _ in range(int(rng.integers(2, 5))):
        t0 = float(rng.uniform(0.0, max(1.0, piece_dur - 12.0)))
        dur = float(rng.uniform(2.0, 10.0))
        typ = str(rng.choice(np.array(["accel", "brake", "wobble"])))
        bell = np.clip(1.0 - np.abs(ts - (t0 + dur / 2)) / (dur / 2), 0.0, 1.0)
        if typ == "accel":
            peak = 1.8 + 2.2 * wild
            rate *= 1.0 + (peak - 1.0) * bell ** 2
        elif typ == "brake":
            floor = max(0.18, 0.65 - 0.35 * wild)
            rate *= 1.0 - (1.0 - floor) * bell ** 2
        else:
            rate *= 1.0 + (0.15 + 0.3 * wild) * bell * \
                np.sin(2 * np.pi * rng.uniform(2.0, 6.0) * ts)
        warps.append(dict(t=round(t0, 1), dur=round(dur, 1), type=typ))
    rate = np.maximum(rate, 0.15)
    p = np.cumsum(rate) / sr                    # input position (seconds) per output sample
    D = (n - 1) / sr
    n_out = int(np.searchsorted(p, D))
    if n_out < sr // 2:
        return y, warps
    pos = p[:n_out] * sr                        # -> input position in SAMPLES
    out = np.stack([_interp_hermite(y[:, c], pos) for c in range(2)], axis=1)
    return out.astype(np.float32), warps


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(
        description="Extreme final stage: beat-locked spine, rate-cycling collage, "
                    "structural silence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", default="audio_lab")
    ap.add_argument("-o", "--out", default="storm_pieces")
    ap.add_argument("--pieces", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bpm", type=float, help="pin the global tempo grid")
    ap.add_argument("--wild", type=float, default=None,
                    help="pin the extremeness arc 0..1 (1 = max speed range, "
                         "stutter, density)")
    ap.add_argument("--min-len", type=float, default=185.0)
    ap.add_argument("--max-len", type=float, default=235.0)
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    files = mg.discover(args.base, Path(args.out).name)
    files = [f for f in files if Path(f).parent.name != "final_pieces"]
    if not files:
        ap.error(f"no wavs found under {args.base}")
    bpms = load_bpm_map(args.base)
    drum_files = [f for f in files if Path(f).parent.name in DRUM_DIRS]
    if not drum_files:
        ap.error("no drum-generation pieces found (need rhythms/alien_rhythms/flux_rhythms)")
    rng0 = np.random.default_rng(args.seed)
    buckets = skim_brightness(files, rng0)
    total = sum(sf.info(f).duration for f in files)
    print(f"corpus: {len(files)} pieces / {total / 60:.0f} min | "
          f"drum spine pool: {len(drum_files)} | silence is material")

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
        pseed = args.seed + 7919 * i + 41
        prng = np.random.default_rng(pseed)
        bpm = args.bpm or float(prng.uniform(120.0, 172.0))
        bar = 16 * 60.0 / bpm / 4.0
        piece_bars = int(prng.uniform(args.min_len, args.max_len) / bar)
        wild_arc = ([args.wild, args.wild] if args.wild is not None else
                    [[0.15, 1.0], [1.0, 0.2], [0.3, 1.0, 0.4],
                     [0.5, 1.0], [0.9, 0.3, 1.0]][int(prng.integers(5))])
        audio, phases, fx = render_piece(prng, piece_bars, bpm, files, drum_files,
                                        buckets, bpms, wild_arc)
        f = int(0.02 * SR)
        audio[:f] *= np.linspace(0, 1, f)[:, None]
        audio[-f:] *= np.linspace(1, 0, f)[:, None]
        pk = float(np.abs(audio).max())
        if pk > 1e-4:
            audio *= 0.97 / pk
        wav = outdir / f"storm{i:02d}_{pseed}.wav"
        gl.write_wav(wav, SR, audio)
        print(f"  storm{i:02d} ({audio.shape[0] / SR / 60:.1f} min, {bpm:.0f}bpm, "
              f"{len(phases)} phases, wild {wild_arc[0]:.2f}->{wild_arc[-1]:.2f}, "
              f"{fx['echo_events']} echoes, {fx['lock_events']} lock-hits, "
              f"{len(fx['warps'])} warps) -> {wav.name}")
        manifest[str(wav)] = dict(
            seed=pseed, bpm=bpm, bars=piece_bars, wild_arc=wild_arc, fx=fx,
            phases=[dict(profile=p.profile, bars=p.bars, mode=p.mode,
                         silence_p=round(p.silence_p, 2),
                         keep_spine_p=round(p.keep_spine_p, 2)) for p in phases])

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"done -> {outdir}/")


if __name__ == "__main__":
    sys.exit(main())