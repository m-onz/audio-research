#!/usr/bin/env python3
"""
club_lab.py — clubbify any piece: stutter-gated kicks + genre-morphing beat
evolved by a genetic algorithm.

The post-processing layer that gives freeform tracks a dance skeleton:

1. STUTTER GATES. A random number of stutter events at random positions and
   lengths. Each event: the track audio is replaced by a rapid CD-skip loop
   (slice size, rate ramp and pitch-ladder randomised per event), and a real
   kick sample plays at the START of every event. The kick marks every
   stutter — the track itself becomes part of the drum kit.

2. BEAT ENGINE. A hard drum layer over the whole track whose genre changes
   every ~5 seconds between HARDCORE (driving 4/4), DNB (two-step), GARAGE
   (swung 2-step) and GRIME (halftime). Each genre has a standard motif —
   and a COMPLEX LAYER on top of it.

3. GENETIC ALGORITHM. Every 5-second slot runs one generation of a GA
   (per-genre populations of 88-bit pattern genomes: extra kicks, ghost
   snares, hat euclid density/rotation, ghost layers, ratchets, swing, pitch
   accents). Fitness rewards structured complexity: onset-interval entropy,
   syncopation, density in a musical range, ratchets, plus novelty vs already
   used patterns. The best individual renders the slot; selection + crossover
   + mutation breed the next slot's population. Complexity compounds across
   the piece.

The track is lightly ducked under the beat kicks — the glue that makes it
all feel like one club record.

Usage:
    python3 club_lab.py audio_lab/final_review -o audio_lab/clubbed
    python3 club_lab.py piece.wav -o out --seed 5
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

SR = 44100
EPS = 1e-10
GENRES = ("hardcore", "dnb", "garage", "grime")
GENRE_BPM = {"hardcore": (160, 170), "dnb": (168, 176),
             "garage": (130, 138), "grime": (138, 144)}
SKELETONS = {
    # kick steps, snare steps, ohat steps, base-hat euclid pulses, swing
    "hardcore": dict(kick=[0, 4, 8, 12], snare=[4, 12], ohat=[2, 6, 10, 14],
                     hatk=13, swing=0.0),
    "dnb":      dict(kick=[0, 10], snare=[4, 12], ohat=[], hatk=11, swing=0.02),
    "garage":   dict(kick=[0, 10], snare=[4, 12], ohat=[2, 6, 10, 14],
                     hatk=9, swing=0.18),
    "grime":    dict(kick=[0, 6, 10], snare=[8], ohat=[4, 12], hatk=7,
                     swing=0.16),
}

# genome layout (88 bits)
NB_KICK, NB_SNARE, NB_HATD, NB_HATR, NB_HATV = 16, 16, 4, 4, 4
NB_GHOST, NB_RAT, NB_SWING, NB_PITCH = 16, 8, 4, 16
N_BITS = NB_KICK + NB_SNARE + NB_HATD + NB_HATR + NB_HATV + NB_GHOST + \
    NB_RAT + NB_SWING + NB_PITCH


def decode(bits):
    b = bits.astype(bool)
    p = 0
    def take(n):
        nonlocal p
        q = b[p:p + n]
        p += n
        return q
    g = dict(kick=take(NB_KICK), snare=take(NB_SNARE),
             hat_d=int(take(NB_HATD).tobytes()[0]) if NB_HATD else 8,
             hat_r=0, hat_v=0, ghost=take(NB_GHOST),
             ratchet=take(NB_RAT), swing=0.0, pitch=take(NB_PITCH))
    # small-int fields: sum of bits (robust to mutation)
    g["hat_d"] = 1 + int(np.sum(b[NB_KICK + NB_SNARE:NB_KICK + NB_SNARE + NB_HATD]))
    g["hat_r"] = int(np.sum(b[NB_KICK + NB_SNARE + NB_HATD:
                              NB_KICK + NB_SNARE + NB_HATD + NB_HATR]))
    g["hat_v"] = 0.2 + 0.5 * np.sum(b[NB_KICK + NB_SNARE + NB_HATD + NB_HATR:
                                        NB_KICK + NB_SNARE + NB_HATD + NB_HATR + NB_HATV]) / NB_HATV
    g["swing"] = 0.06 * np.sum(b[-NB_SWING - NB_PITCH:-NB_PITCH]) / NB_SWING
    return g


def random_genome(rng):
    return rng.random(N_BITS) < 0.5


def mutate(bits, rng, p=0.07):
    b = bits.copy()
    flip = rng.random(N_BITS) < p
    b[flip] = ~b[flip]
    return b


def crossover(a, b, rng):
    m = rng.random(N_BITS) < 0.5
    return np.where(m, a, b), np.where(m, b, a)


def pattern_from(genre, g):
    sk = SKELETONS[genre]
    kick = list(sk["kick"]) + [i for i in range(16) if g["kick"][i]
                                and i not in sk["kick"] and i % 2 == 0]
    snare = list(sk["snare"]) + [i for i in range(16) if g["snare"][i]
                                 and i not in sk["snare"]]
    ghosts = [i for i in range(16) if g["ghost"][i]]
    ratchets = {8 + 2 * j: 2 + j % 2 for j in range(NB_RAT) if g["ratchet"][j]}
    pitch_steps = {i: 1.0 + 0.06 * (j + 1) for j, i in
                   enumerate(np.flatnonzero(g["pitch"])[:4])}
    return dict(kick=sorted(set(kick)), snare=sorted(set(snare)),
                ohat=sk["ohat"], hatk=max(1, g["hat_d"]), hatr=g["hat_r"],
                hatv=g["hat_v"], ghosts=ghosts, ratchets=ratchets,
                swing=max(SKELETONS[genre]["swing"], g["swing"]),
                pitch_steps=pitch_steps)


def fitness(bits, genre, used):
    """Structured complexity: entropy + syncopation + musical density."""
    g = decode(bits)
    pat = pattern_from(genre, g)
    onsets = np.array(sorted(set(pat["kick"] + pat["snare"] + pat["ghosts"] +
                                 [i for i in range(16) if i % 4 == 2])))
    # onset-interval entropy (variety of spacing)
    if len(onsets) > 2:
        iv = np.diff(onsets)
        c = np.bincount(iv, minlength=17)
        p = c / c.sum()
        ent = -(p[p > 0] * np.log2(p[p > 0])).sum() / 4.0
    else:
        ent = 0.0
    # syncopation: non-skeleton hits on weak 16ths
    weak = [i for i in pat["snare"] + pat["ghosts"] if i % 4 != 0]
    sync = len(weak) / 12.0
    # density in a musical range
    dens = (len(pat["kick"]) + len(pat["snare"]) + len(pat["ghosts"]) +
            pat["hatk"]) / 48.0
    dens_score = 1.0 - abs(dens - 0.55) * 2.0
    rat = len(pat["ratchets"]) / 8.0
    nov = 0.0
    if used:
        nov = float(np.mean([np.mean(bits != u) for u in used[-4:]]))
    swing_bonus = pat["swing"] * (2.0 if genre in ("garage", "grime") else 0.5)
    return float(0.30 * ent + 0.22 * sync + 0.22 * max(0, dens_score) +
                 0.12 * rat + 0.08 * nov + 0.06 * swing_bonus)


def ga_step(pop, scores, rng):
    """One generation: tournament selection, crossover, mutation, elitism."""
    order = np.argsort(scores)[::-1]
    elite = [pop[i] for i in order[:2]]
    children = list(elite)
    while len(children) < len(pop):
        def parent():
            idx = rng.integers(len(pop), size=3)
            return pop[idx[int(np.argmax([scores[i] for i in idx]))]]
        a, b1 = crossover(parent(), parent(), rng)
        children.append(mutate(a, rng))
        if len(children) < len(pop):
            children.append(mutate(b1, rng))
    return children


# ------------------------------------------------------------------ drum bank

def load_drum_bank(sounds_dir, rng, per_role=24):
    bank = ar.load_bank(sounds_dir, rng, per_role=per_role)
    return {k: v for k, v in bank.items()
            if k in ("kick", "snare", "hat", "ohat", "clap", "rim")}


def emit(events, pool, t, vel, pan=0.0, pitch=1.0, rng=None,
         phaser_depth=0.0, semis=0.0):
    if not pool:
        return
    sm = pool[int(rng.integers(len(pool)))]
    y = sm.data
    if semis != 0.0:                                   # pitch shift (semitones)
        y = gl.resample_grain(y, float(2 ** (semis / 12.0)), "hermite")
    elif pitch != 1.0:
        y = gl.resample_grain(y, pitch, "hermite")
    f = min(len(y) // 4, int(0.004 * SR))
    y = y.copy()
    y[:f] *= np.linspace(0, 1, f)[:, None]
    y[-f:] *= np.linspace(1, 0, f)[:, None]
    y = phaser(y, rng, phaser_depth)                   # the phasor effect
    th = (pan + 1) * np.pi / 4
    y = y * np.array([np.cos(th), np.sin(th)], np.float32) * vel
    events.append((max(0, int(t * SR)), np.ascontiguousarray(y, np.float32)))


def phaser(y, rng, depth):
    """Modulated-comb phasor: two LFO-driven fractional delay taps ->
    sweeping notches. Fully vectorised; per-hit random LFO rates/phases."""
    if depth <= 0.02 or len(y) < 256:
        return y
    n = len(y)
    t = np.arange(n) / SR
    out = y.astype(np.float32).copy()
    mono = y[:, 0] if y.ndim == 2 else y
    for _ in range(2):                                 # two moving notches
        rate = float(rng.uniform(0.4, 7.0))
        ph0 = float(rng.uniform(0, 2 * np.pi))
        d0 = float(rng.uniform(0.0008, 0.005))
        D = d0 * (1 + 0.8 * np.sin(2 * np.pi * rate * t + ph0))
        pos = np.clip(np.arange(n) - D * SR, 0, n - 1)
        i0 = pos.astype(np.int64)
        fr = (pos - i0).astype(np.float32)
        i1 = np.minimum(i0 + 1, n - 1)
        if y.ndim == 2:
            wet = np.stack([mono[i0] * (1 - fr) + mono[i1] * fr,
                             y[:, 1][i0] * (1 - fr) + y[:, 1][i1] * fr], axis=1)
        else:
            wet = (mono[i0] * (1 - fr) + mono[i1] * fr)[:, None]
        out = out + depth * 0.5 * wet
    return out / (1.0 + depth * 0.5)


def render_beat_slots(track_len, rng, bank, log):
    """Genre-morphing beat across the whole track; GA evolves per 5s slot.
    Starts REALLY simple (bare skeletons) and ramps up: extras, ghosts,
    ratchets, phaser depth and pitch-shifting all grow with piece progress."""
    pops = {g: [random_genome(rng) for _ in range(20)] for g in GENRES}
    used = []
    events = []
    kick_times = []
    t = 0.5
    slot_i = 0
    arc = str(rng.choice(np.array(["rise", "fall", "zigzag", "wobble"])))

    def pitch_semi(p):
        if arc == "rise":
            return 7.0 * p
        if arc == "fall":
            return -5.0 * p
        if arc == "zigzag":
            return 6.0 * np.sin(3 * np.pi * p)
        return 5.0 * p * np.sin(2 * np.pi * 2.5 * p)     # growing wobble

    while t < track_len - 1.0:
        genre = GENRES[slot_i % len(GENRES)] if slot_i < 4 else \
            str(rng.choice(np.array(GENRES)))
        bpm = float(rng.uniform(*GENRE_BPM[genre]))
        bar = 16 * 60.0 / bpm / 4.0
        bars = max(1, int(round(5.0 / bar)))
        slot_dur = bars * bar
        if t + slot_dur > track_len:
            slot_dur = track_len - t
        p = min(1.0, t / track_len)
        ramp = float(np.clip(p * 1.8, 0.0, 1.0))         # full by ~55%
        semi_base = float(pitch_semi(p))

        scores = [fitness(b, genre, used) for b in pops[genre]]
        best = int(np.argmax(scores))
        pat = pattern_from(genre, decode(pops[genre][best]))
        used.append(pops[genre][best].copy())
        pops[genre] = ga_step(pops[genre], scores, rng)
        stepdur = 60.0 / bpm / 4.0
        sk = SKELETONS[genre]
        for b in range(bars):
            base = t + b * bar
            for s in pat["kick"]:
                if s not in sk["kick"] and rng.random() > ramp:
                    continue                            # extras fade in
                tt = base + s * stepdur
                semi = semi_base + rng.uniform(-2, 2) * ramp
                emit(events, bank.get("kick"), tt, 1.0, rng=rng,
                     phaser_depth=0.5 * ramp * rng.uniform(0.5, 1.0),
                     semis=semi)
                kick_times.append(tt)
            for s in pat["snare"]:
                if s not in sk["snare"] and rng.random() > ramp:
                    continue
                tt = base + s * stepdur + pat["swing"] * stepdur * (s % 2)
                semi = semi_base + rng.uniform(-2, 2) * ramp
                ph_d = ramp * rng.uniform(0.35, 0.9)
                emit(events, bank.get("snare"), tt, 0.9,
                     pan=rng.uniform(-0.08, 0.08), rng=rng,
                     phaser_depth=ph_d, semis=semi)
                if rng.random() < 0.3 * ramp:
                    emit(events, bank.get("clap"), tt, 0.5,
                         pan=rng.uniform(-0.2, 0.2), rng=rng,
                         phaser_depth=ph_d, semis=semi)
            if ramp > 0.2:
                for s in pat["ohat"]:
                    emit(events, bank.get("ohat"), base + s * stepdur, 0.55,
                         rng=rng,
                         phaser_depth=ramp * rng.uniform(0.3, 0.8),
                         semis=semi_base + rng.uniform(-1, 1) * ramp)
            hatk_eff = max(1, int(round(pat["hatk"] * (0.3 + 0.7 * ramp))))
            bits16 = gl._euclid_bits(hatk_eff, 16)
            bits16 = bits16[pat["hatr"] % 16:] + bits16[:pat["hatr"] % 16]
            for i, on in enumerate(bits16):
                if not on:
                    continue
                tt = base + i * stepdur + pat["swing"] * stepdur * (i % 2)
                pitch = pat["pitch_steps"].get(i, 1.0)
                emit(events, bank.get("hat"), tt, pat["hatv"],
                     pan=(0.55 if i % 2 else -0.55), rng=rng,
                     phaser_depth=ramp * rng.uniform(0.35, 0.9),
                     semis=semi_base * 0.5 + rng.uniform(-2, 2) * ramp)
            if ramp > 0.3:
                for s in pat["ghosts"]:
                    if rng.random() > ramp * 0.85:
                        continue
                    emit(events, bank.get("rim"), base + s * stepdur, 0.4,
                         pan=rng.uniform(-0.4, 0.4), rng=rng,
                         phaser_depth=ramp * rng.uniform(0.4, 0.9),
                         semis=semi_base + rng.uniform(-3, 3) * ramp)
            if ramp > 0.55:
                for s, k in pat["ratchets"].items():
                    for j in range(k):
                        emit(events, bank.get("snare"),
                             base + s * stepdur + j * stepdur / 2,
                             0.7 - 0.1 * j, rng=rng,
                             phaser_depth=ramp * rng.uniform(0.3, 0.7),
                             semis=semi_base + 1.5 * j * ramp)
        log.append(dict(t=round(t, 1), genre=genre, bpm=round(bpm, 1),
                        bars=bars, fitness=round(float(scores[best]), 3),
                        ramp=round(ramp, 2), pitch_semi=round(semi_base, 2)))
        t += slot_dur
        slot_i += 1
    return events, kick_times


# ------------------------------------------------------------------ stutters

def apply_stutters(track, rng, bank, kick_times_beat, log):
    """Random count/length stutter events; a kick plays at each start."""
    n_events = int(rng.integers(3, 9))
    L = len(track)
    dur = L / SR
    starts = []
    for _ in range(n_events * 3):
        if len(starts) >= n_events:
            break
        s = 2.0 + (dur - 8.0) * (float(rng.random()) ** 0.55)   # later-biased
        if all(abs(s - q) > 4.0 for q in starts):
            starts.append(s)
    out = track.copy()
    extra = []
    for s in sorted(starts):
        ev_len = float(rng.uniform(0.35, 1.6))
        slice_s = float(rng.choice(np.array([0.03, 0.05, 0.08, 0.12])))
        i0 = int(s * SR)
        sl0 = max(0, i0 - int(slice_s * SR))
        sl1 = i0
        if sl1 - sl0 < 64 or i0 + int(ev_len * SR) > L:
            continue
        # snap to the nearest beat kick with probability
        if kick_times_beat and rng.random() < 0.7:
            arr = np.array(kick_times_beat)
            i0 = int(arr[np.argmin(np.abs(arr - s))] * SR)
            sl1 = i0
            sl0 = max(0, i0 - int(slice_s * SR))
            if sl1 - sl0 < 64:
                continue
        n_rep = max(2, int(ev_len / slice_s))
        seg = track[sl0:sl1].copy()
        mode = str(rng.choice(np.array(["const", "ramp", "ladder"])))
        ramp = float(rng.uniform(0.75, 1.3))          # <1 grows, >1 shrinks
        rep = seg
        built = [rep]
        total = len(rep)
        want = int(ev_len * SR)
        while total < want:
            if mode == "ladder":
                rep = gl.resample_grain(rep, 1.059, "hermite")
            elif mode == "ramp":
                n_len = int(np.clip(len(rep) * ramp, 256, int(0.4 * SR)))
                pos = np.linspace(0, len(rep) - 1, n_len)
                rep = np.stack([np.interp(pos, np.arange(len(rep)), rep[:, c])
                                for c in range(2)], axis=1).astype(np.float32)
            built.append(rep)
            total += len(rep)
        block = np.concatenate(built)[:want]
        f = min(len(block) // 4, 512)
        block[:f] *= np.linspace(0, 1, f)[:, None]
        block[-f:] *= np.linspace(1, 0, f)[:, None]
        end = min(L, i0 + len(block))
        out[i0:end] = block[: end - i0]
        # THE KICK at the start of the stutter event (ramped fx like the beat)
        p = min(1.0, (i0 / SR) / dur)
        ramp = float(np.clip(p * 1.8, 0.0, 1.0))
        emit(extra, bank.get("kick"), i0 / SR, 1.0, rng=rng,
             phaser_depth=0.5 * ramp * rng.uniform(0.5, 1.0),
             semis=rng.uniform(-2, 2) * ramp)
        if rng.random() < 0.5:
            extra.append((i0, ar.sub_thump(SR, 0.9, rng)))
        log.append(dict(t=round(i0 / SR, 2), dur=round(len(block) / SR, 2),
                        slice=round(slice_s, 3), mode=mode))
    return out, extra


# ------------------------------------------------------------------ duck + mix

def duck(track, kick_times, depth=0.35, tau=0.14):
    n = len(track)
    env = np.ones(n, np.float32)
    t = np.arange(n) / SR
    for tk in kick_times:
        i0 = int(tk * SR)
        i1 = min(n, i0 + int(0.35 * SR))
        if i0 >= n:
            continue
        seg = np.exp(-(t[i0:i1] - tk) / tau)
        env[i0:i1] = np.minimum(env[i0:i1], 1.0 - depth * seg)
    return track * env[:, None]


def mix(track, events_list, track_gain=0.9):
    L = len(track)
    audio = track.astype(np.float32) * track_gain
    for events in events_list:
        for t, y in events:
            i0 = int(t) if isinstance(t, (int, np.integer)) else int(t * SR)
            if i0 >= L:
                continue
            e = min(L, i0 + len(y))
            audio[i0:e] += y[: e - i0]
    return audio


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description="Clubbify: stutter-kick gates + GA-evolved genre-morphing beat.")
    ap.add_argument("inputs", nargs="*", help="files or folders (default: final_review)")
    ap.add_argument("--sounds", default="audio_lab/sounds")
    ap.add_argument("-o", "--out", default="audio_lab/clubbed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duck", type=float, default=0.35)
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    files = []
    for a in (args.inputs or ["audio_lab/final_review"]):
        p = Path(a)
        files += sorted(p.glob("*.wav")) if p.is_dir() else [p]
    if not files:
        ap.error("no input wavs")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    mpath = outdir / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.load(open(mpath))
        except Exception:
            manifest = {}

    for fi, f in enumerate(files):
        pseed = args.seed + 7919 * fi + 71
        prng = np.random.default_rng(pseed)
        bank = load_drum_bank(args.sounds, prng)
        track, sr = sf.read(str(f), always_2d=True)
        if sr != SR:
            n = int(len(track) * SR / sr)
            track = np.stack([np.interp(np.linspace(0, len(track) - 1, n),
                                        np.arange(len(track)), track[:, c])
                              for c in range(2)], axis=1)
        track = np.ascontiguousarray(track, np.float32)
        print(f"{Path(f).stem}: {len(track)/SR/60:.1f} min")

        slot_log, stutter_log = [], []
        beat_events, kick_times = render_beat_slots(len(track) / SR, prng,
                                                   bank, slot_log)
        track2, stutter_kicks = apply_stutters(track, prng, bank,
                                               kick_times, stutter_log)
        track2 = duck(track2, kick_times, depth=args.duck)
        audio = mix(track2, [beat_events, stutter_kicks])
        f0 = int(0.02 * SR)
        audio[:f0] *= np.linspace(0, 1, f0)[:, None]
        audio[-f0:] *= np.linspace(1, 0, f0)[:, None]
        pk = np.abs(audio).max()
        if pk > 1e-4:
            audio *= 0.97 / pk
        wav = outdir / f"club_{Path(f).stem}.wav"
        gl.write_wav(wav, SR, audio)
        fits = [s["fitness"] for s in slot_log]
        print(f"  {wav.name}: {len(slot_log)} genre slots "
              f"(fitness {fits[0]:.2f} -> {np.mean(fits[-4:]):.2f}), "
              f"{len(stutter_log)} stutter-kick events -> "
              f"{audio.shape[0]/SR/60:.2f} min")
        manifest[str(wav)] = dict(seed=pseed, source=str(f),
                                  slots=slot_log, stutters=stutter_log,
                                  fitness_first=fits[0],
                                  fitness_last_recent=round(float(np.mean(fits[-4:])), 3))
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"done -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())