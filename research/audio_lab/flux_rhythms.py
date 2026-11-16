#!/usr/bin/env python3
"""
flux_rhythms.py — drum core + source textures, patterns in constant flux.

Fourth-generation instrument. The design brief:

- DRUMS ONLY from the sample library: kick, snare, rimshot, hihat (the sub-bass
  thump under kicks is synthesized). Everything else in the mix is...
- ...YOUR ORIGINAL SOUND: the texture voices (tex/air) are sliced from the
  trimmed source recordings (audio_lab/trimmed/), processed and highpassed out
  of the kick's lane.
- CONSTANT FLUX: the piece is a sequence of short phrases (1-3 seconds,
  quantised to beats). At every phrase boundary each voice gets a COMPLETELY
  NEW pattern expression in the alien-lisp language — denser and deeper than
  alien_rhythms (more wrappers, more polymeter, velocity masks, 32nd-note
  hi-hat rates). Small doses of memory (phrase carry-over, theme callbacks)
  keep it musical instead of chaotic.
- Tempo stays put (with gentle wander); the flux does the moving.

Every piece ships a _score.txt with the full phrase-by-phrase lispy score —
pasteable into [alien] in Pure Data.

Usage:
    python3 flux_rhythms.py audio_lab/sounds audio_lab/trimmed -o audio_lab/flux_rhythms
    python3 flux_rhythms.py audio_lab/sounds audio_lab/trimmed -o out --pieces 12 --seed 5
    python3 flux_rhythms.py audio_lab/sounds audio_lab/experiments -o out --bpm 168
    python3 flux_rhythms.py audio_lab/sounds audio_lab/trimmed -o out \
        --voice 'kick=(euclid 4 16)' --voice 'hat=(euclid 11 16 3)'
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import granular_lab as gl
import alien_rhythms as ar

STEP = 16
DRUM_FOLDERS = {"kick": "kick", "snare": "snare", "rimshot": "rim", "hihat": "hat"}

# denser, deeper profiles than alien_rhythms — this stage wants complexity
FLUX_PROFILE = {
    "kick":  dict(k=(3, 9), val="hi", steady=0.18, poly=0.20),
    "snare": dict(k=(2, 7), val="mid", poly=0.25),
    "rim":   dict(k=(3, 10), val="lo", poly=0.40),
    "hat":   dict(k=(6, 15), val="lo", poly=0.45),
    "tex":   dict(k=(2, 6), val="mid", poly=0.35),
    "air":   dict(k=(2, 6), val="lo", poly=0.35),
}
# per-voice chance of running the pattern at double/half step-rate
RATE_TABLE = {
    "kick":  [(1.0, 0.85), (0.5, 0.15)],
    "snare": [(1.0, 0.90), (2.0, 0.10)],
    "rim":   [(2.0, 0.35), (1.0, 0.60), (0.5, 0.05)],
    "hat":   [(2.0, 0.45), (1.0, 0.50), (0.5, 0.05)],
    "tex":   [(0.5, 0.35), (1.0, 0.65)],
    "air":   [(0.5, 0.35), (1.0, 0.65)],
}
BPM_ZONES = [(128.0, 134.0), (136.0, 145.0), (136.0, 145.0), (168.0, 176.0)]


def flux_gen_pattern(rng, prof):
    """Like alien_rhythms.gen_pattern, but denser: mask almost always,
    1-4 structural wrappers, deeper nesting."""
    pat = ar.gen_gate(rng, prof)
    if rng.random() < 0.7:
        pat = ["mask", ar.gen_value_expr(rng, prof["val"]), pat]
    for _ in range(int(rng.integers(1, 5))):
        pat = ar._wrap(rng, pat, prof)
    return pat


def draw_rate(rng, voice):
    table = RATE_TABLE.get(voice, [(1.0, 1.0)])
    r, acc = rng.random(), 0.0
    for rate, p in table:
        acc += p
        if r < acc:
            return rate
    return table[-1][0]


def load_drum_bank(sounds_dir, rng, per_role=32):
    """Only the drum folders: kick, snare, rimshot, hihat."""
    bank = {}
    for folder, role in DRUM_FOLDERS.items():
        sub = Path(sounds_dir) / folder
        if not sub.is_dir():
            print(f"  warning: no {folder}/ folder, voice '{role}' will be silent")
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
            if sr != 44100:
                a = ar._resample_linear(ar._stereo(a), sr, 44100)
            s = ar._prep_sample(a, 44100, role, f.name)
            if s is not None:
                bank.setdefault(role, []).append(s)
    return bank


@dataclass
class Phrase:
    dur: float
    beats: int
    bpm: float
    swing: float
    anchor: bool
    sub_p: float
    fill: bool
    exprs: dict = field(default_factory=dict)
    rates: dict = field(default_factory=dict)


def build_phrases(rng, voices, pin, piece_len, base_bpm, args):
    beat = 60.0 / base_bpm
    phrases, t0, prev, themes = [], 0.0, {}, []
    while t0 < piece_len - 1e-6:
        bpm = base_bpm * (1.0 + rng.uniform(-0.04, 0.04))
        beat = 60.0 / bpm
        target = rng.uniform(args.phrase_min, args.phrase_max)
        n_beats = max(1, int(round(target / beat)))
        dur = n_beats * beat
        if t0 + dur > piece_len:
            if piece_len - t0 < 0.5 * beat:      # tail shorter than half a beat: stop
                break
            dur = piece_len - t0
            n_beats = max(1, int(round(dur / beat)))
            dur = n_beats * beat if t0 + n_beats * beat <= piece_len else dur
        exprs, rates = {}, {}
        for v in voices:
            if v in pin:
                exprs[v] = pin[v]
            elif prev.get(v) is not None and rng.random() < 0.12:
                exprs[v] = prev[v]                      # slight carry-over
            elif themes and rng.random() < 0.10:
                exprs[v] = themes[int(rng.integers(len(themes)))]   # motif callback
            else:
                exprs[v] = flux_gen_pattern(rng, FLUX_PROFILE[v])
                themes.append(exprs[v])
                if len(themes) > 48:
                    themes.pop(0)
            rates[v] = draw_rate(rng, v)
        ph = Phrase(dur, n_beats, bpm,
                     float(np.clip(rng.uniform(-0.02, 0.18), 0.0, 0.2)),
                     rng.random() < 0.35,               # kick anchored to downbeat
                     float(rng.uniform(0.3, 0.7)),       # sub-thump under kick
                     rng.random() < 0.30,               # end-of-phrase fill
                     exprs, rates)
        phrases.append(ph)
        prev = exprs
        t0 += dur
    return phrases


def emit_hit(events, pool, role, t, val, rng, sr, ch, state):
    sm = pool[int(rng.integers(len(pool)))]
    y = ar.process_hit(sm.data, sr, rng, ch, role)
    amp = float(val) * 10.0 ** (rng.uniform(-1.2, 1.2) / 20.0)
    y = y * gl.make_env("const", len(y))[:, None] * amp
    if role == "hat":
        state["hatpan"] = -state.get("hatpan", 1.0)
        pan = 0.55 * state["hatpan"] + rng.uniform(-0.1, 0.1)
    elif role in ("tex", "air"):
        pan = rng.uniform(-0.8, 0.8)
    else:
        pan = rng.uniform(-0.15, 0.15)
    th = (pan + 1.0) * (np.pi / 4.0)
    y[:, 0] *= np.cos(th)
    y[:, 1] *= np.sin(th)
    events.append((max(0, int(t * sr)), y))


def render_piece(phrases, voices, bank, rng, sr, ch):
    events, t0, state = [], 0.0, {"hatpan": 1.0}
    for ph in phrases:
        stepdur = 60.0 / ph.bpm / 4.0
        for v in voices:
            pool = bank.get(v)
            if not pool:
                continue
            try:
                pat = ar.normalize_velocities(ar.ev(ph.exprs[v], rng))
            except Exception as e:
                print(f"  warning: pattern for {v} failed ({e})")
                continue
            L = len(pat)
            if L == 0:
                continue
            rate = ph.rates.get(v, 1.0)
            grid = stepdur / rate
            steps = max(1, int(round(ph.dur / grid)))
            hits = []
            for s in range(steps):
                val = pat[s % L]
                if val is None or val <= 0.02:
                    continue
                hits.append((s, float(val)))
            if v == "kick" and ph.anchor and not any(s == 0 for s, _ in hits):
                hits.append((0, 1.0))
            drop_tail = ph.fill and v == "kick" and rng.random() < 0.4
            for s, val in hits:
                t = t0 + s * grid
                if rate == 1.0 and s % 2 == 1:
                    t += ph.swing * stepdur
                if drop_tail and s * grid > ph.dur - 0.5 * 60.0 / ph.bpm:
                    continue
                t += rng.uniform(-1, 1) * ch["jit_ms"] / 1000.0
                emit_hit(events, pool, v, t, val, rng, sr, ch, state)
                if v == "kick" and rng.random() < ph.sub_p:
                    events.append((max(0, int(t * sr)), ar.sub_thump(sr, val, rng)))
            if ph.fill and v in ("snare", "rim") and rng.random() < 0.75:
                k = int(3 + rng.integers(0, 4))          # accelerating roll
                beat = 60.0 / ph.bpm
                for j in range(k):
                    tt = t0 + ph.dur - beat + beat * (j / k) ** 1.5
                    emit_hit(events, pool, v, tt, 0.9 - 0.08 * j, rng, sr, ch, state)
        t0 += ph.dur
    return events


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(
        description="Drum core (kick/snare/rim/hihat) + source textures, "
                    "patterns regenerating every 1-3 seconds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("sounds", help="one-shot sample folder (uses kick/snare/rimshot/hihat)")
    ap.add_argument("corpus", help="texture source: trimmed recordings or experiments")
    ap.add_argument("-o", "--out", default="flux_rhythms")
    ap.add_argument("--pieces", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bpm", type=float, help="pin tempo")
    ap.add_argument("--piece-len", type=float, default=0.0,
                    help="seconds per piece (0 = random 25-50)")
    ap.add_argument("--phrase-min", type=float, default=1.0)
    ap.add_argument("--phrase-max", type=float, default=3.0)
    ap.add_argument("--tex-boost", type=float, default=1.5,
                    help="gain multiplier on the texture voices")
    ap.add_argument("--voice", dest="voices", action="append", default=[],
                    metavar="NAME=(expr)", help="pin a voice pattern (repeatable)")
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")
    if args.phrase_min >= args.phrase_max:
        ap.error("--phrase-min must be below --phrase-max")

    rng = np.random.default_rng(args.seed)
    print("loading drum bank (kick / snare / rimshot / hihat) ...")
    bank = load_drum_bank(args.sounds, rng)
    if "kick" not in bank:
        ap.error("no kick samples found")

    cfiles = gl.gather_inputs([args.corpus])
    if not cfiles:
        ap.error(f"no wav files found in corpus {args.corpus!r}")
    print(f"loading texture layer from {args.corpus} ({len(cfiles)} files)...")
    tex = ar.load_texture(cfiles, rng, cap=1400, per_file=200)
    for role, boost in (("tex", args.tex_boost), ("air", args.tex_boost)):
        for smp in tex.get(role, []):
            smp.data = smp.data * boost
    bank.update(tex)
    print("bank: " + ", ".join(f"{r}x{len(v)}" for r, v in sorted(bank.items())))

    pin = {}
    for spec in args.voices:
        name, _, src = spec.partition("=")
        try:
            pin[name.strip().lower()] = ar.parse_pattern(src.strip())
        except ValueError as e:
            ap.error(f"bad --voice {spec!r}: {e}")

    voices = [v for v in ("kick", "snare", "rim", "hat", "tex", "air") if v in bank]
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
        pseed = args.seed + 7919 * i + 23
        prng = np.random.default_rng(pseed)
        ch = ar.piece_character(prng)
        zone = ((args.bpm, args.bpm) if args.bpm
                else BPM_ZONES[int(prng.integers(len(BPM_ZONES)))])
        base_bpm = float(prng.uniform(zone[0], zone[1]))
        piece_len = args.piece_len or float(prng.uniform(25.0, 50.0))
        phrases = build_phrases(prng, voices, pin, piece_len, base_bpm, args)
        events = render_piece(phrases, voices, bank, prng, 44100, ch)
        if not events:
            print(f"  flux{i:02d}: no events, skipped")
            continue
        audio = gl.finalize(events, 44100, 0.97)
        wav = outdir / f"flux{i:02d}_{pseed}.wav"
        gl.write_wav(wav, 44100, audio)
        score = outdir / f"flux{i:02d}_{pseed}_score.txt"
        with open(score, "w") as f:
            f.write(f";; flux{i:02d} — seed {pseed} — ~{base_bpm:.0f} bpm, "
                    f"{len(phrases)} phrases of {args.phrase_min:.0f}-{args.phrase_max:.0f}s\n"
                    f";; voices: {' '.join(voices)}\n"
                    ";; paste any pattern into [alien] in Pure Data\n\n")
            for pi, ph in enumerate(phrases):
                f.write(f"[phrase {pi + 1:02d}] {ph.dur:.2f}s ({ph.beats} beats) "
                        f"bpm {ph.bpm:.0f} swing {ph.swing:.2f}"
                        f"{' anchor' if ph.anchor else ''}{' fill' if ph.fill else ''}\n")
                for v in voices:
                    f.write(f"{v}: {ar.to_src(ph.exprs[v])}"
                            f"  ;; rate {ph.rates.get(v, 1.0):g}\n")
                f.write("\n")
        print(f"  flux{i:02d} ({audio.shape[0] / 44100:5.1f}s, {len(phrases)} phrases, "
              f"~{base_bpm:.0f}bpm) -> {wav.name} (+ score)")
        manifest[str(wav)] = dict(
            seed=pseed, bpm=base_bpm, character=ch, voices=voices,
            phrase_len=[args.phrase_min, args.phrase_max],
            phrases=[dict(dur=p.dur, beats=p.beats, bpm=p.bpm, swing=p.swing,
                          anchor=p.anchor, sub_p=p.sub_p, fill=p.fill,
                          rates=p.rates,
                          exprs={v: ar.to_src(p.exprs[v]) for v in voices})
                     for p in phrases],
            score_file=str(score))

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"done -> {outdir}/ (each piece ships a phrase-by-phrase _score.txt)")


if __name__ == "__main__":
    sys.exit(main())