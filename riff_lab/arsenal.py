"""arsenal: the FULL transform arsenal — open-ended, chained, crossbred.

The cleanliness doctrine is retired. Novelty search needs the whole
character space, and ears.defects() is the only safety net (clicks /
silence / clipping / NaN — technical failure, never taste). So:

- the ENTIRE oneshot_lab registry is available to every voice (spectral,
  glitch, databend, timestretch, harmonics, pitch, novel) — the dirt is
  back in;
- transforms CHAIN (oneshot_lab.engine.mutate's idea): 1-3 stages, so a
  sound travels far from its source;
- CROSSBREEDING fuses two different pool sounds into a new timbre
  (novel.spectral_dna / env_splice / formant_walk with a real partner) —
  the move that makes a sound unlike anything it came from;
- partial_resynth / founder_synth tune bass variants onto the riff's
  notes, and tomb-style synth founders grow entirely new timbres.

Every stage is clamped to a tight oneshot and wrapped so it can never be
fatal; a failed/degenerate stage is skipped.
"""
import numpy as np

from oneshot_lab import core
from oneshot_lab import glitch, harmonics, novel, pitchfx, spectral, \
    timestretch, warp
from oneshot_lab.registry import TRANSFORMS, FAMILIES
from oneshot_lab.engine import random_params

MAX_SHOT_S = 1.5

# ---- OPTIONAL: tomb's second transform family (Karplus-Strong resonate,
# reaction-diffusion, formant, granular sieve, wavesort, logistic chaos...)
# bridged in as extra voice transforms — more glitch/timbre character for
# novelty. tomb ships in this repo under research/tomb; it is auto-discovered
# there (or point AUDIO_RESEARCH_TOMB_SRC elsewhere). If it can't be imported
# the arsenal simply runs without these ops.
import os as _os
import sys as _sys
from pathlib import Path as _Path
_tomb_src = _os.environ.get(
    "AUDIO_RESEARCH_TOMB_SRC",
    str(_Path(__file__).resolve().parent.parent / "research" / "tomb" / "src"))
if _tomb_src and _Path(_tomb_src).is_dir() and _tomb_src not in _sys.path:
    _sys.path.insert(0, _tomb_src)
try:
    from tomb import ops as _tomb
    _TOMB_OPS = ["resonate", "comb", "decimate", "logistic", "gate", "wavesort",
                 "diffuse", "propagate", "react", "automate", "rephase",
                 "crush", "bitrot", "sub_resonate", "harmonic_seed",
                 "granular_sieve", "formant", "lowpass_sweep"]
    _TOMB_OPS = [o for o in _TOMB_OPS if o in _tomb.OPS]
except Exception:
    _tomb, _TOMB_OPS = None, []


def _tomb_apply(name, x, sr, rng, max_s):
    """Apply a tomb op (mono-domain) per channel, clamp to an oneshot."""
    try:
        y = core.st(lambda c: _tomb.mutate(c, sr, rng, name)[0], np.asarray(x, np.float32))
        y = core.clamp_dur(core.safe(y), sr, max_s)
        if y is None or y.shape[0] < 256:
            return x, False
        return y.astype(np.float32), True
    except Exception:
        return x, False

# voice-appropriate transform pools — WIDE now (dirt included). Only the
# genuinely wrong-shaped transforms per voice are left out; the rest is
# fair game and the defect gate culls the casualties.
_EXCLUDE = {
    # these make long drones / washes — bad as a punchy BASS oneshot
    "bass": {"paulstretch", "paulstretch_hyper", "spectral_freeze",
             "stretch_modulate", "pv_stretch", "spectral_smear",
             "shimmer_harmonizer", "lpc_reanimate", "freeze_loop_stretch",
             "wavetable_stretch", "oscillating_timestretch"},
    "noise": set(),
    "fill": set(),
}
_CROSSBREED = ("spectral_dna", "env_splice", "formant_walk")


def _voice_names(voice):
    base = [n for n in TRANSFORMS if n not in _EXCLUDE.get(voice, set())]
    # tomb ops are tone/texture creators — fine for every voice
    return base + [f"tomb:{o}" for o in _TOMB_OPS]


def _apply1(name, x, sr, rng, max_s):
    """One transform stage, clamped to an oneshot. Returns (y, ok)."""
    if name.startswith("tomb:"):
        return _tomb_apply(name[5:], x, sr, rng, max_s)
    try:
        fn, fam, defaults, needs_bytes, _, _ = TRANSFORMS[name]
        params = random_params(name, rng)
        if needs_bytes:
            y = fn(x, sr, rng, path=None, **params)
        else:
            y = fn(x, sr, rng, **params)
        y = core.clamp_dur(core.safe(y), sr, max_s)
        if y is None or y.shape[0] < 256:
            return x, False
        return y.astype(np.float32), True
    except Exception:
        return x, False


def transform_chain(x, sr, rng, voice, n_stages, max_s=MAX_SHOT_S):
    """Apply a chain of n_stages random voice-appropriate transforms.
    Returns (audio, [names applied])."""
    names = _voice_names(voice)
    y = np.asarray(x, np.float32)
    used = []
    for _ in range(max(0, int(n_stages))):
        name = str(rng.choice(names))
        y2, ok = _apply1(name, y, sr, rng, max_s)
        if ok:
            y = y2
            used.append(name)
    return y, used


def crossbreed(x, partner, sr, rng, max_s=MAX_SHOT_S):
    """Fuse x with a partner sound into a new timbre (partial-level).
    Returns (audio, name)."""
    name = str(rng.choice(_CROSSBREED))
    fn = getattr(novel, name)
    try:
        if name == "spectral_dna":
            y = fn(x, sr, rng, partner=partner, cross=float(rng.uniform(0.5, 1.0)))
        elif name == "env_splice":
            y = fn(x, sr, rng, partner=partner, graft=float(rng.uniform(0.0, 0.7)))
        else:  # formant_walk
            y = fn(x, sr, rng, partner=partner, one_way=bool(rng.random() < 0.4))
        y = core.clamp_dur(core.safe(y), sr, max_s)
        if y is None or y.shape[0] < 256:
            return x, None
        return y.astype(np.float32), name
    except Exception:
        return x, None


# ----------------------------------------------------------------- bass tuning

def _harm_amplitudes(x, sr, f0, n_partials=10):
    m = core.mono(x)
    seg = m[: int(0.5 * sr)]
    if len(seg) < 256:
        return np.ones(n_partials) / n_partials
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) * 2 / len(seg)
    fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
    amps = []
    for h in range(1, n_partials + 1):
        c = h * f0
        if c > sr / 2 - 200:
            amps.append(0.0); continue
        band = (fr > c * 0.93) & (fr < c * 1.07)
        amps.append(float(S[band].max()) if band.any() else 0.0)
    return np.array(amps) / (max(amps) + 1e-9)


def partial_resynth(x, sr, rng, note, dur_s=None, drive=0.0, detune_drift=0.004):
    """Re-ring the source's harmonic series AT a target MIDI note —
    pitch-perfect tuned bass variant (clean by construction)."""
    if dur_s is None:
        dur_s = float(rng.uniform(0.25, 0.7))
    f0 = core.dominant_f0(x, sr, default=55.0)
    amps = _harm_amplitudes(x, sr, f0, 10)
    f_target = 440.0 * 2 ** ((note - 69) / 12.0)
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    y = np.zeros(n, np.float64)
    for h, a in enumerate(amps, start=1):
        if a < 0.06 or h * f_target > sr / 2 - 300:
            continue
        det = 1.0 + detune_drift * h * rng.uniform(0.3, 1.0) * \
            np.sin(2 * np.pi * rng.uniform(0.8, 4.0) * t)
        y += a * np.sin(2 * np.pi * h * f_target * det * t +
                        rng.uniform(0, 6.28)) / h ** 0.4
    atk_s = max(0.008, 1.2 / f_target)
    atk_n = min(n - 1, int(atk_s * sr))
    env = np.ones(n)
    env[: atk_n] = 0.5 - 0.5 * np.cos(np.pi * np.arange(atk_n) / max(1, atk_n - 1))
    env *= np.exp(-t * float(rng.uniform(4.0, 9.0)))
    rel = int(0.012 * sr)
    if rel < n:
        env[n - rel:] *= np.linspace(1, 0, rel)
    y *= env
    if drive > 0:
        y = np.tanh(y * (1 + drive * 3)) / np.tanh(1 + drive * 3)
    pk = np.abs(y).max()
    if pk > 1e-6:
        y = y / pk * 0.9
    return np.stack([y, np.roll(y, int(0.0015 * sr))], axis=1).astype(np.float32)


def founder_synth(rng, note, sr=44100, dur_s=None, mode=None):
    """Grow a bass/stab variant from the note itself — a small harmonic
    stack (bomb's founder) or a formant/granular seed (tomb). New timbre,
    no source needed."""
    if dur_s is None:
        dur_s = float(rng.uniform(0.25, 0.6))
    f = 440.0 * 2 ** ((note - 69) / 12.0)
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    y = np.zeros(n, np.float64)
    n_part = int(rng.integers(4, 10))
    for h in range(1, n_part + 1):
        a = 1.0 / h ** float(rng.uniform(0.3, 0.8))
        if rng.random() < 0.18:
            a *= 0.25                           # a gap in the stack = character
        y += a * np.sin(2 * np.pi * h * f * t + rng.uniform(0, 6.28))
    if mode == "fm" or (mode is None and rng.random() < 0.4):
        idx = float(rng.uniform(1.0, 6.0))
        rat = float(rng.choice([0.5, 1.0, 1.5, 2.0, 3.0]))
        y += 0.6 * np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * rat * t))
    y *= np.exp(-t * float(rng.uniform(4, 9)))
    atk_n = min(n - 1, int(max(0.008, 1.2 / f) * sr))
    y[: atk_n] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(atk_n) / max(1, atk_n - 1))
    y /= (np.abs(y).max() + 1e-9)
    return np.stack([y, y], axis=1).astype(np.float32)
