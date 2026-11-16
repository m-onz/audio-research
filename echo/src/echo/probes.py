"""probes: the known-answer test sounds.

To measure whether an encoder hears 'same' and 'different' the way a
listener would, we need pairs whose truth we already know:

    NEAR DUPLICATES (must land close): gain, sub-15ms shift, gentle EQ
    tilt, -40 dB noise floor, a 320 kbps mp3 round-trip (the metabolism's
    generational loss — chasing codec artifacts as 'novel' would poison
    the search), a resample round-trip, and same-specimen seed jitter
    (handled by the bench, which renders the specimens).

    CONTRASTS (must land far): synthetic archetypes — sines, noise,
    clicks, sweeps, grains — that no breeding run should ever confuse.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["decode", "write_wav", "perturbations", "PERTURB_NAMES",
           "contrasts", "CONTRAST_NAMES", "SR"]

SR = 44100


# ------------------------------------------------------------------ audio io

def decode(path, sr: int = SR, stereo: bool = False) -> np.ndarray:
    """Any audio file -> float32 mono (n,) or stereo (n, 2). ffmpeg pipe,
    like every decoder in this ecosystem."""
    ac = 2 if stereo else 1
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", str(ac), "-ar", str(sr), "pipe:1"],
        capture_output=True, check=True).stdout
    x = np.frombuffer(out, np.float32)
    return x.reshape(-1, ac) if stereo else x


def write_wav(path, audio, sr: int = SR) -> str:
    """Float audio -> 16-bit wav via ffmpeg (the ecosystem writer)."""
    x = np.asarray(audio, np.float32)
    if x.ndim == 2 and x.shape[1] == 2:
        inter = np.empty(len(x) * 2, np.float32)
        inter[0::2], inter[1::2] = x[:, 0], x[:, 1]
        x, ch = inter, 2
    else:
        x, ch = x.reshape(-1), 1
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(sr),
         "-ac", str(ch), "-i", "pipe:0", str(path)],
        input=pcm.tobytes(), check=True)
    return str(path)


# ------------------------------------------------------------------ near duplicates

def _gain(x, sr, rng):
    return x * float(rng.choice([0.9, 1.1]))


def _shift(x, sr, rng):
    n = int(sr * float(rng.uniform(0.003, 0.015)))
    out = np.zeros_like(x)
    if n > 0:
        out[n:] = x[:-n]
    return out


def _eq_tilt(x, sr, rng):
    """Gentle spectral tilt (~1.5 dB/oct) — a different mastering, same
    performance."""
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    tilt = (f + 1.0) ** (float(rng.choice([-0.25, 0.25])))
    X = X * (tilt[:, None] if X.ndim == 2 else tilt)
    return np.fft.irfft(X, n=len(x), axis=0)


def _noise_floor(x, sr, rng):
    return x + rng.normal(0, 10 ** (-40 / 20), x.shape).astype(x.dtype)


def _mp3_round(x, sr, rng):
    """320 kbps mp3 round-trip: the metabolism's loss, must read as
    'the same sound, slightly aged' — never as novelty."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.wav"
        q = Path(td) / "p.mp3"
        write_wav(p, x, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(p),
             "-b:a", "320k", str(q)], check=True)
        return decode(q, sr, stereo=(x.ndim == 2))


def _resample_round(x, sr, rng):
    """44.1 kHz -> 32 kHz -> 44.1 kHz: bandwidth loss, same identity."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.wav"
        q = Path(td) / "p32.wav"
        write_wav(p, x, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(p),
             "-ar", "32000", str(q)], check=True)
        return decode(q, sr, stereo=(x.ndim == 2))


PERTURBS = {
    "gain": _gain,
    "shift": _shift,
    "eq_tilt": _eq_tilt,
    "noise_floor": _noise_floor,
    "mp3_round": _mp3_round,
    "resample_round": _resample_round,
}
PERTURB_NAMES = list(PERTURBS)


def perturbations(audio, sr: int = SR, rng=None, names=None) -> dict:
    """audio -> {name: perturbed audio} for each requested perturbation."""
    rng = rng or np.random.default_rng(0)
    out = {}
    for n in (names or PERTURB_NAMES):
        try:
            out[n] = PERTURBS[n](audio, sr, rng)
        except Exception:
            continue                                  # ffmpeg missing etc.
    return out


# ------------------------------------------------------------------ contrasts

def _sine(sr, n, rng):
    t = np.arange(n) / sr
    return (0.6 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _sine_harmonics(sr, n, rng):
    t = np.arange(n) / sr
    x = sum((0.5 / (i + 1)) * np.sin(2 * np.pi * 220 * (i + 1) * t)
            for i in range(5))
    return (x / np.abs(x).max() * 0.8).astype(np.float32)


def _bright_sine(sr, n, rng):
    t = np.arange(n) / sr
    return (0.5 * np.sin(2 * np.pi * 2400 * t)).astype(np.float32)


def _white(sr, n, rng):
    return (rng.normal(0, 0.25, n)).astype(np.float32)


def _pinkish(sr, n, rng):
    """1/f-ish noise: cumsum of white (bounded by highpass detrending)."""
    w = rng.normal(0, 0.02, n)
    p = np.cumsum(w)
    p -= np.convolve(p, np.ones(2205) / 2205, mode="same")
    return (p / (np.abs(p).max() + 1e-9) * 0.7).astype(np.float32)


def _clicks(sr, n, rng):
    x = np.zeros(n, np.float32)
    period = int(sr / 8)                                # 8 Hz click weather
    x[::period] = rng.uniform(0.5, 1.0, len(x[::period]))
    return x


def _fm_sweep(sr, n, rng):
    t = np.arange(n) / sr
    f = np.geomspace(200, 2000, n)
    ph = 2 * np.pi * np.cumsum(f) / sr
    return (0.6 * np.sin(ph)).astype(np.float32)


def _grain_cloud(sr, n, rng):
    x = np.zeros(n, np.float32)
    for _ in range(int(n / sr * 24)):                  # 24 grains / sec
        s = int(rng.integers(0, n - 2000))
        d = int(rng.integers(300, 2000))
        g = np.hanning(d) * rng.normal(0, 0.4, d)
        x[s: s + d] += g.astype(np.float32)
    return x / (np.abs(x).max() + 1e-9) * 0.8


def _rumble_whistle(sr, n, rng):
    t = np.arange(n) / sr
    low = 0.5 * np.sin(2 * np.pi * 48 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t))
    hi = 0.15 * np.sin(2 * np.pi * 3150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.3 * t))
    return (low + hi).astype(np.float32)


CONTRASTS = {
    "sine220": _sine,
    "sine_harmonics": _sine_harmonics,
    "sine2400": _bright_sine,
    "white": _white,
    "pinkish": _pinkish,
    "clicks8": _clicks,
    "fm_sweep": _fm_sweep,
    "grain_cloud": _grain_cloud,
    "rumble_whistle": _rumble_whistle,
}
CONTRAST_NAMES = list(CONTRASTS)


def contrasts(sr: int = SR, seconds: float = 3.0, seed: int = 7) -> dict:
    """The synthetic archetypes — pairwise distinct by construction."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    return {name: fn(sr, n, rng) for name, fn in CONTRASTS.items()}