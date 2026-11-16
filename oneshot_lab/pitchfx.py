"""Pitch family: tape-style pitch shift (time+pitch together), phase-vocoder
pitch shift (time holds), glissando sweeps, vibrato (oscillating pitch)."""
import numpy as np

from .core import resample, resample_curve, semis_to_speed, st


def pitch_shift_tape(x, sr, rng, semis=None, max_s=None):
    """Tape pitch: resample — pitch and duration change together."""
    if semis is None:
        semis = float(rng.choice([-12, -7, -5, 5, 7, 12,
                                 rng.uniform(-9, 9)]))
    return resample(x, semis_to_speed(semis))


def pitch_shift_pv(x, sr, rng, semis=None, n_fft=2048, max_s=None):
    """Pitch shift with duration preserved: resample (pitch moves) then
    phase-vocoder-stretch back to the original length."""
    from .timestretch import pv_stretch
    if semis is None:
        semis = float(rng.choice([-12, -7, -5, 5, 7, 12,
                                 rng.uniform(-9, 9)]))
    r = semis_to_speed(semis)
    y = resample(x, r)
    y = pv_stretch(y, sr, rng, factor=r, n_fft=n_fft, max_s=1e9)
    n = np.asarray(x).shape[0]
    if y.ndim > 1:
        out = np.zeros((n, y.shape[1]), np.float32)
        e = min(n, y.shape[0])
        out[:e] = y[:e]
    else:
        out = np.zeros(n, np.float32)
        e = min(n, len(y))
        out[:e] = y[:e]
    return out


def glissando(x, sr, rng, semis0=None, semis1=None, max_s=None):
    """Gliding pitch: rate curve from semis0 to semis1 — the hit swoops."""
    if semis0 is None:
        semis0 = float(rng.uniform(-12, 0))
    if semis1 is None:
        semis1 = float(rng.uniform(-3, 12))
    s0, s1 = semis_to_speed(semis0), semis_to_speed(semis1)
    x = np.asarray(x, np.float32)
    n = x.shape[0]
    speed = np.linspace(s0, s1, int(n / max(0.05, 0.5 * (s0 + s1))) + 1)
    pos = np.cumsum(speed)
    from .core import interp_hermite

    def one(y):
        return interp_hermite(y, pos).astype(np.float32)

    return st(one, x)


def vibrato(x, sr, rng, rate_hz=None, depth_cents=35, dur_mult=1.0, max_s=None):
    """Oscillating pitch: sinusoidal vari-speed — the pitch warbles around
    the original (and time breathes slightly with it)."""
    if rate_hz is None:
        rate_hz = float(rng.uniform(3.0, 9.0))
    x = np.asarray(x, np.float32)
    if dur_mult > 1.01:
        reps = int(np.ceil(dur_mult))
        x = np.tile(x, (reps, 1)) if x.ndim > 1 else np.tile(x, reps)
    n = x.shape[0]
    t = np.arange(n) / sr
    rate = 1.0 + (depth_cents / 1200.0) * np.log(2.0) * np.sin(2 * np.pi * rate_hz * t)
    return resample_curve(x, rate)