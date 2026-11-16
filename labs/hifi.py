"""hifi — shared high-fidelity audio helpers used by every *_lab.

Band-limited (PolyBLEP) oscillators, oversampled nonlinearities (to suppress
aliasing), a DC blocker, a soft limiter, and a 24-bit master/write. Factored
out of neuro_lab3 so all the earlier experiments can share one quality path.
"""
import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

SUBTYPE = "PCM_24"


# ----------------------------------------------------------------- band-limited osc

def _blep(ph, dt):
    y = np.zeros_like(ph)
    dt = np.maximum(dt, 1e-9)
    m = ph < dt
    t = ph[m] / dt[m]
    y[m] = t + t - t * t - 1.0
    m2 = ph > 1.0 - dt
    t = (ph[m2] - 1.0) / dt[m2]
    y[m2] = t * t + t + t + 1.0
    return y


def blsaw(ph, dt):
    """Band-limited sawtooth from normalized phase `ph` (0..1) and per-sample
    phase increment `dt` (= freq/sr)."""
    return (2.0 * ph - 1.0) - _blep(ph, dt)


def blsquare(ph, dt):
    """Band-limited square (two offset band-limited saws)."""
    return blsaw(ph, dt) - blsaw((ph + 0.5) % 1.0, dt)


# ----------------------------------------------------------------- nonlinear / clean

def _os1(x, fn, factor):
    up = resample_poly(x, factor, 1)
    y = fn(up)
    return resample_poly(y, 1, factor)[: len(x)]


def os_apply(x, fn, factor=4):
    """Apply a (possibly aliasing) nonlinearity at `factor`x oversampling,
    polyphase up/down — mono or (n, ch)."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        return _os1(x, fn, factor).astype(np.float32)
    return np.stack([_os1(x[:, c], fn, factor) for c in range(x.shape[1])],
                    axis=1).astype(np.float32)


def dc_block(x, sr, fc=22.0):
    x = np.asarray(x, np.float32)
    sos = butter(2, fc / (sr / 2), btype="high", output="sos")
    if x.ndim == 1:
        return sosfilt(sos, x).astype(np.float32)
    return np.stack([sosfilt(sos, x[:, c]) for c in range(x.shape[1])],
                    axis=1).astype(np.float32)


def soft_limit(x, drive=1.6):
    """Oversampled tanh saturation/limiter — loud without aliasing or hard clip."""
    k = max(1.0, float(drive))
    return os_apply(x, lambda u: np.tanh(u * k) / np.tanh(k))


def normalize(y, peak=0.97):
    y = np.asarray(y, np.float32)
    m = float(np.max(np.abs(y)))
    return (y * (peak / m)).astype(np.float32) if m > 1e-9 else y


def master(y, sr, drive=1.6, peak=0.97):
    """DC-block -> oversampled soft limit -> normalize. Mono or stereo in/out."""
    y = np.nan_to_num(np.asarray(y, np.float32))
    y = dc_block(y, sr)
    y = soft_limit(normalize(y), drive)
    return normalize(y, peak)


def write(path, y, sr):
    """24-bit wav write."""
    sf.write(str(path), np.asarray(y, np.float32), sr, subtype=SUBTYPE)
    return len(y) / sr
