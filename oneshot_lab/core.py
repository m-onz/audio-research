"""oneshot_lab — general-purpose oneshot mutation library.

Turn a folder of oneshots (or any sounds) into a bank of unique, weird
variations: time stretch (paulstretch, phase vocoder, wavetable), pitch and
harmonic effects, spectral transformations, glitch/databend corruption.

Extracts and generalizes the DSP developed across this project's
audio_lab / *_reprocess experiments into a reusable, batchable library.

All transforms share one signature:

    fn(x, sr, rng, **params) -> np.ndarray

    x    : float32 array, shape (n,) or (n, ch)  — usually (n, 2)
    sr   : sample rate of x
    rng  : np.random.Generator
    out  : float32 array (any length; engine normalizes + caps)
"""
from pathlib import Path

import numpy as np
import soundfile as sf

EPS = 1e-9
AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}

# ----------------------------------------------------------------- io


def read_audio(path, max_in_s=None, rng=None):
    """Read any soundfile-supported file. Returns (x (n,2) float32, sr).

    Stereo is preserved; mono is duplicated to stereo so every transform
    can assume (n, 2). Inputs longer than max_in_s are sliced to a random
    window (or centered if rng is None)."""
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    if max_in_s is not None and len(x) > max_in_s * sr:
        w = int(max_in_s * sr)
        if rng is not None:
            st = int(rng.integers(0, len(x) - w))
        else:
            st = (len(x) - w) // 2
        x = np.ascontiguousarray(x[st:st + w])
    return x, sr


def write_audio(path, y, sr, peak=0.97, subtype="PCM_16"):
    """Safe write: fix non-finite, normalize, micro-fade edges, write wav."""
    y = np.asarray(y, np.float32)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    y = safe(y)
    m = float(np.mean(y))
    if abs(m) > 0.005:                       # strip DC offset
        y = y - y.dtype.type(m)
    y = normalize(y, peak)
    y = fade_edges(y, sr=sr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)
    return y.shape[0] / sr


def find_audio(folder, recursive=False):
    """All audio files in a folder, sorted. Optionally recursive — so a
    mutation bank (oneshot_out/<transform>/...) can itself be reprocessed."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it
                  if p.suffix.lower() in AUDIO_SUFFIXES and p.is_file())


# ----------------------------------------------------------------- helpers


def mono(x):
    x = np.asarray(x, np.float32)
    return x.mean(axis=1) if x.ndim > 1 else x


def safe(y):
    y = np.asarray(y, np.float32).copy()
    y[~np.isfinite(y)] = 0.0
    return y


def normalize(y, peak=0.97):
    y = np.asarray(y, np.float32)
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m < EPS:
        return np.zeros_like(y)
    return y * (peak / m)


def fade_edges(y, s=0.005, sr=44100):
    """Tiny fades to kill clicks (s in seconds, applied per edge)."""
    y = np.asarray(y, np.float32)
    f = min(len(y) // 8, int(s * sr))
    if f > 2:
        r = np.linspace(0.0, 1.0, f, dtype=np.float32)
        if y.ndim == 1:
            y[:f] *= r
            y[-f:] *= r[::-1]
        else:
            y[:f] *= r[:, None]
            y[-f:] *= r[::-1, None]
    return y


def clamp_dur(y, sr, max_s, fade=0.03):
    """Truncate to max_s with a short fade-out at the cut."""
    y = np.asarray(y, np.float32)
    n = int(max_s * sr)
    if len(y) <= n:
        return y
    y = y[:n].copy()
    f = min(len(y) // 4, int(fade * sr))
    if f > 1:
        w = np.linspace(1.0, 0.0, f, dtype=np.float32)
        if y.ndim == 1:
            y[-f:] *= w
        else:
            y[-f:] *= w[:, None]
    return y


def tukey(n, alpha=0.5):
    """Tukey window (flat top, cosine edges)."""
    if n <= 1:
        return np.ones(1, np.float32)
    w = np.ones(n, np.float32)
    e = max(1, int(alpha * n / 2))
    r = np.linspace(0, np.pi / 2, e, endpoint=False)
    w[:e] = np.sin(r) ** 2
    w[-e:] = np.cos(r) ** 2
    return w


def st(fn, x, *args, **kw):
    """Apply a mono function per channel of (n, ch)."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        return fn(x, *args, **kw)
    outs = [fn(x[:, c], *args, **kw) for c in range(x.shape[1])]
    L = max(len(o) for o in outs)
    out = np.zeros((L, x.shape[1]), np.float32)
    for c, o in enumerate(outs):
        out[: len(o), c] = o
    return out


# ----------------------------------------------------------------- interpolation / resampling


def interp_hermite(x, pos):
    """Vectorized hermite interpolation of x at (possibly huge) float pos."""
    n = len(x)
    pos = np.clip(np.asarray(pos, np.float64), 0.0, n - 1)
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


def resample(x, speed, method="hermite"):
    """Tape-style resample: speed>1 plays faster/shorter/higher.

    Works on (n,) or (n, ch)."""
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        return st(lambda c: resample(c, speed, method), x)
    if speed <= 0:
        speed = EPS
    n_out = max(4, int(len(x) / speed))
    pos = np.arange(n_out, dtype=np.float64) * speed
    if method == "hermite":
        y = interp_hermite(x, pos)
    else:
        y = np.interp(pos, np.arange(len(x)), x)
    return y.astype(np.float32)


def resample_curve(x, rate):
    """Vari-speed with a per-sample rate curve (len >= n).

    pos[i] = sum(rate[:i]) — output continues until pos exits the input.
    Time AND pitch bend together (tape style)."""
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        return st(lambda c: resample_curve(c, rate), x)
    p = np.cumsum(np.asarray(rate, np.float64))
    n_out = int(np.searchsorted(p, len(x) - 1.001))
    if n_out < 16:
        n_out = min(len(x), 16)
    return interp_hermite(x, p[:n_out]).astype(np.float32)


def glissando(x, s0, s1, method="hermite"):
    """Resample with a speed curve gliding from s0 to s1."""
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        return st(lambda c: glissando(c, s0, s1, method), x)
    n_out = max(4, int(len(x) / max(0.05, 0.5 * (s0 + s1))))
    pos = np.linspace(0, 1, n_out)
    speed = s0 * (1.0 - pos) + s1 * pos
    return interp_hermite(x, np.cumsum(speed)).astype(np.float32)


def semis_to_speed(semis):
    return float(2.0 ** (semis / 12.0))


# ----------------------------------------------------------------- analysis


def detect_pitch(m, sr, lo=40.0, hi=900.0):
    """Autocorrelation pitch (Hz, confidence) via FFT, with parabolic refine.

    From the corpus curation / granular labs."""
    m = np.asarray(m, np.float64)
    if len(m) < 512:
        return 0.0, 0.0
    dec = max(1, int(sr // 12000))
    xs = m[::dec]
    srs = sr / dec
    n = len(xs)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(xs, nfft)
    ac = np.fft.irfft(np.abs(X) ** 2, nfft)[: n // 2]
    if ac[0] <= EPS:
        return 0.0, 0.0
    ac /= ac[0]
    lagmin = max(2, int(srs / hi))
    lagmax = min(int(srs / lo), n - 1)
    if lagmax <= lagmin + 2:
        return 0.0, 0.0
    k = lagmin + int(np.argmax(ac[lagmin:lagmax]))
    conf = float(ac[k])
    if conf < 0.30:
        return 0.0, 0.0
    d = 0.0
    if 0 < k < n - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        if abs(den) > EPS:
            # clamp the parabola offset: perfectly looped synth waveforms
            # give an exactly flat autocorr peak and d can explode
            d = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    f0 = float(srs / (k + d))
    if f0 > hi * 1.05 or f0 < lo * 0.5:       # out of range -> unreliable
        return 0.0, 0.0
    return f0, conf


def analyze(x, sr):
    """Cheap feature analysis of a sound (mono-collapsed)."""
    m = mono(x)
    n = len(m)
    out = {"dur_s": n / sr, "rms": 0.0, "peak": 0.0, "centroid": 0.0,
           "flatness": 0.0, "crest": 0.0, "transient": 0.0,
           "f0": 0.0, "f0_conf": 0.0}
    if n < 64:
        return out
    rms = float(np.sqrt(np.mean(m * m) + EPS))
    peak = float(np.max(np.abs(m)))
    out["rms"], out["peak"], out["crest"] = rms, peak, peak / (rms + EPS)
    seg = m[: min(n, 8 * sr)] * np.hanning(min(n, 8 * sr))
    S = np.abs(np.fft.rfft(seg)) + EPS
    fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
    out["centroid"] = float(np.sum(fr * S) / np.sum(S))
    out["flatness"] = float(np.exp(np.mean(np.log(S))) / (np.mean(S) + EPS))
    w = max(1, int(0.05 * sr))
    nf = n // w
    if nf > 2:
        e = np.sqrt(np.mean(m[: nf * w].reshape(nf, w) ** 2, axis=1) + EPS)
        out["transient"] = float(np.max(np.diff(e)) / (np.median(e) + EPS))
    f0, conf = detect_pitch(m, sr)
    out["f0"], out["f0_conf"] = f0, conf
    return out


def dominant_f0(x, sr, default=55.0):
    """Detected pitch or a sensible low default."""
    f0, conf = detect_pitch(mono(x), sr)
    if f0 <= 0 or conf < 0.30:
        return default
    return f0


# ----------------------------------------------------------------- misc


def rms_env(m, sr, win_s=0.02):
    """Envelope follower (smoothed rectified mono)."""
    m = np.abs(mono(m))
    w = max(2, int(win_s * sr))
    k = np.ones(w) / w
    return np.convolve(m, k, "same").astype(np.float32)


def bell(n):
    """Raised-cosine bell 0..1..0."""
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))