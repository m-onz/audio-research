"""render: one genome, many phenotypes.

The same compositional graph is read four ways. Sync is structural:
the picture and the sound are the same function pointed at different
axes — nothing is synchronized after the fact, there is nothing to
sync.

| traversal | x        | y        | t        | what comes out |
|-----------|----------|----------|----------|----------------|
| spectra   | seconds  | log-freq | seconds  | magnitude field → phase-accumulated inverse-FFT frames |
| wavetable | phase    | morph    | seconds  | a band-limited morphing single-cycle table |
| field     | seconds  | 0        | seconds  | the waveform itself, oversampled ×4, lowpassed, decimated |
| picture   | space    | space    | frozen   | luminance × two chromas |

The sole designed stage is the safety limiter (the house invariant):
rms is normalized, edges are faded, peaks are scaled. Phase precision
is float32 inside the graph — deterministic, and at audio rates the
worst artifact is a slight inharmonic wobble, never a random one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .genome import Genome

SR = 44100
MODES = ("spectra", "wavetable", "field")
TARGET_RMS = 0.18
FADE_S = 0.04
PEAK = 0.95
_EVAL_BLOCK = 1 << 20          # field-mode chunks (keep genome memory sane)


# ------------------------------------------------------------------ helpers


def _pick(outs: dict, prefer: str) -> np.ndarray:
    """Named output, else the first the genome offers."""
    if prefer in outs:
        return np.asarray(outs[prefer])
    if outs:
        return np.asarray(next(iter(outs.values())))
    raise ValueError("genome has no outputs")


def _norm_range(x: np.ndarray, lo: float = 1.0, hi: float = 99.0
                ) -> np.ndarray:
    """Percentile-normalize to [0, 1]; flat fields pass through the
    middle so pictures never go black."""
    flat = x if x.size <= 100_000 else x.reshape(-1)[:: x.size // 50_000 + 1]
    a, b = np.percentile(flat, [lo, hi])
    if b - a < 1e-9:
        return np.full_like(x, 0.5, dtype=np.float32)
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(np.float32)


def _slow_curve(genome: Genome, name: str, length: float,
                n: int = 4096) -> np.ndarray | None:
    """A named output read along the time axis at low resolution —
    pan paths, morph paths. None if the genome has no such output."""
    if name not in genome.outputs:
        return None
    ts = np.linspace(0.0, length, n, dtype=np.float32)
    outs = genome.evaluate({"x": ts, "t": ts,
                            "y": np.zeros_like(ts),
                            "d": np.abs(ts), "b": 1.0})
    v = np.asarray(outs[name], np.float64).reshape(-1)
    v = _norm_range(v) * 2.0 - 1.0                     # [-1, 1]
    fine = np.arange(int(length * SR)) / SR
    return np.interp(fine, ts, v)


def _polish(x: np.ndarray, sr: int = SR, pan: np.ndarray | None = None,
            target_rms: float = TARGET_RMS) -> np.ndarray:
    """rms-normalize, fade the edges, apply the pan field, limit.
    The only shaping any render receives — the safety stage."""
    x = np.asarray(x, np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms < 1e-9:
        return np.zeros((x.size, 2), np.float32)
    x = x * (target_rms / rms)
    nf = int(FADE_S * sr)
    if nf > 0 and x.size > 2 * nf:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, nf))
        x[:nf] *= ramp
        x[-nf:] *= ramp[::-1]
    if pan is None:
        pan = np.zeros(x.size)
    theta = (np.clip(pan, -1.0, 1.0).reshape(-1) + 1.0) * (np.pi / 4.0)
    L = x * np.cos(theta)
    R = x * np.sin(theta)
    peak = max(np.abs(L).max(), np.abs(R).max())
    if peak > PEAK:
        L, R = L * (PEAK / peak), R * (PEAK / peak)
    return np.stack([L, R], axis=1).clip(-0.99, 0.99).astype(np.float32)


# ------------------------------------------------------------------ spectra


def _render_spectra(g: Genome, length: float, sr: int, seed: int,
                    target_rms: float, **_kw) -> np.ndarray:
    """CPPN(time × log-freq) → magnitude field → additive frames.
    Each frame is synthesized by inverse FFT with accumulated phase,
    so partials slide and beat instead of chattering. The picture of
    this genome IS its score."""
    hop, win, F = 1024, 2048, 160
    fmin, fmax = 40.0, 0.46 * sr
    n_frames = int(np.ceil(length * sr / hop)) + 1
    frame_t = (np.arange(n_frames) * hop / sr).astype(np.float32)
    f_axis = np.geomspace(fmin, fmax, F)
    y_axis = (np.log(f_axis / fmin) / np.log(fmax / fmin)
              * 2.0 - 1.0).astype(np.float32)
    X = frame_t[:, None]
    Y = y_axis[None, :]
    outs = g.evaluate({"x": X, "t": X, "y": Y,
                       "d": np.hypot(X, Y).astype(np.float32), "b": 1.0})
    mag = _pick(outs, "amp")
    mag = _norm_range(mag, 1.0, 99.5) ** 1.6

    lin_f = np.fft.rfftfreq(win, 1.0 / sr)            # linear bins
    in_band = (lin_f >= fmin) & (lin_f <= fmax)
    pos = np.zeros(lin_f.size)                        # outside the field's
    pos[in_band] = np.interp(lin_f[in_band], f_axis,  # band: silence,
                             np.linspace(0.0, 1.0, F))  # never clamp
    fi = np.clip(pos * (F - 1), 0.0, F - 1)
    k0 = np.floor(fi).astype(np.int64)
    k1 = np.minimum(k0 + 1, F - 1)
    a = (fi - k0).astype(np.float32)
    W = np.zeros((lin_f.size, F), np.float32)          # F-bands → bins
    rows = np.arange(lin_f.size)
    np.add.at(W, (rows, k0), 1.0 - a)
    np.add.at(W, (rows, k1), a)
    W[~in_band] = 0.0   # no energy was ever mapped here — no leak
    M = mag @ W.T                                      # (frames, bins)

    rng = np.random.default_rng(seed)
    # dense additive synthesis: random initial phases per bin. With
    # hop = win/2 the per-frame phase advance is exactly πk, so a
    # zero-initialized field realigns every even frame into one giant
    # windowed impulse — a broadband phenotype would secretly be a
    # click train (crest ~60). Random starts keep partials incoherent;
    # the frame-to-frame jitter below stays as slow detune wander.
    ph = rng.uniform(0.0, 2.0 * np.pi, lin_f.size)
    advance = lin_f * (2.0 * np.pi * hop / sr)
    jitter = rng.standard_normal((n_frames, lin_f.size)) * 0.03
    tbuf = np.zeros((n_frames - 1) * hop + win)
    hann = np.hanning(win)
    for i in range(n_frames):
        ph = ph + advance + jitter[i]
        spec = M[i] * np.exp(1j * ph)
        tbuf[i * hop:i * hop + win] += np.fft.irfft(spec, win) * hann
    audio = tbuf[: int(length * sr)]
    pan = _slow_curve(g, "pan", length)
    return _polish(audio, sr, pan, target_rms)


# ------------------------------------------------------------------ wavetable


def _render_wavetable(g: Genome, length: float, sr: int, seed: int,
                      target_rms: float, f0: float = 110.0,
                      morph_hz: float = 0.02, **_kw) -> np.ndarray:
    """CPPN(phase × morph) → a bank of single-cycle tables, band-
    limited per harmonic against f0, played back with table crossfade
    along a slow morph path. sin∘sin territory: timbre walks without
    a single designed envelope."""
    TL, K = 2048, 64
    u = np.arange(TL, dtype=np.float32) / TL
    mm = np.linspace(-1.0, 1.0, K, dtype=np.float32)
    outs = g.evaluate({"x": (2.0 * u - 1.0)[None, :],
                       "y": mm[:, None],
                       "d": np.hypot((2.0 * u - 1.0)[None, :],
                                     mm[:, None]).astype(np.float32),
                       "b": 1.0, "t": 0.0})
    bank = _pick(outs, "wave")
    bank = np.asarray(bank, np.float64)
    peak = np.abs(bank).max()
    if peak > 1e-9:
        bank = bank * (0.9 / peak)

    harm = np.arange(TL // 2 + 1) * f0                 # harmonic freqs
    edge = 0.45 * sr
    gain = np.exp(-((harm - edge) / (0.02 * sr)) ** 2)
    gain[harm <= edge] = 1.0
    for k in range(K):                                 # band-limit tables
        S = np.fft.rfft(bank[k])
        bank[k] = np.fft.irfft(S * gain, TL)

    n = int(length * sr)
    t = np.arange(n, dtype=np.float64) / sr
    if "morph" in g.outputs:                           # genome drives path
        curve = _slow_curve(g, "morph", length)
        m_t = 0.5 + 0.5 * np.clip(curve, -1.0, 1.0) if curve is not None \
            else 0.5 + 0.5 * np.sin(2 * np.pi * morph_hz * t)
    else:
        m_t = 0.5 + 0.5 * np.sin(2 * np.pi * morph_hz * t)
    mp = m_t * (K - 1)
    j0 = np.floor(mp).astype(np.int64)
    j0 = np.clip(j0, 0, K - 1)
    j1 = np.minimum(j0 + 1, K - 1)
    al = (mp - j0)

    idx = (f0 * t) % 1.0 * TL
    i0 = np.floor(idx).astype(np.int64) % TL
    i1 = (i0 + 1) % TL
    be = idx - np.floor(idx)
    w00, w01 = bank[j0, i0], bank[j0, i1]
    w10, w11 = bank[j1, i0], bank[j1, i1]
    x = ((1 - be) * ((1 - al) * w00 + al * w10)
         + be * ((1 - al) * w01 + al * w11))
    pan = _slow_curve(g, "pan", length)
    return _polish(x, sr, pan, target_rms)


# ------------------------------------------------------------------ field


def _render_field(g: Genome, length: float, sr: int, seed: int,
                  target_rms: float, rate: float = 1.0, **_kw) -> np.ndarray:
    """CPPN(seconds) at audio rate — the waveform is the pattern.
    Oversampled ×4, FFT-lowpassed below the target Nyquist, decimated:
    composition can alias; the readout is not allowed to."""
    OS = 4
    nos = int(length * sr) * OS
    chunks = []
    for s in range(0, nos, _EVAL_BLOCK):
        e = min(s + _EVAL_BLOCK, nos)
        xs = (np.arange(s, e) / float(sr * OS)).astype(np.float32)
        outs = g.evaluate({"x": xs * rate, "t": xs * rate,
                           "y": np.zeros_like(xs),
                           "d": np.abs(xs), "b": 1.0})
        chunks.append(np.asarray(_pick(outs, "field"), np.float64))
    x = np.concatenate(chunks) if chunks else np.zeros(0)
    if x.size == 0:
        return np.zeros((0, 2), np.float32)

    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1.0 / (sr * OS))
    edge = 0.47 * sr
    gain = np.exp(-((f - edge) / (0.03 * sr)) ** 2)
    gain[f <= edge] = 1.0
    gain[0] = 0.0                                      # no DC
    x = np.fft.irfft(X * gain, x.size)[::OS]
    pan = _slow_curve(g, "pan", length)
    return _polish(x, sr, pan, target_rms)


# ------------------------------------------------------------------ picture


def picture(g: Genome, width: int = 1280, height: int = 720,
            t: float = 0.0) -> np.ndarray:
    """CPPN(space) → luminance tinted by two chroma channels. The
    bellfield picture is bellfield's score, seen from above."""
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    X, Y = xs[None, :], ys[:, None]
    outs = g.evaluate({"x": X, "y": Y,
                       "d": np.hypot(X, Y).astype(np.float32),
                       "b": 1.0, "t": np.float32(t)})
    lum = _norm_range(_pick(outs, "lum"), 0.5, 99.5)
    if "ca" in outs and "cb" in outs:
        ca = _norm_range(outs["ca"], 0.5, 99.5) * 2.0 - 1.0
        cb = _norm_range(outs["cb"], 0.5, 99.5) * 2.0 - 1.0
    else:                                              # achromatic genomes
        ca = np.zeros_like(lum)
        cb = np.zeros_like(lum)
    R = np.clip(lum * (1.0 + 0.55 * ca), 0.0, 1.0)
    G = lum
    B = np.clip(lum * (1.0 + 0.55 * cb), 0.0, 1.0)
    img = np.stack([R, G, B], axis=-1)
    return (np.flipud(img) * 255.0).astype(np.uint8)


# ------------------------------------------------------------------ io


def write_wav(path, audio: np.ndarray, sr: int = SR) -> str:
    """Float audio (mono (n,) or stereo (n, 2)) → 16-bit wav via ffmpeg
    (the ecosystem writer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, np.float32)
    if x.ndim == 2 and x.shape[1] == 2:
        inter = np.empty(len(x) * 2, np.float32)
        inter[0::2], inter[1::2] = x[:, 0], x[:, 1]
        x, ch = inter, 2
    else:
        x, ch = x.reshape(-1), 1
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le",
         "-ar", str(sr), "-ac", str(ch), "-i", "pipe:0", str(path)],
        input=pcm.tobytes(), check=True)
    return str(path)


def write_png(path, img: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(path)
    return str(path)


# ------------------------------------------------------------------ dispatch


_RENDERERS = {"spectra": _render_spectra,
              "wavetable": _render_wavetable,
              "field": _render_field}


def render(g: Genome, length: float = 8.0, sr: int = SR,
           mode: str = "spectra", seed: int = 0, f0: float = 110.0,
           morph_hz: float = 0.02, rate: float = 1.0,
           target_rms: float = TARGET_RMS) -> np.ndarray:
    """The phenotype. Returns stereo (n, 2) float32, loudness-safe."""
    if mode not in _RENDERERS:
        raise ValueError(f"mode '{mode}' not in {MODES}")
    return _RENDERERS[mode](g, float(length), int(sr), int(seed),
                           target_rms, f0=f0, morph_hz=morph_hz, rate=rate)