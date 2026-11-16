"""ops: the material processes — tomb's mutation operators.

Sound is never "played" — it is weathered. Five families:

  GENESIS   structure-creators (break the noise fixed point):
            resonate, comb, decimate, logistic, gate, wavesort
  FIELD     spectrogram physics (transform existing structure):
            diffuse, propagate, react, automate
  STAT      population-as-filterbank (self-reference):
            morph, convmate, rephase, histogram
  ORDER     pure statistics: wavesort, shufflewin
  LOSS      generational decay: mp3_round, resample_war, crush, bitrot
  STRUCTURE complexification: splice, reverse, layer, expand

Every op: (audio float32 mono, sr, rng, **params) -> (audio, sr).
All pure numpy, fully offline, deterministic from rng.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["OPS", "FAMILIES", "mutate", "random_op_recipe"]

SR = 44100
EPS = 1e-9


# ------------------------------------------------------------------ genesis

def _resonate(x, sr, rng, delay_s=0.004, feedback=0.92, taps=14, damp=0.35):
    """Noise through a feedback delay network — Karplus-Strong territory.
    Noise becomes tone: the comb resonates, the damp lowpasses each pass."""
    d = max(2, int(delay_s * sr))
    fb = float(np.clip(feedback, 0.1, 0.97))
    y = x.copy()
    g = 1.0
    for t in range(1, int(taps) + 1):
        g *= fb
        y += np.concatenate([np.zeros(d * t), x[: len(x) - d * t] * g])
        x = _onepole(y.copy(), damp) if t % 4 == 0 else y
    return x


def _onepole(x, alpha):
    """One-pole lowpass (cheap, warm)."""
    y = np.empty_like(x)
    acc = 0.0
    a = float(np.clip(alpha, 0.01, 0.99))
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


def _comb(x, sr, rng, delays=(0.002, 0.005, 0.011), gains=(0.9, 0.7, 0.5)):
    """Layer delayed copies: autocorrelation combs — noise becomes buzz."""
    y = x.copy()
    for d_s, g in zip(delays, gains):
        d = max(2, int(d_s * sr))
        y += np.concatenate([np.zeros(d), x[: len(x) - d] * g])
    return y * 0.5


def _decimate(x, sr, rng, hold=8):
    """Sample-and-hold decimation: noise becomes steppy tonal grit."""
    h = max(2, int(hold))
    idx = (np.arange(len(x)) // h) * h
    return x[idx]


def _logistic(x, sr, rng, r=None, x0=None):
    """Samples through the logistic map — deterministic chaos at specific
    parameters yields structure (periodic islands in the chaos)."""
    r = float(r if r is not None else rng.uniform(3.55, 3.95))
    y = np.empty_like(x)
    v = float(x0 if x0 is not None else 0.5)
    for i in range(len(x)):
        v = np.clip(r * v * (1.0 - v), 0.0, 1.0)
        y[i] = (v - 0.5) * 2.0
    # mix the deterministic stream with the source (the source modulates)
    return y * 0.7 + x * 0.3


def _gate(x, sr, rng, n_bands=6, rate_hz=3.0):
    """Temporal envelope imposition: random-gated bands — flat spectrum,
    real temporal structure (structure without pitch)."""
    win = int(0.02 * sr)
    n_windows = len(x) // win
    env = np.zeros(n_windows)
    t = np.arange(n_windows) / sr * win  # per-window time
    for _ in range(n_bands):
        f = rng.uniform(0.5, rate_hz)
        phase = rng.uniform(0, 2 * np.pi)
        env += np.sin(2 * np.pi * f * t + phase)
    env = np.clip(env / n_bands * 1.4, 0, 1)
    env = np.repeat(env, win)
    if len(env) < len(x):
        env = np.pad(env, (0, len(x) - len(env)))
    return x[: len(env)] * env


def _wavesort(x, sr, rng, window=256):
    """Sort samples within windows: random → monotonic ramps. THE symmetry
    breaker — pixel-sorting, heard. Genesis + ORDER family member."""
    w = max(16, int(window))
    y = x.copy()
    for i in range(0, len(y) - w, w):
        y[i : i + w] = np.sort(y[i : i + w])
    return y


# ------------------------------------------------------------------ field

def _spectrogram(x, n_fft=1024, hop=256):
    frames = (len(x) - n_fft) // hop + 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)), n_fft, hop


def _istft(mag, n_fft, hop, length):
    frames = len(mag)
    phase = np.random.RandomState(0).uniform(0, 2 * np.pi, mag.shape)
    S = mag * np.exp(1j * phase)
    y = np.zeros(length + n_fft)
    wsum = np.zeros(length + n_fft)
    win = np.hanning(n_fft)
    for i in range(frames):
        seg = np.fft.irfft(S[i], n_fft) * win
        y[i * hop : i * hop + n_fft] += seg
        wsum[i * hop : i * hop + n_fft] += win ** 2
    good = wsum > 1e-8
    y[good] /= wsum[good]
    return y[:length]


def _diffuse(x, sr, rng, k=1.2, steps=2):
    """Anisotropic blur on the spectrogram (womb waveplate's damping)."""
    mag, n_fft, hop = _spectrogram(x)
    log_m = np.log1p(mag)
    for _ in range(int(steps)):
        kx = np.random.RandomState(rng.integers(1 << 30)).uniform(0.5, k)
        ky = np.random.RandomState(rng.integers(1 << 30)).uniform(0.5, k)
        lap = (np.roll(log_m, 1, 0) + np.roll(log_m, -1, 0)) * kx \
            + (np.roll(log_m, 1, 1) + np.roll(log_m, -1, 1)) * ky
        log_m = log_m * 0.6 + lap * 0.2
    return _istft(np.expm1(np.clip(log_m, 0, 20)), n_fft, hop, len(x))


def _propagate(x, sr, rng, c=0.4, steps=3):
    """One FDTD wave-equation pass across the spectrogram — spectral
    ripples (womb waveplate as a mutation operator)."""
    mag, n_fft, hop = _spectrogram(x)
    log_m = np.log1p(mag)
    u_prev = log_m.copy()
    u = log_m.copy()
    for _ in range(int(steps)):
        lap = np.roll(u, 1, 0) + np.roll(u, -1, 0) \
            + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
        u_next = (2 * u - u_prev + c * lap) * 0.999
        u_prev, u = u, u_next
    return _istft(np.expm1(np.clip(u, 0, 20)), n_fft, hop, len(x))


def _react(x, sr, rng, f=0.055, k=0.062, steps=8):
    """Gray-Scott reaction-diffusion on the log-magnitude field — organic
    erosion (womb automata/doom RD as an audio shaper)."""
    mag, n_fft, hop = _spectrogram(x)
    log_m = np.log1p(mag)
    A = np.clip(log_m / log_m.max() + EPS, 0, 1)
    B = 1.0 - A
    A, B = B.copy(), A.copy()  # swap so B is the "chemical" seeded by sound
    for _ in range(int(steps)):
        la = (np.roll(A, 1, 0) + np.roll(A, -1, 0)
              + np.roll(A, 1, 1) + np.roll(A, -1, 1) - 4 * A)
        lb = (np.roll(B, 1, 0) + np.roll(B, -1, 0)
              + np.roll(B, 1, 1) + np.roll(B, -1, 1) - 4 * B)
        reaction = A * B * B
        A += 0.16 * la - reaction + f * (1 - A)
        B += 0.08 * lb + reaction - (k + f) * B
        A = np.clip(A, 0, 1)
        B = np.clip(B, 0, 1)
    return _istft(np.expm1(A * 6.0), n_fft, hop, len(x))


def _automate(x, sr, rng, steps=4):
    """One continuous-CA step on the magnitude field (womb automata)."""
    mag, n_fft, hop = _spectrogram(x)
    s = np.clip(mag / (mag.max() + EPS), 0, 1)
    for _ in range(int(steps)):
        lap = (np.roll(s, 1, 0) + np.roll(s, -1, 0)
               + np.roll(s, 1, 1) + np.roll(s, -1, 1) - 4 * s)
        s = np.clip(s + lap * 0.4 - s * s * 0.9, 0, 1)
        s += (rng.random(s.shape) < 0.0005) * rng.uniform(0.3, 0.8)
    return _istft(s * (mag.max() + EPS), n_fft, hop, len(x))


# ------------------------------------------------------------------ stat

def _spectral_envelope(x, n_fft=1024):
    mag, _, _ = _spectrogram(x, n_fft=n_fft, hop=n_fft // 2)
    return mag.mean(axis=0)  # (n_fft/2+1,) average magnitude


def _morph(x, sr, rng, member=None, blend=0.5):
    """Remap this sound's spectral envelope toward a living member's —
    breeding through timbre transfer, not DSP."""
    if member is None:
        return x
    e_self = _spectral_envelope(x)
    e_other = _spectral_envelope(member)
    ratio = (e_other + EPS) / (e_self + EPS)
    mag, n_fft, hop = _spectrogram(x)
    smooth = np.interp(
        np.linspace(0, 1, len(ratio)),
        np.linspace(0, 1, min(len(ratio), len(mag.T))),
        ratio[: min(len(ratio), len(mag.T))],
    )
    target = mag * smooth[None, : len(mag.T)]
    return _istft(target, n_fft, hop, len(x))


def _convmate(x, sr, rng, member=None, mix=0.4):
    """Convolve with a living member (or autoconvolve) — the literal
    self-convoluting process."""
    other = member if member is not None else x
    n = 1 << int(np.ceil(np.log2(len(x) + len(other))))
    X = np.fft.rfft(x, n)
    O = np.fft.rfft(other, n)
    y = np.fft.irfft(X * O, n)[: len(x)]
    y = y / (np.abs(y).max() + EPS)
    return x * (1 - mix) + y * mix


def _rephase(x, sr, rng):
    """Keep the magnitude, randomize phase — same matter, new creature."""
    mag, n_fft, hop = _spectrogram(x)
    return _istft(mag, n_fft, hop, len(x))


def _histogram(x, sr, rng, member=None):
    """Match the sample-value histogram to another member — matter
    redistribution, no musical grammar."""
    if member is None:
        return x
    x_sorted = np.sort(x)
    member_sorted = np.sort(member)
    n = min(len(x_sorted), len(member_sorted))
    ranks = np.searchsorted(x_sorted, x)
    return np.interp(np.clip(ranks, 0, n - 1), np.arange(n),
                     member_sorted[:n])


# ------------------------------------------------------------------ loss

def _mp3_round(x, sr, rng, kbps=96):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(sr),
             "-ac", "1", "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", f"{kbps}k", tmp],
            input=x.astype(np.float32).tobytes(), check=True,
        )
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp, "-f", "f32le", "-ac", "1",
             "-ar", str(sr), "pipe:1"],
            capture_output=True, check=True,
        ).stdout
        y = np.frombuffer(out, np.float32).copy()
        return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    finally:
        Path(tmp).unlink(missing_ok=True)


def _resample_war(x, sr, rng, rate=8000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(sr),
             "-ac", "1", "-i", "pipe:0", "-ar", str(int(rate)), tmp],
            input=pcm.tobytes(), check=True,
        )
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp, "-f", "f32le", "-ac", "1",
             "-ar", str(sr), "pipe:1"],
            capture_output=True, check=True,
        ).stdout
        y = np.frombuffer(out, np.float32).copy()
        return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    finally:
        Path(tmp).unlink(missing_ok=True)


def _crush(x, sr, rng, bits=10):
    q = 2.0 ** (bits - 1)
    return np.round(x * q) / q


def _bitrot(x, sr, rng, amount=0.0005):
    y = x.copy()
    hits = int(len(y) * amount)
    for _ in range(hits):
        i = int(rng.integers(len(y)))
        op = rng.integers(3)
        if op == 0:
            run = int(rng.integers(1, 9))
            y[i: i + run] = 0.0
        elif op == 1 and i + 1 < len(y):
            y[i], y[i + 1] = y[i + 1], y[i]
        else:
            y[i] = -y[i]
    return y


# ------------------------------------------------------------------ structure

def _splice(x, sr, rng, keep=0.6):
    n = len(x)
    start = int(rng.integers(0, max(1, int(n * (1 - keep)))))
    end = start + int(n * keep)
    return x[start:end]


def _reverse(x, sr, rng):
    return x[::-1].copy()


def _layer(x, sr, rng, member=None, mix=0.5):
    if member is None:
        return x
    other = member[: len(x)]
    if len(other) < len(x):
        other = np.pad(other, (0, len(x) - len(other)))
    return x * (1 - mix) + other * mix


def _expand(x, sr, rng, stretch=1.5):
    """Append a copy of the sound — the unbounded-complexity axis."""
    return np.concatenate([x, x])


# ------------------------------------------------------------------ registry

FAMILIES = {
    "genesis": [("_resonate", {}), ("_comb", {}), ("_decimate", {}),
                ("_logistic", {}), ("_gate", {}), ("_wavesort", {"window": 256}),
                ("_sub_resonate", {}), ("_harmonic_seed", {}),
                ("_granular_sieve", {}), ("_formant", {}), ("_lowpass_sweep", {})],
    "field": [("_diffuse", {}), ("_propagate", {}), ("_react", {}), ("_automate", {})],
    "stat": [("_morph", {}), ("_convmate", {}), ("_rephase", {}), ("_histogram", {})],
    "order": [("_wavesort", {"window": 128})],
    "loss": [("_mp3_round", {}), ("_resample_war", {}), ("_crush", {"bits": 10}),
             ("_bitrot", {})],
    "structure": [("_splice", {}), ("_reverse", {}), ("_layer", {}), ("_expand", {})],
}

# OPS registry built at module bottom (after all op definitions)


def random_op_recipe(rng, phase="genesis"):
    """One op from a family, weighted by phase. Genesis phase favors
    structure-creators; later phases favor transformers."""
    if phase == "genesis":
        weights = {"genesis": 6, "field": 1, "stat": 1, "order": 2, "loss": 1, "structure": 1}
    else:
        weights = {"genesis": 1, "field": 3, "stat": 3, "order": 2, "loss": 2, "structure": 2}
    fams = list(weights)
    w = np.array([weights[f] for f in fams], dtype=np.float64)
    fam = fams[int(rng.choice(len(fams), p=w / w.sum()))]
    name, params = FAMILIES[fam][int(rng.integers(len(FAMILIES[fam])))]
    return name.strip("_"), dict(params)  # stripped: matches OPS registry keys


def mutate(x, sr, rng, op_name, member=None, **params):
    """Apply one op. `member` is a random living sound (for stat/structure
    ops that need a partner)."""
    fn = OPS.get(op_name)
    if fn is None:
        raise ValueError(f"unknown op {op_name!r}; have {sorted(OPS)}")
    if op_name in ("morph", "convmate", "histogram", "layer"):
        y = fn(x, sr, rng, member=member, **params)
    else:
        y = fn(x, sr, rng, **params)
    return _normalize(y), sr


def _normalize(x):
    peak = float(np.abs(x).max()) if x.size else 0.0
    return x if peak < 1e-6 else (x / peak * 0.9).astype(np.float32)

# ------------------------------------------------------------------ warm genesis (v1.3)
# The first genesis ops created *high-frequency* structure (resonate/decimate).
# These create low-frequency, warm, pitched material — the sounds a musician
# would actually reach for.

def _sub_resonate(x, sr, rng, f0=None, q=None, sweep=0.5):
    """Low-frequency resonant filter sweep on noise: rumble, drone, sub-bass
    growl. Noise becomes *body*."""
    f0 = float(f0 if f0 is not None else rng.uniform(50, 300))
    q = float(q if q is not None else rng.uniform(2, 12))
    n = len(x)
    t = np.arange(n) / sr
    f_sweep = f0 * 2 ** (np.sin(t * sweep * 2 * np.pi) * 0.5)  # slow octave sweep
    # biquad resonator, coefficients updated every 64 samples
    y = np.zeros_like(x)
    a1 = a2 = b0 = b1 = b2 = 0.0
    for i in range(n):
        if i % 64 == 0:
            w0 = 2 * np.pi * f_sweep[i] / sr
            alpha = np.sin(w0) / (2 * q)
            b0 = alpha / (1 + alpha)
            b1 = 0
            b2 = -alpha / (1 + alpha)
            a1 = -2 * np.cos(w0) / (1 + alpha)
            a2 = (1 - alpha) / (1 + alpha)
        xn = x[i]
        yn = b0 * xn + b1 * 0 + b2 * 0 - a1 * y[i - 1 if i > 0 else 0] - a2 * y[i - 2 if i > 2 else 0]
        y[i] = yn
    return y * 3.0  # gain up the resonance


def _harmonic_seed(x, sr, rng, f0=None, n_partials=8, decay=None):
    """Place harmonic partials in the signal: noise becomes a pitched drone.
    A random fundamental + harmonics with individual decay envelopes."""
    f0 = float(f0 if f0 is not None else rng.uniform(55, 440))
    decay = float(decay if decay is not None else rng.uniform(1.5, 5.0))
    n = len(x)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for k in range(1, int(n_partials) + 1):
        f = f0 * k
        if f > sr / 2:
            break
        amp = 1.0 / k ** rng.uniform(0.8, 1.4)
        env = np.exp(-t * decay / rng.uniform(0.5, 1.5))
        y += amp * env * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    # dominant harmonic content, minimal noise texture
    return y * 0.95 + x * 0.05


def _granular_sieve(x, sr, rng, grain_ms=40, rate_hz=4.0, spread=0.3):
    """Rhythmic granulation: grains placed on a pulse grid with random
    offsets — noise becomes rhythmic texture."""
    grain_n = int(grain_ms / 1000 * sr)
    period = int(1.0 / rate_hz * sr)
    n = len(x)
    y = np.zeros(n)
    pos = 0
    while pos + grain_n < n:
        src_start = int(rng.integers(0, max(1, len(x) - grain_n)))
        grain = x[src_start: src_start + grain_n] * np.hanning(grain_n)
        # filter each grain: resonant lowpass gives the grain a pitch
        fc = rng.uniform(80, 1200)
        w0 = 2 * np.pi * fc / sr
        alpha = np.sin(w0) / (2 * 4.0)
        b0 = alpha / (1 + alpha)
        b2 = -alpha / (1 + alpha)
        a1 = -2 * np.cos(w0) / (1 + alpha)
        a2 = (1 - alpha) / (1 + alpha)
        flt = np.zeros_like(grain)
        for j in range(2, grain_n):
            flt[j] = b0 * grain[j] + b2 * grain[j - 2] - a1 * flt[j - 1] - a2 * flt[j - 2]
        grain = flt * 2.0
        jitter = int(rng.integers(-period * spread, period * spread))
        at = max(0, pos + jitter)
        y[at: at + grain_n] += grain
        pos += period
    return y * 2.0


def _formant(x, sr, rng, f1=None, f2=None, q=6.0):
    """Noise through two formant filters (vowel territory): noise becomes
    voice-like. Random vowel positions."""
    f1 = float(f1 if f1 is not None else rng.uniform(200, 800))
    f2 = float(f2 if f2 is not None else rng.uniform(800, 2500))
    def bandpass_at(x, fc, q):
        w0 = 2 * np.pi * fc / sr
        alpha = np.sin(w0) / (2 * q)
        b0 = alpha / (1 + alpha)
        b2 = -alpha / (1 + alpha)
        a1 = -2 * np.cos(w0) / (1 + alpha)
        a2 = (1 - alpha) / (1 + alpha)
        # simple recursive filter
        y = np.zeros_like(x)
        for i in range(2, len(x)):
            y[i] = b0 * x[i] + b2 * x[i - 2] - a1 * y[i - 1] - a2 * y[i - 2]
        return y
    return bandpass_at(bandpass_at(x, f1, q), f2, q) * 4.0


def _lowpass_sweep(x, sr, rng, cut_lo=200, cut_hi=3000, sweep_hz=0.1):
    """Noise through a slowly-sweeping lowpass: rumble, ocean, distant
    thunder. The sweep gives temporal movement without rhythm."""
    n = len(x)
    t = np.arange(n) / sr
    cut = cut_lo + (cut_hi - cut_lo) * (0.5 + 0.5 * np.sin(t * sweep_hz * 2 * np.pi))
    y = np.zeros_like(x)
    a1 = a2 = b0 = b2 = 0.0
    y1 = y2 = x1 = x2 = 0.0
    for i in range(n):
        if i % 64 == 0:
            w0 = 2 * np.pi * cut[i] / sr
            alpha = np.sin(w0) / (2 * 0.8)
            b0 = (1 - np.cos(w0)) / 2 / (1 + alpha)
            b2 = (1 + np.cos(w0)) / 2 / (1 + alpha)
            a1 = -2 * np.cos(w0) / (1 + alpha)
            a2 = (1 - alpha) / (1 + alpha)
        xn = x[i]
        yn = b0 * xn + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yn
        x2 = x1; x1 = xn; y2 = y1; y1 = yn
    return y * 2.0


# ------------------------------------------------------------------ registry (must be after all defs)
OPS = {}
for _fam, _members in FAMILIES.items():
    for _name, _params in _members:
        OPS[_name.strip("_")] = globals()[_name]
