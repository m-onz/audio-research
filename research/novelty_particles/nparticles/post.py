"""Post-processing chain for the soundtrack (pure numpy DSP).

Effects (all vectorised, no sample-by-sample Python loops):

  exciter[:drive]     parallel tanh drive on the high band + |x|*x even
                      harmonic term -> spectral richness without boom
  flanger[:rate]      LFO-swept multi-tap delay (FIR approximation of a
                      feedback flanger/phaser): moving notches, stereo
                      phases offset for width
  phasing[:eps]       Steve Reich tape phasing: the right channel plays
                      the mono texture bed at rate 1-eps, slowly sliding
                      out of sync with the left -> evolving comb patterns
  ghost[:stretch]     paulstretch (phase-randomised overlap-add) of the
                      texture bed, stretched `stretch`x, truncated back
                      to the reel and mixed underneath.  Extreme
                      time-stretch as an ambient layer *parallel* to the
                      frame-synced events, so AV sync is never broken.
  shimmer[:mix]       granular pitch-shift x2 of the ghost -> octave-up
                      sparkle in the upper spectrum
  tilt[:db]           linear spectral tilt (brighten/darken via one FFT)

Chain order is fixed and sensible: exciter -> flanger -> (reverb happens
in media.py) -> loudness.  phasing/ghost/shimmer are parallel layers on
the texture bed, assembled by media.py.
"""

from __future__ import annotations

import numpy as np

from .sonify import SR, TWO_PI

# ----------------------------------------------------------------- helpers


def box_lp(x: np.ndarray, w: int) -> np.ndarray:
    """Fast box lowpass via cumulative sum. Works on (N,) or (N, ch)."""
    if w < 2:
        return x.copy()
    c = np.cumsum(x, axis=0, dtype=np.float64)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), c], axis=0)
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(i - w // 2, 0)
    hi = np.minimum(i + (w + 1) // 2, n)
    if x.ndim == 1:
        return ((c[hi] - c[lo]) / (hi - lo)).astype(x.dtype)
    return ((c[hi] - c[lo]) / (hi - lo)[:, None]).astype(x.dtype)


# ----------------------------------------------------------------- effects


def exciter(x: np.ndarray, drive: float = 3.0, mix: float = 0.85) -> np.ndarray:
    """Parallel saturation on the high band + even-harmonic term."""
    out = x.copy()
    for ch in range(x.shape[1]):
        hp = x[:, ch] - box_lp(x[:, ch], 320)
        shaped = np.tanh(drive * hp) / np.tanh(drive)
        even = hp * np.abs(hp) * 0.6
        out[:, ch] += mix * 0.5 * (shaped + even)
    return out


def flanger(x: np.ndarray, rate: float = 0.07, depth_ms: float = 5.0,
            g: float = 0.55, taps: int = 4) -> np.ndarray:
    """LFO-swept multi-tap delay: y = sum_k g^k * x(n - k*d(n))  (FIR
    approximation of a feedback flanger -> phaser-like moving notches)."""
    n = np.arange(len(x), dtype=np.float64)
    out = x.copy()
    for ch in range(x.shape[1]):
        ph = ch * 2.1                                   # stereo offset
        d = depth_ms * 1e-3 * SR * (0.5 + 0.5 * np.sin(TWO_PI * rate * n / SR + ph))
        d += 1.5 * SR * 1e-3                            # keep min delay > 0
        src = x[:, ch]
        base = np.arange(len(src), dtype=np.float64)
        acc = np.zeros(len(src), dtype=np.float32)
        for k in range(1, taps + 1):
            idx = np.clip(n - k * d, 0, len(src) - 1.001)
            acc += (g ** k) * np.interp(idx, base, src).astype(np.float32)
        out[:, ch] += acc
    return out


def tape_phasing(mono: np.ndarray, eps: float = 0.003) -> np.ndarray:
    """Reich-style tape phasing: L = original, R plays at rate 1-eps.
    The right channel lags eps*t seconds behind -- a slow comb drift that
    never resolves."""
    n = np.arange(len(mono), dtype=np.float64)
    right = np.interp(n * (1.0 - eps), n, mono).astype(np.float32)
    return np.stack([mono, right], axis=1)


def tape_phasing_tv(mono: np.ndarray, eps: float,
                    wobble_sig: np.ndarray) -> np.ndarray:
    """Tape phasing with a genome-driven *breathing* drift: the right
    channel's playback rate is 1 - eps*(1 + wobble(n)), so the comb
    interference speeds up, slows, almost resolves, and drifts again."""
    w = np.clip(wobble_sig, -0.9, 0.9)
    lag = np.cumsum(eps * (1.0 + w), dtype=np.float64)
    n = np.arange(len(mono), dtype=np.float64)
    p = np.clip(n - lag, 0.0, len(mono) - 1.001)
    right = np.interp(p, n, mono).astype(np.float32)
    return np.stack([mono, right], axis=1)


def paulstretch(x: np.ndarray, stretch: float = 6.0,
                window_s: float = 0.30) -> np.ndarray:
    """Extreme time-stretch with per-frame phase randomisation (the
    'paulstretch' ambient wash)."""
    return paulstretch_vary(x, np.full(len(x), stretch, dtype=np.float32),
                            window_s)


def paulstretch_vary(x: np.ndarray, stretch_ctrl: np.ndarray,
                     window_s: float = 0.30) -> np.ndarray:
    """paulstretch with a per-sample stretch control signal: the input
    read pointer advances by hop_out/stretch(n) each frame, so the wash
    can breathe between tight and enormous stretches over time."""
    N = 1 << int(round(np.log2(max(window_s * SR, 256))))
    hop_out = N // 2
    win = np.hanning(N).astype(np.float32)
    smax = float(np.max(stretch_ctrl)) if len(stretch_ctrl) else 1.0
    out_cap = int(len(x) * max(smax, 1.0)) + N
    out = np.zeros(out_cap, dtype=np.float32)
    rng = np.random.default_rng(31)
    base = np.arange(len(x), dtype=np.float64)
    pos, fi = 0.0, 0
    while pos + N <= len(x) and fi * hop_out + N < out_cap:
        st = float(np.clip(np.interp(pos + N / 2, base, stretch_ctrl),
                           0.6, 14.0))
        ip = int(pos)
        S = np.fft.rfft(x[ip:ip + N] * win)
        ph = rng.uniform(0.0, TWO_PI, len(S))
        frame = np.fft.irfft(np.abs(S) * np.exp(1j * ph), N) * win
        out[fi * hop_out:fi * hop_out + N] += frame
        pos += hop_out / st
        fi += 1
    return out


def pitch_shift(x: np.ndarray, ratio: float = 2.0, grain: int = 2048) -> np.ndarray:
    """Granular pitch shift (duration preserved).  ratio=2 -> octave up.
    Accepts (N,) or (N, ch)."""
    if x.ndim == 2:
        return np.stack([pitch_shift(x[:, c], ratio, grain) for c in range(x.shape[1])], axis=1)
    hop = grain // 2
    win = np.hanning(grain).astype(np.float32)
    out = np.zeros_like(x)
    for p in range(0, max(len(x) - grain, 0), hop):
        read = min(int(grain * ratio), len(x) - p)   # ratio>1 reads MORE -> pitch up
        if read < 16:
            break
        src = x[p:p + read]
        idx = np.linspace(0, read - 1, grain)
        out[p:p + grain] += np.interp(idx, np.arange(read), src).astype(np.float32) * win
    return out


def shimmer(ghost: np.ndarray, mix: float = 0.30, ratio: float = 2.0) -> np.ndarray:
    """Octave-up sparkle: granular pitch-shifted, softened copy of the ghost."""
    sh = pitch_shift(box_lp(ghost, 8), ratio)
    return ghost + mix * sh


def shimmer_tv(ghost: np.ndarray, mix_sig: np.ndarray) -> np.ndarray:
    """Shimmer with a per-sample mix automation signal."""
    sh = pitch_shift(box_lp(ghost, 8), 2.0)
    return ghost + np.clip(mix_sig, 0, 0.7)[:, None] * sh


def exciter_tv(x: np.ndarray, drive_sig: np.ndarray, mix: float = 0.85) -> np.ndarray:
    """Exciter with per-sample drive automation."""
    out = x.copy()
    d = np.maximum(drive_sig, 0.9)
    for ch in range(x.shape[1]):
        hp = x[:, ch] - box_lp(x[:, ch], 320)
        shaped = np.tanh(d * hp) / np.tanh(d)
        even = hp * np.abs(hp) * 0.6
        out[:, ch] += mix * 0.5 * (shaped + even)
    return out


def flanger_tv(x: np.ndarray, rate_sig: np.ndarray, depth_sig: np.ndarray,
               g: float = 0.55, taps: int = 4) -> np.ndarray:
    """Flanger with per-sample LFO rate (Hz) and depth (ms) automation --
    the notch sweeps themselves get played by the genome."""
    n = np.arange(len(x), dtype=np.float64)
    out = x.copy()
    for ch in range(x.shape[1]):
        phase = TWO_PI * np.cumsum(rate_sig) / SR + ch * 2.1
        depth = np.maximum(depth_sig, 0.15) * 1e-3 * SR
        d = 1.2 * SR * 1e-3 + depth * (0.5 + 0.5 * np.sin(phase))
        src = x[:, ch]
        base = np.arange(len(src), dtype=np.float64)
        acc = np.zeros(len(src), dtype=np.float32)
        for k in range(1, taps + 1):
            idx = np.clip(n - k * d, 0, len(src) - 1.001)
            acc += (g ** k) * np.interp(idx, base, src).astype(np.float32)
        out[:, ch] += acc
    return out


def spectral_tilt(x: np.ndarray, db: float = 1.5) -> np.ndarray:
    """Linear spectral tilt via one FFT pass (db per Nyquist)."""
    n = len(x)
    S = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    gain = 10.0 ** (db * 20 / 20.0 * (f / f[-1]) / 20.0)
    return np.fft.irfft(S * gain[:, None], n, axis=0).astype(np.float32)


# ----------------------------------------------------------------- chain


DEFAULT_POST = "excite:3,phasing:0.003,ghost:6"
KNOWN = ("excite", "flanger", "phasing", "ghost", "shimmer", "tilt")


def parse_post(spec: str) -> list[tuple[str, float]]:
    if spec.strip().lower() == "none":
        return []
    chain = []
    for tok in filter(None, [t.strip() for t in spec.split(",")]):
        name, _, arg = tok.partition(":")
        if name not in KNOWN:
            raise ValueError(f"unknown post effect: {name} (known: {KNOWN})")
        chain.append((name, float(arg) if arg else _default_arg(name)))
    return chain


def _default_arg(name: str) -> float:
    return {"excite": 3.0, "flanger": 0.07, "phasing": 0.003,
            "ghost": 6.0, "shimmer": 0.3, "tilt": 1.5}[name]


def run_chain(x: np.ndarray, chain: list[tuple[str, float]],
              tv: dict | None = None) -> np.ndarray:
    """Apply the serial effects (excite/flanger/tilt).  `tv` optionally
    provides per-sample automation arrays keyed 'drive', 'rate', 'depth'
    (genome-driven FX); phasing/ghost/shimmer are parallel texture-bed
    layers handled by media.py."""
    tv = tv or {}
    for name, arg in chain:
        if name == "excite":
            if "drive" in tv:
                x = exciter_tv(x, tv["drive"])
            else:
                x = exciter(x, drive=arg)
        elif name == "flanger":
            if "rate" in tv and "depth" in tv:
                x = flanger_tv(x, tv["rate"], tv["depth"])
            else:
                x = flanger(x, rate=arg)
        elif name == "tilt":
            x = spectral_tilt(x, db=arg)
    return x