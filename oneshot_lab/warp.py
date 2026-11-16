"""Warp: pitch-preserving time stretch with a MODULATED stretch factor —
the Ableton 'automate the sample's tempo down to 20 bpm without changing
pitch' effect — layered as slightly detuned voices whose stretch curves
drift in and out of phase (phasor / shimmer).

How it works:
- Grains are always read from the input at 1x speed (pitch holds exactly,
  like warp mode), but the input position advances slower and slower as the
  local stretch factor grows — the sound "slows down" without transposing.
- The local stretch factor follows a curve (ramp / brake / wave / breathe)
  AND a phasor LFO — the stretch breathes and wobbles.
- Several voices run the same scan with tiny rate detunes and LFO phase
  offsets: they phase against each other -> shimmer/comb movement.
- When the scan passes the end of the (tiled) input, the hit softly
  re-fires in slow motion."""
import numpy as np

from .core import clamp_dur, safe


def _tile_xfade(x, need, sr):
    """Tile input to `need` samples, crossfading copies (Hann 50% COLA)."""
    n = x.shape[0]
    if n >= need:
        return np.ascontiguousarray(x[:need])
    xf = int(min(0.03 * sr, max(8, n // 2)))
    step = max(1, n - xf)
    w = np.hanning(2 * xf).astype(np.float32)
    rise, fall = w[:xf], w[xf:]
    out = np.zeros((need + xf, x.shape[1]), np.float32)
    pos = 0
    first = True
    while pos < need + xf:
        seg = x.copy()
        if not first:
            seg[:xf] *= rise[:, None]
        if pos + n < need + xf:
            seg[-xf:] *= fall[:, None]
        e = min(pos + n, need + xf)
        out[pos:e] += seg[: e - pos]
        pos += step
        first = False
    return out[:need]


def _base_curve(curve, t, T, max_stretch, curve_shape, stretch0=1.0):
    """Stretch-factor automation over output time [0, T]."""
    if curve == "ramp":            # tempo automation: stretch0 -> max
        base = stretch0 + (max_stretch - stretch0) * (t / T) ** curve_shape
    elif curve == "brake":          # starts super-slow, accelerates to normal
        base = max_stretch - (max_stretch - 1.0) * (t / T) ** curve_shape
    elif curve == "wave":           # slow down into the middle, recover out
        bell = np.exp(-0.5 * ((t - T * 0.5) / (T * 0.22)) ** 2)
        base = stretch0 + (max_stretch - stretch0) * bell
    elif curve == "breathe":        # constant mean stretch, LFO does the rest
        base = np.full(len(t), 0.5 * (1.0 + max_stretch))
    else:
        base = np.ones(len(t))
    return np.maximum(base, 0.3)


def stretch_modulate(x, sr, rng, max_stretch=10.0, curve="ramp",
                     curve_shape=1.5, rate_hz=0.15, depth=0.2, voices=3,
                     voice_drift=0.005, grain_ms=70.0, jitter_ms=3.0,
                     out_s=7.0, pan_spread=0.7, tile=True, stretch0=1.0):
    """Pitch-preserving stretch-modulation with phasor shimmer.

    curve        ramp | brake | wave | breathe — the tempo-automation shape
    max_stretch  how deep the slow-down gets (1 = none)
    stretch0     starting stretch for ramp/wave (1 = begin at natural speed;
                 higher = the hit is already slow out of the gate)
    curve_shape  ramp/brake exponent (higher = stays fast longer, then dives)
    rate_hz      phasor LFO rate modulating the local stretch factor
    depth        LFO depth (0.2 = +-20% stretch wobble)
    voices       layered scan voices; voice 0 is the center anchor
    voice_drift  per-voice stretch detune — the phasing/shimmer amount
    grain_ms     grain size (smaller = cloudier, bigger = silkier)
    jitter_ms    per-grain input jitter (decorrelates grain boundaries)
    out_s        output duration
    tile         True: crossfade-tile the input so the scan can outrun it
                 (the hit softly re-fires). False: each voice stops when it
                 reaches the end of the input — a natural oneshot ending,
                 with the trailing silence trimmed.
    """
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n_ch = x.shape[1]
    n_out = int(out_s * sr)
    g = max(128, int(grain_ms / 1000.0 * sr)) // 2 * 2
    hop = g // 2
    win = np.hanning(g).astype(np.float32)
    T = n_out / sr
    t = np.arange(n_out) / sr

    base = _base_curve(curve, t, T, max_stretch, curve_shape, stretch0)
    if tile:
        # how much input could the fastest voice consume? tile for safety
        s_min = max(0.25, float(base.min()) * max(0.1, 1.0 - depth - 2 * voice_drift))
        need = min(int(n_out / s_min) + 4 * g, int(120 * sr))   # sane cap
        xt = _tile_xfade(x, need, sr)
    else:
        need = x.shape[0]
        xt = x

    out = np.zeros((n_out + g, n_ch), np.float32)
    jitter = (jitter_ms / 1000.0) * sr
    for v in range(int(voices)):
        drift = 0.0 if v == 0 else \
            (voice_drift if v % 2 else -voice_drift) * (1.0 + 0.3 * (v // 2))
        lfo_ph = rng.uniform(0, 2 * np.pi)
        s_v = np.maximum(base * (1.0 + depth * np.sin(2 * np.pi * rate_hz * t
                                                      + lfo_ph))
                         * (1.0 + drift), 0.25)
        if v == 0:
            pan = 0.0
        else:
            k = max(1, voices - 1)
            pan = pan_spread * (((v - 1) / k) * 2 - 1)
        th = (pan + 1) * np.pi / 4
        gv = np.float32([np.cos(th), np.sin(th)])
        in_pos = 0.0
        for i in range(0, n_out, hop):
            p0 = int(in_pos)
            off = int(rng.normal(0, jitter)) if jitter > 0 else 0
            src = max(0, p0 + off)
            if src + g >= need:
                break
            e = min(n_out, i + g)
            out[i:e] += xt[src: src + (e - i)] * win[: e - i, None] * gv
            in_pos += hop / s_v[i]
    out = out[:n_out] / max(1.0, float(voices))
    if not tile:
        # natural ending: trim trailing silence (keep a 40ms pad)
        m = np.max(np.abs(out), axis=1)
        nz = np.flatnonzero(m > 1e-3)
        if len(nz):
            out = out[: min(n_out, int(nz[-1]) + int(0.04 * sr))]
    return clamp_dur(safe(out), sr, out_s, fade=0.04)