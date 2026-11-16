"""Harmonic family: octaver (interval stack layering), drifting detune
chorus, shimmer harmonizer (pitched delay-tap feedback cascades), reich
phasing (loops at slightly mismatched speeds), resonator (measure the hit's
own harmonic series and re-ring it), and a tuned sub-octave layer."""
import numpy as np

from .core import (EPS, dominant_f0, mono, resample,
                   rms_env, safe)


def _pan(voice_i, n_voices, spread=0.6):
    pan = ((voice_i / max(1, n_voices - 1)) * 2 - 1) * spread
    th = (pan + 1) * np.pi / 4
    return np.cos(th), np.sin(th)


def octaver(x, sr, rng, intervals=(-12, -5, 12), gains=(0.5, 0.35, 0.3),
            detune_cents=8.0, max_s=None):
    """Layer resampled copies at musical intervals (tape pitch), each
    slightly detuned and panned — instant dark octaver / fifth stack."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    out = x * 0.9
    for i, (iv, g) in enumerate(zip(intervals, gains)):
        det = 2.0 ** (rng.uniform(-detune_cents, detune_cents) / 1200.0)
        sp = 2.0 ** (iv / 12.0) * det
        v = resample(x, sp)
        gl, gr = _pan(i, len(intervals))
        if v.ndim == 1:
            v = np.stack([v, v], axis=1)
        L = min(len(out), len(v))
        out[:L] += v[:L] * np.float32(g) * np.float32([gl, gr])
    return safe(out)


def detune_chorus(x, sr, rng, voices=3, detune_cents=6.0, drift_ms=6.0,
                  spread=0.7, max_s=None):
    """Wide drifting chorus: voices resampled at ±cents with slow sinusoidal
    speed wobble and pan spread — the hit blooms into a swarm."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    out = x * 0.75
    n = x.shape[0]
    t = np.arange(n) / sr
    for v in range(voices):
        cents = rng.uniform(0.5, detune_cents) * (1 if v % 2 else -1)
        wob = rng.uniform(0.3, 1.5)  # drift LFO Hz... slow
        speed = 2.0 ** (cents / 1200.0) * \
            (1 + (drift_ms / 1000.0 * 0.1) * np.sin(2 * np.pi * wob * t))
        pos = np.cumsum(speed)
        from .core import interp_hermite
        vv = np.stack([interp_hermite(x[:, c], pos)[: n]
                       for c in range(x.shape[1])], axis=1)
        # small modulated delay for shimmer motion
        d0 = rng.uniform(0.002, 0.012)
        dm = drift_ms / 1000.0
        dl = int(d0 * sr) + (dm * sr * 0.5 *
                             (0.5 + 0.5 * np.sin(2 * np.pi * wob * t))).astype(int)
        wet = np.zeros_like(vv)
        idx = np.arange(n)
        dl_c = np.clip(dl, 0, n - 1)
        wet = vv[np.clip(idx - dl_c, 0, n - 1)]
        gl, gr = _pan(v, voices, spread)
        out += wet * np.float32(0.35) * np.float32([gl, gr])
    return safe(out)


def shimmer_harmonizer(x, sr, rng, taps=None, fb=0.8, iters=12, max_s=8.0):
    """Delay taps with tape-pitch-shifted feedback at interval ratios — the
    harmonizer cascade plays the oneshot's harmonic space. From the
    harmonic lab shimmer."""
    if taps is None:
        taps = tuple((float(rng.uniform(0.15, 0.6)),
                      float(rng.choice([2.0, 1.5, 0.5, 3.0, 4.0])),
                      float(rng.uniform(0.35, 0.55)))
                     for _ in range(int(rng.integers(2, 4))))
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    n = int(min(max(len(x), 2.0 * sr), max_s * sr))
    buf = np.zeros((n, 2), np.float32)
    m = min(len(x), n)
    buf[:m] = x[:m] * 0.7
    acc = buf.copy()
    for _ in range(int(iters)):
        nxt = np.zeros_like(buf)
        for d, ratio, g in taps:
            off = int(d * sr)
            if off >= n:
                continue
            shifted = resample(buf, ratio)
            if len(shifted) >= n - off:
                tail = shifted[: n - off]
            else:
                tail = np.pad(shifted, ((0, n - off - len(shifted)), (0, 0)))
            nxt[off:] += tail * np.float32(g)
        buf = nxt * np.float32(fb)
        acc += buf
        if float(np.abs(buf).max()) < 1e-4:
            break
    return safe(acc)


def reich_phase(x, sr, rng, dur_s=None, rates=None, pan_spread=0.6, max_s=8.0):
    """Classic gradual phasing: copies of the oneshot looped at slightly
    different speeds, drifting in and out of sync. From the harmonic lab."""
    x = np.asarray(x, np.float32)
    if rates is None:
        rates = (1.0, 1.0 + rng.uniform(0.004, 0.012),
                 1.0 - rng.uniform(0.004, 0.012))
    if dur_s is None:
        dur_s = min(max_s, max(4.0, 2.2 * len(x) / sr))
    loop = x.copy()
    f = min(len(loop) // 8, int(0.05 * sr))
    if f > 2:
        r = np.linspace(0, 1, f)
        if loop.ndim == 1:
            loop[:f] *= r
            loop[-f:] *= r[::-1]
        else:
            loop[:f] *= r[:, None]
            loop[-f:] *= r[::-1, None]
    out = []
    need = int(dur_s * sr)
    for vi, r in enumerate(rates):
        tiles = np.concatenate([loop] * int(np.ceil(need * r / len(loop)) + 1))
        voice = resample(tiles, r)[:need]
        if voice.ndim == 1:
            voice = np.stack([voice, voice], axis=1)
        gl, gr = _pan(vi, len(rates), pan_spread)
        out.append(voice * np.float32([gl, gr]))
    L = min(len(v) for v in out)
    return safe(np.sum([v[:L] for v in out], axis=0) / len(out))


def resonator(x, sr, rng, dur_s=None, n_partials=None, decay=None,
              detune=0.0012, follow=0.35, max_s=8.0):
    """Measure the oneshot's own harmonic series at its detected f0 and
    re-ring it as decaying detuned partials — the sound rings its own pitch
    back. From the harmonic lab resonator."""
    x = np.asarray(x, np.float32)
    m = mono(x)
    f0 = dominant_f0(x, sr, default=55.0)
    if dur_s is None:
        dur_s = min(max_s, max(3.0, 2.2 * len(x) / sr))
    if n_partials is None:
        n_partials = int(rng.integers(5, 10))
    if decay is None:
        decay = float(rng.uniform(1.5, 4.0))
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    # measure partial amplitudes/phases near h*f0
    seg = m[: min(len(m), int(0.5 * sr))] * np.hanning(min(len(m), int(0.5 * sr)))
    S = np.abs(np.fft.rfft(seg)) * 2.0 / max(2, len(seg))
    fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
    amps, phases = [], []
    for h in range(1, n_partials + 1):
        c = h * f0
        if c > sr / 2 - 200:
            break
        band = (fr > c * 0.94) & (fr < c * 1.06)
        if not band.any():
            amps.append(0.0)
            phases.append(0.0)
            continue
        i0 = np.flatnonzero(band)[int(np.argmax(S[band]))]
        amps.append(float(S[i0]))
        phases.append(float(np.angle(np.fft.rfft(seg))[i0]))
    if not amps:                               # f0 too high for any partial
        f0 = max(30.0, sr / 30.0 / n_partials)
        amps = [1.0 / h for h in range(1, n_partials + 1)]
        phases = [0.0] * n_partials
    mx = max(amps) + EPS
    ring = np.zeros(n)
    for h, (a, p) in enumerate(zip(amps, phases), start=1):
        if a < 0.01 * mx:
            continue
        det = 1 + detune * (1 if h % 2 else -1) * rng.uniform(0.5, 1.5)
        ring += (a / mx) * np.sin(2 * np.pi * h * f0 * det * t + p) / h ** 0.3
    ring *= np.exp(-t * decay) * np.minimum(1.0, t / 0.01)
    # blend a little of the original at the head
    out = np.stack([ring, np.roll(ring, int(0.002 * sr))], axis=1)
    if follow > 0:
        olen = min(len(x), n)
        out[:olen] += x[:olen] * np.float32(follow)
    return safe(out)


def sub_octave(x, sr, rng, div=2, gain=0.7, follow=True, max_s=None):
    """Add a tuned pure sine sub (at the detected f0, or f0/div) whose
    envelope follows the hit's own — instant sub-weight."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    f0 = dominant_f0(x, sr, default=50.0) / div
    if f0 < 18:
        f0 = 18.0
    n = x.shape[0]
    t = np.arange(n) / sr
    sub = np.sin(2 * np.pi * f0 * t +
                 rng.uniform(0, 0.02) * np.sin(2 * np.pi * 0.7 * t)) \
        .astype(np.float32)
    if follow:
        env = rms_env(mono(x), sr, 0.03)
        pk = env.max() + EPS
        sub *= (env / pk)
    sub = sub[:, None]
    out = x + sub * np.float32(gain)
    return safe(out)