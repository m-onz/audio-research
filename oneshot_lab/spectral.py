"""Spectral transformations: freeze, temporal blur/smear, frame shuffle,
spectral reverse (envelope backwards, waveform forward), bin roll (metallic
frequency rotation), spectral hybrid (cross-synthesis with self variants),
NMF self-stamping (re-project onto its own learned dictionary), spectral
morph (magnitudes A->B->A with held phases), LPC reanimation (formant
resynthesis), and a tuned comb bank."""
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import istft, lfilter, stft

from .core import (EPS, clamp_dur, dominant_f0, mono, safe, st)

_NP = 2048          # analysis window
_NOV = 1536         # 75% overlap


def _stft(y, sr):
    _, _, Z = stft(y, sr, nperseg=_NP, noverlap=_NOV, boundary="even")
    return Z


def _istft(Z, sr):
    _, y = istft(Z, sr, nperseg=_NP, noverlap=_NOV, boundary="even")
    return y


def _frames_to_mag(Z):
    return np.abs(Z).astype(np.float32), np.angle(Z).astype(np.float32)


def _rebuild(mag, ph, sr):
    return _istft(mag * np.exp(1j * ph.astype(np.float64)), sr)


def spectral_freeze(x, sr, rng, at=0.4, dur_s=6.0, shimmer=0.12,
                    detune=0.002, max_s=10.0):
    """Spectral freeze: hold the magnitude frame around `at` forever, phases
    drifting forward with a slow random detune — a frozen ghost of the hit."""
    def one(y):
        Z = _stft(y, sr)
        mag, _ = _frames_to_mag(Z)
        nf = mag.shape[1]
        if nf < 2:
            return y
        k = min(nf - 1, max(0, int(at * nf)))
        m0 = mag[:, max(0, k - 2): k + 3].mean(axis=1) if nf > 4 else mag[:, k]
        n_out = max(2, int(min(dur_s, max_s) * sr / (_NP - _NOV)))
        fr = np.fft.rfftfreq(_NP, 1.0 / sr)
        adv = 2 * np.pi * fr * (_NP - _NOV) / sr
        drift = rng.uniform(-1, 1, len(fr)) * detune
        syn_ph = rng.uniform(0, 2 * np.pi, len(fr))
        env = np.abs(y)
        w = max(2, int(0.05 * sr))
        env = np.convolve(env, np.ones(w) / w, "same")
        n = int(min(dur_s, max_s) * sr)
        out = np.zeros(n, np.float32)
        # follow the original's amplitude envelope with the frozen texture
        env_out = np.interp(np.linspace(0, 1, n),
                            np.linspace(0, 1, len(env)),
                            np.maximum(env, EPS)).astype(np.float32)
        for i in range(n_out):
            syn_ph = syn_ph + adv * (1.0 + drift)
            syn_ph = (syn_ph + np.pi) % (2 * np.pi) - np.pi
            fr_mag = m0 * (1.0 + shimmer * rng.uniform(-1, 1, len(m0))) \
                if shimmer else m0
            frame = np.fft.irfft(fr_mag * np.exp(1j * syn_ph), _NP)
            p = i * (_NP - _NOV)
            e = min(n, p + _NP)
            out[p:e] += frame[: e - p] * np.hanning(_NP)[: e - p]
        pk = float(np.max(np.abs(out)))
        if pk > EPS:
            out = out / pk * env_out
        return out

    y = st(one, x)
    return clamp_dur(safe(y), sr, max_s, fade=0.05)


def spectral_smear(x, sr, rng, blur_s=0.35, mix=0.75, max_s=None):
    """Temporal blur: each frame's magnitudes become a moving average over
    +/- blur_s of frames; transients melt, ghosts remain."""
    def one(y):
        Z = _stft(y, sr)
        mag, ph = _frames_to_mag(Z)
        hop_t = (_NP - _NOV) / sr
        k = max(1, int(blur_s / hop_t))
        kern = np.hanning(2 * k + 1).astype(np.float32)
        kern /= kern.sum()
        blurred = np.zeros_like(mag)
        for b in range(mag.shape[0]):
            blurred[b] = np.convolve(
                np.pad(mag[b], (k, k), mode="edge"), kern, "same")[k:-k]
        mag2 = (1.0 - mix) * mag + mix * blurred
        return _rebuild(mag2, ph, sr)

    return safe(st(one, x))


def spectral_shuffle(x, sr, rng, block_ms=120.0, reverse_p=0.25, mag_only=False):
    """Spectral glitch: chop the STFT into time blocks and permute them;
    some blocks land reversed. mag_only keeps original phase (ghostlier)."""
    def one(y):
        Z = _stft(y, sr)
        mag, ph = _frames_to_mag(Z)
        hop_t = (_NP - _NOV) / sr
        B = max(1, int(block_ms / 1000.0 / hop_t))
        nf = mag.shape[1]
        nb = nf // B
        if nb < 2:
            return y
        mag = mag[:, : nb * B].reshape(mag.shape[0], nb, B)
        ph = ph[:, : nb * B].reshape(ph.shape[0], nb, B)
        order = rng.permutation(nb)
        for i in range(nb):
            if rng.random() < reverse_p:
                mag[:, i] = mag[:, i][:, ::-1]
                if not mag_only:
                    ph[:, i] = ph[:, i][:, ::-1]
        mag = mag[:, order].reshape(mag.shape[0], nb * B)
        ph = ph[:, order].reshape(ph.shape[0], nb * B)
        out = _rebuild(mag, ph, sr)
        return out[: len(y)] if len(out) > len(y) else \
            np.pad(out, (0, len(y) - len(out)))

    return safe(st(one, x))


def spectral_reverse(x, sr, rng, bin_flip_p=0.35, max_s=None):
    """Envelope backwards, waveform forwards: magnitudes' time order is
    reversed while the original phases march on — the 'reverse without
    reversing' possession. Optionally also flip frequency bins (metallic)."""
    def one(y):
        Z = _stft(y, sr)
        mag, ph = _frames_to_mag(Z)
        mag = mag[:, ::-1]
        if bin_flip_p and rng.random() < bin_flip_p:
            mag = mag[::-1, :]
        return _rebuild(mag, ph, sr)[: len(y)]

    return safe(st(one, x))


def spectral_binroll(x, sr, rng, shift_bins=37, drift_bins=10, max_s=None):
    """Roll the frequency axis by shift bins (wrapping) with slow drift —
    comb-metallic alien recasting of the partials."""
    def one(y):
        Z = _stft(y, sr)
        mag, ph = _frames_to_mag(Z)
        nf = mag.shape[1]
        offs = shift_bins + drift_bins * np.sin(
            np.linspace(0, rng.uniform(1, 9), nf))
        offs = np.round(offs).astype(int)
        for i in range(nf):
            mag[:, i] = np.roll(mag[:, i], offs[i])
        return _rebuild(mag, ph, sr)[: len(y)]

    return safe(st(one, x))


def spectral_hybrid(x, sr, rng, ref="reverse", mix=0.6, max_s=None):
    """Spectral convolution: keep x's phases, multiply magnitudes by the
    time-averaged magnitude envelope of a variant of itself ('reverse',
    'shuffle', 'roll'). From the databend conv_stamp."""
    def one(y):
        Z = _stft(y, sr)
        mag, ph = _frames_to_mag(Z)
        if ref == "reverse":
            Zr = _stft(y[::-1].copy(), sr)
        elif ref == "shuffle":
            Zr = Z.copy()
            nf = Zr.shape[1]
            B = max(1, nf // 8)
            nb = nf // B
            Zr = Zr[:, : nb * B].reshape(Zr.shape[0], nb, B)[:, rng.permutation(nb)]
            Zr = Zr.reshape(Zr.shape[0], nb * B)
        else:  # 'roll': metallic frequency rotation of the reference
            Zr = np.roll(Z, 37, axis=0)
        env = np.abs(Zr).mean(axis=1, keepdims=True).astype(np.float32)
        # compress the per-bin envelope so peaky bass doesn't blow up:
        # peak-normalize, soften, renormalize to mean 1, clamp
        env = env / (env.max() + EPS)
        env = np.power(env, 0.75)
        env = env / (env.mean() + EPS)
        env = np.clip(env, 0.0, 6.0)
        mag2 = mag * (1.0 - mix + mix * env)
        return _rebuild(mag2, ph, sr)[: len(y)]

    return safe(st(one, x))


def spectral_nmf(x, sr, rng, k=24, iters=35, mix=1.0, max_s=None):
    """Learn a spectral dictionary from the oneshot's OWN STFT, then rebuild
    it as a mixture of those atoms — self-requantized texture. From the
    databend NMF stamp."""
    def one(y):
        Z = _stft(y, sr)
        V = np.abs(Z).astype(np.float32)
        if V.shape[1] < 4:
            return y
        Vn = V / (V.mean() + EPS)
        W = (rng.random((V.shape[0], k)) * 0.1 + 0.01).astype(np.float32)
        H = (rng.random((k, V.shape[1])) * 0.1 + 0.01).astype(np.float32)
        for _ in range(iters):
            H *= (W.T @ Vn) / (W.T @ W @ H + EPS)
            W *= (Vn @ H.T) / (W @ H @ H.T + EPS)
            nrm = np.linalg.norm(W, axis=0, keepdims=True) + EPS
            W /= nrm
            H *= nrm.T
        V2 = (W @ H).astype(np.float32)
        V2 *= (V.mean() + EPS) / (V2.mean() + EPS)
        mag2 = (1.0 - mix) * V + mix * V2
        return _rebuild(mag2, np.angle(Z).astype(np.float32), sr)[: len(y)]

    return safe(st(one, x))


def spectral_morph(x, sr, rng, target="reverse", cycles=2.0, max_s=None):
    """Magnitudes crossfade original -> transformed variant -> original over
    `cycles` round trips while the original phases hold steady. From the
    crossover 'possession chain'."""
    def one(y):
        Z = _stft(y, sr)
        mag, ph = _frames_to_mag(Z)
        Zt = _stft(y[::-1].copy(), sr) if target == "reverse" else \
            np.roll(Z, 37, axis=0)
        mag_t = np.abs(Zt).astype(np.float32)
        nf = min(mag.shape[1], mag_t.shape[1])
        mag, mag_t, ph = mag[:, :nf], mag_t[:, :nf], ph[:, :nf]
        tt = np.linspace(0, cycles, nf, endpoint=False) % 1.0
        w = 0.5 - 0.5 * np.cos(2 * np.pi * tt)          # 0->1->0 morph window
        mag2 = mag * (1 - w)[None, :] + mag_t * w[None, :]
        return _rebuild(mag2, ph, sr)[: len(y)]

    return safe(st(one, x))


# ----------------------------------------------------------------- LPC


def _lpc_frame(y, order=24):
    """Stable LPC coefficients for one frame (poles pulled inside 0.985)."""
    r = y * np.hanning(len(y))
    ac = np.correlate(r, r, "full")[len(r) - 1:]
    if ac[0] < EPS:
        return None
    ac = ac / ac[0]
    try:
        a = solve_toeplitz(ac[:order], ac[1:order + 1])
    except Exception:
        return None
    poly = np.concatenate([[1.0], -a])
    roots = np.roots(poly)
    mx = np.max(np.abs(roots)) if len(roots) else 0.0
    if mx > 0.985:
        a = a * (0.985 / mx)
    return a.astype(np.float64)


def lpc_reanimate(x, sr, rng, dur_s=None, excitation="pulse", order=24,
                  max_s=8.0):
    """LPC reanimation: build a time-varying formant filter from the
    oneshot's own LPC frames, then re-excite it — the hit 'speaks' with a
    new source: pitched glottal buzz at its own f0, breath noise, or its own
    whitened self. From the databend LPC stamp."""
    if dur_s is None:
        dur_s = min(max_s, max(2.0, len(x) / sr))
    n_out = int(dur_s * sr)
    m = mono(x)
    f0 = dominant_f0(x, sr)
    hop, fl = 1024, 2048
    frames = []
    for i in range(0, max(1, len(m) - fl), hop):
        a = _lpc_frame(m[i:i + fl], order)
        if a is not None:
            frames.append(a)
    if not frames:
        return x
    A = np.stack(frames)
    # excitation source
    t = np.arange(n_out) / sr
    if excitation == "noise":
        exc = rng.standard_normal(n_out).astype(np.float64) * 0.25
    elif excitation == "self":
        exc = np.diff(m, prepend=m[0])
        exc = np.pad(exc, (0, max(0, n_out - len(exc))))[:n_out]
        exc = exc / (np.std(exc) + EPS) * 0.25
    else:  # pulse: glottal buzz at f0 with jitter
        jit = 1.0 + 0.01 * np.sin(2 * np.pi * rng.uniform(0.5, 6) * t)
        phase = np.cumsum(f0 * jit) / sr
        exc = (np.sin(2 * np.pi * phase) * 0.6 +
               rng.standard_normal(n_out) * 0.05)
    # drive the interpolated LPC filter blockwise, following original env
    pos = np.linspace(0, len(A) - 1, max(2, int(n_out / hop) + 1))
    i0 = np.clip(pos.astype(int), 0, len(A) - 1)
    A_interp = A[i0] * (1 - (pos - i0))[:, None] + \
        A[np.clip(i0 + 1, 0, len(A) - 1)] * (pos - i0)[:, None]
    env = np.abs(m)
    w = max(2, int(0.03 * sr))
    env = np.convolve(env, np.ones(w) / w, "same")
    env_out = np.interp(np.linspace(0, 1, n_out),
                        np.linspace(0, 1, len(env)),
                        np.maximum(env, EPS))
    out = np.zeros(n_out)
    zi = np.zeros(order)
    for bi in range(len(A_interp)):
        a = A_interp[bi].copy()
        roots = np.roots(np.concatenate([[1.0], -a]))
        mx = np.max(np.abs(roots)) if len(roots) else 0.0
        if mx > 0.99:
            a *= 0.99 / mx
        s = bi * hop
        e = min(n_out, s + hop)
        if s >= n_out:
            break
        yb, zi = lfilter([1.0], np.concatenate([[1.0], -a]), exc[s:e], zi=zi)
        if (not np.all(np.isfinite(yb))) or float(np.max(np.abs(yb))) > 50.0:
            zi = np.zeros(order)
            yb = exc[s:e]
        out[s:e] = yb
    out *= env_out / (np.std(out) + EPS)
    out = np.clip(out, -1.0, 1.0)
    return safe(np.stack([out, np.roll(out, int(0.002 * sr))], axis=1)
                ).astype(np.float32)


# ----------------------------------------------------------------- combs


def comb_bank(x, sr, rng, n_combs=6, feedback=0.72, f0=None,
              ratios=None, mix=0.6, max_s=None):
    """Bank of tuned feedback combs at f0 × (harmonic or inharmonic) ratios —
    metallic ring wash locked to the hit's own pitch."""
    x = np.asarray(x, np.float32)
    f0 = f0 or dominant_f0(x, sr)
    if ratios is None:
        ratios = list(range(1, n_combs + 1))
    out = np.zeros_like(x)
    for ch in range(x.shape[1]):
        y = x[:, ch].astype(np.float64)
        acc = np.zeros_like(y)
        for r in ratios:
            f = f0 * r * rng.uniform(0.999, 1.001)
            d = max(2, int(sr / max(20.0, f)))
            fb = feedback * rng.uniform(0.85, 1.05)
            z = y.copy()
            # block-recursion comb: y[i] += fb * y[i-d]
            for s in range(d, len(y), d):
                e = min(len(y), s + d)
                z[s:e] += fb * z[s - d: e - d]
            acc += z
        out[:, ch] = acc
    out = (1.0 - mix) * x + mix * out / (len(ratios) + EPS)
    return safe(out).astype(np.float32)