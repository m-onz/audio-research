"""Novel analysis-driven transformations — partial-level operations.

The library's spectral transforms work on whole magnitude frames; these go
finer: they TRACK individual partials (sinusoidal peaks) and operate on them
directly:

- spectral_dna      partial-level crossbreeding: keeper A's peak structure,
                    keeper B's amplitudes grafted at the nearest partials
- partial_resequence each partial's amplitude trajectory is independently
                    reversed / rotated / held / stuttered / swapped / re-rung
- env_splice        keeper A's spectral body wearing keeper B's amplitude
                    envelope (plus optional stolen-attack graft)
- formant_walk      A's fine spectral structure, its 1/3-octave envelope
                    gliding into B's formant shape and back
- harmonic_grid     re-tune the detected f0 onto a scale, shear odd/even
                    partials, add an octave shadow + sub
"""
import numpy as np

from .core import EPS, clamp_dur, dominant_f0, mono, safe, st
from .spectral import _NP, _NOV, _istft, _stft

HOP = _NP - _NOV          # 512
K_DEF = 16                # partials tracked per frame
SIGMA = 1.6               # peak gaussian width (bins) ~ hann mainlobe


# ----------------------------------------------------------------- partials


def _frame_peaks(mag, k, sr, fmin=50.0, floor=0.02):
    """Top-k parabolic-interpolated spectral peaks of one magnitude frame.
    Returns array (k', 2): [freq_hz, amp]."""
    mx = float(mag.max())
    if mx < EPS:
        return np.zeros((0, 2))
    idx = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1
    if len(idx) == 0:
        return np.zeros((0, 2))
    a, b, c = mag[idx - 1], mag[idx], mag[idx + 1]
    den = (a - 2 * b + c)
    d = np.where(np.abs(den) > EPS, 0.5 * (a - c) / np.where(
        np.abs(den) > EPS, den, 1.0), 0.0)
    d = np.clip(d, -0.5, 0.5)
    freqs = (idx + d) * sr / _NP
    amps = b - 0.25 * (a - c) * d
    keep = (freqs >= fmin) & (amps >= floor * mx)
    freqs, amps = freqs[keep], amps[keep]
    if len(freqs) == 0:
        return np.zeros((0, 2))
    order = np.argsort(amps)[::-1][: k]
    return np.stack([freqs[order], amps[order]], axis=1)


def _all_peaks(mags, k, sr):
    """Peaks for every STFT frame. mags: (bins, frames)."""
    return [_frame_peaks(mags[:, j], k, sr) for j in range(mags.shape[1])]


def _add_gauss(mag, freqs, amps, sr, sigma=SIGMA):
    """Add gaussian bumps into a magnitude frame at freqs (Hz)."""
    n = len(mag)
    for f, a in zip(freqs, amps):
        c = f * _NP / sr
        lo = max(0, int(c - 4 * sigma))
        hi = min(n, int(c + 4 * sigma) + 1)
        if hi <= lo:
            continue
        bins = np.arange(lo, hi)
        mag[lo:hi] += float(a) * np.exp(-0.5 * ((bins - c) / sigma) ** 2)


def _peaks_to_frames_mag(shape, peaks, sr):
    """Render peak lists [(freq, amp) per frame] into a magnitude array.
    Rows with non-finite freq or zero amp are skipped."""
    out = np.zeros(shape, np.float32)
    for j, pk in enumerate(peaks):
        if len(pk):
            pk = np.asarray(pk, np.float64)
            ok = np.isfinite(pk[:, 0]) & (pk[:, 1] > 0)
            if ok.any():
                _add_gauss(out[:, j], pk[ok, 0], pk[ok, 1], sr)
    return out


def _link_tracks(peaks, tol_hz_base=15.0, tol_rel=0.06, max_tracks=24,
                 min_life_frac=0.25):
    """Greedy nearest-frequency partial tracking across frames.
    Returns tracks: list of dict(freqs (nF,), amps (nF,), life)."""
    nF = len(peaks)
    tracks = []
    for j, pk in enumerate(peaks):
        if len(pk) == 0:
            continue
        for f, a in sorted(pk, key=lambda p: -p[1]):
            t0, best = None, 1e9
            for t in tracks:
                lf = t["last"]
                if lf is None:
                    continue
                df = abs(lf - f)
                if df < best:
                    best, t0 = df, t
            if t0 is not None and best <= max(tol_hz_base, tol_rel * f):
                t0["freqs"][j] = f
                t0["amps"][j] = a
                t0["last"] = f
                continue
            if len(tracks) < max_tracks:
                tracks.append(dict(freqs=np.full(nF, np.nan),
                                   amps=np.zeros(nF), last=f))
                tracks[-1]["freqs"][j] = f
                tracks[-1]["amps"][j] = a
    for t in tracks:
        t["life"] = int(np.sum(t["amps"] > 0))
        f = t["freqs"]               # forward-fill freq gaps
        last = None
        for j in range(len(f)):
            if not np.isnan(f[j]):
                last = f[j]
            elif last is not None:
                f[j] = last
    min_life = max(4, int(min_life_frac * nF))
    return [t for t in tracks if t["life"] >= min_life]


def _smooth_env(mag, sr, n_bands=64, fmin=50.0):
    """1/3-octave-ish spectral envelope of a magnitude frame (bins,)
    -> envelope over the same bins."""
    n = len(mag)
    fmax = sr / 2
    band_f = np.geomspace(fmin, fmax, n_bands)
    band_v = np.interp(band_f, np.arange(n) * sr / _NP, mag)
    return np.interp(np.arange(n) * sr / _NP, band_f, band_v)


def _resample_frames(vals, n):
    """Resample a per-frame series to n frames."""
    v = np.asarray(vals, np.float64)
    if len(v) < 2:
        return np.full(n, v[0] if len(v) else 1.0)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)


# ----------------------------------------------------------------- transforms


def spectral_dna(x, sr, rng, partner=None, k=K_DEF, cross=0.8,
                 residual_mix=0.45, tol_cents=60.0, max_s=None):
    """PARTIAL-LEVEL CROSSBREEDING: child = keeper A's partial structure
    (frequencies, timing, residual body) with keeper B's partial amplitudes
    grafted onto the nearest matching partials. B's loud partials silence
    A's quiet ones and vice versa — a genetic splice of two timbres.

    partner=None breeds A with its own reversed ghost."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if partner is None:
        partner = x[::-1].copy()
    partner = np.asarray(partner, np.float32)
    mA, mB = mono(x), mono(partner)
    peaksA = None
    peaksB = _all_peaks(np.abs(_stft(mB, sr)), k, sr)

    def one(y):
        nonlocal peaksA
        Z = _stft(y, sr)
        mag, ph = np.abs(Z), np.angle(Z).astype(np.float32)
        if peaksA is None:                     # analyze once (mono ref)
            peaksA = _all_peaks(np.abs(_stft(mA, sr)), k, sr)
        nF = mag.shape[1]
        child = np.zeros_like(mag)
        for j in range(nF):
            jb = int(j / max(1, nF - 1) * max(0, len(peaksB) - 1))
            pkA = peaksA[min(j, len(peaksA) - 1)]
            pkB = peaksB[jb]
            if len(pkA) == 0:
                child[:, j] = mag[:, j] * residual_mix
                continue
            # A's frame minus its own partial energy = the residual body
            res = mag[:, j].copy()
            _add_gauss(res, pkA[:, 0], -pkA[:, 1], sr)
            new_pk = np.empty_like(pkA)
            for i, (f, a) in enumerate(pkA):
                amp_new = a * (1.0 - cross) * 0.6      # orphan: attenuated
                if len(pkB):
                    tol = f * (2.0 ** (tol_cents / 1200.0)) - f
                    d = np.abs(pkB[:, 0] - f)
                    bi = int(np.argmin(d))
                    if d[bi] <= max(12.0, tol):
                        amp_new = cross * pkB[bi, 1] + \
                            (1 - cross) * a * 0.5
                new_pk[i] = (f, max(0.0, amp_new))
            frame = np.maximum(res, 0.0) * residual_mix
            _add_gauss(frame, new_pk[:, 0], new_pk[:, 1], sr)
            child[:, j] = frame
        out = _istft(child * np.exp(1j * ph), sr)
        return out[: len(y)] if len(out) > len(y) else \
            np.pad(out, (0, len(y) - len(out)))

    return safe(st(one, x))


def partial_resequence(x, sr, rng, k=14, ops=None, residual_mix=0.6, max_s=None):
    """Each partial's amplitude TRAJECTORY is independently re-sequenced:
    reversed / time-rotated / held / block-stuttered / swapped with another
    partial's line / re-rung with a new decay. Phases keep marching —
    the timing feel stays but the spectral evolution scrambles."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    m = mono(x)
    Zm = _stft(m, sr)
    magsM = np.abs(Zm)
    peaks = _all_peaks(magsM, k, sr)
    tracks = _link_tracks(peaks)
    if not tracks:
        return x
    nF = magsM.shape[1]
    ops = ops or ["reverse", "rotate", "hold", "stutter", "swap", "redecay"]
    # snapshot trajectories, apply ops, then handle swaps in pairs
    track_amps = [t["amps"].astype(np.float64).copy() for t in tracks]
    plan = [str(rng.choice(ops)) for _ in tracks]
    for ti, op in enumerate(plan):
        a = track_amps[ti]
        if op == "reverse":
            track_amps[ti] = a[::-1].copy()
        elif op == "rotate":
            track_amps[ti] = np.roll(a, int(rng.integers(1, max(2, nF))))
        elif op == "hold":
            track_amps[ti] = np.full(nF, float(a.mean()))
        elif op == "stutter":
            block = int(rng.integers(2, 8))
            b = a.copy()
            for s in range(0, nF, block):
                b[s: s + block] = a[s]
            track_amps[ti] = b
        elif op == "redecay":
            track_amps[ti] = np.linspace(float(a.max()), 0.0, nF) ** \
                float(rng.uniform(0.6, 2.5))
    idxs = [ti for ti, op in enumerate(plan) if op == "swap"]
    for i in range(0, len(idxs) - 1, 2):
        i1, i2 = idxs[i], idxs[i + 1]
        track_amps[i1], track_amps[i2] = track_amps[i2], track_amps[i1]
    new_tracks = [(t["freqs"], track_amps[ti])
                  for ti, t in enumerate(tracks)]
    # rebuild
    def one(y):
        Z = _stft(y, sr)
        mag, ph = np.abs(Z), np.angle(Z).astype(np.float32)
        nFy = mag.shape[1]
        # original partial skeleton from the mono reference
        orig = _peaks_to_frames_mag(mag.shape, [
            np.stack([[t["freqs"][min(j, nF - 1)], t["amps"][min(j, nF - 1)]]
                      for t in tracks], axis=0) if tracks else np.zeros((0, 2))
            for j in range(nFy)], sr)
        new = np.zeros_like(mag)
        for j in range(nFy):
            jm = min(j, nF - 1)
            pk = np.stack([[t[0][jm], t[1][jm]] for t in new_tracks
                           if not np.isnan(t[0][jm])], axis=0) \
                if any(not np.isnan(t[0][jm]) for t in new_tracks) else np.zeros((0, 2))
            frame = np.maximum(mag[:, j] - orig[:, j], 0.0) * residual_mix
            if len(pk):
                _add_gauss(frame, pk[:, 0], pk[:, 1], sr)
            new[:, j] = frame
        out = _istft(new * np.exp(1j * ph), sr)
        return out[: len(y)] if len(out) > len(y) else \
            np.pad(out, (0, len(y) - len(out)))

    return safe(st(one, x))


def env_splice(x, sr, rng, partner=None, graft=0.45, smooth=3, max_s=None):
    """Keeper A's spectral body wearing keeper B's amplitude envelope —
    A plays B's dynamic shape. Optionally B's attack transient is grafted
    onto the front (the classic stolen-attack move)."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if partner is None:
        partner = x[::-1].copy()
    partner = np.asarray(partner, np.float32)
    mB = mono(partner)
    magsB = np.abs(_stft(mB, sr))
    envB = magsB.mean(axis=0) if magsB.size else np.array([1.0])

    def one(y):
        Z = _stft(y, sr)
        mag, ph = np.abs(Z), np.angle(Z).astype(np.float32)
        envA = mag.mean(axis=0)
        # normalize both envelopes to peak 1 so the ratio truly imposes B's
        # shape instead of saturating the clip on loud material
        envA_n = envA / (envA.max() + EPS)
        envB_r = _resample_frames(envB, mag.shape[1])
        envB_n = envB_r / (envB_r.max() + EPS)
        if smooth > 1:
            k = np.ones(smooth) / smooth
            envA_n = np.convolve(envA_n, k, "same")
            envB_n = np.convolve(envB_n, k, "same")
        ratio = envB_n / (envA_n + 0.02)
        ratio = np.clip(ratio, 0.01, 30.0)
        out = _istft(mag * ratio[None, :] * np.exp(1j * ph), sr)
        y2 = out[: len(y)] if len(out) >= len(y) else np.pad(out,
               (0, len(y) - len(out)))
        return y2

    y = st(one, x)
    if rng.random() < graft:
        # stolen attack: partner's first transient (up to 120ms), xfaded in
        eB = np.abs(mB)
        w = max(2, int(0.005 * sr))
        e = np.convolve(eB, np.ones(w) / w, "same")
        atk = int(np.argmax(e > 0.85 * e.max())) + int(0.01 * sr) \
            if (e > 0.85 * e.max()).any() else int(0.03 * sr)
        atk = int(np.clip(atk, int(0.01 * sr), int(0.12 * sr)))
        graft_seg = partner[:atk].copy()
        xf = min(atk // 3, int(0.012 * sr))
        graft_seg[-xf:] *= np.linspace(1, 0, xf)[:, None]
        tail = y.copy()
        tail[:xf] *= np.linspace(0.4, 1.0, xf)[:, None]
        y = np.concatenate([graft_seg, tail], axis=0)
    return safe(y)


def formant_walk(x, sr, rng, partner=None, one_way=False, hold_end=0.3,
                 max_s=None):
    """A's fine partial structure is preserved while its smoothed spectral
    envelope (the 'mouth shape') glides into B's formant field — 0->B->0
    (possession) or a one-way drift that ends living in B's mouth."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if partner is None:
        partner = x[::-1].copy()
    partner = np.asarray(partner, np.float32)
    mB = mono(partner)
    magsB = np.abs(_stft(mB, sr))
    envB = np.stack([_smooth_env(magsB[:, j], sr) for j in range(magsB.shape[1])],
                    axis=1) if magsB.size else None

    def one(y):
        Z = _stft(y, sr)
        mag, ph = np.abs(Z), np.angle(Z).astype(np.float32)
        nF = mag.shape[1]
        envA = np.stack([_smooth_env(mag[:, j], sr) for j in range(nF)],
                        axis=1)
        envB_r = np.stack([_resample_frames(envB[b], nF)
                           for b in range(envB.shape[0])], axis=0) \
            if envB is not None else envA
        # morph weight over time
        tt = np.linspace(0, 1, nF)
        if one_way:
            w = np.clip(tt / max(0.4, 1.0 - hold_end), 0, 1) ** 0.8
        else:
            w = 0.5 - 0.5 * np.cos(2 * np.pi * tt)
        w = w[None, :]
        M = np.power(np.maximum(envA, EPS), 1 - w) * \
            np.power(np.maximum(envB_r, EPS), w)
        ratio = np.clip(M / (envA + EPS), 0.08, 12.0)
        out = _istft(mag * ratio * np.exp(1j * ph), sr)
        return out[: len(y)] if len(out) >= len(y) else \
            np.pad(out, (0, len(y) - len(out)))

    return safe(st(one, x))


SCALES = {
    "minor_pent": [0, 3, 5, 7, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "hirajoshi": [0, 2, 3, 7, 8],
    "detuned": None,          # special: snap f0 by +-6 cents only
}
ROOTS = {"A": 55.0, "C": 65.41, "D": 73.42, "E": 82.41,
         "F": 87.31, "G": 98.0}


def harmonic_grid(x, sr, rng, scale=None, root=None, shear=0.3,
                  octave_shadow=0.35, sub=True, max_s=None):
    """Re-tune the keeper's detected f0 onto a scale grid (root seeded),
    shear odd/even partials against each other, add an octave shadow and
    a tuned sub layer — the hit 'plays in key' with a supersaw-ish split
    personality."""
    from . import pitchfx
    from .core import rms_env
    x = np.asarray(x, np.float32)
    scale = scale or str(rng.choice(sorted(SCALES)))
    root = root or str(rng.choice(sorted(ROOTS)))
    f0 = dominant_f0(x, sr, default=0.0)
    y = x
    target = None
    if f0 > 0:
        root_hz = ROOTS[root]
        if SCALES[scale] is None:               # detuned: +-6 cents
            target = f0 * 2 ** (float(rng.choice([-1, 1]) * 6 / 1200.0))
        else:
            cands = []
            for s in SCALES[scale]:
                for octv in range(-2, 3):
                    cands.append(root_hz * 2 ** (s / 12.0) * 2 ** octv)
            target = min(cands, key=lambda c: abs(np.log2(c / f0)))
        semis = 12.0 * float(np.log2(target / f0))
        y = pitchfx.pitch_shift_pv(x, sr, rng, semis=semis)
    else:
        # unpitched: seeded params must still shape the sound, or every
        # variant comes out identical — randomize the spectral surgery
        shear = float(rng.uniform(0.2, 0.55))
        octave_shadow = float(rng.uniform(0.2, 0.5))
    m = mono(y)
    Zm = _stft(m, sr)
    magsM = np.abs(Zm)
    peaks = _all_peaks(magsM, 18, sr)

    def one(ch):
        Z = _stft(ch, sr)
        mag, ph = np.abs(Z), np.angle(Z).astype(np.float32)
        nF = mag.shape[1]
        out = mag.copy()
        for j in range(nF):
            pk = peaks[min(j, len(peaks) - 1)]
            if len(pk) == 0:
                continue
            res = mag[:, j].copy()
            _add_gauss(res, pk[:, 0], -pk[:, 1], sr)
            new_pk = []
            for rank, (f, a) in enumerate(pk):
                a2 = a * (1 + shear * (1 if rank % 2 == 0 else -1))
                new_pk.append((f, max(0.0, a2)))
                if rank < 6 and octave_shadow > 0:
                    new_pk.append((f * 2.0, a * octave_shadow *
                                   (1 - rank / 6)))
            frame = np.maximum(res, 0.0)
            _add_gauss(frame, [p[0] for p in new_pk],
                       [p[1] for p in new_pk], sr)
            out[:, j] = frame
        o = _istft(out * np.exp(1j * ph), sr)
        return o[: len(ch)] if len(o) >= len(ch) else \
            np.pad(o, (0, len(ch) - len(o)))

    y2 = st(one, y)
    if sub:
        # tuned sub layer at the KNOWN target/2 (no re-detection after the
        # shear — the detector is unreliable on doctored spectra)
        sub_f = (target / 2.0) if target else dominant_f0(y2, sr, default=55.0) / 2.0
        n = y2.shape[0]
        t = np.arange(n) / sr
        env = rms_env(mono(y2), sr, 0.03)
        env /= (env.max() + EPS)
        s = np.sin(2 * np.pi * sub_f * t).astype(np.float32)[:, None] * env[:, None]
        y2 = safe(y2 + s * np.float32(0.4))
    return safe(y2)