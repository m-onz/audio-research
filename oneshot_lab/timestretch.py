"""Time-stretch family: paulstretch (canonical Paul Nasca algorithm, public
domain, with blur / glide / phase-coherence extensions), phase vocoder,
granular stretch, wavetable stretch (a sound scans through ITSELF as
single-cycle wavetables), oscillating vari-speed, tape stop, freeze-lock."""
import numpy as np

from .core import (EPS, clamp_dur, dominant_f0, interp_hermite, mono,
                   resample, resample_curve, st)


def _power_window(n):
    """Paulstretch OLA window: pow(1 - x^2, 1.25), sums to constant at half-hop."""
    return (1.0 - np.linspace(-1.0, 1.0, n) ** 2) ** 1.25


def _smooth_size(n):
    """Round up to a 5-smooth size (2/3/5 factors) for fast FFTs — as paulstretch."""
    orig = max(16, n)
    while True:
        m = orig
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m < 2:
            return orig
        orig += 1


def _blur_mags(mags, blur, rng):
    """Blur magnitudes across frequency: random-width smoothing + jitter."""
    out = np.empty_like(mags)
    w = max(1, int(blur * rng.uniform(0.5, 3.0) * mags.shape[0] / 64))
    w = min(w, max(1, (mags.shape[0] - 1) // 2))   # kernel must fit 'same'
    k = np.exp(-0.5 * (np.arange(-w, w + 1) / max(1.0, w / 2.5)) ** 2)
    k /= k.sum()
    for c in range(mags.shape[1]):
        sm = np.convolve(mags[:, c], k, "same")
        jit = 1.0 + blur * rng.uniform(-0.35, 0.35, len(sm))
        out[:, c] = np.maximum(sm * jit, 0.0)
    return out


def paulstretch(x, sr, rng, stretch=8.0, window_s=0.15, blur=0.0,
                glide=0.6, phase_coherence=0.0, max_s=8.0):
    """Paul's Extreme Sound Stretch.

    Large windows, input creeping at half-window / stretch, magnitudes kept,
    phases randomized each output frame — extreme stretches become ambient
    smear instead of grain-mush.

    Extensions over the canonical algorithm:
      blur             random-width spectral smoothing of the magnitudes
      glide            crossfade magnitudes between adjacent input frames
                       while the stretch dwells on them (smoother evolution,
                       adapted from paulstretch 'newmethod')
      phase_coherence  0 = fully random phases (canonical); higher keeps an
                       evolving phase state (more tonal, stable, dronier)
    """
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n_ch = x.shape[1]
    W = _smooth_size(int(window_s * sr))
    half = W // 2
    if len(x) < 2 * W:                       # tiny input: loop it up
        reps = int(np.ceil(2 * W / max(1, len(x))))
        x = np.tile(x, (reps, 1))[: 2 * W]
    win = _power_window(W).astype(np.float32)
    max_out = int(max_s * sr)
    old = np.zeros((W, n_ch), np.float32)
    ph_state = np.exp(1j * rng.uniform(0, 2 * np.pi,
                                       (W // 2 + 1, n_ch))).astype(np.complex64)
    chunks = []
    have = 0
    kf = 0.0                                  # input position in half-window units

    def frame_mags(k):
        p = int(k) * half
        buf = x[p: p + W]
        if len(buf) < W:
            buf = np.pad(buf, ((0, W - len(buf)), (0, 0)))
        return np.abs(np.fft.rfft(buf * win[:, None], axis=0)).astype(np.float32)

    while have < max_out:
        k0 = int(kf)
        frac = kf - k0
        mags = frame_mags(k0)
        if glide > 0.0:
            nxt = frame_mags(k0 + 1)
            mags = (1.0 - glide) * mags + glide * ((1 - frac) * mags + frac * nxt)
        if blur > 0.0:
            mags = _blur_mags(mags, blur, rng)
        # phases: random per frame (canonical), blended with evolving state
        rand_ph = np.exp(1j * rng.uniform(0, 2 * np.pi,
                                          mags.shape)).astype(np.complex64)
        if phase_coherence > 0.0:
            ph_state = ph_state * phase_coherence + rand_ph * (1.0 - phase_coherence)
            ph_state /= (np.abs(ph_state) + EPS)
        else:
            ph_state = rand_ph
        buf2 = np.fft.irfft(mags * ph_state, n=W, axis=0) * win[:, None]
        chunk = (buf2[:half] + old[half:]).astype(np.float32)
        old = buf2.astype(np.float32)
        chunks.append(chunk)
        have += half
        kf += 1.0 / max(stretch, 0.05)
        if k0 * half >= len(x) - W:
            break
    out = np.concatenate(chunks, axis=0)
    return clamp_dur(out, sr, max_s, fade=0.15)


def pv_stretch(x, sr, rng, factor=4.0, n_fft=2048, max_s=8.0):
    """Phase-vocoder stretch: duration × factor, pitch holds, harmonics bloom.

    From the harmonic lab; frame-domain resampling with phase accumulation."""
    x = np.asarray(x, np.float32)
    hop = n_fft // 4
    win = np.hanning(n_fft)

    def one(y):
        if len(y) < n_fft * 2 or factor <= 1.05:
            return y
        pos = np.arange(0, len(y) - n_fft, hop)
        if len(pos) < 4:
            return y
        F = np.stack([y[p: p + n_fft] * win for p in pos])
        S = np.fft.rfft(F, axis=1)
        mags = np.abs(S).astype(np.float32)
        ph = np.angle(S).astype(np.float32)
        exp_adv = 2 * np.pi * hop * np.fft.rfftfreq(n_fft)  # rad/hop per bin
        dph = np.diff(ph, axis=0)
        dph = dph - np.round((dph - exp_adv[None, :]) / (2 * np.pi)) * 2 * np.pi
        nf = mags.shape[0]
        n_out = min(int(nf * factor), int(max_s * sr) // hop + 2)
        out = np.zeros((n_out - 1) * hop + n_fft, np.float32)
        syn_ph = ph[0].copy()
        for i in range(n_out):
            j = min(i / factor, len(dph) - 2)   # keep j0, j0+1 in range for dph
            j0 = int(j)
            fr = j - j0
            mag = (1.0 - fr) * mags[j0] + fr * mags[j0 + 1]
            adv = (1.0 - fr) * dph[j0] + fr * dph[j0 + 1]
            syn_ph = (syn_ph + adv + np.pi) % (2 * np.pi) - np.pi
            p = i * hop
            out[p: p + n_fft] += np.fft.irfft(mag * np.exp(1j * syn_ph), n_fft) * win
        return out / 1.5

    return st(one, x)


def granular_stretch(x, sr, rng, factor=3.0, grain_ms=60.0, jitter=0.3,
                     reverse_p=0.1, pitch_p=0.1, max_s=8.0):
    """Granular time stretch: grains read at 1x (pitch holds) but spaced
    factor× further apart; optional jitter / reversed / pitched grains."""
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        return st(lambda c: granular_stretch(c, sr, rng, factor,
                  grain_ms, jitter, reverse_p, pitch_p, max_s), x)
    g = max(64, int(grain_ms / 1000.0 * sr))
    hop_out = g // 2
    hop_in = max(1, int(hop_out / max(factor, 0.05)))
    w = np.hanning(g).astype(np.float32)
    n = len(x)
    chunks = []
    out_pos = 0
    in_pos = 0
    max_out = int(max_s * sr)
    while out_pos < max_out:
        j = int(rng.uniform(-jitter, jitter) * g) if jitter else 0
        s = max(0, in_pos + j)
        if s + g > n:
            break
        gr = x[s: s + g].copy()
        if reverse_p and rng.random() < reverse_p:
            gr = gr[::-1]
        if pitch_p and rng.random() < pitch_p:
            gr = resample(gr, 2.0 ** rng.uniform(-0.5, 0.5))
            if len(gr) < g:
                gr = np.pad(gr, (0, g - len(gr)))
            gr = gr[:g]
        chunks.append((out_pos, gr * w))
        out_pos += hop_out
        in_pos += hop_in
    if not chunks:
        return x
    out = np.zeros(out_pos + g, np.float32)
    for p, gr in chunks:
        out[p: p + g] += gr
    return clamp_dur(out, sr, max_s, fade=0.05)


def wavetable_stretch(x, sr, rng, f0=None, scan_per_s=10.0,
                      scan_mode="pingpong", vib_cents=8.0, max_s=10.0):
    """TIME-STRETCH WAVETABLE: the sound becomes a stack of single-cycle
    wavetables; an oscillator at f0 reads them while a slow scan head walks
    through the original — time is re-cast as wavetable position.

    From the databend wavetable_voice. scan_mode: forward | pingpong | random.
    """
    m = mono(x)
    f0 = f0 or dominant_f0(x, sr)
    table = max(64, int(sr / f0))
    if len(m) < table * 4:
        reps = int(np.ceil(4 * table / len(m)))
        m = np.tile(m, reps)
    n_out = int(max_s * sr)
    t = np.arange(n_out) / sr
    # vibrato: slow LFO on f0
    f0_t = f0 * (1.0 + (vib_cents / 100.0) *
                 np.sin(2 * np.pi * rng.uniform(2.0, 7.0) * t))
    phase = np.cumsum(f0_t) / sr
    pos_in_table = (phase % 1.0) * table
    # scan head through the source
    n_tables = (len(m) - table - 2) / table
    scan_pos = t * scan_per_s
    if scan_mode == "pingpong":
        cyc = scan_pos % max(2.0, 2 * n_tables)
        scan_pos = np.where(cyc < n_tables, cyc, 2 * n_tables - cyc)
    elif scan_mode == "random":
        pts = max(8, int(100 * max_s))
        walk = np.cumsum(rng.uniform(-1, 1, pts))
        walk -= walk.min()
        walk /= (walk.max() + EPS)
        scan_pos = np.interp(np.linspace(0, 1, n_out),
                             np.linspace(0, 1, pts), walk) * n_tables
    scan_pos = np.clip(scan_pos, 0.0, n_tables)
    p = scan_pos * table + pos_in_table
    y = interp_hermite(m, p).astype(np.float32)
    env = (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t +
                             rng.uniform(0, 6.28))).astype(np.float32)
    y *= env
    return np.stack([y, np.roll(y, int(0.0025 * sr))], axis=1)


def oscillating_timestretch(x, sr, rng, depth=0.25, rate_hz=0.6,
                            drift=0.35, wobble_p=0.5, dur_mult=1.0, max_s=10.0):
    """OSCILLATING TIME STRETCH: a phasor-LFO rate curve breathes over the
    whole sound — time AND pitch bend together (tape style). The LFO's own
    frequency drifts, and an optional wobble storm punctuates it.
    From the storm vari-speed.

    dur_mult > 1 loops the input first so the wobble has room to travel."""
    x = np.asarray(x, np.float32)
    if dur_mult > 1.01:
        reps = int(np.ceil(dur_mult))
        x = np.tile(x, (reps, 1))
    n = x.shape[0]
    ts = np.arange(n, dtype=np.float64) / sr
    rate = np.ones(n, np.float64)
    f0 = rate_hz * (1.0 + drift * np.sin(2 * np.pi * 0.013 * ts +
                                         rng.uniform(0, 6.28)))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    rate += depth * np.sin(phase)
    if rng.random() < wobble_p:  # occasional mid-rate wobble storm
        t0 = rng.uniform(0.05, 0.6)
        dur = rng.uniform(0.1, 0.3)
        b = np.clip(1.0 - np.abs(ts - (t0 + dur / 2)) / (dur / 2), 0, 1)
        rate *= 1.0 + (0.3 + 0.5 * depth) * b * \
            np.sin(2 * np.pi * rng.uniform(2.0, 7.0) * ts)
    rate = np.maximum(rate, 0.05)
    return resample_curve(x, rate)


def tape_stop(x, sr, rng, floor=0.02, shape=2.5, hold=0.3):
    """Tape stop / vinyl brake: rate sags 1 -> floor, time+pitch dive,
    then the dead-slow tail rides out. From the storm 'brake' warp."""
    x = np.asarray(x, np.float32)
    n = x.shape[0]
    hold_n = int(hold * n)
    tt = np.linspace(0, 1, max(2, n - hold_n)) ** shape
    rate = np.concatenate([
        np.ones(hold_n),
        (1.0 - floor) * (1.0 - tt) + floor,
    ])
    return resample_curve(x, rate)


def freeze_loop_stretch(x, sr, rng, factor=6.0, max_s=8.0):
    """Stretch by granular freeze-locking: the loudest window of the sound
    is looped with crossfaded, micro-jittered tails — a hypnotic lock-groove
    that slowly releases back into the original ending. Storm 'loop-lock'
    flavor without the beat grid."""
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        return st(lambda c: freeze_loop_stretch(c, sr, rng, factor, max_s), x)
    n = len(x)
    w = max(256, min(n, int(rng.uniform(0.06, 0.25) * sr)))
    e = np.abs(x)
    win = max(0, int(np.argmax(np.convolve(e, np.ones(w) / w, "same")) - w / 2))
    core = x[win: win + w].copy() * np.hanning(w).astype(np.float32)
    out_n = int(min(factor * n, max_s * sr))
    out = np.zeros(out_n, np.float32)
    pos = 0
    while pos < out_n:
        j = int(rng.uniform(-0.04, 0.04) * sr)      # micro-jitter per loop
        seg = np.roll(core, j)
        if rng.random() < 0.15:
            seg = resample(seg, 2.0 ** rng.uniform(-0.2, 0.2))
        e2 = min(out_n - pos, len(seg))
        out[pos: pos + e2] += seg[:e2]
        pos += w
    # splice the real tail back in over the last 20%
    tail_start = int(out_n * 0.8)
    tail = x[n - min(n, out_n - tail_start):]
    if len(tail):
        g = np.linspace(0, 1, len(tail)) ** 2
        out[tail_start: tail_start + len(tail)] *= (1 - g)
        out[tail_start: tail_start + len(tail)] += tail * g
    return out