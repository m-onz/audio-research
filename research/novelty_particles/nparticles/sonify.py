"""Sonification of evolved particle effects — three coupled layers.

Layer 1: PARTICLES-AS-INSTRUMENTS (event grains)
    Every rendered frame, the fastest live particles trigger short FM grains:
    distance-from-centre -> pitch, speed -> velocity, colour -> FM timbre
    index, x-position -> stereo pan.  The soundtrack follows the animation.

Layer 2: THE GENOME PLAYS ITSELF (ANN-as-synth)
    The evolved ANN *is* a synthesizer: instead of particle positions, feed
    it a slow Lissajous traversal of its own input space and read the colour
    outputs as a 3-oscillator FM drone, (vx, vy) as stereo/pitch drift and
    `size` as brightness.  Timbre variety is free because the nodes use the
    paper's 8 activation functions: sin/cos -> tonal drones, gauss -> bell
    dings, step -> digital chatter, tan -> harsh grit.  One genotype, two
    phenotypes (image and sound).

Layer 3: BEHAVIOUR-STAT MACROS
    Each clip's musical role comes from its measured behaviour statistics:
    mean distance -> register, entropy -> event density, mirror symmetry ->
    pentatonic quantisation vs continuous glissando, colour dynamics -> FM
    index.  Chronological reels also raise the root with generation number,
    so the soundtrack audibly narrates complexification.

Also here: fingerprint-as-spectrogram sonification (for slideshow mode) and
a vectorised Schroeder reverb to glue the grains together.
"""

from __future__ import annotations

import numpy as np

SR = 44100
TWO_PI = 2.0 * np.pi

# --------------------------------------------------------------- wavetables
# Band-limited-ish additive wavetables: saw, square, triangle (16/15/15
# harmonics).  Grain carriers pick a table by dominant colour channel, so
# red-heavy particles grind (saw), green chatter (square), blue ring (tri).

_TAB_N = 4096


def _build_tables():
    t = np.arange(_TAB_N) / _TAB_N
    saw = sum(np.sin(TWO_PI * k * t) / k for k in range(1, 17))
    sqr = sum(np.sin(TWO_PI * k * t) / k for k in range(1, 32, 2))
    tri = sum(((-1) ** ((k - 1) // 2)) * np.sin(TWO_PI * k * t) / (k * k)
              for k in range(1, 16, 2))
    tabs = np.stack([saw, sqr, tri])
    tabs = tabs / np.abs(tabs).max(axis=1, keepdims=True)
    return tabs.astype(np.float32)


TABLES = _build_tables()  # (3, _TAB_N): 0=saw 1=square 2=triangle

def _wt_lookup(table_idx: int, phase: np.ndarray) -> np.ndarray:
    """phase in radians -> wavetable sample (cheap int indexing)."""
    idx = (phase * (_TAB_N / TWO_PI)).astype(np.int64) % _TAB_N
    return TABLES[table_idx][idx]

# stats vector order (must match sim.STATS)
S_MEAN_SPEED, S_STD_SPEED, S_MEAN_DIST, S_STD_DIST, S_RADIAL_DRIFT, \
    S_ANG_MEAN, S_ANG_STD, S_ALIVE_FRAC, S_SPREAD, S_ENTROPY, S_SYM_H, \
    S_SYM_V, S_TEMPORAL, S_COLOR_TEMP, S_COLOR_SPATIAL = range(15)

PENTA = (0, 3, 5, 7, 10)          # minor-pentatonic semitone set


def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def snap_pentatonic(freq: float, root: float) -> float:
    """Snap a frequency to the minor-pentatonic scale built on `root`."""
    s = 12.0 * np.log2(max(freq, 1e-6) / root)
    octv = np.floor(s / 12.0)
    rem = s - 12.0 * octv
    idx = int(np.argmin(np.abs(np.asarray(PENTA) - rem)))
    snapped = 12.0 * octv + PENTA[idx]
    return float(root * 2.0 ** (snapped / 12.0))


class Macro:
    """Musical parameters derived from a genome's behaviour statistics."""

    def __init__(self, stats: np.ndarray, register_shift: float = 0.0):
        ms = stats[S_MEAN_SPEED]
        self.root = 55.0 * 2.0 ** (_clip(stats[S_MEAN_DIST], 0.0, 1.2) * 0.9
                                    + register_shift)          # 55..~130 Hz
        self.snap = stats[S_SYM_H] > 0.45 or stats[S_SYM_V] > 0.45
        self.pitch_range = 1.5 + 2.5 * _clip(stats[S_SPREAD] / 0.8, 0, 1)
        self.events_per_frame = int(round(1.0 + 3.0 * _clip(stats[S_ENTROPY], 0, 1)))
        self.fm_index = 0.4 + 2.0 * _clip(stats[S_COLOR_TEMP] / 0.12, 0, 1.6)
        self.detune = _clip(stats[S_STD_SPEED] / 0.010, 0.1, 1.5)
        self.grain_gain = 0.10 + 0.22 * _clip(ms / 0.012, 0, 1.6)
        # Lissajous traversal rates for the ANN-as-synth layer
        rate = 0.03 + 0.10 * _clip(ms / 0.012, 0.1, 2.0)
        self.liss_a = rate
        self.liss_b = rate * (1.618 + 0.30 * np.sign(stats[S_RADIAL_DRIFT])
                              + 0.05 * np.sin(stats[S_ANG_MEAN] * 500.0))
        self.pan_spin = _clip(abs(stats[S_ANG_MEAN]) / 0.01, 0, 1)
        self.texture_gain = 0.16 + 0.10 * _clip(stats[S_STD_SPEED] / 0.01, 0, 1)


# ---------------------------------------------------------------------------
# layer 1: event grains
# ---------------------------------------------------------------------------

def event_layer(frames: list[dict], macro: Macro) -> np.ndarray:
    """frames: per-video-frame Sim.step() dicts -> stereo buffer (N, 2).
    Each grain: colour-selected wavetable carrier, nested two-stage FM,
    a sub-octave sine and a two-tap resonant body -- rich but short."""
    total = int(len(frames) / 30.0 * SR) + int(0.45 * SR)
    buf = np.zeros((total, 2), dtype=np.float32)
    if not frames:
        return buf
    for fi, st in enumerate(frames):
        na = st["n_alive"]
        if not na:
            continue
        t0 = int(fi / 30.0 * SR)
        spd = st["spd"]
        k = min(macro.events_per_frame, na)
        idx = np.argsort(spd)[::-1][:k]                 # the fastest particles
        px, py = st["px"][idx], st["py"][idx]
        d = np.sqrt(px * px + py * py) / 1.35
        col = st["col"][idx]
        size = st["size"][idx]
        smax = max(float(spd[idx].max()), 1e-9)
        for j in range(k):
            v = _clip(d[j], 0.0, 1.0)
            freq = macro.root * 4.0 * 2.0 ** (v * macro.pitch_range)
            if macro.snap:
                freq = snap_pentatonic(freq, macro.root * 2.0)
            amp = macro.grain_gain * (0.25 + 0.75 * float(spd[idx][j]) / smax)
            dur = 0.025 + 0.075 * float(size[j])
            L = int(dur * SR)
            if L < 8:
                continue
            t = np.arange(L, dtype=np.float32) / SR
            env = 0.5 - 0.5 * np.cos(TWO_PI * t / L)     # hann grain
            rj, gj, bj = (float(col[j, 0]), float(col[j, 1]), float(col[j, 2]))
            # nested FM: modulator on modulator -> dense sideband families
            ratio1 = 1.5 + 1.5 * gj
            ratio2 = 2.0 + bj
            I1 = macro.fm_index * (0.5 + rj)
            I2 = 0.5 * macro.fm_index * bj
            cph = TWO_PI * freq * t
            mod = np.sin(TWO_PI * freq * ratio1 * t
                         + I2 * np.sin(TWO_PI * freq * ratio2 * t))
            tab = int(np.argmax(col[j]))                 # saw / square / triangle
            wave = _wt_lookup(tab, cph + I1 * mod)
            wave += 0.45 * np.sin(cph * 0.5)             # sub octave
            g = (wave * env * amp).astype(np.float32)
            body = max(int(SR / max(freq, 20.0)), 2)     # 2-tap resonant body
            if body * 2 < L:
                g[body:] += 0.30 * g[:-body]
                g[2 * body:] += 0.15 * g[:-2 * body]
            end = min(t0 + L, total)
            if end <= t0:
                continue
            g = g[:end - t0]
            pan = _clip(0.5 + 0.5 * float(px[j]) / 1.35, 0, 1)
            theta = pan * np.pi / 2.0
            buf[t0:end, 0] += g * np.cos(theta)
            buf[t0:end, 1] += g * np.sin(theta)
    return buf


# ---------------------------------------------------------------------------
# layer 2: the genome plays itself
# ---------------------------------------------------------------------------

def texture_layer(genome, dur_s: float, macro: Macro, seed: int = 5) -> np.ndarray:
    """Evaluate the ANN over a Lissajous input trajectory and read its
    outputs as a 7-voice detuned supersaw drone (MONO -- the media pipeline
    turns this into stereo via tape phasing).  Voices sit at root/fifth/
    octave, each a 7-oscillator detuned saw stack; drive adds harmonics."""
    n = int(dur_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    ph = np.random.default_rng(seed).uniform(0, TWO_PI, 2)
    xa = 1.35 * np.sin(TWO_PI * macro.liss_a * t + ph[0])
    yb = 1.35 * np.sin(TWO_PI * macro.liss_b * t + ph[1])
    d = np.sqrt(xa * xa + yb * yb)
    X = np.stack([xa, yb, d, np.ones(n, dtype=np.float32)], axis=1)
    Y = genome.evaluate(X)                              # (n, 6)
    rgb = np.clip(Y[:, 2:5] * 0.5 + 0.5, 0, 1)
    amp = (0.5 + 0.5 * rgb) * np.array([0.5, 0.35, 0.25])   # root/fifth/octave
    ratios = np.array([1.0, 1.5, 2.0])
    drift = 0.5 + 0.20 * np.clip(Y[:, 0], -1, 1)         # vx -> slow pitch bend
    bright = 0.25 + 0.75 * np.clip(Y[:, 5] * 0.5 + 0.5, 0, 1)
    det_amt = 0.3 + 1.2 * _clip(macro.detune, 0.0, 1.5)  # cents scale
    detunes = np.array([-12, -7, -4, 0, 4, 7, 12]) * 0.01 * det_amt
    voice = np.zeros(n, dtype=np.float32)
    for k in range(3):
        f = macro.root * 2.0 * ratios[k] * drift
        vk = np.zeros(n, dtype=np.float32)
        for dt in detunes:
            phase = TWO_PI * np.cumsum(f, dtype=np.float32) / SR * (2.0 ** (dt / 12.0))
            vk += _wt_lookup(0, phase)                   # saw stack
        voice += vk * amp[:, k] / len(detunes)
    voice = np.tanh(1.7 * voice) / 1.7 * 1.6            # drive: harmonic richness
    # brightness: blend raw vs lowpassed (moving average)
    voice_lp = np.convolve(voice, np.ones(64, np.float32) / 64.0, mode="same")
    mix = bright * voice + (1.0 - bright) * voice_lp
    return (mix * macro.texture_gain).astype(np.float32)


def genome_mod_signals(genome, dur_s: float, macro: Macro, seed: int = 5) -> np.ndarray:
    """Second, faster performance of the same genome: a brisk Lissajous
    traversal of the ANN input space (plus a ripple octave for movement),
    returning raw outputs (n, 6) in [-1, 1].  media.py smooths these and
    maps them onto post-FX parameters -- the genome not only plays itself,
    it plays the mixing desk."""
    n = int(dur_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, TWO_PI, 4)
    fa, fb = macro.liss_a * 6.7, macro.liss_b * 4.3
    xa = (1.15 * np.sin(TWO_PI * fa * t + ph[0])
          + 0.30 * np.sin(TWO_PI * fa * 2.63 * t + ph[2]))
    yb = (1.15 * np.sin(TWO_PI * fb * t + ph[1])
          + 0.30 * np.sin(TWO_PI * fb * 1.77 * t + ph[3]))
    xa = np.clip(xa, -1.35, 1.35)
    yb = np.clip(yb, -1.35, 1.35)
    d = np.sqrt(xa * xa + yb * yb)
    X = np.stack([xa, yb, d, np.ones(n, dtype=np.float32)], axis=1)
    return np.clip(genome.evaluate(X), -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# slideshow: fingerprint-as-spectrogram
# ---------------------------------------------------------------------------

def spectrogram_layer(grids: np.ndarray, dur_s: float, macro: Macro,
                      seed: int = 9) -> np.ndarray:
    """Inverse-spectrogram sonification of the 2x32x32 occupancy fingerprints:
    rows -> log-spaced fundamentals with a small harmonic stack (1x/2.01x/
    2.99x, so each partial family is rich), columns -> time, R/G channel of
    the fingerprint = early/late phase of the run."""
    n = int(dur_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(seed)
    rows = grids.shape[1]
    freqs = 80.0 * (8000.0 / 80.0) ** (np.arange(rows) / max(rows - 1, 1))
    buf = np.zeros((n, 2), dtype=np.float32)
    for r in range(rows):
        spec = np.concatenate([grids[0, r], grids[1, r]])   # 64 columns over time
        tt = np.linspace(0, len(spec) - 1, n)
        A = np.interp(tt, np.arange(len(spec)), spec.astype(np.float64))
        A = np.convolve(A, np.ones(9) / 9.0, mode="same")    # smooth envelope
        pan = 0.5 + 0.42 * (r / max(rows - 1, 1) - 0.5) * (1 if r % 2 else -1)
        th = pan * np.pi / 2.0
        for mult, wg in ((1.0, 1.0), (2.01, 0.45), (2.99, 0.20)):
            det = 1.0 + rng.uniform(-0.002, 0.002) + 0.003 * (mult - 1.0)
            ph = rng.uniform(0, TWO_PI)
            w = (np.sin(TWO_PI * freqs[r] * mult * det * t + ph)
                 * (A * wg if mult == 1.0 else A ** 2 * wg * 2.0)).astype(np.float32)
            buf[:, 0] += w * np.cos(th)
            buf[:, 1] += w * np.sin(th)
    energy = max(np.abs(grids).sum() / grids.size, 1e-6)
    return buf * (0.35 / energy) * 2.0 / 1.65


# ---------------------------------------------------------------------------
# finishing: reverb, loudness, wav
# ---------------------------------------------------------------------------

def _comb(x: np.ndarray, d: int, g: float) -> np.ndarray:
    y = np.zeros_like(x)
    for off in range(0, len(x), d):
        end = min(off + d, len(x))
        if off >= d:
            y[off:end] = x[off:end] + g * y[off - d:end - d]
        else:
            y[off:end] = x[off:end]
    return y


def _allpass(x: np.ndarray, d: int, g: float) -> np.ndarray:
    y = np.zeros_like(x)
    for off in range(0, len(x), d):
        end = min(off + d, len(x))
        if off >= d:
            y[off:end] = -g * x[off:end] + x[off - d:end - d] + g * y[off - d:end - d]
        else:
            y[off:end] = -g * x[off:end]
    return y


def schroeder_reverb(x: np.ndarray, wet=0.30) -> np.ndarray:
    """Classic Schroeder reverb (4 combs + 2 allpass), fully vectorised
    blockwise; operates on a (N, ch) buffer."""
    out = np.zeros_like(x)
    ms = [(29.7, 0.742), (37.1, 0.733), (41.1, 0.715), (43.7, 0.697)]
    for c in range(x.shape[1]):
        wet_c = np.zeros_like(x[:, c])
        for delay, g in ms:
            wet_c += _comb(x[:, c], max(int(delay * SR / 1000.0), 1), g)
        wet_c = _allpass(wet_c, int(5.0 * SR / 1000.0), 0.7)
        wet_c = _allpass(wet_c, int(1.7 * SR / 1000.0), 0.7)
        out[:, c] = x[:, c] + wet * wet_c / 2.0
    return out


def rms_balance(buf: np.ndarray, target_rms=0.10) -> np.ndarray:
    """Loudness matching: scale to a target RMS, then a gentle tanh soft-knee
    folds peaks down so dense and sparse material end up comparable."""
    r = float(np.sqrt(np.mean(buf ** 2)) + 1e-9)
    x = buf * (target_rms / r)
    return (np.tanh(x * 1.2) / 1.2).astype(buf.dtype)


def to_wav(path: str, buf: np.ndarray, sr=SR):
    import wave
    peak = float(np.abs(buf).max())
    if peak > 0.95:
        buf = buf * (0.95 / peak)          # only ever attenuate
    x = np.clip(buf, -0.98, 0.98)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(buf.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())