"""organism: the wavetable rig — a genome read through four axes.

The frames ARE the medium (born, not designed); the paths are how the
medium is played. Four evolvable axes, the grammar of three synths
plus the modern fix:

    table content   frames of harmonic partials (PPG's ROM, evolved)
    morph path      smooth scan (PPG) | free curve | stepped
                    wavesequencing (Wavestation, duty = crossfade)
    vector mix      1-4 detuned voices on the corners of a 2-D
                    joystick path (Prophet VS)
    pitch path      f0 over time — fixed | glide | vib | walk | step
                    (the axis prism froze; here it is a gene)

Readout: integer-mip band-limiting. Every frame is pre-materialised
at every partial count (a cumsum in the partial domain — all variants
for the price of one), and the read blends adjacent counts against
the instantaneous pitch, with a hard Nyquist gate on the blend. A
high note reads a short table; a sweep reads them in order; nothing
aliases but the transition partial, and that is gated.

House laws honored: phases never align at zero (eden's lesson — every
partial gets a seeded random phase); np.interp never silences what it
should clamp (the band mapping clamps explicitly); the only designed
stage is the safety limiter. Deterministic from seed: same genome,
same bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import subprocess
from pathlib import Path

import numpy as np

__all__ = ["WtGenome", "render", "random_genome", "mutate", "tonality",
           "viable", "write_wav", "ARCHETYPES", "SR", "SECONDS",
           "to_dict", "from_dict"]

SR = 44100
RENDER_SR = 11025                  # synthesis rate, exact FFT upsample x4
UP = SR // RENDER_SR
SECONDS = 2.0
TABLE_N = 2048                     # samples per single-cycle frame
MIN_FRAMES, MAX_FRAMES = 4, 64
MIN_PARTIALS, MAX_PARTIALS = 2, 32
MIN_VOICES, MAX_VOICES = 1, 4
F0_LO, F0_HI = 30.0, 2000.0
EDGE_FRAC = 0.45                   # conservative band edge
TARGET_RMS = 0.18
FADE_S = 0.04
PEAK = 0.95

MORPH_MODES = ("lfo", "curve", "step")
PITCH_MODES = ("fixed", "glide", "vib", "walk", "step")
SHAPES = ("sin", "tri", "saw")

ARCHETYPES = ("tone", "organ", "metal", "formant", "scan", "sequence",
              "ensemble", "chord", "glide", "wash")


# ------------------------------------------------------------------ genome

@dataclass
class WtGenome:
    """Every field is plain json-able data; genes have names."""
    f0: float = 110.0
    pitch: dict = field(default_factory=lambda: {
        "type": "fixed", "rate": 0.5, "depth": 200.0, "end": 1.0})
    frames: list = field(default_factory=list)
    #   each frame: {"partials": [amps by partial 1..P], "inharm": float}
    morph: dict = field(default_factory=lambda: {
        "type": "lfo", "rate": 0.2, "depth": 0.8, "shape": "saw",
        "duty": 0.0, "bps": [0.1, 0.5, 0.9]})
    voices: list = field(default_factory=list)
    #   each voice: {detune(cents), level, pan, morph_offset, attack, decay_tau}
    vector: dict = field(default_factory=lambda: {
        "type": "fixed", "x0": 0.0, "y0": 0.0, "rate_x": 0.1,
        "rate_y": 0.13, "depth": 0.0, "shape_x": "sin", "shape_y": "tri"})
    seed: int = 0

    # ------------------------------------------------------------ labels
    def archetype(self) -> str:
        """Ground truth off the genome — fixed cascade, no audio."""
        K = len(self.frames)
        if K == 0:
            return "tone"
        acts = [sum(1 for a in fr["partials"] if a > 0.03)
                for fr in self.frames]
        p_mean = float(np.mean(acts))
        inh = float(np.mean([fr["inharm"] for fr in self.frames]))
        mp = self.morph
        dets = [abs(v["detune"]) for v in self.voices]
        det_mean = float(np.mean(dets)) if dets else 0.0

        if p_mean <= 2.2 and mp["depth"] < 0.15:
            return "tone"
        if inh > 0.4:
            return "metal"
        if mp["type"] == "step":
            return "sequence"
        if mp["depth"] > 0.6 and mp["rate"] > 2.0:
            return "wash"
        if mp["depth"] > 0.4:
            return "scan"
        # formant: energy concentrated in a high, narrow partial band
        n_form = 0
        for fr in self.frames:
            amps = np.asarray(fr["partials"])
            if amps.size < 4:
                continue
            idx = np.nonzero(amps > 0.05 * amps.max() + 1e-6)[0] + 1
            if idx.size >= 2 and idx.min() >= 3 and idx.max() - idx.min() <= 5:
                n_form += 1
        if n_form > K / 2:
            return "formant"
        # chord: two voices a near-equal-tempered interval apart
        if len(dets) >= 2:
            ds = sorted(dets)
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    d = abs(ds[j] - ds[i])
                    if d > 60.0 and abs(d - 100.0 * round(d / 100.0)) < 40.0:
                        return "chord"
        if len(self.voices) >= 3 and det_mean > 8.0:
            return "ensemble"
        if self.pitch["type"] in ("glide", "walk", "step"):
            return "glide"
        return "organ"

    def anatomy(self) -> dict:
        acts = [len(fr["partials"]) for fr in self.frames]
        return {
            "frames": len(self.frames),
            "partials": f"{min(acts)}-{max(acts)}" if acts else "0",
            "inharm_mean": round(float(np.mean(
                [fr["inharm"] for fr in self.frames])) , 3) if self.frames else 0.0,
            "f0": round(self.f0, 1),
            "pitch": self.pitch["type"],
            "morph": (f'{self.morph["type"]} rate={self.morph["rate"]:.2f} '
                      f'depth={self.morph["depth"]:.2f}'),
            "voices": len(self.voices),
            "detunes": [round(v["detune"], 1) for v in self.voices],
            "vector": self.vector["type"],
            "archetype": self.archetype(),
        }


def tonality(g: WtGenome) -> float:
    """The greedy objective: structural purity (partial count, inharm,
    detune, movement). Converges to two-partial sines at fixed pitch —
    the trap, honest and audible, and a true fixed point: counts
    cannot fall below the floor, so the optimum is stable."""
    if not g.frames:
        return 0.0
    counts = [len(fr["partials"]) for fr in g.frames]
    inh = float(np.mean([fr["inharm"] for fr in g.frames]))
    det = float(np.mean([abs(v["detune"]) for v in g.voices])) \
        if g.voices else 0.0
    return -(float(np.mean(counts)) + 8.0 * inh + det / 100.0
             + 2.0 * g.morph["depth"] + 0.5 * (len(g.voices) - 1))


# ------------------------------------------------------------------ helpers

def _rng(seed: int, tag: int) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0x7FFFFFFF, int(tag)])


def _wave(name: str, x: np.ndarray, phi: float) -> np.ndarray:
    """Unit shapes in [-1, 1]; x is cycles, phi in [0, 1)."""
    u = np.mod(x + phi, 1.0)
    if name == "sin":
        return np.sin(2.0 * np.pi * u)
    if name == "tri":
        return 4.0 * np.abs(u - 0.5) - 1.0
    return 2.0 * u - 1.0                              # saw: scan + wrap


def _jit(rng, v, lo, hi, rate=0.3):
    """Multiplicative log jitter, clipped — dx's kernel."""
    return float(np.clip(v * 10 ** rng.normal(0, rate), lo, hi))


def _add(rng, v, lo, hi, span=0.1):
    """Additive jitter for signed/bounded genes (pan, detune, x0...)."""
    return float(np.clip(v + rng.normal(0, span), lo, hi))


def _clip_param(g, key, sub, lo, hi):
    g[sub][key] = float(np.clip(g[sub][key], lo, hi))


# ------------------------------------------------------------------ tables

def _partial_freqs(p: int, inharm: float) -> np.ndarray:
    """Partial ratios: p * (p+1)^inharm — 0 gives pure harmonics,
    >0 stretches progressively (the metal axis)."""
    k = np.arange(1, p + 1, dtype=np.float64)
    return k * (k + 1.0) ** inharm


def _build_variants(g: WtGenome, levels: list) -> np.ndarray | None:
    """All frames at every requested partial count.

    Harmonic frames: masked inverse FFT per level (exact). Inharmonic
    frames: direct partial sums via one cumsum — every variant for the
    price of one render. Returns (L, K, N) float32, peak-normalised to
    0.9 against the FULL tables (level Pmax is always built so the
    normalisation is render-invariant), or None for a silent genome.
    """
    K = len(g.frames)
    pmax = max(max((len(fr["partials"]) for fr in g.frames), default=0), 1)
    vmax = max(levels)
    n = TABLE_N
    V = np.zeros((len(levels), K, n), np.float32)
    ph_all = _rng(g.seed, 5000).uniform(0.0, 1.0, (K, pmax))

    for k, fr in enumerate(g.frames):
        amps = np.asarray(fr["partials"], np.float64)
        P = amps.size
        if P == 0:
            continue
        ph = ph_all[k, :P]
        if fr["inharm"] < 1e-9:                       # harmonic: irfft
            spec = np.zeros(n // 2 + 1, np.complex128)
            spec[1:P + 1] = amps * np.exp(2j * np.pi * ph)
            for li, v in enumerate(levels):
                m = spec.copy()
                m[min(v, P) + 1:] = 0                  # partials 1..v
                V[li, k] = np.fft.irfft(m, n)
        else:                                          # metal: cumsum
            need = min(vmax, P)
            r = _partial_freqs(need, float(fr["inharm"]))
            S = np.sin(2.0 * np.pi * np.outer(r, np.arange(n)) / n
                       + 2.0 * np.pi * ph[:need, None]) * amps[:need, None]
            C = np.cumsum(S, axis=0)
            for li, v in enumerate(levels):
                V[li, k] = C[min(v, P) - 1]

    full_li = levels.index(pmax)
    peak = float(np.abs(V[full_li]).max())
    if peak < 1e-12:
        return None
    V *= 0.9 / peak
    return V


# ------------------------------------------------------------------ paths

def _pitch_curve(g: WtGenome, t: np.ndarray, seconds: float,
                 f_hi_cap: float) -> np.ndarray:
    """f(t) in Hz, clamped so the fundamental never crosses the edge."""
    pp = g.pitch
    if pp["type"] == "fixed" or t.size == 0:
        cents = np.zeros_like(t)
    elif pp["type"] == "glide":
        cents = 1200.0 * np.log2(max(pp["end"], 1e-6)) * (t / seconds)
    elif pp["type"] == "vib":
        phi = float(_rng(g.seed, 700).uniform(0.0, 1.0))
        cents = pp["depth"] * _wave("sin", pp["rate"] * t, phi)
    elif pp["type"] == "walk":
        w = _rng(g.seed, 901).normal(0.0, 1.0, t.size)
        w = np.cumsum(w)
        k = max(3, t.size // 64)
        w = np.convolve(w, np.ones(k) / k, mode="same")
        w -= w.mean()
        sd = float(w.std())
        cents = pp["depth"] * (w / sd) if sd > 1e-9 else np.zeros_like(t)
    else:                                             # step
        vals = _rng(g.seed, 902).uniform(-1.0, 1.0,
                                        int(np.ceil(seconds * pp["rate"])) + 1)
        seg = np.clip((t * pp["rate"]).astype(int), 0, vals.size - 1)
        cents = vals[seg] * pp["depth"]
    f = g.f0 * 2.0 ** (cents / 1200.0)
    return np.clip(f, 20.0, f_hi_cap)


def _morph_base(g: WtGenome, t: np.ndarray, seconds: float, K: int):
    """The morph scan before per-voice offsets.

    Returns ("cont", m) with m in [0, K-1] for lfo/curve, or
    ("step", cur, prev, xf) — frame indices held per segment with a
    duty-fraction crossfade from the previous frame (the Wavestation's
    crossfade parameter; duty 0 = hard wavesequencing).
    """
    mp = g.morph
    span = float(K - 1)
    if mp["type"] == "step":
        rate = max(mp["rate"], 1e-6)
        seg = t * rate
        cur = np.floor(seg).astype(np.int64) % K
        prev = (cur - 1) % K
        if mp["duty"] <= 1e-6:
            xf = np.ones_like(t)
        else:
            xf = np.clip((seg - np.floor(seg)) / mp["duty"], 0.0, 1.0)
        return ("step", cur, prev, xf)
    if mp["type"] == "curve":
        bps = np.asarray(mp["bps"], np.float64)
        times = np.linspace(0.0, seconds, bps.size)
        v = np.interp(t, times, bps)
        m = span * (0.5 + mp["depth"] * (v - 0.5))
    else:                                             # lfo
        phi = float(_rng(g.seed, 600).uniform(0.0, 1.0))
        s = _wave(mp["shape"], mp["rate"] * t, phi)
        m = span * (0.5 + 0.5 * mp["depth"] * s)
    return ("cont", np.clip(m, 0.0, span))


def _voice_morph(base, offset: float, K: int, t: np.ndarray):
    """Per-voice (j0, j1, a): the frame pair and blend at each sample."""
    span = float(K - 1)
    if base[0] == "step":
        _, cur, prev, xf = base
        off = int(round(offset * K)) % K
        cur = (cur + off) % K
        prev = (cur - 1) % K
        return cur, prev, xf
    m = np.clip(base[1] + offset * span, 0.0, span)
    j0 = np.floor(m).astype(np.int64)
    j1 = np.minimum(j0 + 1, K - 1)
    return j0, j1, (m - j0)


def _vector_xy(g: WtGenome, t: np.ndarray):
    vp = g.vector
    if vp["type"] == "fixed" or vp["depth"] <= 1e-6 or t.size == 0:
        return (np.full(t.size, vp["x0"]), np.full(t.size, vp["y0"]))
    ph = _rng(g.seed, 610).uniform(0.0, 1.0, 2)
    x = vp["x0"] + vp["depth"] * _wave(vp["shape_x"], vp["rate_x"] * t, ph[0])
    y = vp["y0"] + vp["depth"] * _wave(vp["shape_y"], vp["rate_y"] * t, ph[1])
    return np.clip(x, -1.0, 1.0), np.clip(y, -1.0, 1.0)


def _vector_weights(g: WtGenome, x: np.ndarray, y: np.ndarray):
    """Bilinear square-corner weights (the VS joystick), equal-power.
    Fewer voices than corners: renormalise the used corners."""
    corners = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
    w = [((1.0 + corners[i][0] * x) / 2.0) * ((1.0 + corners[i][1] * y) / 2.0)
         for i in range(len(g.voices))]
    W = np.stack(w) + 1e-9
    W = W / W.sum(axis=0)
    return np.sqrt(W)                                 # (n_voices, n)


# ------------------------------------------------------------------ render

def _gather(V, il, j0, j1, a, i0, i1, fr):
    """Bilinear read: (variant level, frame, position) — one corner set."""
    T00 = V[il, j0, i0]
    T01 = V[il, j0, i1]
    T10 = V[il, j1, i0]
    T11 = V[il, j1, i1]
    return ((1.0 - a) * ((1.0 - fr) * T00 + fr * T01)
            + a * ((1.0 - fr) * T10 + fr * T11))


def render(g: WtGenome, seconds: float = SECONDS, sr: int = SR,
           hi: bool = False) -> np.ndarray:
    """Genome -> stereo (n, 2) float32, loudness-safe, deterministic."""
    rs = int(sr) if hi else RENDER_SR
    up = max(1, sr // rs)
    edge = EDGE_FRAC * rs
    nyq = rs / 2.0
    f_cap = edge / 1.05
    n = int(seconds * rs)
    if n <= 0 or not g.frames or not g.voices:
        return np.zeros((max(0, int(seconds * sr)), 2), np.float32)
    t = np.arange(n, dtype=np.float64) / rs

    K = len(g.frames)
    pmax = max(max(len(fr["partials"]) for fr in g.frames), 1)

    f = _pitch_curve(g, t, seconds, f_cap)
    mults = [2.0 ** (v["detune"] / 1200.0) for v in g.voices]
    f_top = float(f.max()) * max(mults)
    f_bot = float(f.min()) * min(mults)

    # ---- the integer mip: every partial count the episode can ask for
    a_hi = edge / max(f_bot, 1e-6)
    a_lo = edge / max(f_top, 1e-6)
    v_min = int(np.floor(np.clip(a_lo, 1.0, pmax)))
    v_max = int(np.floor(np.clip(a_hi, 1.0, pmax)))
    levels = sorted({int(np.clip(v, 1, pmax))
                     for v in range(v_min, v_max + 2)} | {pmax})
    V = _build_variants(g, levels)
    if V is None:
        return np.zeros((int(seconds * sr), 2), np.float32)
    Larr = np.asarray(levels, np.int64)

    base = _morph_base(g, t, seconds, K)
    xv, yv = _vector_xy(g, t)
    sqw = _vector_weights(g, xv, yv)
    vph = _rng(g.seed, 8000).uniform(0.0, 1.0, MAX_VOICES)

    L = np.zeros(n)
    R = np.zeros(n)
    for vi, vo in enumerate(g.voices):
        fv = np.clip(f * mults[vi], 20.0, f_cap)
        A = np.clip(edge / fv, 1.0, pmax)
        v_lo = np.floor(A).astype(np.int64)
        v_hi = np.minimum(v_lo + 1, pmax)
        b = np.clip(A - v_lo, 0.0, 1.0)
        b = np.where(v_hi * fv > nyq, 0.0, b)         # the Nyquist gate
        il0 = np.searchsorted(Larr, v_lo)
        il1 = np.searchsorted(Larr, v_hi)

        j0, j1, a = _voice_morph(base, vo["morph_offset"], K, t)
        ph = np.cumsum(fv) / rs + vph[vi]
        pos = (ph % 1.0) * TABLE_N
        i0 = np.floor(pos).astype(np.int64) % TABLE_N
        i1 = (i0 + 1) % TABLE_N
        fr = pos - np.floor(pos)

        x = _gather(V, il0, j0, j1, a, i0, i1, fr)
        if b.any():                                    # variant blend
            x = (1.0 - b) * x + b * _gather(V, il1, j0, j1, a, i0, i1, fr)

        e = np.minimum(t / max(vo["attack"], 1e-4), 1.0) \
            * np.exp(-t / max(vo["decay_tau"], 1e-3))
        gain = vo["level"] * e * sqw[vi]
        theta = (np.clip(vo["pan"], -0.9, 0.9) + 1.0) * (np.pi / 4.0)
        L += x * gain * np.cos(theta)
        R += x * gain * np.sin(theta)

    # ---- exact band-limited upsample (maze dx's trick)
    want = int(seconds * sr)
    if up > 1:
        N = n * up
        upL = np.zeros(N)
        upR = np.zeros(N)
        for ch, sig in ((0, L), (1, R)):
            X = np.fft.rfft(sig)
            XL = np.zeros(N // 2 + 1, np.complex128)
            XL[:len(X)] = X
            out = np.fft.irfft(XL, N) * up
            if ch == 0:
                upL = out
            else:
                upR = out
        L, R = upL, upR
    if len(L) < want:                                # exact length, always
        pad = np.zeros(want - len(L))
        L, R = np.concatenate([L, pad]), np.concatenate([R, pad])
    L, R = L[:want], R[:want]

    return _polish(L, R, sr)


def _polish(L: np.ndarray, R: np.ndarray, sr: int) -> np.ndarray:
    """The safety limiter — the only designed stage any render gets."""
    x = np.stack([L, R], axis=1).astype(np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=0)
    rms = float(np.sqrt((x * x).mean())) if x.size else 0.0
    if rms < 1e-9:
        return np.zeros_like(x, np.float32)
    x = x * (TARGET_RMS / rms)
    nf = int(FADE_S * sr)
    if nf > 0 and len(x) > 2 * nf:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, nf))
        x[:nf] *= ramp[:, None]
        x[-nf:] *= ramp[::-1][:, None]
    peak = float(np.abs(x).max())
    if peak > PEAK:
        x = x * (PEAK / peak)
    return x.clip(-0.99, 0.99).astype(np.float32)


# ------------------------------------------------------------------ viability

def viable(audio) -> bool:
    """The minimal criterion, identical for every arm: audible + alive."""
    x = np.asarray(audio, np.float64)
    if not np.isfinite(x).all():
        return False
    rms = float(np.sqrt((x ** 2).mean()))
    if not (0.02 <= rms <= 0.6):
        return False
    blocks = np.array_split(x, 4, axis=0)
    rms_b = [float(np.sqrt((b ** 2).mean())) for b in blocks]
    if max(rms_b) / (min(rms_b) + 1e-9) < 1.05:       # static drone gate
        return False
    return True


# ------------------------------------------------------------------ genetics

def _rand_frame(rng) -> dict:
    p = int(rng.integers(MIN_PARTIALS, 13))
    return {"partials": [float(10 ** rng.uniform(-2.6, 0.0))
                         for _ in range(p)],
            "inharm": float(rng.choice([0.0, 0.0, 0.0,
                                        rng.uniform(0.05, 0.9)]))}


def random_genome(rng: np.random.Generator, seed: int) -> WtGenome:
    K = int(rng.integers(MIN_FRAMES, 17))
    frames = []
    for _ in range(K):
        fr = _rand_frame(rng)
        frames.append(fr)
    morph_type = str(rng.choice(MORPH_MODES))
    morph = {"type": morph_type,
             "rate": float(10 ** rng.uniform(-1.5, 0.7)),
             "depth": float(rng.choice([rng.uniform(0.0, 0.3),
                                        rng.uniform(0.3, 1.0)])),
             "shape": str(rng.choice(SHAPES)),
             "duty": float(rng.uniform(0.0, 0.5)),
             "bps": [float(rng.uniform(0.0, 1.0))
                     for _ in range(int(rng.integers(3, 6)))]}
    pitch_type = str(rng.choice(PITCH_MODES,
                                p=[0.4, 0.15, 0.15, 0.15, 0.15]))
    pitch = {"type": pitch_type,
             "rate": float(10 ** rng.uniform(-1.5, 0.7)),
             "depth": float(rng.uniform(5.0, 600.0)),
             "end": float(2 ** rng.uniform(-2.0, 2.0))}
    n_v = int(rng.choice([1, 2, 2, 3]))
    voices = []
    for _ in range(n_v):
        voices.append({
            "detune": float(rng.choice([0.0, rng.uniform(-30, 30),
                                        rng.uniform(-700, 700),
                                        rng.uniform(-1200, 1200)])),
            "level": float(10 ** rng.uniform(-0.7, 0.0)),
            "pan": float(rng.uniform(-0.8, 0.8)),
            "morph_offset": float(rng.uniform(0.0, 0.3)),
            "attack": float(10 ** rng.uniform(-2.5, -0.7)),
            "decay_tau": float(10 ** rng.uniform(-0.5, 0.9)),
        })
    vector = {"type": str(rng.choice(["fixed", "lfo2d"], p=[0.4, 0.6])),
              "x0": float(rng.uniform(-0.6, 0.6)),
              "y0": float(rng.uniform(-0.6, 0.6)),
              "rate_x": float(10 ** rng.uniform(-2.0, 0.3)),
              "rate_y": float(10 ** rng.uniform(-2.0, 0.3)),
              "depth": float(rng.uniform(0.0, 1.0)),
              "shape_x": str(rng.choice(("sin", "tri"))),
              "shape_y": str(rng.choice(("sin", "tri")))}
    return WtGenome(f0=float(2 ** rng.uniform(5.9, 9.97)),
                    pitch=pitch, frames=frames, morph=morph,
                    voices=voices, vector=vector, seed=int(seed))


# -- structural ops (factored so tests can hit each one directly) ------

def op_partial_birth(g: WtGenome, rng) -> bool:
    k = int(rng.integers(len(g.frames)))
    fr = g.frames[k]
    if len(fr["partials"]) >= MAX_PARTIALS:
        return False
    fr["partials"].append(float(10 ** rng.uniform(-2.5, -0.3)))
    return True


def op_partial_death(g: WtGenome, rng) -> bool:
    k = int(rng.integers(len(g.frames)))
    fr = g.frames[k]
    if len(fr["partials"]) <= MIN_PARTIALS:
        return False
    fr["partials"].pop(int(rng.integers(len(fr["partials"]))))
    return True


def op_frame_interp(g: WtGenome, rng) -> bool:
    if len(g.frames) >= MAX_FRAMES or len(g.frames) < 2:
        return False
    k = int(rng.integers(len(g.frames) - 1))
    a, b = g.frames[k], g.frames[k + 1]
    P = max(len(a["partials"]), len(b["partials"]))
    pa = a["partials"] + [0.0] * (P - len(a["partials"]))
    pb = b["partials"] + [0.0] * (P - len(b["partials"]))
    mid = {"partials": [float(0.5 * (x + y)) for x, y in zip(pa, pb)],
           "inharm": float(0.5 * (a["inharm"] + b["inharm"]))}
    g.frames.insert(k + 1, mid)
    return True


def op_frame_dup(g: WtGenome, rng) -> bool:
    if len(g.frames) >= MAX_FRAMES:
        return False
    k = int(rng.integers(len(g.frames)))
    g.frames.insert(k + 1, {"partials": list(g.frames[k]["partials"]),
                           "inharm": float(g.frames[k]["inharm"])})
    return True


def op_frame_birth(g: WtGenome, rng) -> bool:
    if len(g.frames) >= MAX_FRAMES:
        return False
    g.frames.insert(int(rng.integers(len(g.frames) + 1)), _rand_frame(rng))
    return True


def op_frame_death(g: WtGenome, rng) -> bool:
    if len(g.frames) <= MIN_FRAMES:
        return False
    g.frames.pop(int(rng.integers(len(g.frames))))
    return True


def op_voice_birth(g: WtGenome, rng) -> bool:
    if len(g.voices) >= MAX_VOICES:
        return False
    g.voices.append({
        "detune": float(rng.uniform(-700, 700)),
        "level": float(10 ** rng.uniform(-0.7, 0.0)),
        "pan": float(rng.uniform(-0.8, 0.8)),
        "morph_offset": float(rng.uniform(0.0, 0.3)),
        "attack": float(10 ** rng.uniform(-2.5, -0.7)),
        "decay_tau": float(10 ** rng.uniform(-0.5, 0.9)),
    })
    return True


def op_voice_death(g: WtGenome, rng) -> bool:
    if len(g.voices) <= MIN_VOICES:
        return False
    g.voices.pop(int(rng.integers(len(g.voices))))
    return True


def _cycle(tup, cur):
    return tup[(tup.index(cur) + 1) % len(tup)]


def drift_params(g, rng: np.random.Generator, rate: float = 0.3) -> None:
    """Drift the path genes every scan organism shares — f0, pitch,
    morph, vector, voices — in place, including the law swaps and the
    voice structural ops. wt frames and sub substrates both call
    this; the content axes mutate separately."""
    g.f0 = _jit(rng, g.f0, F0_LO, F0_HI, rate)
    g.pitch["rate"] = _jit(rng, g.pitch["rate"], 0.02, 8.0, rate)
    g.pitch["depth"] = _jit(rng, g.pitch["depth"], 0.0, 2400.0, rate)
    g.pitch["end"] = _jit(rng, g.pitch["end"], 0.25, 4.0, rate)
    g.morph["rate"] = _jit(rng, g.morph["rate"], 0.01, 16.0, rate)
    g.morph["depth"] = _jit(rng, g.morph["depth"], 0.0, 1.0, rate)
    g.morph["duty"] = _add(rng, g.morph["duty"], 0.0, 1.0, 0.08)
    g.morph["bps"] = [_add(rng, v, 0.0, 1.0, 0.15)
                      for v in g.morph["bps"]]
    if rng.random() < 0.05:
        g.morph["shape"] = str(rng.choice(SHAPES))
    g.vector["x0"] = _add(rng, g.vector["x0"], -1.0, 1.0, 0.1)
    g.vector["y0"] = _add(rng, g.vector["y0"], -1.0, 1.0, 0.1)
    g.vector["rate_x"] = _jit(rng, g.vector["rate_x"], 0.005, 8.0, rate)
    g.vector["rate_y"] = _jit(rng, g.vector["rate_y"], 0.005, 8.0, rate)
    g.vector["depth"] = _jit(rng, g.vector["depth"], 0.0, 1.0, rate)
    if rng.random() < 0.04:
        g.vector["type"] = _cycle(("fixed", "lfo2d"), g.vector["type"])
    for v in g.voices:
        v["detune"] = _add(rng, v["detune"], -1200.0, 1200.0, 25.0)
        v["level"] = _jit(rng, v["level"], 0.05, 1.5, rate)
        v["pan"] = _add(rng, v["pan"], -0.9, 0.9, 0.08)
        v["morph_offset"] = _add(rng, v["morph_offset"], 0.0, 0.5, 0.04)
        v["attack"] = _jit(rng, v["attack"], 1e-3, 1.0, rate)
        v["decay_tau"] = _jit(rng, v["decay_tau"], 0.05, 8.0, rate)
    # law swaps (the grammar toggles)
    if rng.random() < 0.05:
        g.morph["type"] = _cycle(MORPH_MODES, g.morph["type"])
        if g.morph["type"] == "curve" and len(g.morph["bps"]) < 3:
            g.morph["bps"] = [float(rng.uniform(0, 1)) for _ in range(4)]
    if rng.random() < 0.04:
        g.pitch["type"] = _cycle(PITCH_MODES, g.pitch["type"])
    if rng.random() < 0.06:
        (op_voice_birth if rng.random() < 0.5 else op_voice_death)(g, rng)


def mutate(g: WtGenome, rng: np.random.Generator, rate: float = 0.3,
           seed: int = 0) -> WtGenome:
    """One birth: every continuous gene drifts; structure moves by ops."""
    child = WtGenome(
        f0=g.f0,
        pitch={k: v for k, v in g.pitch.items()},
        frames=[{"partials": list(fr["partials"]),
                 "inharm": float(fr["inharm"])} for fr in g.frames],
        morph=dict(g.morph),
        voices=[dict(v) for v in g.voices],
        vector=dict(g.vector),
        seed=int(seed),
    )
    drift_params(child, rng, rate)
    # ---- frames drift (the wt content axis)
    for fr in child.frames:
        fr["partials"] = [_jit(rng, a, 1e-3, 3.0, rate)
                          for a in fr["partials"]]
        fr["inharm"] = _jit(rng, fr["inharm"], 0.0, 1.5, rate)

    # ---- structural ops (the design table's probabilities)
    if rng.random() < 0.10:
        (op_partial_birth if rng.random() < 0.5 else op_partial_death)(child, rng)
    if rng.random() < 0.08:
        op_frame_interp(child, rng)
    if rng.random() < 0.04:
        op_frame_dup(child, rng)
    if rng.random() < 0.06:
        (op_frame_birth if rng.random() < 0.6 else op_frame_death)(child, rng)

    # ---- the leap
    if rng.random() < 0.02:
        axis = int(rng.integers(4))
        if axis == 0 and child.frames:
            child.frames[int(rng.integers(len(child.frames)))] = _rand_frame(rng)
        elif axis == 1:
            child.morph["type"] = str(rng.choice(("lfo", "step")))
            child.morph["rate"] = float(10 ** rng.uniform(0.0, 1.3))
            child.morph["depth"] = 1.0
        elif axis == 2:
            child.f0 = float(np.clip(child.f0 * rng.choice([0.5, 2.0]),
                                     F0_LO, F0_HI))
        else:
            child.vector.update({
                "type": "lfo2d",
                "x0": float(rng.uniform(-0.6, 0.6)),
                "y0": float(rng.uniform(-0.6, 0.6)),
                "rate_x": float(10 ** rng.uniform(-1.0, 0.3)),
                "rate_y": float(10 ** rng.uniform(-1.0, 0.3)),
                "depth": float(rng.uniform(0.5, 1.0)),
            })
    return child


# ------------------------------------------------------------------ io

def to_dict(g: WtGenome) -> dict:
    return {"genome": {"f0": g.f0, "pitch": g.pitch, "frames": g.frames,
                       "morph": g.morph, "voices": g.voices,
                       "vector": g.vector, "seed": g.seed},
            "scan": "genome", "version": 1}


def from_dict(d: dict) -> WtGenome:
    g = d["genome"] if "genome" in d else d
    return WtGenome(f0=float(g["f0"]), pitch=dict(g["pitch"]),
                   frames=[{"partials": [float(a) for a in fr["partials"]],
                            "inharm": float(fr["inharm"])}
                           for fr in g["frames"]],
                   morph=dict(g["morph"]),
                   voices=[dict(v) for v in g["voices"]],
                   vector=dict(g["vector"]), seed=int(g["seed"]))


def write_wav(path, audio: np.ndarray, sr: int = SR) -> str:
    """Float audio (n, 2) -> 16-bit wav via ffmpeg (the ecosystem writer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, np.float32)
    inter = np.empty(len(x) * 2, np.float32)
    inter[0::2], inter[1::2] = x[:, 0], x[:, 1]
    pcm = (np.clip(inter, -1.0, 1.0) * 32767.0).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le",
         "-ar", str(sr), "-ac", "2", "-i", "pipe:0", str(path)],
        input=pcm.tobytes(), check=True)
    return str(path)