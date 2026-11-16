"""dx: the FM organism — a compact DX-style operator genome.

Chowning's gift: a handful of genes (ratios, indices, envelopes) maps
to an enormous, NONLINEAR perceptual manifold. Integer ratios at low
index sound brass-ish, inharmonic ratios at high index ring like
bells, deep slow index LFOs make washes, low-ratio modulators at high
index growl. Every region sounds like a different country — exactly
what an exploration demo needs: a country big enough that undirected
diffusion (√t) cannot cross it in the budget, but directed novelty
(t) can.

Genome:
    f0            fundamental (55–1000 Hz)
    ops           2–5 operators in feed-forward chains:
                    ratio        frequency multiple of f0
                    index        modulation depth (Hz-deviation = index*ratio*f0)
                    level        audible gain (0 = pure modulator)
                    target       which LATER op it modulates (target > own index)
                    attack, decay_tau, pan
    lfo           (rate, depth) index weather
    noise_amp     a pinch of noise floor
    seed          deterministic render

Render: Chowning phase integration in topological order (op i may be
modulated only by ops k < i), at 11.025 kHz with deviation clamped
below Nyquist, then exact FFT upsample x4 — a 2 s render costs ~3 ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DxGenome", "render", "random_genome", "mutate", "tonality",
           "ARCHETYPES", "SR", "SECONDS", "MAX_OPS"]

SR = 44100
SECONDS = 2.0
RENDER_SR = 11025                  # synthesis rate, upsampled x4 after
NYQ = RENDER_SR / 2.0
MAX_OPS = 5
MAX_F = 4800.0                     # clamp for fundamental*ratio
MAX_DEV = 3400.0                   # clamp for FM deviation


@dataclass
class DxGenome:
    f0: float = 220.0
    ops: list = field(default_factory=list)
    # each op: dict(ratio, index, level, target, attack, decay_tau, pan)
    lfo_rate: float = 0.3
    lfo_depth: float = 0.0
    noise_amp: float = 0.0
    seed: int = 0

    # ------------------------------------------------------------ labels
    def archetype(self) -> str:
        """Ground truth off the genome — fixed cascade, no audio."""
        idx = [o["index"] for o in self.ops]
        ratios = [o["ratio"] for o in self.ops if o["index"] > 0.6]
        carriers = [o for o in self.ops if o["level"] > 0.05]
        imax = max(idx) if idx else 0.0
        slow = [o for o in self.ops if o["decay_tau"] > 1.2]
        if self.lfo_depth > 0.5 and self.lfo_rate < 1.5:
            return "wash"
        if imax < 0.6:
            return "tone"
        if any(o["index"] > 6.0 and o["ratio"] < 0.9 for o in self.ops):
            return "growl"
        inharmonic = any(abs(r - round(r)) > 0.08 for r in ratios)
        if imax > 6.0 and inharmonic:
            return "bell"
        if imax > 6.0 and len(carriers) >= 2:
            return "complex"
        if slow and 0.6 <= imax:
            return "drone"
        if imax >= 0.6:
            return "keys"                   # decaying mid-index: e-piano-ish
        return "drone"

    def tonality(self) -> float:
        """The greedy objective: purity. Converges to index-0 sines —
        the trap, honest and audible."""
        return float(-sum(o["index"] for o in self.ops)
                     - 3.0 * self.noise_amp)


ARCHETYPES = ("tone", "keys", "bell", "growl", "wash", "drone", "complex")


def tonality(g: DxGenome) -> float:
    return g.tonality()


# ------------------------------------------------------------------ render

def render(g: DxGenome, seconds: float = SECONDS, sr: int = SR) -> np.ndarray:
    """Genome -> stereo (n, 2) float32. Deterministic from seed."""
    rng = np.random.default_rng(g.seed & 0x7FFFFFFF)
    n = int(seconds * RENDER_SR)
    t = np.arange(n) / RENDER_SR
    ops = g.ops

    # index LFO: the weather on every modulator
    if g.lfo_depth > 0.001:
        lfo = 1.0 - g.lfo_depth * 0.5 * (
            1.0 + np.sin(2 * np.pi * g.lfo_rate * t
                         + rng.uniform(0, 2 * np.pi)))
    else:
        lfo = 1.0

    freqs = [float(np.clip(g.f0 * o["ratio"], 30.0, MAX_F)) for o in ops]
    phases = []
    for i in range(len(ops)):
        dev = np.zeros(n)
        for k in range(i):                          # feed-forward only
            o = ops[k]
            if o["target"] == i and o["index"] > 0.01:
                dev += min(o["index"] * freqs[k], MAX_DEV) * lfo * np.sin(phases[k])
        np.clip(dev, -MAX_DEV, MAX_DEV, out=dev)
        rate = 2 * np.pi * (freqs[i] + dev) / RENDER_SR
        phases.append(np.cumsum(rate) + rng.uniform(0, 2 * np.pi))

    L = np.zeros(n)
    R = np.zeros(n)
    for o, p in zip(ops, phases):
        if o["level"] <= 0.01:
            continue
        atk = int(np.clip(o["attack"] * RENDER_SR, 1, n))
        e = np.exp(-t / max(0.02, o["decay_tau"]))
        e[:atk] *= np.linspace(0, 1, atk)
        w = o["level"] * e * np.sin(p)
        pan = float(np.clip(o.get("pan", 0.0), -0.9, 0.9))
        L += w * np.cos((pan + 1) * np.pi / 4)
        R += w * np.sin((pan + 1) * np.pi / 4)
    if g.noise_amp > 0.01:
        L += g.noise_amp * rng.normal(0, 1, n)
        R += g.noise_amp * rng.normal(0, 1, n)

    # exact band-limited upsample x4 (spectral zero-padding)
    up = sr // RENDER_SR
    N = n * up
    out = np.zeros((N, 2), np.float64)
    for ch, sig in ((0, L), (1, R)):
        X = np.fft.rfft(sig)
        XL = np.zeros(N // 2 + 1, np.complex128)
        XL[:len(X)] = X
        out[:, ch] = np.fft.irfft(XL, N) * up
    stereo = out.astype(np.float32)
    peak = float(np.abs(stereo).max())
    if peak > 1e-9:
        stereo = stereo / peak * 0.95
    if not np.isfinite(stereo).all():
        stereo = np.zeros_like(stereo)
    return stereo[:int(seconds * sr)]


# ------------------------------------------------------------------ genetics

def random_genome(rng: np.random.Generator, seed: int) -> DxGenome:
    n_ops = int(rng.integers(2, MAX_OPS + 1))
    ops = []
    for i in range(n_ops):
        carrier = (i == 0) or rng.random() < 0.5
        ops.append({
            "ratio": float(round(2 ** rng.uniform(-2, 4.0)))
                        if rng.random() < 0.4 else float(2 ** rng.uniform(-2, 4.0)),
            "index": float(10 ** rng.uniform(-1.5, 1.2)),
            "level": float(10 ** rng.uniform(-0.8, 0.0)) if carrier else 0.0,
            "target": int(rng.integers(i + 1, n_ops)) if i + 1 < n_ops else i,
            "attack": float(10 ** rng.uniform(-2.5, -0.5)),
            "decay_tau": float(10 ** rng.uniform(-1.3, 0.8)),
            "pan": float(rng.uniform(-0.8, 0.8)),
        })
    return DxGenome(
        f0=float(2 ** rng.uniform(5.8, 10.0)),
        ops=ops,
        lfo_rate=float(10 ** rng.uniform(-1.5, 0.7)),
        lfo_depth=float(rng.choice([0.0, rng.uniform(0, 0.9)])),
        noise_amp=float(10 ** rng.uniform(-3, -0.8)),
        seed=int(seed),
    )


def _jit(rng, v, lo, hi, rate=0.3):
    return float(np.clip(v * 10 ** rng.normal(0, rate), lo, hi))


def _valid_target(rng, k, n):
    """Feed-forward: op k may only modulate a LATER op."""
    return int(rng.integers(k + 1, n)) if k + 1 < n else k


def mutate(g: DxGenome, rng: np.random.Generator, rate: float = 0.3,
           seed: int = 0) -> DxGenome:
    child = DxGenome(
        f0=_jit(rng, g.f0, 55.0, 1000.0, rate),
        ops=[dict(o) for o in g.ops],
        lfo_rate=_jit(rng, g.lfo_rate, 0.02, 8.0, rate),
        lfo_depth=_jit(rng, g.lfo_depth, 0.0, 1.0, rate),
        noise_amp=_jit(rng, g.noise_amp, 0.0, 0.5, rate),
        seed=int(seed),
    )
    for k, o in enumerate(child.ops):
        o["ratio"] = _jit(rng, o["ratio"], 0.25, 24.0, rate)
        o["index"] = _jit(rng, o["index"], 0.0, 25.0, rate)
        o["level"] = _jit(rng, o["level"], 0.0, 1.5, rate)
        o["attack"] = _jit(rng, o["attack"], 0.001, 1.0, rate)
        o["decay_tau"] = _jit(rng, o["decay_tau"], 0.05, 8.0, rate)
        o["pan"] = float(np.clip(o["pan"] + rng.normal(0, 0.1), -0.9, 0.9))
        if o["target"] >= len(child.ops):
            o["target"] = _valid_target(rng, k, len(child.ops))
    r = rng.random()
    if r < 0.12 and len(child.ops) < MAX_OPS:                 # op birth
        child.ops.append({
            "ratio": float(2 ** rng.uniform(-2, 4.0)),
            "index": float(10 ** rng.uniform(-1.5, 1.2)),
            "level": float(10 ** rng.uniform(-0.8, 0.0)),
            "target": len(child.ops),                         # inert until rewired
            "attack": float(10 ** rng.uniform(-2.5, -0.5)),
            "decay_tau": float(10 ** rng.uniform(-1.3, 0.8)),
            "pan": float(rng.uniform(-0.8, 0.8)),
        })
    elif r < 0.22 and len(child.ops) > 2:                     # op death
        dead = int(rng.integers(len(child.ops)))
        child.ops.pop(dead)
        for k, o in enumerate(child.ops):
            if o["target"] >= len(child.ops):
                o["target"] = _valid_target(rng, k, len(child.ops))
    if rng.random() < 0.10:                                   # rewire
        k = int(rng.integers(len(child.ops)))
        child.ops[k]["target"] = _valid_target(rng, k, len(child.ops))
    if rng.random() < 0.02:                                    # the leap
        axis = int(rng.integers(4))
        if axis == 0 and child.ops:
            child.ops[0]["ratio"] = float(np.round(2 ** rng.uniform(-2, 4.0)))
        elif axis == 1 and child.ops:
            child.ops[0]["index"] = float(10 ** rng.uniform(0.8, 1.4))
        elif axis == 2:
            child.lfo_depth = float(rng.uniform(0.6, 1.0))
            child.lfo_rate = float(10 ** rng.uniform(-1.5, 0.0))
        elif axis == 3 and child.ops:
            child.ops[0]["decay_tau"] = float(10 ** rng.uniform(0.3, 0.9))
    return child