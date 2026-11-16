"""voice: the organism — a minimal spectral genome rendered in numpy.

The genome is a handful of numbers you can point at:

    bumps        [(freq, amp, bw_oct), ...]   gaussian spectral peaks;
                 bw 0.01 = pure tone, bw 0.5 = noise band
    noise        (amp, color)                 colored noise floor
                 (color 0 white .. 2 brown)
    lfo          (rate_hz, depth)             amplitude weather
    vib          (rate_hz, depth)              pitch weather (per voice)
    clicks       (rate_hz, amp, jitter)       stochastic click layer
    env          (attack_s, decay_exp)         global envelope shape
    seed                                    deterministic render

Everything is audible structure: tone-ness, noise-ness, rhythm-ness,
drone-ness are all direct functions of the axes, which is exactly why
this is a fair arena — the ground truth of 'what kind of sound' is
readable from the genome, independent of any audio analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["VoiceGenome", "render", "random_genome", "mutate",
           "tonality", "ARCHETYPES", "SR", "SECONDS"]

SR = 44100
SECONDS = 4.0
MAX_BUMPS = 6
MAX_VOICES = 14                      # total detuned sinusoid budget


@dataclass
class VoiceGenome:
    bumps: list = field(default_factory=list)   # [(f0, amp, bw_oct)]
    noise_amp: float = 0.0
    noise_color: float = 1.0
    lfo_rate: float = 0.2
    lfo_depth: float = 0.0
    vib_rate: float = 0.5
    vib_depth: float = 0.0
    click_rate: float = 0.0
    click_amp: float = 0.0
    click_jitter: float = 0.5
    attack: float = 0.05
    decay_exp: float = 1.0
    seed: int = 0

    # ------------------------------------------------------------ labels
    def archetype(self) -> str:
        """Ground truth straight from the genome — no audio analysis,
        no circularity. The taxonomy is arbitrary but fixed."""
        b_max = max((a for _, a, _ in self.bumps), default=0.0)
        b_bw = max((b for _, _, b in self.bumps), default=0.0)
        n_bumps = len(self.bumps)
        rhythmic = (self.click_rate > 6 and self.click_amp > 0.2) or \
                   (self.lfo_rate > 1.2 and self.lfo_depth > 0.5)
        tonal = n_bumps > 0 and b_max > 0.4 and b_bw < 0.12
        noisy = self.noise_amp > 0.45
        textural = n_bumps > 0 and b_bw >= 0.15 and b_max > 0.3
        layers = sum((tonal, noisy, rhythmic))
        if layers >= 3:
            return "hybrid"
        if rhythmic and (tonal or noisy or textural):
            return "rhythmic_mix"
        if rhythmic:
            return "rhythm"
        if tonal and n_bumps == 1 and self.noise_amp < 0.15 \
                and self.click_amp < 0.1:
            return "pure_tone"
        if tonal:
            return "chord"
        if noisy:
            return "noise_wash"
        if textural:
            return "grain_cloud"
        return "sparse"

    def tonality(self) -> float:
        """The greedy arm's objective: a plausible 'make it tonal'
        score, computable from the genome (and honest about it)."""
        bump_e = sum(a for _, a, _ in self.bumps)
        return float(bump_e - 1.2 * self.noise_amp - 0.5 * self.click_amp)


ARCHETYPES = ("pure_tone", "chord", "noise_wash", "rhythm",
              "grain_cloud", "rhythmic_mix", "hybrid", "sparse")


def tonality(g: VoiceGenome) -> float:
    """Module-level objective (the greedy arm's trap)."""
    return g.tonality()


# ------------------------------------------------------------------ render

def render(g: VoiceGenome, seconds: float = SECONDS, sr: int = SR) -> np.ndarray:
    """Genome -> stereo audio (n, 2) float32. Deterministic from seed."""
    rng = np.random.default_rng(g.seed & 0x7FFFFFFF)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    L = np.zeros(n)
    R = np.zeros(n)

    # --- sinusoid voices: each bump becomes 1..k detuned partials
    voices = []                       # (freq_hz, amp, pan)
    for f0, amp, bw in g.bumps:
        f0 = float(np.clip(f0, 30.0, 9000.0))
        k = max(1, min(int(round(bw * 12.0)), 6))
        for i in range(k):
            off = (i / max(1, k - 1) - 0.5) if k > 1 else 0.0
            det = 2.0 ** (off * bw)
            voices.append((f0 * det, amp / np.sqrt(k),
                           float(rng.uniform(-0.8, 0.8))))
    if len(voices) > MAX_VOICES:      # budget: thin uniformly
        idx = np.linspace(0, len(voices) - 1, MAX_VOICES).astype(int)
        voices = [voices[i] for i in idx]

    if voices:
        F0 = np.array([v[0] for v in voices])[:, None]         # (V, 1)
        A = np.array([v[1] for v in voices])[:, None]
        P = rng.uniform(0, 2 * np.pi, len(voices))[:, None]
        if g.vib_depth > 0.001:
            F = F0 * (1.0 + 0.12 * g.vib_depth * np.sin(
                2 * np.pi * g.vib_rate * t[None, :] + P))
        else:
            F = np.broadcast_to(F0, (len(voices), n))         # (V, N)
        phase = 2 * np.pi * np.cumsum(F, axis=1) / sr
        waves = A * np.sin(phase + rng.uniform(0, 2 * np.pi, len(voices))[:, None])
        for i, (_, _, pan) in enumerate(voices):
            L += pan_from(waves[i], pan, 0)
            R += pan_from(waves[i], pan, 1)

    # --- colored noise floor (FFT-shaped white)
    if g.noise_amp > 0.001:
        w = rng.normal(0, 1, n)
        W = np.fft.rfft(w)
        fr = np.fft.rfftfreq(n, 1 / sr)
        W *= (fr + 40.0) ** (-float(np.clip(g.noise_color, 0, 2)) / 2.0)
        nz = np.fft.irfft(W, n)
        nz /= (np.abs(nz).max() + 1e-9)
        pan = float(rng.uniform(-0.5, 0.5))
        L += g.noise_amp * nz * (1.0 - max(0.0, pan))
        R += g.noise_amp * nz * (1.0 + min(0.0, pan))

    # --- click weather: jittered impulse train with 2 ms decays
    if g.click_rate > 0.05 and g.click_amp > 0.01:
        klen = int(0.002 * sr)
        kernel = np.exp(-np.linspace(0, 6, klen))
        expected = g.click_rate * seconds
        n_clicks = max(1, int(rng.poisson(expected))) if expected > 0.5 else 0
        for _ in range(n_clicks):
            pos = int(rng.uniform(0, n - klen - 1))
            if g.click_jitter > 0:
                pos = int(np.clip(
                    pos * (1.0 + rng.normal(0, 0.02 + 0.1 * g.click_jitter)),
                    0, n - klen - 1))
            a = g.click_amp * float(np.exp(rng.normal(-0.3, 0.6)))
            pan = float(rng.uniform(-0.9, 0.9))
            L[pos: pos + klen] += a * kernel * (1.0 - max(0.0, pan))
            R[pos: pos + klen] += a * kernel * (1.0 + min(0.0, pan))

    # --- global envelope + LFO amplitude weather
    atk = int(np.clip(g.attack, 0.001, seconds) * sr)
    env = np.ones(n)
    env[:atk] = np.linspace(0, 1, atk) ** 0.5
    m = max(1, n - atk)
    pos = np.linspace(0, 1, m)                        # 0..1 through tail
    sustain = 0.5
    env[atk:] = 1.0 - (1.0 - sustain) * pos ** max(0.05, g.decay_exp)
    if g.lfo_depth > 0.001:
        env *= 1.0 - g.lfo_depth * 0.5 * (1 + np.sin(
            2 * np.pi * g.lfo_rate * t + rng.uniform(0, 2 * np.pi)))
    L *= env
    R *= env

    stereo = np.stack([L, R], axis=1).astype(np.float32)
    peak = float(np.abs(stereo).max())
    if peak > 1e-9:
        stereo = stereo / peak * 0.95
    if not np.isfinite(stereo).all():
        stereo = np.zeros_like(stereo)
    return stereo


def pan_from(wave, pan, channel):
    """Constant-power pan: pan in [-1, 1] (L .. R)."""
    if channel == 0:
        return wave * np.cos((pan + 1) * np.pi / 4)
    return wave * np.sin((pan + 1) * np.pi / 4)


# ------------------------------------------------------------------ genetics

def random_genome(rng: np.random.Generator, seed: int) -> VoiceGenome:
    """A founder: 1..3 bumps, mostly quiet, some noise — nothing
    extreme; the arms must EARN the far corners."""
    n_bumps = int(rng.integers(1, 4))
    bumps = [[float(2 ** rng.uniform(5.3, 12.5)),
              float(rng.uniform(0.2, 0.7)),
              float(10 ** rng.uniform(-2, -0.7))] for _ in range(n_bumps)]
    return VoiceGenome(
        bumps=bumps,
        noise_amp=float(10 ** rng.uniform(-2, -0.3)),
        noise_color=float(rng.uniform(0, 2)),
        lfo_rate=float(10 ** rng.uniform(-1.3, 0.6)),
        lfo_depth=float(rng.choice([0.0, rng.uniform(0, 0.6)])),
        vib_rate=float(10 ** rng.uniform(-1, 0.8)),
        vib_depth=float(rng.choice([0.0, rng.uniform(0, 0.7)])),
        click_rate=float(rng.choice([0.0, 10 ** rng.uniform(-1, 1.2)])),
        click_amp=float(10 ** rng.uniform(-2, 0)),
        click_jitter=float(rng.uniform(0, 1)),
        attack=float(10 ** rng.uniform(-2, 0.3)),
        decay_exp=float(rng.uniform(0.3, 2.5)),
        seed=int(seed),
    )


def _jit(rng, v, lo, hi, rate=0.35):
    return float(np.clip(v * 10 ** rng.normal(0, rate), lo, hi))


def mutate(g: VoiceGenome, rng: np.random.Generator, rate: float = 0.35,
           seed: int = 0) -> VoiceGenome:
    """Small-step drift: log-jitter everything, occasionally grow/shrink
    the bump list, occasionally leap (so corners are reachable)."""
    child = VoiceGenome(
        bumps=[list(b) for b in g.bumps],
        noise_amp=g.noise_amp, noise_color=g.noise_color,
        lfo_rate=g.lfo_rate, lfo_depth=g.lfo_depth,
        vib_rate=g.vib_rate, vib_depth=g.vib_depth,
        click_rate=g.click_rate, click_amp=g.click_amp,
        click_jitter=g.click_jitter, attack=g.attack,
        decay_exp=g.decay_exp, seed=int(seed),
    )
    for b in child.bumps:
        b[0] = _jit(rng, b[0], 30.0, 9000.0, rate)
        b[1] = _jit(rng, b[1], 0.01, 1.0, rate)
        b[2] = _jit(rng, b[2], 0.008, 0.8, rate)
    child.noise_amp = _jit(rng, child.noise_amp, 0.0, 1.0, rate)
    child.noise_color = _jit(rng, child.noise_color, 0.0, 2.0, rate)
    child.lfo_rate = _jit(rng, child.lfo_rate, 0.02, 8.0, rate)
    child.lfo_depth = _jit(rng, child.lfo_depth, 0.0, 1.0, rate)
    child.vib_rate = _jit(rng, child.vib_rate, 0.02, 12.0, rate)
    child.vib_depth = _jit(rng, child.vib_depth, 0.0, 1.0, rate)
    child.click_rate = _jit(rng, child.click_rate, 0.0, 30.0, rate)
    child.click_amp = _jit(rng, child.click_amp, 0.0, 1.0, rate)
    child.attack = _jit(rng, child.attack, 0.002, 3.0, rate)
    child.decay_exp = _jit(rng, child.decay_exp, 0.1, 4.0, rate)
    r = rng.random()
    if r < 0.15 and len(child.bumps) < MAX_BUMPS:            # bump birth
        child.bumps.append([float(2 ** rng.uniform(5.3, 12.5)),
                            float(rng.uniform(0.05, 0.8)),
                            float(10 ** rng.uniform(-2, -0.7))])
    elif r < 0.3 and len(child.bumps) > 1:                    # bump death
        child.bumps.pop(int(rng.integers(len(child.bumps))))
    if rng.random() < 0.05:                                   # the leap
        axis = int(rng.integers(6))
        leaps = [
            lambda: child.bumps.__setitem__(0, [float(2 ** rng.uniform(5.3, 12.5)),
                                                float(rng.uniform(0.1, 1.0)), 0.01]),
            lambda: setattr(child, "noise_amp", float(rng.uniform(0.3, 1.0))),
            lambda: setattr(child, "click_rate", float(10 ** rng.uniform(0.5, 1.4))),
            lambda: setattr(child, "click_amp", float(rng.uniform(0.3, 1.0))),
            lambda: setattr(child, "bumps", [[float(2 ** rng.uniform(5.3, 12.5)),
                                              float(rng.uniform(0.2, 0.9)),
                                              float(rng.uniform(0.15, 0.6))]]),
            lambda: setattr(child, "lfo_rate", float(10 ** rng.uniform(0.3, 0.9))),
        ]
        leaps[axis]()
    return child