"""voice: the organism — determinism, bounds, genetics, labels."""

import numpy as np
import pytest

from maze.voice import (ARCHETYPES, VoiceGenome, mutate, random_genome,
                        render, tonality)


def test_render_deterministic():
    g = random_genome(np.random.default_rng(1), seed=42)
    a = render(g, seconds=1.0)
    b = render(g, seconds=1.0)
    assert np.array_equal(a, b)


def test_render_stereo_finite_bounded():
    g = random_genome(np.random.default_rng(2), seed=7)
    x = render(g, seconds=1.0)
    assert x.shape == (44100, 2) and x.dtype == np.float32
    assert np.isfinite(x).all() and float(np.abs(x).max()) <= 1.0


def test_seed_changes_render():
    g1 = random_genome(np.random.default_rng(3), seed=11)
    g2 = random_genome(np.random.default_rng(3), seed=11)
    g2.seed = 12
    assert not np.array_equal(render(g1, 1.0), render(g2, 1.0))


def test_mutation_small_steps_mostly():
    rng = np.random.default_rng(4)
    g = random_genome(rng, seed=5)
    child = mutate(g, rng, seed=99)
    # same gene count structure unless a birth/death fired; params drift
    assert 1 <= len(child.bumps) <= 6
    assert child.attack > 0 and child.decay_exp > 0


def test_archetypes_label_sensibly():
    pure = VoiceGenome(bumps=[[440.0, 0.9, 0.01]], seed=1)
    wash = VoiceGenome(bumps=[], noise_amp=0.8, seed=1)
    rhythm = VoiceGenome(bumps=[], click_rate=10.0, click_amp=0.6, seed=1)
    cloud = VoiceGenome(bumps=[[800.0, 0.7, 0.3]], seed=1)
    assert pure.archetype() == "pure_tone"
    assert wash.archetype() == "noise_wash"
    assert rhythm.archetype() == "rhythm"
    assert cloud.archetype() == "grain_cloud"
    for a in (pure.archetype(), wash.archetype(), rhythm.archetype(),
              cloud.archetype()):
        assert a in ARCHETYPES


def test_tonality_prefers_tones():
    pure = VoiceGenome(bumps=[[440.0, 0.9, 0.01]], seed=1)
    noisy = VoiceGenome(bumps=[[440.0, 0.9, 0.01]], noise_amp=0.8, seed=1)
    assert tonality(pure) > tonality(noisy)


def test_render_spans_the_space():
    """The arena has real perceptual structure: a pure tone, a noise
    wash and a click train must differ grossly (zero crossings,
    spectral flatness, sparsity) — the demo's premise."""
    sr = 44100
    pure = render(VoiceGenome(bumps=[[440.0, 0.9, 0.01]], seed=3), 1.0)
    wash = render(VoiceGenome(noise_amp=0.9, noise_color=1.0, seed=3), 1.0)
    clicks = render(VoiceGenome(click_rate=14.0, click_amp=0.9,
                                click_jitter=0.8, seed=3), 1.0)
    def zcr(x):
        m = x.mean(axis=1)
        return np.abs(np.diff(np.signbit(m))).mean()
    def flat(x):
        m = x.mean(axis=1)
        S = np.abs(np.fft.rfft(m * np.hanning(len(m)))) + 1e-9
        return float(np.exp(np.log(S).mean()) / S.mean())
    def rms(x):
        return float(np.sqrt((x.mean(axis=1) ** 2).mean()))
    assert zcr(pure) < 0.15 * zcr(wash)
    assert flat(pure) < 0.1 * flat(wash)
    # clicks are sparse positive weather: loud peaks, tiny rms,
    # and (being unipolar) no zero crossings of their own
    assert float(np.abs(clicks).max()) > 0.5
    assert rms(clicks) < 0.4 * rms(wash)