"""dx: the FM organism — determinism, span, genetics, labels."""

import numpy as np
import pytest

from maze.dx import (ARCHETYPES, DxGenome, mutate, random_genome, render,
                     tonality)
from maze.engine import viable


def _op(ratio, index, level=0.0, target=1, decay=1.0, attack=0.01):
    return {"ratio": ratio, "index": index, "level": level,
            "target": target, "attack": attack, "decay_tau": decay,
            "pan": 0.0}


def test_render_deterministic_and_bounded():
    g = random_genome(np.random.default_rng(1), seed=42)
    a = render(g)
    b = render(g)
    assert np.array_equal(a, b)
    assert a.shape == (int(2.0 * 44100), 2) and a.dtype == np.float32
    assert np.isfinite(a).all() and float(np.abs(a).max()) <= 1.0


def test_carrier_required_for_sound():
    silent = DxGenome(f0=220, ops=[_op(1.0, 0.0, level=0.0)], seed=1)
    assert float(np.abs(render(silent)).max()) < 1e-6
    loud = DxGenome(f0=220, ops=[_op(1.0, 0.0, level=0.9)], seed=1)
    assert float(np.abs(render(loud)).max()) > 0.5


def test_index_changes_the_sound_a_lot():
    """FM nonlinearity: a pure tone vs a deeply modulated carrier must
    differ grossly — most of the wild one's energy lives in the FM
    sidebands above 1 kHz (a sparse sideband stack is invisible to
    bin-dominated flatness, so measure the energy split directly)."""
    def hi_energy(g):
        x = render(g).mean(axis=1)
        S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-9
        fr = np.fft.rfftfreq(len(x), 1 / 44100)
        return float(S[fr > 1000].sum() / S.sum())
    pure = DxGenome(f0=220, ops=[_op(1.0, 0.0, level=0.9, target=0)], seed=1)
    wild = DxGenome(f0=220, ops=[_op(3.01, 14.0, level=0.0, target=1),
                                 _op(1.0, 0.0, level=0.9, target=1)], seed=1)
    assert hi_energy(wild) > 0.5
    assert hi_energy(pure) < 0.01
    assert hi_energy(wild) > 50 * hi_energy(pure)


def test_archetypes_label_sensibly():
    tone = DxGenome(f0=220, ops=[_op(1.0, 0.0, level=0.9, target=0)], seed=1)
    bell = DxGenome(f0=220, ops=[_op(3.41, 12.0, level=0.0, target=1),
                                 _op(1.0, 0.0, level=0.9, target=1)], seed=1)
    growl = DxGenome(f0=110, ops=[_op(0.5, 10.0, level=0.0, target=1),
                                  _op(1.0, 0.0, level=0.9, target=1)],
                     seed=1)
    wash = DxGenome(f0=220, lfo_rate=0.3, lfo_depth=0.8,
                    ops=[_op(2.0, 4.0, level=0.0, target=1),
                         _op(1.0, 0.0, level=0.9, target=1)], seed=1)
    assert tone.archetype() == "tone"
    assert bell.archetype() == "bell"
    assert growl.archetype() == "growl"
    assert wash.archetype() == "wash"
    for g in (tone, bell, growl, wash):
        assert g.archetype() in ARCHETYPES


def test_mutation_keeps_feed_forward():
    rng = np.random.default_rng(9)
    g = random_genome(rng, seed=5)
    for _ in range(30):
        g = mutate(g, rng, seed=int(rng.integers(1 << 30)))
        assert 2 <= len(g.ops) <= 5
        for k, o in enumerate(g.ops):
            assert 0.25 <= o["ratio"] <= 24.0
            assert o["index"] >= 0.0
    # modulators feed forward: rendering must not blow up
    x = render(g)
    assert np.isfinite(x).all()


def test_render_speed():
    """The long-run budget: a 2 s render must stay in the few-ms range."""
    import time
    g = random_genome(np.random.default_rng(2), seed=13)
    t0 = time.perf_counter()
    for _ in range(20):
        render(g)
    dt = (time.perf_counter() - t0) / 20
    assert dt < 0.05, f"render too slow: {dt * 1000:.1f} ms"


def test_founders_are_viable_often():
    """The arena must be livable: most random genomes pass the MC."""
    rng = np.random.default_rng(3)
    ok = 0
    for i in range(20):
        g = random_genome(rng, seed=i)
        if viable(render(g)):
            ok += 1
    assert ok >= 10