"""phase E: the substrate organism — DeepHyperNEAT repurposed.

Known answers, determinism (per device), batch-vs-single bit-stability
on CPU, depth/breadth bounds, complexification, the fixed-point trap,
and the engine running the new organism end to end.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scan import substrate as sub
from scan import viable, ARCHETYPES
from scan.substrate import (SubGenome, render, render_batch, random_genome,
                            mutate, tonality, champion, SUB_CHAMPIONS,
                            op_inc_depth, op_dec_depth, op_inc_breadth,
                            op_dec_breadth, decode, amp_matrix, amp_batch,
                            MAX_DEPTH, MAX_WIDTH, P_LO, P_HI, K_LO, K_HI,
                            to_dict, from_dict)


@pytest.fixture(autouse=True)
def _cpu():
    sub.set_device("cpu")            # CI-stable default; mps tested separately


def _founder(seed=3):
    return random_genome(np.random.default_rng(seed), seed=seed * 101)


# ------------------------------------------------------------------ known answers

def test_rampbank_is_a_diagonal_ridge():
    """Constant CPPN -> every weight clamps to +MAX_WEIGHT -> the
    substrate mixes: amp(p, m) = |tanh(c tanh(p + m))| - structure
    along the p+m diagonal. Known answers: frames' spectral centroid
    rises along the morph axis (brightening), and the amp matrix
    correlates with (p + m) - geometry's absence still paints
    geometry."""
    g = champion("rampbank")
    amp = amp_matrix(g)
    assert amp.shape == (g.P, g.K)
    cen = (amp * np.arange(g.P)[:, None]).sum(axis=0) \
        / (amp.sum(axis=0) + 1e-9)
    assert cen[-1] > cen[0] + 2.0, \
        f"centroid should rise along morph: {cen[0]:.1f} -> {cen[-1]:.1f}"
    pp, mm = np.meshgrid(np.linspace(-1, 1, g.P),
                         np.linspace(-1, 1, g.K), indexing="ij")
    r = np.corrcoef(amp.ravel(), np.abs(pp + mm).ravel())[0, 1]
    assert r > 0.5, f"amp should track the diagonal valley, r={r:.2f}"
    x = render(g, seconds=2.0)
    assert viable(x)


def test_comb_weights_oscillate_with_distance():
    """w = sin(4d): the decode must paint weights that vary with the
    geometric distance — not a constant, not random."""
    g = champion("comb")
    W = decode(g, "cpu")
    assert len(W) == 3                     # in -> h1 -> h2 -> out
    w = W[0].numpy()
    assert w.min() < -1.0 < 1.0 < w.max()  # oscillating, clamped to +-3
    assert viable(render(g, seconds=2.0))


def test_sub_champions_distinct_and_viable():
    auds = {}
    for name in SUB_CHAMPIONS:
        g = champion(name)
        x = render(g, seconds=2.0)
        assert viable(x), f"{name} failed the minimal criterion"
        assert g.archetype() in ARCHETYPES
        auds[name] = x
    a, b, c = (auds[n] for n in SUB_CHAMPIONS)
    def corr(x, y):
        u, v = x.mean(axis=1), y.mean(axis=1)
        u, v = u - u.mean(), v - v.mean()
        return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    assert corr(a, b) < 0.9 and corr(b, c) < 0.9


# ------------------------------------------------------------------ physics

def test_determinism_per_device():
    g = _founder(1)
    assert np.array_equal(render(g), render(g))
    # the materialised frames are cached but mutation makes fresh objects
    c = mutate(g, np.random.default_rng(2), seed=5)
    assert np.array_equal(render(c), render(c))


def test_batch_equals_single_on_cpu():
    """CPU bmm is bit-stable: a genome's amp depends only on its genes,
    never on the company it keeps in the batch."""
    rng = np.random.default_rng(7)
    gs = [random_genome(rng, seed=int(rng.integers(1 << 30)))
          for _ in range(4)]
    singles = [amp_matrix(g) for g in gs]
    batched = amp_batch(gs)
    for s, b in zip(singles, batched):
        assert np.array_equal(s, b)
    # and through the full render path
    assert np.array_equal(render(gs[0]), render_batch(gs)[0])


def test_mps_deterministic_when_available():
    if not torch.backends.mps.is_available():
        pytest.skip("no mps")
    sub.set_device("mps")
    try:
        g = champion("comb")
        assert np.array_equal(render(g), render(g))
    finally:
        sub.set_device("cpu")


def test_bounds_and_finiteness():
    for seed in (0, 1, 2):
        x = render(_founder(seed), seconds=1.0)
        assert x.shape == (44100, 2)
        assert np.isfinite(x).all()
        assert float(np.abs(x).max()) <= 1.0


def test_structural_ops_respect_bounds():
    g = _founder(6)
    rng = np.random.default_rng(0)
    for _ in range(300):
        op_inc_depth(g, rng)
        op_dec_depth(g, rng)
        op_inc_breadth(g, rng)
        op_dec_breadth(g, rng)
        assert 1 <= len(g.widths) <= MAX_DEPTH
        assert all(1 <= w <= MAX_WIDTH for w in g.widths)


def test_mutate_respects_grid_bounds():
    g = _founder(8)
    rng = np.random.default_rng(1)
    for _ in range(50):
        g = mutate(g, rng, seed=1)
        assert P_LO <= g.P <= P_HI
        assert K_LO <= g.K <= K_HI
        assert len(g.widths) <= MAX_DEPTH


def test_complexification_grows():
    """The open-ended claim: graphs grow under mutation pressure —
    the ledger prism keeps, transplanted."""
    rng = np.random.default_rng(4)
    g = _founder(9)
    c0 = g.cppn.complexity() + sum(g.widths)
    grew = 0
    for _ in range(40):
        g = mutate(g, rng, seed=1)
    assert g.cppn.complexity() + sum(g.widths) > c0


def test_the_trap_is_a_fixed_point():
    """Minimal graph + min partials + static morph + one voice: mutation
    almost never purifies further (structural tonality)."""
    g = champion("rampbank")
    g.P = P_LO
    g.morph["depth"] = 0.0
    g.widths = [1]
    # strip the cppn to the constant answer
    from scan.cppn import Cppn
    g.cppn = Cppn(seed=1)
    g.cppn.add_output("w", fn="lin", bias=1.0)
    t0 = tonality(g)
    rng = np.random.default_rng(2)
    purer = 0
    for _ in range(10):
        c = mutate(g, rng, seed=1)
        if tonality(c) > t0:
            purer += 1
    assert purer <= 1


def test_json_roundtrip_bits():
    g = _founder(11)
    d = to_dict(g)
    g2 = from_dict(json.loads(json.dumps(d)))
    assert np.array_equal(render(g), render(g2))


# ------------------------------------------------------------------ engine

def test_sub_engine_smoke(tmp_path):
    """The whole loop with the substrate organism: four arms, bank
    admission, artifacts, report."""
    pytest.importorskip("echo")
    from scan.engine import run_experiment
    summary = run_experiment(out=str(tmp_path / "run"), seeds=(0,),
                             gens=3, pop=6, births=4, verbose=False,
                             jobs=1, organism_name="sub")
    for arm in ("greedy", "random", "cell", "novelty"):
        r = summary["results"][arm]
        assert r["kinds_ever_mean"] >= 1
        assert (tmp_path / "run" / f"{arm}_s0" / "mixtape.wav").exists()
    report = (tmp_path / "run" / "report.md").read_text()
    assert "sub" in report and "parent-recall" in report


def test_render_speed():
    g = _founder(12)
    render(g)                                        # warm
    import time
    t0 = time.time()
    for _ in range(4):
        render(g)
    per = (time.time() - t0) / 4 * 1000
    assert per < 400, f"{per:.0f} ms per render — too slow for breeding"