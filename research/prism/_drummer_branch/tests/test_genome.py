"""genome tests — the senses, the mutation zoo, JSON round-trips."""

import numpy as np
import pytest

from prism.champions import answerer, follower, splitter
from prism.genome import FUNCTIONS, INPUTS, INPUT_IDS, Genome, random_genome


def test_inputs_include_scaffold_senses():
    assert INPUTS[:5] == ("x", "y", "d", "b", "t")
    for k in ("s0", "s1", "s2", "s3", "s4",
              "live_rms", "live_centroid", "live_flux"):
        assert k in INPUT_IDS
    # hidden/output ids never collide with inputs
    g = random_genome(np.random.default_rng(0))
    for nid in g.nodes:
        assert nid not in INPUT_IDS.values() or nid < len(INPUTS)


def test_json_roundtrip():
    for g in (follower(), answerer(), splitter(),
              random_genome(np.random.default_rng(5))):
        d = g.to_dict()
        h = Genome.from_dict(d)
        assert h.to_dict() == d
        # and the round-tripped genome renders identically
        feeds = {"b": np.float32(1.0),
                 "s0": np.linspace(0, 1, 16, dtype=np.float32)}
        a = g.evaluate(feeds)
        b = h.evaluate(feeds)
        for k in a:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]))


def test_mutations_keep_forward_order():
    rng = np.random.default_rng(2)
    for _ in range(4):
        g = random_genome(np.random.default_rng(int(rng.integers(1 << 20))))
        for _ in range(60):
            g.mutate(rng)
        for c in g.conns:
            if c.on:
                assert g.pos(c.src) < g.pos(c.dst), \
                    "a mutation broke the forward order"


def test_mutations_survive_eval_nan_club():
    # adversarial: huge weights, step functions, zeros — eval stays finite
    rng = np.random.default_rng(9)
    g = random_genome(np.random.default_rng(3))
    for _ in range(40):
        g.mutate(rng)
    for c in g.conns:
        c.w = 1e6
    out = g.evaluate({"x": np.float32(0.5), "b": np.float32(1.0),
                      "t": np.arange(32, dtype=np.float32)})
    for v in out.values():
        assert np.isfinite(v).all()


def test_zoo_functions_bounded_or_finite():
    u = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    for fn in FUNCTIONS.values():
        assert np.isfinite(fn(u)).all()


def test_add_output_grows_the_alphabet():
    g = follower()
    rng = np.random.default_rng(4)
    for _ in range(6):
        g.mut_add_output(rng)
    assert len(g.outputs) == 7
    feeds = {"b": np.float32(1.0), "s0": np.ones(8, np.float32) * 0.8}
    out = g.evaluate(feeds)
    assert set(out) == set(g.outputs)