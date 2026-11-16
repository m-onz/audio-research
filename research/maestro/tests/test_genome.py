"""genome tests — the senses, the mutation zoo, JSON round-trips.

Invariants (feed-forward order, finite evaluation, NEAT splits,
dispatch reachability) adapted from the prism spec; champions here
are maestro's score organisms.
"""

import numpy as np
import pytest

from maestro.champions import (CHAMPIONS, answerer, build_champion,
                               follower, random_score_genome, splitter)
from maestro.genome import FUNCTIONS, INPUTS, INPUT_IDS, Genome, random_genome


def _order_ok(g: Genome) -> bool:
    pos = {nid: i for i, nid in enumerate(g.order)}
    return all(pos[c.src] < pos[c.dst] for c in g.conns if c.on)


def test_inputs_present():
    g = Genome()
    assert all(nid in g.nodes for nid in INPUT_IDS.values())
    assert g.order[: len(INPUTS)] == [INPUT_IDS[n] for n in INPUTS]


def test_inputs_include_scaffold_senses():
    for k in ("s0", "s1", "s2", "s3", "s4",
              "live_rms", "live_centroid", "live_flux"):
        assert k in INPUT_IDS
    # hidden/output ids never collide with inputs
    g = random_genome(np.random.default_rng(0))
    for nid in g.nodes:
        assert nid not in INPUT_IDS.values() or nid < len(INPUTS)


def test_evaluate_shapes_and_defaults():
    g = follower()
    scalar = g.evaluate({"s0": np.float32(0.9)})
    assert np.asarray(scalar["pulse"]).shape == ()
    arr = g.evaluate({"s0": np.linspace(0, 1, 7, dtype=np.float32)})
    assert np.asarray(arr["pulse"]).shape == (7,)
    out = g.evaluate({})                     # everything defaults
    assert np.isfinite(out["pulse"]).all()


def test_functions_finite():
    grid = np.linspace(-8.0, 8.0, 257, dtype=np.float32)
    for f in FUNCTIONS.values():
        v = f(grid)
        assert np.isfinite(v).all()


@pytest.mark.parametrize("op", ["mut_drift", "mut_add_conn", "mut_add_node",
                                "mut_add_output", "mut_swap_fn", "mut_toggle",
                                "mut_duplicate"])
def test_mutations_preserve_feedforward(op):
    rng = np.random.default_rng(7)
    for _ in range(6):
        g = random_score_genome(rng, n_hidden=int(rng.integers(1, 4)))
        for _ in range(12):
            getattr(g, op)(rng)
            assert _order_ok(g), f"{op} broke the feed-forward order"
        out = g.evaluate({"s0": np.linspace(0, 1, 64, dtype=np.float32)})
        for v in out.values():
            assert np.isfinite(v).all(), f"{op} produced non-finite output"


def test_add_node_splits_connection():
    rng = np.random.default_rng(0)
    g = answerer()
    n_conns, n_nodes = len(g.conns), len(g.nodes)
    op = g.mut_add_node(rng)
    assert op == "add_node"
    assert len(g.conns) == n_conns + 2     # one disabled, two grown
    assert len(g.nodes) == n_nodes + 1


def test_add_output_grows_alphabet():
    rng = np.random.default_rng(1)
    g = follower()
    before = set(g.outputs)
    g.mut_add_output(rng)
    assert set(g.outputs) > before
    out = g.evaluate({"s0": np.zeros(4, np.float32)})
    assert all(np.isfinite(v).all() for v in out.values())


def test_duplicate_copies_and_hooks():
    rng = np.random.default_rng(2)
    g = random_score_genome(np.random.default_rng(20))   # has hiddens
    n_before = len(g.nodes)
    op = g.mut_duplicate(rng)
    assert op == "duplicate"
    assert len(g.nodes) > n_before
    assert _order_ok(g)
    out = g.evaluate({"s0": np.linspace(0, 1, 32, dtype=np.float32)})
    assert all(np.isfinite(v).all() for v in out.values())


def test_mutate_dispatch():
    rng = np.random.default_rng(3)
    g = random_genome(rng)
    seen = {g.mutate(rng) for _ in range(80)}
    assert "drift" in seen                # always reachable
    assert seen <= {"drift", "add_conn", "add_node", "add_output",
                    "swap_fn", "toggle", "duplicate"}


def test_json_roundtrip():
    for g in (follower(), answerer(), splitter(),
              random_score_genome(np.random.default_rng(5))):
        d = g.to_dict()
        h = Genome.from_dict(d)
        assert h.to_dict() == d
        feeds = {"b": np.float32(1.0),
                 "s0": np.linspace(0, 1, 16, dtype=np.float32)}
        a = g.evaluate(feeds)
        b = h.evaluate(feeds)
        for k in a:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]))


def test_copy_independent():
    g = follower()
    h = g.copy()
    rng = np.random.default_rng(9)
    h.mutate(rng)
    assert h.to_dict() != g.to_dict()


def test_conn_must_point_forward():
    g = Genome()
    nid = g.add_hidden("sin")
    out = g.add_output("amp")
    with pytest.raises(AssertionError):
        g.conn(out, nid, 0.5)             # backwards: forbidden


def test_champions_catalogue_matches_builds():
    for name in CHAMPIONS:
        g = build_champion(name)
        assert isinstance(g, Genome)
        assert len(g.outputs) >= 1