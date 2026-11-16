"""genome tests: invariants, mutations, json, evaluation."""

import numpy as np
import pytest

from prism.genome import FUNCTIONS, INPUTS, INPUT_IDS, Genome, random_genome
from prism.champions import champion


def _order_ok(g: Genome) -> bool:
    pos = {nid: i for i, nid in enumerate(g.order)}
    return all(pos[c.src] < pos[c.dst] for c in g.conns if c.on)


def test_inputs_present():
    g = Genome()
    assert all(nid in g.nodes for nid in INPUT_IDS.values())
    assert g.order[: len(INPUTS)] == [INPUT_IDS[n] for n in INPUTS]


def test_evaluate_shapes():
    g = champion("glassvoice")
    scalar = g.evaluate({"x": 0.3, "y": 0.1})
    assert np.asarray(scalar["wave"]).shape == ()
    arr = g.evaluate({"x": np.zeros(7, np.float32),
                      "y": np.ones(7, np.float32)})
    assert np.asarray(arr["wave"]).shape == (7,)


def test_evaluate_defaults():
    g = champion("tide")
    out = g.evaluate({})                     # everything defaults
    assert np.isfinite(out["field"]).all()


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
        g = random_genome(rng, n_hidden=int(rng.integers(1, 4)))
        for _ in range(12):
            getattr(g, op)(rng)
            assert _order_ok(g), f"{op} broke the feed-forward order"
        out = g.evaluate({"x": np.linspace(0, 1, 64, dtype=np.float32)})
        for v in out.values():
            assert np.isfinite(v).all(), f"{op} produced non-finite output"


def test_add_node_splits_connection():
    rng = np.random.default_rng(0)
    g = champion("tide")
    n_conns = len(g.conns)
    op = g.mut_add_node(rng)
    assert op == "add_node"
    assert len(g.conns) == n_conns + 2
    assert len(g.nodes) == 8 + 5 + 2 + 1   # inputs + hiddens + outs + new


def test_add_output_grows_alphabet():
    rng = np.random.default_rng(1)
    g = champion("glassvoice")
    before = set(g.outputs)
    g.mut_add_output(rng)
    assert set(g.outputs) > before
    out = g.evaluate({"x": np.zeros(4, np.float32)})
    assert all(np.isfinite(v).all() for v in out.values())


def test_duplicate_copies_and_hooks():
    rng = np.random.default_rng(2)
    g = champion("bellfield")
    n_before = len(g.nodes)
    op = g.mut_duplicate(rng)
    assert op == "duplicate"
    assert len(g.nodes) > n_before
    assert _order_ok(g)
    out = g.evaluate({"x": np.linspace(0, 1, 32, dtype=np.float32)})
    assert np.isfinite(out["amp"]).all()


def test_mutate_dispatch():
    rng = np.random.default_rng(3)
    g = random_genome(rng)
    seen = {g.mutate(rng) for _ in range(80)}
    assert "drift" in seen                # always reachable
    assert seen <= {"drift", "add_conn", "add_node", "add_output",
                    "swap_fn", "toggle", "duplicate"}


def test_json_roundtrip():
    g = champion("bellfield")
    g2 = Genome.from_dict(g.to_dict())
    assert g2.to_dict() == g.to_dict()
    x = np.linspace(0, 1, 33, dtype=np.float32)
    a = g.evaluate({"x": x}, names=["amp"])
    b = g2.evaluate({"x": x}, names=["amp"])
    assert np.array_equal(a["amp"], b["amp"])


def test_copy_independent():
    g = champion("tide")
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