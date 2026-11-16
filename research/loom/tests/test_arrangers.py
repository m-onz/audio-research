import pytest

from loom import Seeded, open_sources
from loom.arrangers import _euclid_pattern, cloud, euclid, layers, make_arranger, sequence
from loom.catalog import Catalog
from loom.chunkers import fixed
from loom.processors import build_chain
from loom.store import MemoryStore


def _ctx(corpus_dir, seconds=0.5):
    srcs = open_sources(corpus_dir)
    cat = Catalog(srcs, MemoryStore())
    cat.add_refs(fixed(srcs, Seeded(0), seconds=seconds))
    return cat, srcs, list(range(cat.n))


def test_euclid_pattern_spread():
    p = _euclid_pattern(8, 3)
    assert sum(p) == 3
    assert p == [1, 0, 0, 1, 0, 0, 1, 0] or p == [1, 0, 1, 0, 0, 1, 0, 0]
    assert _euclid_pattern(4, 0) == [0, 0, 0, 0]


def test_sequence_fills_length(corpus_dir):
    cat, srcs, ids = _ctx(corpus_dir)
    evs = sequence(cat, ids, 10.0, srcs.sr, Seeded(1), {"gap": 0.0}, None)
    assert evs[0].t == 0.0
    assert evs[-1].t < 10.0
    # chunks tile time: each event starts at prev start + prev dur
    for a, b in zip(evs[:-1], evs[1:]):
        assert b.t == pytest.approx(a.t + a.dur, abs=1e-4)


def test_sequence_negative_gap_overlaps(corpus_dir):
    cat, srcs, ids = _ctx(corpus_dir)
    evs = sequence(cat, ids, 10.0, srcs.sr, Seeded(1), {"gap": -0.2}, None)
    for a, b in zip(evs[:-1], evs[1:]):
        assert b.t < a.t + a.dur


def test_cloud_density_curve(corpus_dir):
    cat, srcs, ids = _ctx(corpus_dir)
    evs = cloud(cat, ids, 10.0, srcs.sr, Seeded(2), {"density": 600.0, "pan_spread": 0.5}, None)
    assert len(evs) > 10
    assert all(e.t < 10.0 for e in evs)
    assert any(e.pan != 0 for e in evs)


def test_euclid_grid(corpus_dir):
    cat, srcs, ids = _ctx(corpus_dir)
    evs = euclid(cat, ids, 8.0, srcs.sr, Seeded(3), {"bpm": 120, "steps": 8, "pulses": 3}, None)
    # bar = 8 sixteenth-steps @120bpm = 1.0 s -> 8 bars x 3 pulses
    assert len(evs) == 3 * int(8.0 / 1.0)
    step = 60.0 / 120 / 4
    assert evs[0].t == 0.0
    assert all(e.truncate for e in evs)
    # events sit on the grid
    for e in evs:
        assert (e.t / step) == pytest.approx(round(e.t / step), abs=1e-6)


def test_layers_voices(corpus_dir):
    cat, srcs, ids = _ctx(corpus_dir)
    params = {"voices": [
        {"select": "rms>0.5", "arranger": "euclid", "params": {"bpm": 100, "steps": 4, "pulses": 2}},
        {"select": "all", "arranger": "cloud", "params": {"density": 60}, "gain": 0.3},
    ]}
    evs = layers(cat, ids, 6.0, srcs.sr, Seeded(4), params, None)
    notes = {e.note for e in evs}
    assert notes == {"voice0", "voice1"}
    # gains applied
    assert all(e.gain <= 1.0 for e in evs)


def test_layers_empty_voice_silent(corpus_dir):
    cat, srcs, ids = _ctx(corpus_dir)
    params = {"voices": [{"select": "tag=nosuchtag", "arranger": "sequence", "params": {}}]}
    evs = layers(cat, ids, 4.0, srcs.sr, Seeded(5), params, None)
    assert evs == []


def test_make_arranger_registry():
    assert callable(make_arranger("sequence"))
    with pytest.raises(ValueError):
        make_arranger("waltz")

def test_layers_voice_chains(corpus_dir):
    """A voice with its own `chain` gets it; without, the shared chain."""
    cat, srcs, ids = _ctx(corpus_dir)
    shared = build_chain(["gain"])  # gain(amount=1.0) no-op
    params = {"voices": [
        {"select": "all", "arranger": "cloud", "params": {"density": 200},
         "chain": [["gain", {"amount": 0.05}]]},   # near-silent voice chain
    ]}
    evs = layers(cat, ids, 4.0, srcs.sr, Seeded(9), params, shared)
    assert evs
    import numpy as np
    probe = np.ones((100, 2), np.float32)
    for e in evs:
        assert e.chain is not None
        out = e.chain.run(probe, sr=srcs.sr, t=0.0, rng=Seeded(0))
        # the voice chain's gain(0.05) must have been applied per event
        assert float(np.max(np.abs(out))) == pytest.approx(0.05, abs=0.01)
