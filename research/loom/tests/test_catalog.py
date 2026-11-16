import numpy as np
import pytest

from loom.catalog import Catalog, order_ids
from loom.chunkers import fixed
from loom.rng import Seeded
from loom.sources import open_sources
from loom.store import MemoryStore, open_store


def _catalog(corpus_dir, seconds=1.0):
    srcs = open_sources(corpus_dir)
    cat = Catalog(srcs, MemoryStore())
    cat.add_refs(fixed(srcs, Seeded(0), seconds=seconds))
    return cat


def test_catalog_build(corpus_dir):
    cat = _catalog(corpus_dir)
    # 6s clicks -> 6, 2.0s dark -> 2, 1.5s bright -> 2 (1.0 + 0.5), 1s sine -> 1
    assert cat.n == 11
    assert cat.ranks().shape == (cat.n, 14)


def test_ranks_are_01(corpus_dir):
    cat = _catalog(corpus_dir)
    r = cat.ranks()
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert r[:, 0].max() == pytest.approx(1.0)  # loudest chunk exists


def test_query_quantiles(corpus_dir):
    cat = _catalog(corpus_dir)
    loud = cat.query("rms>0.8")
    quiet = cat.query("rms<0.2")
    assert set(loud) & set(quiet) == set()
    assert len(loud) >= 1 and len(quiet) >= 1


def test_query_and_tags(corpus_dir):
    cat = _catalog(corpus_dir)
    assert cat.query("rms>0.5 and air<0.5") is not None
    with pytest.raises(ValueError):
        cat.query("notafeature>0.5")
    assert cat.query("") == list(range(cat.n))


def test_query_tag_after_derive(corpus_dir):
    cat = _catalog(corpus_dir)
    parent = cat.chunks[0]
    audio = parent.materialize(cat.sources, cat.store)
    cid = cat.add_derived(parent, audio * 0.5, tag="dark", lineage=("gain",))
    assert cat.query("tag=dark") == [cid]
    d = cat.chunks[cid]
    assert d.parent == 0 and d.lineage == ("gain",)


def test_cluster_labels(corpus_dir):
    cat = _catalog(corpus_dir)
    labels = cat.cluster(4)
    assert len(labels) == cat.n
    assert len(set(labels)) >= 2


def test_order_kinds(corpus_dir):
    cat = _catalog(corpus_dir)
    ids = list(range(cat.n))
    for kind in ["random", "by:rms", "farthest", "chase", "markov", "markov:4"]:
        out = order_ids(cat, ids, kind=kind, rng=Seeded(0))
        assert sorted(out) == sorted(ids), f"{kind} lost chunks"
    # n > len(ids) cycles
    out = order_ids(cat, ids[:3], kind="random", rng=Seeded(0), n=10)
    assert len(out) == 10


def test_store_backends(tmp_path, corpus_dir):
    mem = open_store("memory")
    disk = open_store("disk", root=tmp_path / "store")
    x = np.zeros((100, 2), np.float32)
    for s in (mem, disk):
        s.put("k", x)
        y = s.get("k")
        assert y.shape == (100, 2)
    with pytest.raises(ValueError):
        open_store("cloud")

def test_harmonic_ordering(corpus_dir):
    """Harmonic walk: permutation + consecutive intervals tend consonant."""
    from loom.rng import Seeded
    from loom.catalog import order_ids
    cat = _catalog(corpus_dir, seconds=1.0)
    # synthesize pitched knowledge: pitched for most chunks, leave two
    # original (noise -> pitch 0) to exercise the unpitched append path
    import numpy as np
    from dataclasses import replace
    rng = np.random.default_rng(0)
    for i, c in enumerate(cat.chunks[:-2]):
        f = c.features
        cat.chunks[i] = replace(c, features=replace(f, pitch=float(110 * 2 ** (rng.integers(0, 25) / 12)), pitch_conf=0.9))
    cat._invalidate()
    ids = cat.query("all")
    pitched = [i for i in ids if cat.chunks[i].features.pitch > 40]
    assert len(pitched) >= 8
    out = order_ids(cat, ids, kind="harmonic", rng=Seeded(0))
    assert sorted(out) == sorted(ids)  # permutation, nothing lost
    # consecutive intervals: majority should be consonant classes
    cons = {0, 7, 5, 4, 3, 8, 9, 12}
    good = 0
    checked = 0
    for a, b in zip(out[:-1], out[1:]):
        fa = cat.chunks[a].features.pitch
        fb = cat.chunks[b].features.pitch
        if fa <= 40 or fb <= 40:
            continue  # unpitched stragglers appended at the end
        checked += 1
        d = int(round(abs(12 * np.log2(fb / fa))))
        good += (d % 12 in cons) or (d in (12, 24))
    assert checked >= 6
    # greedy walk on a tiny random pool: consonance-majority, not perfection
    assert good >= 0.55 * checked
