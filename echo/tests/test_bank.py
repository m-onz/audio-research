"""bank: exact k-NN, novelty, persistence, normalization."""

import os
import tempfile

import numpy as np

from echo.bank import EchoBank, EchoMetric
from echo.encode import DIM


def _vecs(n=20, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, DIM))


def test_knn_matches_brute_force():
    bank = EchoBank(":memory:")
    X = _vecs(30)
    bank.add_many(X)
    v = _vecs(3, seed=99)[0]
    got = bank.knn(v, k=7)
    metric = bank.metric
    bf = np.sort(metric.dists(v, X))
    assert np.allclose([g["dist"] for g in got], bf[:7], atol=1e-5)
    assert got[0]["dist"] <= got[-1]["dist"]


def test_novelty_is_knn_mean():
    bank = EchoBank(":memory:")
    X = _vecs(25)
    bank.add_many(X)
    v = X[0] + 0.001
    nov = bank.novelty(v, k=5, )
    bf = np.sort(bank.metric.dists(v, X))[:5]
    assert abs(nov - bf.mean()) < 1e-6
    # empty bank: infinite novelty (nothing heard yet)
    assert EchoBank(":memory:").novelty(v) == float("inf")


def test_min_dist():
    bank = EchoBank(":memory:")
    bank.add_many(_vecs(10))
    v = np.zeros(DIM)
    d = bank.min_dist(v)
    assert d > 0 and np.isfinite(d)


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "b.sqlite")
        bank = EchoBank(path)
        X = _vecs(12)
        bank.fit(X, save=True)
        bank.add_many(X, [{"file": f"{i}.wav"} for i in range(12)], gen=3)
        bank.close()

        bank2 = EchoBank(path)
        assert len(bank2) == 12
        assert bank2.metas[0]["file"] == "0.wav"
        assert bank2.gens[0] == 3
        # metric survived: mu fitted on X
        assert np.allclose(bank2.metric.mu, X.mean(axis=0), atol=1e-6)
        v = X[0]
        assert bank2.min_dist(v) < bank2.metric.dists(v, X[1:].mean(axis=0))


def test_fit_tames_dead_axes():
    """A dim that only jitters by 1e-4 across the corpus must not
    explode distances (sigma floored); a real perturbation on it
    should barely move the needle."""
    X = _vecs(50)
    X[:, 100] = 7.0 + _vecs(50, seed=7)[:, 100] * 1e-4   # barely alive axis
    metric = EchoMetric().fit(X)
    assert metric.sigma[100] >= 1e-3                    # floored, not 1e-4
    a = X[0].copy()
    b = a.copy()
    b[100] += 1e-3                                      # real jitter, dead axis
    assert metric.dist(a, b) < 0.01
    # sanity: the same jitter on a live axis also stays small but the
    # floor is exactly what keeps the dead one from exploding
    unfitted = EchoMetric()                              # sigma = 1 everywhere
    assert unfitted.dist(a, b) < 0.01


def test_dim_mismatch_guard():
    bank = EchoBank(":memory:")
    try:
        bank.add(np.zeros(100))
        assert False, "should have raised"
    except ValueError:
        pass


def test_roles_filter():
    bank = EchoBank(":memory:")
    bank.add_many(_vecs(6, seed=5), role="corpus")
    bank.add_many(_vecs(4, seed=6), role="archive")
    assert len(bank.knn(np.zeros(DIM), k=100, roles="corpus")) == 6
    assert len(bank.knn(np.zeros(DIM), k=100, roles="archive")) == 4
    assert len(bank) == 10