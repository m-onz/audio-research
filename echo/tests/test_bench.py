"""bench: the measurement machinery itself (stat fns, probes, arms)."""

import numpy as np
import pytest

from echo.bench import (ab_experiment, d_prime, locality, roc_auc,
                        spearman)
from echo.encode import DIM, encode
from echo.probes import contrasts, perturbations


def test_roc_auc_extremes():
    pos = [0.1, 0.2, 0.3]
    neg = [0.7, 0.8, 0.9]
    assert roc_auc(pos, neg) == 1.0
    assert roc_auc(neg, pos) == 0.0
    assert roc_auc([1.0], [1.0]) == 0.5


def test_d_prime_sign():
    assert d_prime([1, 1, 1], [5, 5, 5]) > 0
    assert d_prime([5, 5, 5], [1, 1, 1]) < 0


def test_spearman_perfect():
    assert abs(spearman([1, 2, 3], [10, 20, 30]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3], [30, 20, 10]) + 1.0) < 1e-9


def test_contrasts_are_distinct():
    syn = contrasts(44100, seconds=1.0)
    assert len(syn) >= 8
    vecs = {k: encode(np.asarray(v)) for k, v in syn.items()}
    keys = list(vecs)
    d_cross = np.mean([np.linalg.norm(vecs[a] - vecs[b])
                       for a in keys[:4] for b in keys[:4] if a != b])
    d_self = 0.0
    assert d_cross > 0.1


def test_perturbations_stay_close():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.3, (44100, 2)).astype(np.float32)
    perts = perturbations(x, 44100, rng,
                          names=["gain", "shift", "noise_floor", "eq_tilt"])
    base = encode(x)
    for name, p in perts.items():
        d = np.linalg.norm(base - encode(p))
        assert d < np.linalg.norm(base) * 0.5, name


def test_locality_on_synthetic_rows():
    """Fabricated corpus: two takes of two chunks each, identical within
    a take — the truth must be retrieved."""
    rng = np.random.default_rng(3)
    rows = []
    for take in ("a", "b"):
        base = rng.normal(0, 1, DIM)
        for k in range(2):
            rows.append({"file": f"take{take}_{k:04d}.wav",
                         "start_s": take == "b" and 1000.0 or float(k * 5),
                         "dur_s": 5.0,
                         "vec": base + rng.normal(0, 0.05, DIM)})
    res = locality(rows)
    assert res["n_queries"] == 4
    assert res["median_rank"] == 1.0
    assert res["recall_at_1"] == 1.0


def test_extras_plus_anatomy():
    """The librosa tier: same core blocks + the four extras, and the
    anatomy is pinned (592 = 556 core + 36 extras)."""
    pytest.importorskip("librosa")
    from echo.extras import DIM_PLUS, LAYOUT_PLUS, encode_plus
    assert DIM_PLUS == 592
    assert DIM_PLUS == sum(d for _, d, _ in LAYOUT_PLUS)
    x = np.random.default_rng(1).normal(0, 0.3, (44100, 2)).astype(np.float32)
    v = encode_plus(x)
    assert v.shape == (DIM_PLUS,) and np.isfinite(v).all()
    assert np.array_equal(v, encode_plus(x))            # deterministic


def test_ab_experiment_smoke():
    """The harness runs end to end (needs the womb sibling install)."""
    pytest.importorskip("womb.breed")
    res = ab_experiment(seeds=(0,), gens=2, pop=6, births=3, pairs=1,
                        seconds=1.0, strictness=0.5, verbose=False)
    for arm in ("drift", "cell", "novelty"):
        assert arm in res
        runs = res[arm]["runs"]
        assert runs and runs[0].get("log") is not None
        assert len(runs[0]["log"]) == 2