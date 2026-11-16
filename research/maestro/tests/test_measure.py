"""measure tests — the independent judge stays independent."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from echo.probes import SR

from maestro.champions import answerer, follower
from maestro.measure import (anatomy, atlas_png, complexity, plain_coverage,
                             plain_features, plain_radius, score_coverage,
                             score_features)
from maestro.score import score_from
from maestro.scaffold import scaffold_from_array

TICK_HZ = 8.0


def click_spine(onsets_s, seconds=3.5, sr=SR):
    x = np.zeros(int(seconds * sr), np.float32)
    for t in onsets_s:
        p0, klen = int(t * sr), int(0.02 * sr)
        x[p0: p0 + klen] += np.exp(-np.linspace(0, 6, klen)) * 0.8
    return x


def scaf():
    return scaffold_from_array(click_spine([0.5, 1.5, 2.5]),
                               sr=SR, bpm=None, tick_hz=TICK_HZ)


# ------------------------------------------------------------------ audio


def test_plain_features_bounded():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((SR * 2, 2)) * 0.2).astype(np.float32)
    f = plain_features(x)
    assert f.shape == (8,) and np.isfinite(f).all()
    assert f.min() >= 0.0 and f.max() <= 1.01


def test_plain_coverage_and_radius():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (10, 8))
    cov = plain_coverage(X)
    assert cov["n"] == 10 and 0 < cov["pairwise"] < 2
    r = plain_radius(X, np.zeros(8))
    assert r >= 0


# ------------------------------------------------------------------ score


def test_score_features_known_answers():
    sc = score_from(follower(), scaf(), mode="strike")
    f = score_features(sc)
    assert f.shape == (8,) and np.isfinite(f).all()
    assert 0.0 <= f.min() and f.max() <= 1.01
    assert f[0] == pytest.approx(1 / 8)          # one voice of eight
    assert f[2] == pytest.approx(1.0)            # step: all velocities 1.0
    # the answerer is sparse relative to the follower (mid-decay only)
    fa = score_features(score_from(answerer(), scaf(), mode="strike"))
    assert fa[1] < f[1]                          # density axis


def test_score_coverage_cells():
    a = score_features(score_from(follower(), scaf(), mode="strike"))
    b = score_features(score_from(answerer(), scaf(), mode="strike"))
    cov = score_coverage(np.stack([a, b]))
    assert cov["n"] == 2 and cov["cells"] >= 1


# ------------------------------------------------------------------ genome


def test_complexity_and_anatomy():
    g = follower()
    cx = complexity(g)
    assert cx["nodes"] == 1 and cx["conns"] == 1 and cx["outputs"] == 1
    an = anatomy(g)
    assert an["senses"] == ["s0"]                # purity: scaffold only
    assert an["outputs"] == ["pulse"]
    from maestro.genome import INPUT_IDS
    from maestro.genome import Genome
    g2 = Genome()
    h = g2.add_hidden("sin")
    g2.conn(INPUT_IDS["t"], h, 1.0)              # impure: time wired
    assert "t" in anatomy(g2)["senses"]


# ------------------------------------------------------------------ atlas


def test_atlas_png(tmp_path):
    scores = [score_from(follower(), scaf(), mode="strike"),
              score_from(answerer(), scaf(), mode="strike")]
    out = atlas_png(tmp_path / "atlas.png", scores)
    img = np.asarray(Image.open(out))
    assert img.ndim == 2 and img.shape[0] > 0 and img.shape[1] > 0
    atlas_png(tmp_path / "atlas2.png", scores)   # deterministic pixels
    assert (tmp_path / "atlas2.png").read_bytes() == \
        Path(out).read_bytes()