"""engine + measure: the loop and the independent judge."""

import numpy as np
import pytest

from maze.engine import run_experiment, viable
from maze.measure import ascii_map, plain_coverage, plain_features, plain_radius
from maze.voice import render, VoiceGenome


def test_viable_gate():
    assert not viable(np.zeros((44100, 2), np.float32))       # silence
    assert not viable(np.full((44100, 2), 0.3, np.float32))    # dc
    x = render(VoiceGenome(bumps=[[440.0, 0.9, 0.05]], click_rate=4.0,
                           click_amp=0.4, seed=2), 2.0)
    assert viable(x)


def test_plain_features_independent_of_echo():
    from echo.encode import DIM
    x = render(VoiceGenome(bumps=[[440.0, 0.8, 0.02]], noise_amp=0.1,
                           seed=4), 2.0)
    f = plain_features(x)
    assert f.shape == (8,) and np.isfinite(f).all()
    assert f.shape != (DIM,)                                   # not echo's


def test_coverage_and_radius():
    tones = [plain_features(render(VoiceGenome(
        bumps=[[f, 0.8, 0.02]], seed=1), 1.0)) for f in (220, 440, 880)]
    washes = [plain_features(render(VoiceGenome(
        noise_amp=0.5 + 0.2 * i, seed=1 + i), 1.0)) for i in range(3)]
    same = np.stack([plain_features(render(VoiceGenome(
        bumps=[[440.0, 0.8, 0.02]], seed=1), 1.0))] * 3)
    cov_wash = plain_coverage(np.stack(washes))
    cov_same = plain_coverage(same)
    assert cov_wash["pairwise"] > 3 * cov_same["pairwise"]
    assert plain_radius(np.stack(washes), tones[0]) > \
        plain_radius(same, tones[0])


def test_ascii_map_renders():
    feats = {"greedy": np.zeros((2, 8)), "novelty": np.ones((2, 8))}
    art = ascii_map(feats)
    assert "G" in art and "N" in art and "|" in art


def test_experiment_smoke(tmp_path):
    """The whole demonstration, tiny: all four arms must run, produce
    independent measurements, mixtapes, and a report."""
    pytest.importorskip("echo")
    summary = run_experiment(out=str(tmp_path / "run"), seeds=(0,),
                             gens=3, pop=6, births=4, verbose=False)
    for arm in ("greedy", "random", "cell", "novelty"):
        r = summary["results"][arm]
        assert r["kinds_ever_mean"] >= 1
        assert r["runs"][0]["log"] and len(r["runs"][0]["log"]) == 3
        assert (tmp_path / "run" / f"{arm}_s0" / "mixtape.wav").exists()
    report = (tmp_path / "run" / "report.md").read_text()
    assert "kinds ever" in report and "parent-recall" in report