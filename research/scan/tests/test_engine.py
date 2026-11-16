"""engine + measure: the loop and the independent judge (phase B)."""

import numpy as np
import pytest

from scan.engine import run_experiment, ARMS
from scan.measure import (ascii_map, plain_coverage, plain_features,
                          plain_radius)
from scan import render, viable, champion, random_genome


def test_viable_gate():
    assert not viable(np.zeros((44100, 2), np.float32))        # silence
    assert not viable(np.full((44100, 2), 0.3, np.float32))    # dc
    assert viable(render(champion("sequencer"), 2.0))


def test_plain_features_independent_of_echo():
    from echo.encode import DIM
    f = plain_features(render(champion("surfer"), 2.0))
    assert f.shape == (8,) and np.isfinite(f).all()
    assert f.shape != (DIM,)                                   # not echo's


def test_coverage_separates_countries():
    rng = np.random.default_rng(1)
    scans, tones = [], []
    while len(scans) < 3 or len(tones) < 3:
        g = random_genome(rng, seed=int(rng.integers(1 << 30)))
        x = render(g, 1.5)
        (scans if g.archetype() in ("scan", "sequence", "wash")
         else tones).append(plain_features(x))
    X = np.stack(scans)
    same = np.stack(tones[:3])
    assert plain_coverage(X)["pairwise"] > plain_coverage(same)["pairwise"]


def test_ascii_map_renders():
    feats = {"greedy": np.zeros((2, 8)), "novelty": np.ones((2, 8))}
    art = ascii_map(feats)
    assert "G" in art and "N" in art and "|" in art


def test_arms_are_the_four_memories():
    assert ARMS == ("greedy", "random", "cell", "novelty")


def test_experiment_smoke(tmp_path):
    """The whole demonstration, tiny: all four arms must run, produce
    independent measurements, mixtapes, journeys, and a report — the
    phase B acceptance."""
    pytest.importorskip("echo")
    summary = run_experiment(out=str(tmp_path / "run"), seeds=(0,),
                             gens=3, pop=6, births=4, verbose=False,
                             jobs=1)          # serial: tiny runs don't
                                              # amortize worker startup
    for arm in ("greedy", "random", "cell", "novelty"):
        r = summary["results"][arm]
        assert r["kinds_ever_mean"] >= 1
        assert r["runs"][0]["log"] and len(r["runs"][0]["log"]) == 3
        assert (tmp_path / "run" / f"{arm}_s0" / "mixtape.wav").exists()
        # the arms differ only in memory policy, never in budget
        for run in r["runs"]:
            for g in run["log"]:
                assert g["admits"] + g["rej_mc"] + g["rej_nov"] == 4
    report = (tmp_path / "run" / "report.md").read_text()
    assert "kinds ever" in report and "parent-recall" in report
    # the novelty arm admits at most half its viable children
    nov = summary["results"]["novelty"]["runs"][0]["log"]
    assert all(g["admits"] <= 2 + 1 for g in nov)   # keep = births // 2


def test_parallel_equals_serial(tmp_path):
    """The (arm, seed) grid never shares state, so running it across
    processes must produce bit-identical science — the speedup must
    not cost the experiment."""
    pytest.importorskip("echo")
    ser = run_experiment(out=str(tmp_path / "ser"), seeds=(0, 1), gens=2,
                         pop=5, births=3, verbose=False, jobs=1)
    par = run_experiment(out=str(tmp_path / "par"), seeds=(0, 1), gens=2,
                         pop=5, births=3, verbose=False, jobs=2)
    for arm in ("greedy", "random", "cell", "novelty"):
        for run_s, run_p in zip(ser["results"][arm]["runs"],
                                par["results"][arm]["runs"]):
            for a, b in zip(run_s["log"], run_p["log"]):
                assert {k: v for k, v in a.items() if k != "elapsed"} == \
                       {k: v for k, v in b.items() if k != "elapsed"}, \
                    f"{arm}.log differs beyond elapsed telemetry"
            for k in ("census_final", "kinds_ever_list",
                      "cells_cum_final", "coverage", "radius",
                      "parent_recall_pct"):
                assert run_s[k] == run_p[k], f"{arm}.{k} differs"
        # the arm-level mean population features (stripped from runs)
        assert ser["results"][arm]["feats_final"] == \
            par["results"][arm]["feats_final"], f"{arm}.feats differs"


def test_founder_identity_across_arms(tmp_path):
    """Same seed -> same founder in every arm (the fair comparison)."""
    pytest.importorskip("echo")
    from scan.engine import run_arm
    from pathlib import Path
    outs = {arm: run_arm(arm, 0, 1, 4, 2, Path(tmp_path) / arm,
                         verbose=False) for arm in ("random", "novelty")}
    for arm, r in outs.items():
        assert r["log"], f"{arm} produced no generations"