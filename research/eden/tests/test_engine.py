"""engine: the loop, both arms, tiny smoke."""

import numpy as np
import pytest

from eden.engine import run_experiment, viable


def test_viable_gate():
    assert not viable(np.zeros((44100 * 4, 2), np.float32))     # silence
    assert not viable(np.full((44100 * 4, 2), 0.3, np.float32))  # dc


def test_experiment_smoke(tmp_path):
    """Both arms must run end to end: crossover, render, encode,
    novelty admission, telemetry, artifacts, report."""
    pytest.importorskip("echo")
    summary = run_experiment(out=str(tmp_path / "run"), seeds=(0,),
                             gens=2, pop=4, births=3, verbose=False)
    for arm in ("random", "novelty"):
        r = summary["results"][arm]
        assert r["runs"][0]["log"] and len(r["runs"][0]["log"]) == 2
        assert (tmp_path / "run" / f"{arm}_s0" / "mixtape.wav").exists()
    report = (tmp_path / "run" / "report.md").read_text()
    assert "cells ever" in report
    # complexification telemetry present
    assert "agents_mean" in summary["results"]["random"]["runs"][0]["log"][0]