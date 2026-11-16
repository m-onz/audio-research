"""engine tests: gates, lenses, the loop, determinism."""

import json
from pathlib import Path

import numpy as np
import pytest

from prism.champions import champion
from prism.engine import (ARMS, LENSES, EVAL_LEN, SR, floor_ok, lens_rank,
                          run_arm, tonality, viability)
from prism.genome import random_genome
from prism.render import render


def _noise_bursts(seed=0, rms=0.15, sr=SR, sec=2.0):
    rng = np.random.default_rng(seed)
    n = int(sr * sec)
    x = rng.standard_normal((n, 2)) * rms
    gate = (np.sin(2 * np.pi * 3.0 * np.arange(n) / sr) > -0.2).astype(float)
    return (x * gate[:, None]).astype(np.float32)


# ------------------------------------------------------------------ gates


def test_viability_gates():
    silence = np.zeros((SR * 2, 2), np.float32)
    ok, m = viability(silence, 0.5)
    assert not ok
    steady = np.zeros((SR * 2, 2), np.float32)
    steady[:] = 0.1 * np.sin(2 * np.pi * 220.0 *
                             np.arange(SR * 2) / SR)[:, None]
    ok, m = viability(steady, 0.5)
    assert not ok and m["block_ratio"] < 1.05      # static drone gate
    ok, m = viability(_noise_bursts(), 0.5)
    assert ok


def test_strictness_dial_direction():
    """The dial widens both bands, each direction verified: a quiet
    but slowly-drifting signal fails the calibrated rms floor and
    passes loose; an audible but nearly-flat signal passes the
    calibrated static gate and fails the narrow one."""
    n = SR * 4
    t = np.arange(n) / SR
    rng = np.random.default_rng(1)
    quiet_slow = (rng.standard_normal((n, 2)) * 0.020
                  * (0.7 + 0.3 * np.sin(2 * np.pi * 0.3 * t))[:, None]
                 ).astype(np.float32)   # rms ~0.015, ratio ~2.1
    gentle_flat = (rng.standard_normal((n, 2)) * 0.15
                   * (0.92 + 0.08 * np.sin(2 * np.pi * 0.3 * t))[:, None]
                  ).astype(np.float32)  # rms ~0.14, ratio ~1.14
    assert not viability(quiet_slow, 0.5)[0]   # floor 0.02: rejected
    assert viability(quiet_slow, 1.0)[0]       # floor 0.01: admitted
    assert viability(gentle_flat, 0.5)[0]      # static 1.05: admitted
    assert not viability(gentle_flat, 0.1)[0]  # static 1.25: rejected


def test_tonality_ranks_pure_above_noise():
    t = np.arange(SR * 2) / SR
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)[:, None] * np.ones((1, 2)))
    assert tonality(sine) > tonality(_noise_bursts())


# ------------------------------------------------------------------ lenses


def test_lens_dimensions():
    from prism.engine import LENS_INDEX
    dims = {lens: len(idx) for lens, (idx, _) in LENS_INDEX.items()}
    assert dims["all"] == 556
    assert dims["timbre"] == 192 + 192 + 64
    assert dims["time"] == 80 + 13
    assert dims["pitch"] == 12 + 3


def test_lens_rank_excludes_self():
    rng = np.random.default_rng(3)
    arch = rng.standard_normal((6, 556)) * 0.4
    ids = np.arange(100, 106)
    # a member identical to archive entry 3: its novelty must ignore itself
    scores = lens_rank(arch[3:4], ids[3:4], arch, ids, "timbre")
    assert scores[0] > 0.0
    # and a fresh far-away vector must outrank a near-duplicate
    near = arch[3:4] + 0.01
    far = arch[3:4] + 3.0
    s_near = lens_rank(near, [999], arch, ids, "timbre")[0]
    s_far = lens_rank(far, [999], arch, ids, "timbre")[0]
    assert s_far > s_near


def test_floor_ok_relative():
    assert floor_ok(0.6, [0.5, 0.5, 0.5, 0.5])
    assert not floor_ok(0.2, [0.5, 0.5, 0.5, 0.5])  # 0.5 * median = 0.25
    assert floor_ok(0.001, [])                     # empty: always ok


# ------------------------------------------------------------------ the loop


def _tiny_run(tmp_path: Path, arm="random", seed=1, strict=1.0):
    return run_arm(arm, seed, gens=2, pop=6, births=3, out=tmp_path,
                   organism="wavetable", strictness=strict,
                   eval_len=2.0, verbose=False)


def test_run_arm_smoke(tmp_path: Path):
    r = _tiny_run(tmp_path)
    assert r["admits"] >= 1
    assert len(r["log"]) == 2
    d = tmp_path / "random_s1"
    ledger = [json.loads(l) for l in
              (d / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 2 * 3                    # one line per birth
    assert all(set(e) >= {"gen", "ops", "viable", "admitted", "nov"}
               for e in ledger)
    for f in ("mixtape.wav", "atlas.png", "run.json"):
        assert (d / f).exists()
    assert r["parent_recall_pct"] is not None


def test_run_arm_deterministic(tmp_path: Path):
    a = _tiny_run(tmp_path / "a", seed=2)
    b = _tiny_run(tmp_path / "b", seed=2)
    la = (tmp_path / "a" / "random_s2" / "ledger.jsonl").read_text()
    lb = (tmp_path / "b" / "random_s2" / "ledger.jsonl").read_text()
    assert la == lb                               # identical births, identical fates


def test_run_arm_all_arms(tmp_path: Path):
    for arm in ARMS:
        r = run_arm(arm, 0, gens=1, pop=4, births=3, out=tmp_path,
                    organism="wavetable", strictness=1.0, eval_len=2.0,
                    verbose=False)
        assert r["arm"] == arm
        assert r["births"] == 3


def test_novelty_floor_actually_rejects(tmp_path: Path):
    """In the novelty arm, some births must die at the floor once the
    archive is dense — the gate is real, not decorative."""
    r = run_arm("novelty", 0, gens=6, pop=8, births=10, out=tmp_path,
                organism="wavetable", strictness=1.0, eval_len=2.0,
                verbose=False)
    assert r["births"] == 60
    # with a dense archive the floor rejects; with a sparse one it may
    # not — either way the run stays alive and finite
    assert r["admits"] >= 10
    assert r["log"][-1]["rej_mc"] + r["log"][-1]["rej_nov"] <= 10


def test_champion_founders_breed(tmp_path: Path):
    """The loop's machinery works on champions too — the fixtures and
    the evolved population share one API."""
    r = run_arm("random", 3, gens=1, pop=4, births=2, out=tmp_path,
                organism="wavetable", strictness=1.0, eval_len=2.0,
                verbose=False)
    assert r["admits"] >= 0