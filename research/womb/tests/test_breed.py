"""breed tests: the MCC loop, the strictness dial, the drift floor, and
the listenable atlas. Phase B acceptance from CRYPTID.md."""

import json

import numpy as np
import pytest

import womb
from womb.breed import (GRIDS, descriptors, distance, drift_eps, montage,
                        mutate_ear, mutate_process, random_ear, random_process,
                        report_run, seed_run, signature)
from womb.cryptid import render, viability, law_channels
from womb.cryptid import build_champion

SR = 44100


def test_strictness_bands():
    """Soros' dial: loose lowers the bars, strict raises them."""
    sil = np.zeros((SR, 2), np.float32)
    assert not viability(sil, SR, strictness=0.5)["viable"]
    assert not viability(sil, SR, strictness=0.1)["viable"]
    b_loose = viability(sil, SR, strictness=0.1)["bands"]
    b_mid = viability(sil, SR, strictness=0.5)["bands"]
    b_strict = viability(sil, SR, strictness=1.0)["bands"]
    assert b_loose["rms_lo"] < b_mid["rms_lo"] < b_strict["rms_lo"]
    assert b_loose["alive_lo"] < b_mid["alive_lo"] < b_strict["alive_lo"]
    assert drift_eps(0.1) < drift_eps(0.5) < drift_eps(1.0)


def test_descriptors_bounded_and_deterministic():
    st = render(build_champion("clickstorm"), length=2.0)
    d1 = descriptors(st["audio"], SR)
    d2 = descriptors(st["audio"], SR)
    assert d1 == d2
    for k, v in d1.items():
        assert 0.0 <= v <= 1.0, k
    # champions land in different cells on every grid
    cells = {}
    for name in ("chimera", "clickstorm", "jerk_voice"):
        st = render(build_champion(name), length=2.0)
        d = descriptors(st["audio"], SR)
        cells[name] = {g: (int(d[a] * 8), int(d[b] * 8)) for g, (a, b) in GRIDS.items()}
    for g in GRIDS:
        assert len({cells[n][g] for n in cells}) > 1, g


def test_signature_and_drift_floor():
    st_a = render(build_champion("jerk_voice"), length=2.0)
    st_b = render(build_champion("clickstorm"), length=2.0)
    s1 = signature(st_a["audio"], SR)
    s2 = signature(st_a["audio"], SR)
    assert distance(s1, s2) < 1e-9                      # self-distance zero
    assert distance(s1, signature(st_b["audio"], SR)) > 0.02


def test_process_mutation_invariants():
    rng = np.random.default_rng(3)
    proc = random_process(rng)
    for _ in range(20):
        child, _ = mutate_process(rng, proc)
        assert 1 <= len(child["nodes"]) <= 6
        for e in child["edges"]:                          # edges stay valid
            assert 0 <= e["src"] < len(child["nodes"])
            assert 0 <= e["dst"] < len(child["nodes"])
            assert e["ch"] < law_channels(child["nodes"][e["src"]]["law"])
        for node in child["nodes"]:                       # osc_pop stays consistent
            if node["law"] == "osc_pop":
                assert node["params"]["n"] % node["params"]["windows"] == 0
                assert 4 <= node["params"]["radius"] <= node["params"]["n"] // 2
        proc = child


def test_ear_mutation_bounds():
    rng = np.random.default_rng(5)
    ear = random_ear(rng)
    for _ in range(20):
        ear, _ = mutate_ear(rng, ear)
        assert 1 <= len(ear) <= 8
        for op in ear:
            assert op["op"] in ("tap", "diff", "embed", "events", "chimera")


def test_seed_run_smoke(tmp_path):
    """The loop runs, checkpoints, logs, and admits at mid-strictness."""
    out = tmp_path / "crypt1"
    r = seed_run(out=out, pop=6, ears=4, gens=2, strictness=0.5,
                 pairs=2, seconds=1.5, champion_seeds=2, seed=1)
    state = json.loads((out / "state.json").read_text())
    archive = json.loads((out / "archive.json").read_text())
    assert len(state["log"]) == 2
    assert r["specimens"] == len(archive["entries"]) > 0
    assert sum(g["proc_admissions"] for g in state["log"]) > 0
    for g in GRIDS:
        assert (out / f"atlas-{g}.png").exists()
    # every archive entry has a renderable specimen + a thumbnail + a cell,
    # and its STORED ear still hears it alive (regression: the founder loop
    # once archived the loop's last pairing, not the viable one)
    from womb.breed import render_pair, viability as _viab
    for e in archive["entries"][:: max(1, len(archive["entries"]) // 8)]:
        assert set(GRIDS) <= set(e["cells"])
        assert (out / "specimens" / f"s{e['id']:04d}.json").exists()
        assert (out / "thumbs" / f"s{e['id']:04d}.png").exists()
        audio = render_pair(e["proc"], e["ear"], 1.5)
        assert _viab(audio, SR, strictness=0.5)["viable"], e["id"]


def test_seed_run_resume(tmp_path):
    out = tmp_path / "crypt2"
    r1 = seed_run(out=out, pop=6, ears=4, gens=1, strictness=0.5,
                  pairs=2, seconds=1.5, champion_seeds=2, seed=2)
    r2 = seed_run(out=out, gens=1, strictness=0.5, resume=True,
                  pairs=2, seconds=1.5, champion_seeds=2, seed=2)
    state = json.loads((out / "state.json").read_text())
    assert len(state["log"]) == 2                        # continued the ledger
    assert r2["specimens"] >= r1["specimens"]
    print(report_run(out))                               # the regime report
    assert "regime hint" in report_run(out)


def test_montage_smoke(tmp_path):
    """The listenable atlas: panels draw, stereo audio muxes."""
    out = tmp_path / "crypt3"
    seed_run(out=out, pop=6, ears=4, gens=1, strictness=0.5,
             pairs=2, seconds=1.5, champion_seeds=2, seed=9)
    vid = montage(out, out=str(tmp_path / "previews.mp4"), total=8.0)
    import subprocess

    info = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                            "-show_streams", vid], capture_output=True, text=True,
                           check=True).stdout
    streams = json.loads(info)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    a = next(s for s in streams if s["codec_type"] == "audio")
    assert v["codec_name"] == "h264" and int(v["width"]) == 960
    assert a["channels"] == 2                             # stereo survived the mux