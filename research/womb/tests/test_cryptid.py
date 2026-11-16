"""cryptid tests: the invariant holds, the physics is real, the render
is deterministic. Phase A acceptance from CRYPTID.md."""

import time

import numpy as np
import pytest

import womb
from womb.cryptid import (CHAMPIONS, build_champion, describe_genome,
                          load_specimen, make_law, render, save_specimen,
                          simulate, viability)
from womb.sonify import write_wav

SR = 44100


# ------------------------------------------------------------------ laws

def test_law_registry():
    for name in ("jerk", "osc_pop", "map", "sand", "excitable", "fire",
                 "bool", "delay", "clamp", "bytefeed"):
        assert name in womb.cryptid.LAWS
    with pytest.raises(ValueError):
        make_law("no_such_law", {}, 0, 11025)


def test_laws_step_finite():
    """Every law in the zoo runs, emits finite channels, and survives
    a short network step loop."""
    import base64

    genomes = {
        "jerk": {"nodes": [{"law": "jerk", "params": dict(
            a=-0.44, b=-1.97, c=-2.97, d=0.68, e=0.63, f=-2.91, g=1.06,
            speed=50.0, ode_rate=60.0, x0=0.4)}], "ears": [
            {"op": "tap", "node": 0, "ch": 0, "gain": 0.5}]},
        "osc_pop": {"nodes": [{"law": "osc_pop", "params": dict(
            n=32, radius=8, k=1.6, alpha=1.38, freq=110.0, windows=8)}], "ears": [
            {"op": "chimera", "node": 0, "tone": 0.5, "noise": 0.5}]},
        "map+excitable+fire": {"nodes": [
            {"law": "map", "params": dict(period_s=0.05)},
            {"law": "excitable", "params": dict(eps=0.02)},
            {"law": "fire", "params": dict(bias=1.0)},
        ], "edges": [{"src": 0, "ch": 0, "dst": 1, "gain": 1.0},
                     {"src": 0, "ch": 0, "dst": 2, "gain": 1.0}],
            "ears": [{"op": "tap", "node": 1, "ch": 0, "gain": 0.4},
                     {"op": "events", "node": 2, "gain": 0.5}]},
        "bool+delay+clamp": {"nodes": [
            {"law": "bool", "params": dict(k=1, table=(0, 1), period_s=0.1)},
            {"law": "delay", "params": dict(length_s=0.05)},
            {"law": "clamp", "params": dict(gain=1.5, steps=4)},
        ], "edges": [{"src": 0, "ch": 0, "dst": 1, "gain": 1.0},
                     {"src": 1, "ch": 0, "dst": 2, "gain": 1.0}],
            "ears": [{"op": "tap", "node": 2, "ch": 0, "gain": 0.4}]},
        "sand": {"nodes": [{"law": "sand", "params": dict(
            w=24, h=24, interval_s=0.05, warmup=500)}], "ears": [
            {"op": "events", "node": 0, "gain": 0.8},
            {"op": "tap", "node": 0, "ch": 0, "gain": 0.1}]},
        "bytefeed": {"nodes": [{"law": "bytefeed", "params": dict(
            b64=base64.b64encode(b"womb cryptid").decode(), rate=2000)}],
            "ears": [{"op": "tap", "node": 0, "ch": 0, "gain": 0.4}]},
    }
    for name, g in genomes.items():
        g.setdefault("seed", 0)
        g.setdefault("sim_rate", 11025)
        L, R, rate, nodes = simulate(g, 1.0)
        assert np.isfinite(L).all() and np.isfinite(R).all(), name


def test_ear_unknown_op_rejected():
    with pytest.raises(ValueError):
        womb.cryptid.Ear([{"op": "reverb", "node": 0}], [], 11025)


def test_chimera_needs_osc_pop():
    g = build_champion("clickstorm")
    with pytest.raises(ValueError):
        womb.cryptid.Ear([{"op": "chimera", "node": 0}],
                         [make_law("map", {}, 0, 11025)], 11025)


# ------------------------------------------------------------------ champions

@pytest.mark.parametrize("name", sorted(CHAMPIONS))
def test_champion_renders(name):
    st = render(build_champion(name), length=2.0)
    audio = st["audio"]
    assert audio.ndim == 2 and audio.shape[1] == 2      # stereo
    assert len(audio) == int(2.0 * SR)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 1.0                   # safety stage held
    v = viability(audio, SR)
    assert v["viable"], (name, v)


@pytest.mark.parametrize("name", sorted(CHAMPIONS))
def test_champion_determinism(name):
    """Two renders of the same genome are bit-identical (seed = identity)."""
    a1 = render(build_champion(name), length=1.0)["audio"]
    a2 = render(build_champion(name), length=1.0)["audio"]
    assert np.array_equal(a1, a2)


def test_specimen_roundtrip(tmp_path):
    g = build_champion("clickstorm")
    p = save_specimen(g, tmp_path / "specimen.json")
    g2 = load_specimen(p)
    a1 = render(g, length=1.0)["audio"]
    a2 = render(g2, length=1.0)["audio"]
    assert np.array_equal(a1, a2)          # json roundtrip is render-exact


def test_describe_genome():
    text = describe_genome(build_champion("jerk_voice"))
    assert "jerk_voice" in text
    assert "jerk(" in text and "ear:" in text and "notes:" in text


def test_render_speed():
    """Phase A acceptance: 60 s in < 30 s wall. We test the slowest
    champion (chimera) at a scaled-down slice."""
    t0 = time.time()
    render(build_champion("chimera"), length=3.0)
    wall = time.time() - t0
    assert wall < 4.0, f"chimera 3s render took {wall:.1f}s"


# ------------------------------------------------------------------ viability

def test_viability_rejects_silence_and_clipping():
    v = viability(np.zeros((SR, 2), np.float32), SR)
    assert not v["viable"]
    v = viability(np.full((SR, 2), 4.0, np.float32), SR)
    assert not v["viable"]


def test_write_wav_stereo(tmp_path):
    audio = np.stack([np.linspace(-1, 1, 44100, dtype=np.float32)] * 2, axis=1)
    p = write_wav(tmp_path / "s.wav", audio, SR)
    import subprocess

    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_streams",
                          "-print_format", "json", str(p)],
                         capture_output=True, text=True, check=True).stdout
    import json

    info = json.loads(out)
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert a["channels"] == 2