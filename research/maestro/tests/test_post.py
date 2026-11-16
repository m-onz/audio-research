"""post tests — the damage is exact where it claims to be, random
everywhere else, and deterministic about both."""

import numpy as np
import pytest

from echo.probes import SR

from maestro.post import (FAST_BPM, SLOW_BPM, apply_post, duck, plan_post,
                          post_process, seg_crush, seg_gate, seg_reverse,
                          seg_speed, seg_stretch, seg_stutter)

BOUNDS = [
    {"t0": 0.0, "t1": 20.0, "bpm": 90.0, "label": "slow"},
    {"t0": 20.0, "t1": 40.0, "bpm": 300.0, "label": "fast"},
    {"t0": 40.0, "t1": 60.0, "bpm": 80.0, "label": "slow2"},
]


def _tone(seconds=4.0, f=440.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    x = (0.5 * np.sin(2 * np.pi * f * t))[:, None] * np.ones((1, 2))
    return x.astype(np.float64)


def _dominant_hz(x, sr=SR):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[:, 0]
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(fr[np.argmax(X)])


# ------------------------------------------------------------------ segments


def test_seg_speed_is_tape():
    x = _tone(2.0, 220.0)
    y = seg_speed(x, SR, 0.5)                       # half speed: longer+lower
    assert y.shape[0] == pytest.approx(2 * x.shape[0], abs=100)
    assert _dominant_hz(y) == pytest.approx(110.0, abs=3.0)
    y = seg_speed(x, SR, 2.0)
    assert y.shape[0] == pytest.approx(0.5 * x.shape[0], abs=100)
    assert _dominant_hz(y) == pytest.approx(440.0, abs=6.0)


def test_seg_stretch_preserves_pitch_and_time():
    x = _tone(4.0, 220.0)
    y = seg_stretch(x, SR, 2.0)
    assert y.shape[0] == pytest.approx(2 * x.shape[0], rel=0.05)
    assert _dominant_hz(y) == pytest.approx(220.0, abs=8.0)


def test_seg_stutter_fills_the_window():
    x = _tone(2.0)
    rng = np.random.default_rng(0)
    y = seg_stutter(x, SR, rng, slice_s=0.05, want_s=0.8,
                    mode="ladder", step=1.0)
    assert y.shape[0] == pytest.approx(0.8 * SR, abs=2048)
    assert np.isfinite(y).all()
    # const mode repeats identical slices: strong self-similarity
    yc = seg_stutter(x, SR, np.random.default_rng(0), slice_s=0.05,
                     want_s=0.4, mode="const")
    k = int(0.05 * SR)
    assert np.abs(yc[k:2 * k] - yc[2 * k:3 * k]).max() < 0.05


def test_seg_reverse_and_gate_and_crush():
    x = _tone(1.0)
    y = seg_reverse(x)
    assert np.allclose(y, x[::-1])
    g = seg_gate(x, SR, steps=8, pulses=2, rot=0)
    assert (g == 0).mean() > 0.5                     # mostly silence
    c = seg_crush(x, bits=3)
    assert len(np.unique(np.round(c, 6))) < 32


# ------------------------------------------------------------------ planning


def test_plan_post_section_aware():
    rng = np.random.default_rng(7)
    ev = plan_post(rng, BOUNDS, length_s=60.0, intensity=2.0,
                   n_stutters=6, n_stretches=4, n_tapes=0,
                   n_reverses=0, n_gates=0)
    stut = [e for e in ev if e["type"] == "stutter"]
    stre = [e for e in ev if e["type"] == "stretch"]
    assert len(stut) >= 3 and len(stre) >= 2
    # stutters live in the fast section, stretches in the slow ones
    for e in stut:
        assert 20.0 <= e["t0"] <= 40.0
    for e in stre:
        assert not (20.0 < e["t0"] < 40.0)
    # non-overlap with 1.5 s gaps
    for a, b in zip(ev, ev[1:]):
        assert b["t0"] - a["t0"] >= a["dur"] + 0.0 or b["t0"] >= a["t0"] + 1.5


def test_apply_post_changes_length_and_stays_clean():
    x = _tone(40.0, 110.0)
    events = [
        {"type": "stretch", "t0": 2.0, "dur": 4.0, "factor": 2.0},
        {"type": "stutter", "t0": 15.0, "dur": 1.0, "slice": 0.05,
         "mode": "ladder", "step": 1.0},
        {"type": "reverse", "t0": 32.0, "dur": 2.0},
    ]
    out, ledger = apply_post(x, SR, events)
    assert out.shape[0] > x.shape[0]                 # stretch grew it
    assert np.isfinite(out).all()
    assert {e["type"] for e in ledger} == {"stretch", "stutter", "reverse"}
    assert ledger[0]["out_dur"] == pytest.approx(8.0, abs=0.1)


def test_duck_dips_at_times():
    x = np.ones((SR * 2, 2))
    y = duck(x, SR, [0.5], depth=0.5, tau=0.05)
    i = int(0.5 * SR)
    assert y[i, 0] < 0.6 and y[int(1.0 * SR), 0] > 0.95


def test_post_process_deterministic():
    x = _tone(60.0)
    a, la = post_process(x, SR, BOUNDS, seed=3, intensity=8.0)
    b, lb = post_process(x, SR, BOUNDS, seed=3, intensity=8.0)
    assert np.array_equal(a, b) and la == lb and len(la) > 0
    c, _ = post_process(x, SR, BOUNDS, seed=4, intensity=8.0)
    assert not np.array_equal(a, c)                  # the seed matters
    # intensity 0: nothing happens
    d, led = post_process(x, SR, BOUNDS, seed=3, intensity=0.0)
    assert led == []


def test_post_respects_the_piece_edges():
    x = _tone(10.0)
    ev = plan_post(np.random.default_rng(1),
                   [{"t0": 0.0, "t1": 10.0, "bpm": 300.0, "label": "f"}],
                   length_s=10.0, n_stutters=5, n_stretches=0, n_tapes=0,
                   n_reverses=0, n_gates=0)
    for e in ev:
        assert e["t0"] > 0.0 and e["t0"] + e["dur"] < 9.0