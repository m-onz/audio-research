"""scaffold tests — known answers for the spike-decay encoder.

The click spine is the paper's fig. 2 made testable: onsets at
known times, IOI = 1 s, so the spike at any tick is a closed-form
number. The tone spine exercises the band split and the leakage
guard. Determinism is a contract, like everywhere in this ecosystem.
"""

import numpy as np
import pytest

from echo.probes import SR, write_wav

from maestro.scaffold import (SENSES, Scaffold, estimate_bpm, load_spine,
                              pick_onsets, scaffold_from_array, spike_train)

TICK_HZ = 8.0                       # bpm 120, 4 ticks per beat


def click_spine(onsets_s, seconds=3.5, sr=SR):
    """Silence with exponentially-decaying clicks at the given times."""
    x = np.zeros(int(seconds * sr), np.float32)
    for t in onsets_s:
        p0 = int(t * sr)
        klen = int(0.02 * sr)
        x[p0: p0 + klen] += (np.exp(-np.linspace(0, 6, klen)) * 0.8)
    return x


def tone_spine(events, seconds=3.5, sr=SR):
    """events: list of (t0, freq, dur_s, amp)."""
    x = np.zeros(int(seconds * sr), np.float32)
    for t0, f, d, a in events:
        p0, klen = int(t0 * sr), int(d * sr)
        tt = np.arange(klen) / sr
        env = np.hanning(klen)
        x[p0: p0 + klen] += (np.sin(2 * np.pi * f * tt) * env * a)
    return x


# ------------------------------------------------------------------ spikes


def test_spike_train_known_answer():
    v = spike_train([1.0], n_ticks=16, tick_hz=4.0, max_decay_s=2.0)
    assert v[4] == pytest.approx(1.0)            # at the onset
    assert v[8] == pytest.approx(0.5)            # half decayed
    assert v[12] == pytest.approx(0.0, abs=0.01)  # done
    assert v[2] == 0.0                            # before


def test_spike_train_ioi_is_duration():
    # three onsets 1 s apart: each spike reaches zero exactly at the
    # next, whose fresh attack wins the shared tick
    v = spike_train([1.0, 2.0, 3.0], n_ticks=16, tick_hz=4.0)
    assert v[4] == pytest.approx(1.0)            # attack 1
    assert v[8] == pytest.approx(1.0)            # attack 2 at the shared tick
    assert v[7] == pytest.approx(0.25)           # tail of the first
    assert v[9] == pytest.approx(0.75)           # decay of the second
    assert v[12] == pytest.approx(1.0)           # attack 3
    assert all(v[i] > v[i + 1] for i in range(4, 7))   # linear decay
    assert v.min() >= 0.0 and v.max() <= 1.0


def test_spike_train_clamps_and_empty():
    v = spike_train([], n_ticks=8, tick_hz=4.0)
    assert not v.any()
    # single onset: decay clamps at max (does not decay forever)
    v = spike_train([0.5], n_ticks=64, tick_hz=4.0, max_decay_s=1.0)
    assert v.max() == pytest.approx(1.0)
    assert v[-1] == pytest.approx(0.0, abs=0.01)


def test_pick_onsets_refractory_and_floor():
    t = np.arange(100) * 0.01
    nov = np.zeros(100)
    nov[10] = 5.0
    nov[12] = 5.0                     # inside refractory — suppressed
    nov[50] = 4.0
    nov[80] = 1e-9                    # below the relative floor
    on = pick_onsets(nov, t, min_ioi_s=0.08, rel_thr=0.3)
    assert len(on) == 2
    assert abs(on[0] - 0.10) < 0.02
    assert abs(on[1] - 0.50) < 0.02


# ------------------------------------------------------------------ scaffold


def test_scaffold_clicks_known_answer():
    x = click_spine([0.5, 1.5, 2.5])
    scaf = scaffold_from_array(x, sr=SR, bpm=None, tick_hz=TICK_HZ)
    assert scaf.n_ticks == pytest.approx(3.5 * TICK_HZ, abs=2)
    on = scaf.onsets["s0"]
    assert len(on) == 3
    for want, got in zip((0.5, 1.5, 2.5), on):
        assert abs(got - want) < 0.06        # frame resolution
    s0 = scaf.streams["s0"]
    # tick 4 = t 0.5 s: fresh spike; tick 8 = t 1.0: mid-decay
    assert s0[4] > 0.9
    assert 0.3 < s0[8] < 0.7
    # every stream and weather array is grid-shaped and bounded
    for k, v in scaf.streams.items():
        assert v.shape == (scaf.n_ticks,) and v.dtype == np.float32
        assert v.min() >= 0.0 and v.max() <= 1.0
    for k, v in scaf.weather.items():
        assert v.shape == (scaf.n_ticks,)
        assert v.min() >= 0.0 and v.max() <= 1.0


def test_scaffold_determinism():
    x = click_spine([0.5, 1.5, 2.5])
    a = scaffold_from_array(x, sr=SR, bpm=None, tick_hz=TICK_HZ)
    b = scaffold_from_array(x, sr=SR, bpm=None, tick_hz=TICK_HZ)
    for k in a.streams:
        assert np.array_equal(a.streams[k], b.streams[k])
    for k in a.weather:
        assert np.array_equal(a.weather[k], b.weather[k])


def test_scaffold_bands_and_leak_guard():
    # low bursts at 0.5 / 1.5 (100 Hz), high bursts at 1.0 / 2.0 (3 kHz)
    x = tone_spine([(0.5, 100.0, 0.3, 0.6), (1.5, 100.0, 0.3, 0.6),
                    (1.0, 3000.0, 0.3, 0.4), (2.0, 3000.0, 0.3, 0.4)])
    scaf = scaffold_from_array(x, sr=SR, bpm=None, tick_hz=TICK_HZ)
    assert len(scaf.onsets["s1"]) >= 2          # the bass register fires
    assert len(scaf.onsets["s3"]) >= 2          # the air register fires
    low_t = scaf.onsets["s1"]
    assert min(abs(t - 0.5) for t in low_t) < 0.1
    # a pure 100 Hz sine must not fire the air train (leakage guard)
    x = tone_spine([(0.5, 100.0, 0.3, 0.6), (1.5, 100.0, 0.3, 0.6)])
    scaf = scaffold_from_array(x, sr=SR, bpm=None, tick_hz=TICK_HZ)
    assert len(scaf.onsets["s1"]) >= 2
    assert scaf.onsets["s3"] == [] or scaf.streams["s3"].max() < 0.05


def test_scaffold_length_and_grid():
    x = click_spine([0.5, 1.5, 2.5])
    # explicit shorter length truncates the grid
    scaf = scaffold_from_array(x, sr=SR, bpm=None, tick_hz=TICK_HZ,
                               length_s=2.0)
    assert scaf.n_ticks == 16
    assert scaf.length_s == pytest.approx(2.0)


def test_feeds_senses():
    x = click_spine([0.5])
    scaf = scaffold_from_array(x, sr=SR, bpm=None, tick_hz=TICK_HZ)
    f = scaf.feeds()                            # default: spikes + weather
    assert set(f) >= {"s0", "s1", "s2", "s3", "s4", "live_rms",
                      "live_centroid", "live_flux", "b"}
    assert float(f["b"]) == 1.0
    assert float(f["x"]) == 0.0 and float(f["t"]) == 0.0
    assert len(f["s0"]) == scaf.n_ticks
    f = scaf.feeds(senses=("spikes", "position", "time"))
    assert "live_rms" not in f                  # weather opted out
    assert len(f["x"]) == scaf.n_ticks and f["x"][0] == 0.0
    assert f["t"][0] == pytest.approx(0.0) and f["t"][-1] > 3.0
    with pytest.raises(ValueError):
        scaf.feeds(senses=("spikes", "vibes"))


def test_load_spine_file(tmp_path):
    # the ffmpeg path: file -> sha256 -> onsets
    path = tmp_path / "spine.wav"
    write_wav(path, click_spine([0.5, 1.5, 2.5]), SR)
    scaf = load_spine(path, bpm=None, tick_hz=TICK_HZ)
    assert scaf.path == str(path)
    assert len(scaf.sha256) == 64
    assert len(scaf.onsets["s0"]) == 3
    assert isinstance(scaf, Scaffold)
    assert "onsets" in scaf.info().lower()


# ------------------------------------------------------------------ tempo


def test_estimate_bpm_known_answers():
    # clicks every 0.5 s -> 120 bpm
    on = list(np.arange(0.5, 10.0, 0.5))
    assert estimate_bpm(on) == pytest.approx(120.0, abs=1.0)
    # every 3/7 s (odd IOI, no subdivisions) -> ~140 bpm
    on = list(np.arange(0.5, 10.0, 3.0 / 7.0))
    assert estimate_bpm(on) == pytest.approx(140.0, abs=1.5)
    # alternating quarter/eighth at 90 — symmetric evidence, the tie
    # must break toward the beat (the slower near-tied period)
    pat = [0.667, 0.333] * 5
    on = [0.5] + list(0.5 + np.cumsum(pat))
    assert estimate_bpm(on) == pytest.approx(90.0, abs=2.0)
    # the honest octave limit, locked as behavior: UNIFORM eighths are
    # indistinguishable from faster quarters (no beat ioi exists) —
    # the estimate lands on an octave of the truth
    on = list(np.arange(0.5, 10.0, 1.0 / 3.0))
    est = estimate_bpm(on)
    assert est is not None and 90.0 / est in (0.5, 1.0, 2.0)
    # subdivided: quarter + two eighths at 120 (0.5, 0.25, 0.25 repeating)
    pat = [0.5, 0.25, 0.25] * 4
    on = [0.5] + list(0.5 + np.cumsum(pat))
    assert estimate_bpm(on) == pytest.approx(120.0, abs=2.0)


def test_estimate_bpm_degenerate():
    assert estimate_bpm([]) is None
    assert estimate_bpm([1.0, 2.0]) is None          # too few intervals