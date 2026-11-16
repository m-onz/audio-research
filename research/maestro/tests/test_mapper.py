"""mapper tests — the piece is renderable, velocity is gain,
determinism holds, truncate and spine mix behave."""

from pathlib import Path

import numpy as np

from echo.probes import SR, decode, write_wav

from maestro.champions import follower, swell
from maestro.mapper import render_mapped
from maestro.score import score_from
from maestro.scaffold import load_spine, scaffold_from_array

TICK_HZ = 8.0


def click_spine(onsets_s, seconds=3.5, sr=SR):
    x = np.zeros(int(seconds * sr), np.float32)
    for t in onsets_s:
        p0, klen = int(t * sr), int(0.02 * sr)
        x[p0: p0 + klen] += np.exp(-np.linspace(0, 6, klen)) * 0.8
    return x


def make_score(tmp_path, with_spine_file=False):
    if with_spine_file:
        path = tmp_path / "spine.wav"
        write_wav(path, click_spine([0.5, 1.5, 2.5]), SR)
        s = load_spine(path, bpm=None, tick_hz=TICK_HZ)
    else:
        s = scaffold_from_array(click_spine([0.5, 1.5, 2.5]),
                                sr=SR, bpm=None, tick_hz=TICK_HZ)
    return score_from(follower(), s, mode="strike")


def make_sample(tmp_path, name, freq=220.0, seconds=0.5):
    """A little sine ping sample, the stand-in for corpus material."""
    path = tmp_path / name
    t = np.arange(int(seconds * SR)) / SR
    x = (np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.2) * 0.8)
    write_wav(path, x.astype(np.float32), SR)
    return str(path)


def test_map_renders_the_piece(tmp_path):
    sc = make_score(tmp_path)
    kick = make_sample(tmp_path, "kick.wav", freq=80.0)
    out = render_mapped(sc, {"voices": {"pulse": kick},
                             "spine_gain": 0.0}, tmp_path / "piece.wav")
    assert Path(out).exists()
    x = decode(out, sr=SR)
    assert x.max() > 0.3                       # audible and normalized
    assert np.isfinite(x).all()
    assert len(x) > int(3.5 * SR)


def test_map_deterministic_bytes(tmp_path):
    sc = make_score(tmp_path)
    kick = make_sample(tmp_path, "kick.wav")
    vm = {"voices": {"pulse": kick}, "spine_gain": 0.0}
    a = render_mapped(sc, vm, tmp_path / "a.wav")
    b = render_mapped(sc, vm, tmp_path / "b.wav")
    assert Path(a).read_bytes() == Path(b).read_bytes()


def test_velocity_is_gain(tmp_path):
    """The paper's semantics through the whole stack: output level is
    strike volume — a swell score's loud events ring louder than its
    tail events in the rendered piece."""
    scaf = scaffold_from_array(click_spine([0.5, 1.5, 2.5]),
                               sr=SR, bpm=None, tick_hz=TICK_HZ)
    sc = score_from(swell(), scaf, mode="strike")
    ev = [e for e in sc["events"] if e["voice"] == "swell"]
    assert ev and (max(e["velocity"] for e in ev)
                   - min(e["velocity"] for e in ev)) > 0.3
    sample = make_sample(tmp_path, "tap.wav")
    out = render_mapped(sc, {"voices": {"swell": sample},
                             "spine_gain": 0.0}, tmp_path / "swell.wav")
    x = decode(out, sr=SR)
    # energy around the loudest event exceeds the tail event's
    def _rms_around(t0, win=0.08):
        a, b = int(t0 * SR), int((t0 + win) * SR)
        return float(np.sqrt((x[a:b] ** 2).mean()))
    first = max(ev, key=lambda e: e["velocity"])
    last = min(ev, key=lambda e: e["velocity"])
    assert _rms_around(first["t"]) > _rms_around(last["t"])


def test_map_truncate(tmp_path):
    # one gate event only: truncation must silence the ring-out tail
    scaf = scaffold_from_array(click_spine([0.5]), sr=SR, bpm=None,
                               tick_hz=TICK_HZ)
    sc = score_from(follower(), scaf, mode="gate")
    ev = [e for e in sc["events"] if e["voice"] == "pulse"]
    assert len(ev) == 1
    long_sample = make_sample(tmp_path, "long.wav", seconds=2.0)
    full = render_mapped(sc, {"voices": {"pulse": {"file": long_sample}},
                              "spine_gain": 0.0}, tmp_path / "full.wav")
    cut = render_mapped(sc, {"voices": {"pulse": {"file": long_sample,
                                                  "truncate": True}},
                             "spine_gain": 0.0}, tmp_path / "cut.wav")
    xf, xc = decode(full, sr=SR), decode(cut, sr=SR)
    # window after the truncated run but inside the full ring-out
    a, b = int(2.4 * SR), int(2.9 * SR)
    rms_full = float(np.sqrt((xf[a:b] ** 2).mean()))
    rms_cut = float(np.sqrt((xc[a:b] ** 2).mean()))
    assert rms_full > 10.0 * max(rms_cut, 1e-9)


def test_map_spine_mix(tmp_path):
    sc = make_score(tmp_path, with_spine_file=True)
    kick = make_sample(tmp_path, "kick.wav")
    alone = render_mapped(sc, {"voices": {"pulse": kick},
                               "spine_gain": 0.0}, tmp_path / "alone.wav")
    mixed = render_mapped(sc, {"voices": {"pulse": kick},
                               "spine_gain": 0.7}, tmp_path / "mixed.wav")
    xa, xm = decode(alone, sr=SR), decode(mixed, sr=SR)
    assert np.corrcoef(xa, xm)[0, 1] < 1.0       # the spine changed it


def test_map_unknown_voice_ignored(tmp_path):
    sc = make_score(tmp_path)
    kick = make_sample(tmp_path, "kick.wav")
    out = render_mapped(sc, {"voices": {"nosuchvoice": kick},
                             "spine_gain": 0.0}, tmp_path / "silent.wav")
    x = decode(out, sr=SR)
    assert np.abs(x).max() < 0.01                # nothing mapped: silence