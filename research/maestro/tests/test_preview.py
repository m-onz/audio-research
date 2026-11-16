"""preview tests — the artifact is listenable, finite, deterministic."""

from pathlib import Path

import numpy as np

from echo.probes import SR, decode, write_wav

from maestro.champions import follower
from maestro.preview import render_preview
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


def test_preview_wav_listenable(tmp_path):
    sc = make_score(tmp_path)
    out = render_preview(sc, tmp_path / "pv.wav", spine_gain=0.0)
    assert Path(out).exists()
    x = decode(out, sr=SR)
    assert x.max() > 0.05                       # audible
    assert np.isfinite(x).all()
    assert len(x) > int(3.5 * SR)               # grid + tail


def test_preview_deterministic_bytes(tmp_path):
    sc = make_score(tmp_path)
    a = render_preview(sc, tmp_path / "a.wav", spine_gain=0.0)
    b = render_preview(sc, tmp_path / "b.wav", spine_gain=0.0)
    assert Path(a).read_bytes() == Path(b).read_bytes()


def test_preview_spine_mix(tmp_path):
    sc = make_score(tmp_path, with_spine_file=True)
    alone = render_preview(sc, tmp_path / "alone.wav", spine_gain=0.0)
    mixed = render_preview(sc, tmp_path / "mixed.wav", spine_gain=0.6)
    xa, xm = decode(alone, sr=SR), decode(mixed, sr=SR)
    assert np.corrcoef(xa, xm)[0, 1] < 1.0       # the spine changed it
    assert xm.max() > 0.05


def test_preview_empty_score(tmp_path):
    sc = make_score(tmp_path)
    sc["events"] = []
    out = render_preview(sc, tmp_path / "empty.wav", spine_gain=0.0)
    assert Path(out).exists()                    # silence is a valid render