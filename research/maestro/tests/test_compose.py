"""compose tests — the whole pipeline on tiny fixtures, plus the
scaffold's form sampling (non-uniform grids, looped spines)."""

import json
from pathlib import Path

import numpy as np
import pytest

from echo.probes import SR, write_wav

from maestro.compose import compose, resolve_genome
from maestro.form import build_form
from maestro.scaffold import load_spine, scaffold_from_array

TICK_HZ = 8.0


def click_spine(onsets_s, seconds=6.0, sr=SR):
    x = np.zeros(int(seconds * sr), np.float32)
    for t in onsets_s:
        p0, klen = int(t * sr), int(0.02 * sr)
        x[p0: p0 + klen] += np.exp(-np.linspace(0, 6, klen)) * 0.8
    return x


def make_spine(tmp_path):
    path = tmp_path / "spine.wav"
    write_wav(path, click_spine([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), SR)
    return str(path)


def make_corpus(tmp_path):
    """A miniature corpus: kick + hat + texture families via manifest."""
    cdir = tmp_path / "corpus"
    cdir.mkdir(exist_ok=True)
    t = np.arange(int(0.3 * SR)) / SR
    kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.08)
    hat = np.random.default_rng(0).standard_normal(len(t)) * np.exp(-t / 0.01)
    tex = np.sin(2 * np.pi * 330 * t) * np.hanning(len(t))
    manifest = []
    for name, sig, fam in (("k.wav", kick, "audio_lab/sounds/kick"),
                           ("h.wav", hat, "audio_lab/sounds/hihat"),
                           ("oh.wav", hat, "audio_lab/sounds/openhh"),
                           ("r.wav", hat, "audio_lab/sounds/rimshot"),
                           ("p.wav", hat, "audio_lab/sounds/perc"),
                           ("t.wav", tex, "audio_lab/harmonic_corpus")):
        write_wav(cdir / name, sig.astype(np.float32) * 0.8, SR)
        manifest.append({"file": name, "family": fam})
    (cdir / "manifest.json").write_text(json.dumps(manifest))
    return str(cdir)


# ------------------------------------------------------- scaffold sampling


def test_sample_non_uniform_grid(tmp_path):
    path = make_spine(tmp_path)
    scaf = load_spine(path, bpm=None, tick_hz=TICK_HZ)
    f = build_form([[80, 4, "slow"], (240, 4, "fast")])
    tt = f.tick_times()
    s2 = scaf.sample(tt, loop=False)
    assert s2.n_ticks == len(tt)
    assert s2.tick_times is not None
    assert s2.streams["s0"].shape == (len(tt),)
    # events land on the form's times, not a uniform grid
    from maestro.champions import follower
    from maestro.score import score_from
    sc = score_from(follower(), s2, mode="strike")
    for e in sc["events"]:
        assert any(abs(e["t"] - t) < 1e-9 for t in tt)


def test_sample_loop_tiles_the_spine(tmp_path):
    path = make_spine(tmp_path)
    scaf = load_spine(path, bpm=None, tick_hz=TICK_HZ)
    tt = np.linspace(0.0, 24.0, 200)                 # 4x the 6 s spine
    s2 = scaf.sample(tt, loop=True)
    # onsets repeat every spine period
    on = s2.onsets["s0"]
    assert len(on) > 3 * len(scaf.onsets["s0"])
    assert max(on) > 18.0                            # well past one loop
    # a non-looped sample of the same span has no repeats
    s3 = scaf.sample(tt, loop=False)
    assert len(s3.onsets["s0"]) == len(scaf.onsets["s0"])


def test_sample_rejects_bad_grids(tmp_path):
    scaf = scaffold_from_array(click_spine([0.5]), sr=SR, bpm=None,
                               tick_hz=TICK_HZ)
    with pytest.raises(ValueError):
        scaf.sample(np.zeros(8))                     # not increasing
    with pytest.raises(ValueError):
        scaf.sample(np.array([0.0, 1.0]))           # too short


# ------------------------------------------------------------ the pipeline


def test_resolve_genome_exactly_one():
    from maestro.champions import follower
    assert resolve_genome(champion="follower").to_dict() == \
        follower().to_dict()
    with pytest.raises(SystemExit):
        resolve_genome()
    with pytest.raises(SystemExit):
        resolve_genome(champion="follower", random_seed=3)


def test_compose_end_to_end(tmp_path):
    spine = make_spine(tmp_path)
    corpus = make_corpus(tmp_path)
    man = compose(spine=spine, form="flat", length=6.0, seed=3,
                  champion="splitter", corpus=corpus,
                  spine_gain=0.3, duck_depth=0.2, post_intensity=0.5,
                  out=tmp_path / "piece", verbose=False)
    piece = tmp_path / "piece" / "piece.wav"
    assert piece.exists()
    from echo.probes import decode
    x = decode(piece, sr=SR)
    assert abs(x).max() > 0.1                       # audible
    assert np.isfinite(x).all()
    assert len(x) > int(5.0 * SR)
    # manifest provenance: voices resolved to corpus families
    fams = {v: d["family"] for v, d in man["voices"].items()}
    assert "kick" in fams["low"] and "hihat" in fams["high"]
    assert man["form"]["sections"]
    assert (tmp_path / "piece" / "score.json").exists()
    assert (tmp_path / "piece" / "manifest.json").exists()


def test_compose_deterministic_bytes(tmp_path):
    spine = make_spine(tmp_path)
    corpus = make_corpus(tmp_path)
    kw = dict(spine=spine, form="flat", length=6.0, seed=9,
              champion="splitter", corpus=corpus, spine_gain=0.3,
              post_intensity=1.0, verbose=False)
    a = compose(out=tmp_path / "a", **kw)
    b = compose(out=tmp_path / "b", **kw)
    assert (tmp_path / "a" / "piece.wav").read_bytes() == \
        (tmp_path / "b" / "piece.wav").read_bytes()
    assert a["post"] == b["post"]


def test_compose_form_makes_tempo_sections(tmp_path):
    spine = make_spine(tmp_path)
    corpus = make_corpus(tmp_path)
    man = compose(spine=spine, form="flip", length=10.0, seed=1,
                  random_seed=5, corpus=corpus, spine_gain=0.0,
                  duck_depth=0.0, post_intensity=0.0,
                  out=tmp_path / "p2", verbose=False)
    assert any(s["bpm"] >= 240.0 for s in man["form"]["sections"])
    assert any(s["bpm"] <= 95.0 for s in man["form"]["sections"])

def test_compose_kit_ensemble(tmp_path):
    """The kit champion through the whole pipeline: six voices resolve
    to six different kit families, with the role image (gain/pan)."""
    spine = make_spine(tmp_path)
    corpus = make_corpus(tmp_path)
    man = compose(spine=spine, form="flat", length=6.0, seed=3,
                  champion="kit", corpus=corpus, spine_gain=0.3,
                  duck_depth=0.2, post_intensity=0.5,
                  out=tmp_path / "kitpiece", verbose=False)
    fams = {v: d["family"] for v, d in man["voices"].items()}
    assert set(fams) == {"kick", "rim", "percA", "percB", "hat", "openhat"}
    assert "kick" in fams["kick"]
    assert "rimshot" in fams["rim"]
    assert "openhh" in fams["openhat"]
    assert "hihat" in fams["hat"]
    assert "perc" in fams["percA"] and "perc" in fams["percB"]
    # the kit image: pans in the manifest
    assert man["voices"]["hat"]["pan"] < 0 < man["voices"]["rim"]["pan"]
    assert abs(man["voices"]["kick"]["pan"]) < 1e-9
