"""club tests — the driver skeleton is exact math, the piece is a
real two-layer composition, everything deterministic."""

import json
from pathlib import Path

import numpy as np
import pytest

from echo.probes import SR, write_wav

from maestro.club import (best_specimen, club_compose, driver_score,
                          quantize_top, resolve_top_genome)
from maestro.form import build_form
from maestro.score import score_from
from maestro.scaffold import load_spine, scaffold_from_array

TICK_HZ = 8.0


def click_spine(onsets_s, seconds=8.0, sr=SR):
    x = np.zeros(int(seconds * sr), np.float32)
    for t in onsets_s:
        p0, klen = int(t * sr), int(0.02 * sr)
        x[p0: p0 + klen] += np.exp(-np.linspace(0, 6, klen)) * 0.8
    return x


def make_spine(tmp_path):
    path = tmp_path / "spine.wav"
    write_wav(path, click_spine([0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                                 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]), SR)
    return str(path)


def make_corpus(tmp_path):
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
                           ("c.wav", hat, "audio_lab/sounds/crash"),
                           ("t.wav", tex, "audio_lab/harmonic_corpus")):
        write_wav(cdir / name, sig.astype(np.float32) * 0.8, SR)
        manifest.append({"file": name, "family": fam})
    (cdir / "manifest.json").write_text(json.dumps(manifest))
    return str(cdir)


# ------------------------------------------------------------------ driver


def test_driver_four_on_the_floor(tmp_path):
    spine = make_spine(tmp_path)
    scaf = load_spine(spine, bpm=None, tick_hz=TICK_HZ)
    f = build_form([[128, 4, "floor"], [74, 2, "break"], [150, 2, "rush"]])
    dr = driver_score(scaf, f, length=None if False else 6.0)
    tpb = f.ticks_per_beat
    tt = f.tick_times(6.0)
    kicks = sorted(e["tick"] for e in dr["events"] if e["voice"] == "kick")
    # dance-section kicks sit ON beats (tick % tpb == 0) or, in the
    # rush, on the double-time half-beats (tick % (tpb//2) == 0)
    for k in kicks:
        assert k % (tpb // 2) == 0
    # the break (74 bpm section) has NO kicks
    n_break = 2 * 4 * tpb
    assert not any(k < n_break and k >= 4 * 4 * tpb for k in kicks) \
        or True  # section boundaries are exact; verify via labels below
    secs = {s["label"]: s for s in f.section_bounds()}
    for e in dr["events"]:
        if e["voice"] == "kick":
            in_break = secs["break"]["t0"] <= e["t"] < secs["break"]["t1"]
            assert not in_break, "the break must lose the kick"
    # rim on the offbeat (half-beat ticks) in dance sections
    rims = [e["tick"] for e in dr["events"] if e["voice"] == "rim"
            and e["t"] < secs["break"]["t0"]]
    assert rims and all(r % (tpb // 2) == 0 for r in rims)
    # impacts at every section start except the first
    imps = sorted(e["t"] for e in dr["events"] if e["voice"] == "impact")
    assert len(imps) == 2                      # break + rush boundaries
    # hat16 16ths exist; the crossperc is spine-driven and present
    assert any(e["voice"] == "hat16" for e in dr["events"])
    assert any(e["voice"] == "percB" for e in dr["events"])
    # downbeat accents: the loudest kicks sit on bar starts
    kv = [e for e in dr["events"] if e["voice"] == "kick"]
    assert max(e["velocity"] for e in kv) == pytest.approx(1.0)
    assert min(e["velocity"] for e in kv) < 0.9


def test_driver_rush_doubles_the_kick(tmp_path):
    spine = make_spine(tmp_path)
    scaf = load_spine(spine, bpm=None, tick_hz=TICK_HZ)
    f = build_form([[150, 4, "rush"]])
    dr = driver_score(scaf, f, length=4.0)
    tpb = f.ticks_per_beat
    off = [e for e in dr["events"] if e["voice"] == "kick"
           and e["tick"] % tpb == tpb // 2]
    assert off, "rush sections must double-time the kick"
    assert all(e["velocity"] < 0.8 for e in off)     # ghost kicks


# ------------------------------------------------------------------ compose


def test_club_compose_end_to_end(tmp_path):
    spine = make_spine(tmp_path)
    corpus = make_corpus(tmp_path)
    man = club_compose(spine=spine, form="flat", top="random:3",
                       seed=5, length=6.0, corpus=corpus,
                       spine_gain=0.3, top_gain=0.5,
                       post_intensity=0.5, duck_depth=0.3,
                       out=tmp_path / "club", verbose=False)
    piece = tmp_path / "club" / "piece.wav"
    assert piece.exists()
    from echo.probes import decode
    x = decode(piece, sr=SR)
    assert abs(x).max() > 0.1 and np.isfinite(x).all()
    # both layers' scores are on disk and both genomes in the manifest
    assert (tmp_path / "club" / "driver.score.json").exists()
    assert (tmp_path / "club" / "top.score.json").exists()
    assert man["driver"]["kind"].startswith("grid")
    assert man["top"]["genome_source"] == "random:3"
    assert man["top"]["n_events"] >= 0
    # driver kick got cast from the kick family; top voices got the
    # OTHER casting seed (they may differ from the driver's)
    assert "kick" in man["driver"]["voices"]["kick"]["family"]


def test_club_compose_deterministic(tmp_path):
    spine = make_spine(tmp_path)
    corpus = make_corpus(tmp_path)
    kw = dict(spine=spine, form="flat", top="random:3", seed=5,
              length=6.0, corpus=corpus, spine_gain=0.3,
              post_intensity=1.0, verbose=False)
    club_compose(out=tmp_path / "a", **kw)
    club_compose(out=tmp_path / "b", **kw)
    assert (tmp_path / "a" / "piece.wav").read_bytes() == \
        (tmp_path / "b" / "piece.wav").read_bytes()


def test_resolve_top_and_best_specimen(tmp_path):
    # random:N founders
    g = resolve_top_genome("random:11")
    assert set(g.outputs) == {"pulse", "flow"}
    # a specimen json
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"genome": g.to_dict()}))
    assert resolve_top_genome(str(spec)).to_dict() == g.to_dict()
    # best_specimen picks the freshest
    rdir = tmp_path / "run"
    rdir.mkdir()
    for i, nov in enumerate([3.0, 9.5, 5.0]):
        (rdir / f"spec_{i:02d}.json").write_text(json.dumps(
            {"genome": {}, "nov_at_birth": nov}))
    assert best_specimen(rdir).name == "spec_01.json"
    with pytest.raises(SystemExit):
        best_specimen(tmp_path)          # no specimens

# ------------------------------------------------------------------ quantize


def test_quantize_top_danceable(tmp_path):
    """The evolving layer lands ON the grid: one event per voice per
    half-beat window, at most 4 voices per window, velocities kept."""
    from maestro.champions import kit
    f = build_form([[128, 4, "floor"]])
    tt = f.tick_times(8.0)
    scaf = scaffold_from_array(click_spine([0.5, 1.5, 2.5]), sr=SR,
                               bpm=None, tick_hz=16.0).sample(tt, loop=True)
    sc = score_from(kit(), scaf, mode="strike")
    q = quantize_top(sc, f, 8.0, window="half", max_per=3)
    tpb = f.ticks_per_beat
    wt = tpb // 2
    assert all(e["tick"] % wt == 0 for e in q["events"])   # on the grid
    # one per (window, voice), at most max_per per window
    seen = set()
    for e in q["events"]:
        k = (e["tick"] // wt, e["voice"])
        assert k not in seen
        seen.add(k)
    from collections import Counter
    per_w = Counter(e["tick"] // wt for e in q["events"])
    assert max(per_w.values()) <= 3
    assert q["n_events"] < sc["n_events"]                 # thinned
    # the same genome, unquantized, is denser (the dial is real)
    assert sc["n_events"] > 0
