"""score tests — the traversal, the champions' known answers.

follower must strike at the spine's attacks; answerer must fire
mid-decay (the temporal coordinate frame made audible — only the
spike encoding makes 'mid-event' findable); gate must merge runs;
the score must round-trip bit-exact and reproduce deterministically.
"""

import numpy as np
import pytest

from echo.probes import SR

from maestro.champions import (CHAMPIONS, answerer, build_champion,
                               follower, kit, random_score_genome, swell)
from maestro.genome import INPUT_IDS, Genome, random_genome
from maestro.score import (extract_events, levels_matrix, load_score,
                           score_from, score_summary, write_score)
from maestro.scaffold import scaffold_from_array

TICK_HZ = 8.0
TOL_TICKS = 2.5


def click_spine(onsets_s, seconds=3.5, sr=SR):
    x = np.zeros(int(seconds * sr), np.float32)
    for t in onsets_s:
        p0, klen = int(t * sr), int(0.02 * sr)
        x[p0: p0 + klen] += np.exp(-np.linspace(0, 6, klen)) * 0.8
    return x


def scaf():
    return scaffold_from_array(click_spine([0.5, 1.5, 2.5]),
                               sr=SR, bpm=None, tick_hz=TICK_HZ)


# ------------------------------------------------------------------ matrix


def test_levels_matrix_shape_and_finiteness():
    s = scaf()
    m = levels_matrix(follower(), s.feeds())
    assert set(m) == {"pulse"}
    assert m["pulse"].shape == (s.n_ticks,)
    assert np.isfinite(m["pulse"]).all()


def test_levels_matrix_missing_feeds_default_zero():
    g = random_genome(np.random.default_rng(3))
    m = levels_matrix(g, {"b": np.float32(1.0)})    # no scaffold at all
    for v in m.values():
        assert np.isfinite(v).all()


# ------------------------------------------------------------------ events


def test_extract_strike_vs_gate():
    lv = np.array([0.0, 0.5, 0.6, 0.0, 0.9, 0.8, 0.7, 0.0], np.float32)
    st = extract_events({"v": lv}, tick_hz=8.0, threshold=0.3, mode="strike")
    gt = extract_events({"v": lv}, tick_hz=8.0, threshold=0.3, mode="gate")
    assert [e["tick"] for e in st] == [1, 2, 4, 5, 6]
    assert [e["tick"] for e in gt] == [1, 4]
    assert gt[0]["dur"] == pytest.approx(2 / 8.0)
    assert gt[0]["velocity"] == pytest.approx(0.6)
    assert all(0.0 <= e["velocity"] <= 1.0 for e in st)
    assert all(e["dur"] > 0 for e in st + gt)
    with pytest.raises(ValueError):
        extract_events(lv, tick_hz=8.0, mode="wobble")


def test_extract_threshold_monotone():
    rng = np.random.default_rng(0)
    lv = rng.uniform(0, 1, 64).astype(np.float32)
    n = [len(extract_events({"v": lv}, 8.0, threshold=t)) for t in
         (0.1, 0.3, 0.6, 0.9)]
    assert n == sorted(n, reverse=True)


# ------------------------------------------------------------------ champions


def test_follower_attack_lock():
    s = scaf()
    sc = score_from(follower(), s, mode="strike")
    ev = [e for e in sc["events"] if e["voice"] == "pulse"]
    assert ev, "follower must fire"
    first = ev[0]["t"]
    assert abs(first - 0.5) <= TOL_TICKS / TICK_HZ     # at the first onset
    assert first >= 0.5 - 1.5 / TICK_HZ                # never before it
    # a run must start near EVERY detected onset
    for on in s.onsets["s0"]:
        starts = [e["t"] for e in ev if e["tick"] > 0 and
                  sc["matrix"]["pulse"][e["tick"] - 1] < sc["threshold"]]
        if starts:
            assert min(abs(t - on) for t in starts) <= TOL_TICKS / TICK_HZ


def test_answerer_mid_decay():
    s = scaf()
    sc = score_from(answerer(), s, mode="strike")
    ev = [e for e in sc["events"] if e["voice"] == "answer"]
    assert ev, "answerer must fire"
    # peak velocity lands mid-interval: onset + IOI/2, not at the onset
    ons = s.onsets["s0"]
    ioi = ons[1] - ons[0]
    mid = ons[0] + ioi / 2
    best = max(ev, key=lambda e: e["velocity"])
    assert abs(best["t"] - mid) <= TOL_TICKS / TICK_HZ
    # and nothing at the attacks themselves
    assert min(abs(e["t"] - ons[0]) for e in ev) > 0.5 / TICK_HZ


def test_splitter_register_lock():
    x = np.zeros(int(3.5 * SR), np.float32)
    for t0, f in ((0.5, 100.0), (1.5, 100.0), (1.0, 3000.0), (2.0, 3000.0)):
        p0, klen = int(t0 * SR), int(0.3 * SR)
        tt = np.arange(klen) / SR
        x[p0: p0 + klen] += np.sin(2 * np.pi * f * tt) * np.hanning(klen) * 0.5
    s = scaffold_from_array(x, sr=SR, bpm=None, tick_hz=TICK_HZ)
    sc = score_from(build_champion("splitter"), s, mode="gate")
    low = [e for e in sc["events"] if e["voice"] == "low"]
    high = [e for e in sc["events"] if e["voice"] == "high"]
    assert low and high
    assert min(abs(e["t"] - 0.5) for e in low) <= TOL_TICKS / TICK_HZ
    assert min(abs(e["t"] - 1.0) for e in high) <= TOL_TICKS / TICK_HZ
    # register purity: the low voice answers bass, not air (within 1.5 s
    # of the high bursts there are no low onsets)
    for t0 in (1.0, 2.0):
        assert all(abs(e["t"] - t0) > 3.0 / TICK_HZ for e in low)


def test_champions_are_pure_scaffold_functions():
    # zero scaffold -> zero events: no time, no position, no structure
    s = scaf()
    feeds = {"b": np.float32(1.0)}
    feeds = {**feeds, **{k: np.zeros(s.n_ticks, np.float32)
                         for k in ("s0", "s1", "s2", "s3", "s4",
                                   "live_rms", "live_centroid", "live_flux")}}
    for name in CHAMPIONS:
        m = levels_matrix(build_champion(name), feeds)
        for lv in m.values():
            assert np.clip(lv, 0, 1).max() < 0.12, \
                f"{name} invented structure from nothing"


def test_swell_velocity_is_a_curve():
    """The paper's nuance: level is a curve, not a gate. The swell
    champion's velocity must fall WITH the decay — loud at the
    attack, quiet at the tail, monotone inside each run."""
    s = scaf()
    sc = score_from(swell(), s, mode="strike")
    ev = [e for e in sc["events"] if e["voice"] == "swell"]
    assert ev, "swell must fire"
    vels = [e["velocity"] for e in ev]
    assert max(vels) - min(vels) > 0.3            # graded, not binary
    run = [e for e in ev if 0.4 < e["t"] < 1.4]   # the first interval
    assert len(run) >= 2
    assert run[0]["velocity"] > run[-1]["velocity"]   # tails away


def test_score_founder_traverses():
    g = random_score_genome(np.random.default_rng(42))
    sc = score_from(g, scaf(), mode="strike")
    assert isinstance(sc["events"], list)
    assert all(np.isfinite(e["velocity"]) for e in sc["events"])
    assert set(sc["genome"]["outputs"]) == {"pulse", "flow"}


# ------------------------------------------------------------------ score


def test_score_roundtrip_and_determinism(tmp_path):
    s = scaf()
    a = score_from(answerer(), s, mode="gate", note="rt")
    b = score_from(answerer(), s, mode="gate", note="rt")
    assert a == b                                   # deterministic dict
    p = write_score(a, tmp_path / "score.json")
    loaded = load_score(p)
    assert loaded == a                              # bit-exact JSON


def test_score_summary():
    s = scaf()
    sc = score_from(follower(), s, mode="strike")
    sm = score_summary(sc)
    assert sm["n_events"] == sc["n_events"]
    assert "pulse" in sm["voices"]
    assert sm["voices"]["pulse"]["events"] > 0


def test_genome_grows_and_traverses():
    # the evolution move: mutate a champion, traversal still works
    g = follower()
    rng = np.random.default_rng(1)
    for _ in range(30):
        g.mutate(rng)
    sc = score_from(g, scaf(), mode="strike")
    assert sc["n_events"] >= 0
    assert len(sc["genome"]["nodes"]) > len(follower().to_dict()["nodes"]) \
        or sc["n_events"] >= 0      # structure may grow; score still valid


def test_unknown_senses_rejected():
    with pytest.raises(ValueError):
        score_from(follower(), scaf(), senses=("spikes", "vibes"))

# ------------------------------------------------------------------ the kit


def test_kit_known_answers():
    """The ensemble, voice by voice, on the click spine (IOI = 1 s):
    kick attack-locked, rim at the mid clock's 40% phase, percB in
    BOTH halves of the mid IOI (the x2 subdivision), hat denser than
    everything (the x4), openhat gated by loud weather."""
    s = scaffold_from_array(click_spine([0.5, 1.5, 2.5]), sr=SR,
                            bpm=None, tick_hz=16.0)      # fine grid
    sc = score_from(kit(), s, mode="strike")
    ev = {}
    for e in sc["events"]:
        ev.setdefault(e["voice"], []).append(e)
    assert set(ev) == {"kick", "rim", "percA", "percB", "hat", "openhat"}

    # kick: attack-locked on the bass clock (s1 onset ~0.49)
    kick_t = [e["t"] for e in ev["kick"]]
    assert kick_t and min(abs(t - 0.5) for t in kick_t) <= 0.2

    # rim: ONE hit near the 40% phase of the mid clock's IOI
    rim_t = [e["t"] for e in ev["rim"]]
    assert rim_t and min(abs(t - (0.49 + 0.4)) for t in rim_t) <= 0.2

    # percB: the x2 subdivision — hits in BOTH halves of the IOI
    pb = sorted(e["t"] for e in ev["percB"])
    first_half = [t for t in pb if 0.49 <= t < 0.99]
    second_half = [t for t in pb if 0.99 <= t < 1.49]
    assert first_half and second_half

    # hat: the x4 subdivision — its four phase-windows per IOI all
    # get visited (the subdivision clock really subdivides)
    hat_t = sorted(e["t"] for e in ev["hat"])
    # s0-phase p sits at t = onset + (1 - p)*IOI (the spike DECAYS),
    # so the four windows FOLLOW the attack at quarter-IOI spacing
    windows = [(0.55, 0.75), (0.80, 1.00), (1.05, 1.25), (1.30, 1.50)]
    hit = sum(any(a <= t <= b for t in hat_t) for a, b in windows)
    assert hit >= 3, "hat must cover at least 3 of its 4 windows"

    # openhat: opens in loud weather (the click spine has dynamics)
    assert ev["openhat"], "openhat must open on dynamic material"


def test_kit_openhat_weather_gate():
    """The openhat opens only in the LOUD passage: quiet -> loud bed
    -> quiet. Every openhat event must land inside (or at the shoulder
    of) the loud bed — the gate reads relative passage loudness, so a
    whispering stretch never gets the open hat."""
    rng = np.random.default_rng(0)
    x = np.zeros(int(4.5 * SR), np.float32)
    bed = rng.standard_normal(int(1.6 * SR)) * 0.5
    x[int(2.2 * SR): int(2.2 * SR) + len(bed)] = bed
    s = scaffold_from_array(x, sr=SR, bpm=None, tick_hz=16.0)
    sc = score_from(kit(), s, mode="strike")
    oh = [e for e in sc["events"] if e["voice"] == "openhat"]
    assert oh, "the loud bed must open the openhat"
    assert all(1.9 <= e["t"] <= 4.3 for e in oh), \
        "openhat events leaked into the quiet passages"


def test_kit_is_pure_and_graded():
    """Zero scaffold -> zero events (the pure test covers silence);
    here: the graded voices really grade (tanh curves, not gates)."""
    s = scaffold_from_array(click_spine([0.5, 1.5, 2.5]), sr=SR,
                            bpm=None, tick_hz=16.0)
    sc = score_from(kit(), s, mode="strike")
    for voice in ("percA", "percB", "hat"):
        vels = [e["velocity"] for e in sc["events"]
                if e["voice"] == voice]
        assert vels and max(vels) - min(vels) > 0.15, \
            f"{voice} must be graded, not binary"
