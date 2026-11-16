"""phase A acceptance: the organism's physics, proven.

Determinism, bounds, the integer-mip anti-alias claim, mode
distinctness (the three grammars must sound different), label sanity,
structural ops, heritability (the 20-mutation walk), json round-trip,
speed.
"""

import json
import time

import numpy as np
import pytest

from scan import (WtGenome, render, random_genome, mutate, viable,
                   write_wav, to_dict, from_dict, tonality, champion,
                   CHAMPIONS)
from scan.organism import (op_frame_interp, op_frame_dup, op_frame_birth,
                            op_frame_death, op_partial_birth,
                            op_partial_death, op_voice_birth,
                            op_voice_death, MIN_FRAMES, MAX_FRAMES,
                            MIN_PARTIALS, MAX_PARTIALS, MIN_VOICES,
                            MAX_VOICES, SECONDS)


def _founder(seed=3):
    return random_genome(np.random.default_rng(seed), seed=seed * 101)


# ------------------------------------------------------------------ physics

def test_determinism():
    g = _founder(1)
    a = render(g)
    b = render(g)
    assert a.shape == b.shape
    assert np.array_equal(a, b)                       # bit-identical


def test_shapes_and_bounds():
    for seed in (0, 1, 2):
        g = _founder(seed)
        x = render(g, seconds=1.0)
        assert x.shape == (44100, 2)
        assert np.isfinite(x).all()
        assert float(np.abs(x).max()) <= 1.0
        rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
        assert 0.05 < rms < 0.6                        # limiter target 0.18


def test_silent_genome_renders_zero():
    g = _founder(4)
    for fr in g.frames:
        fr["partials"] = [0.0] * len(fr["partials"])
    x = render(g, seconds=0.5)
    assert float(np.abs(x).max()) == 0.0


def test_render_length_exact():
    g = _founder(5)
    for secs in (0.5, 2.0, 3.3):
        x = render(g, seconds=secs)
        assert x.shape[0] == int(secs * 44100)


# ------------------------------------------------------------------ the mip

def _band_energy(x, sr, lo, hi):
    m = x.mean(axis=1)
    S = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(m.size, 1.0 / sr)
    band = (f >= lo) & (f < hi)
    total = S.sum() + 1e-12
    return float(S[band].sum() / total)


def test_mipmap_kills_high_partial_alias():
    """f0=2000 with 32 flat partials: everything above partial 2 must
    be gone or gated — no aliased partials may survive in-band."""
    g = WtGenome(
        f0=2000.0,
        pitch={"type": "fixed", "rate": 0.5, "depth": 0.0, "end": 1.0},
        frames=[{"partials": [1.0] * 32, "inharm": 0.0}] * 8,
        morph={"type": "lfo", "rate": 0.1, "depth": 0.0, "shape": "sin",
               "duty": 0.0, "bps": [0.5, 0.5, 0.5]},
        voices=[{"detune": 0.0, "level": 1.0, "pan": 0.0,
                 "morph_offset": 0.0, "attack": 0.01, "decay_tau": 8.0}],
        vector={"type": "fixed", "x0": 0.0, "y0": 0.0, "rate_x": 0.1,
                "rate_y": 0.13, "depth": 0.0, "shape_x": "sin",
                "shape_y": "tri"},
        seed=9,
    )
    x = render(g, seconds=1.0)
    # partials 1,2 at 2k,4k survive; partial 3 (6k > edge 4961) must not
    leak = _band_energy(x, 44100, 4400.0, 5400.0)
    assert leak < 1e-3, f"alias leak {leak:.4f} in the 4.4-5.4k band"


def test_mipmap_keeps_low_pitch_harmonics():
    """The mirror check: at f0=100 all 32 partials are legal and must
    be present — the mip must not over-truncate."""
    g = WtGenome(
        f0=100.0,
        pitch={"type": "fixed", "rate": 0.5, "depth": 0.0, "end": 1.0},
        frames=[{"partials": [1.0] * 32, "inharm": 0.0}] * 8,
        morph={"type": "lfo", "rate": 0.1, "depth": 0.0, "shape": "sin",
               "duty": 0.0, "bps": [0.5, 0.5, 0.5]},
        voices=[{"detune": 0.0, "level": 1.0, "pan": 0.0,
                 "morph_offset": 0.0, "attack": 0.01, "decay_tau": 8.0}],
        vector={"type": "fixed", "x0": 0.0, "y0": 0.0, "rate_x": 0.1,
                "rate_y": 0.13, "depth": 0.0, "shape_x": "sin",
                "shape_y": "tri"},
        seed=9,
    )
    x = render(g, seconds=1.0)
    hi = _band_energy(x, 44100, 2500.0, 3600.0)        # partials ~26-36
    assert hi > 1e-3, f"high partials missing: {hi:.5f}"


def test_sweep_matches_hi_render_band_content():
    """A glide across the mip boundaries at standard rate must stay
    close to the full-rate render where both are band-legal."""
    g = champion("surfer")
    lo = render(g, seconds=2.0)
    hi = render(g, seconds=2.0, hi=True)
    assert float(np.abs(lo).max()) > 0.1
    assert float(np.abs(hi).max()) > 0.1
    # both must be audible and finite; exact equality is not the claim
    # (different edges), similar loudness is
    rms = lambda a: float(np.sqrt((a.astype(np.float64) ** 2).mean()))
    assert abs(rms(lo) - rms(hi)) < 0.05


# ------------------------------------------------------------------ grammar

def _corr(a, b):
    x, y = a.mean(axis=1), b.mean(axis=1)
    x = x - x.mean()
    y = y - y.mean()
    return float(np.abs(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)
                                  + 1e-12))


def test_champion_labels():
    assert champion("organpipe").archetype() == "organ"
    assert champion("surfer").archetype() == "scan"
    assert champion("sequencer").archetype() == "sequence"


def test_modes_are_distinct():
    """Same frames, three grammars: lfo, curve, step must sound different."""
    frames = champion("sequencer").frames

    def mk(mtype, duty=0.0):
        return WtGenome(
            f0=220.0,
            pitch={"type": "fixed", "rate": 0.5, "depth": 0.0, "end": 1.0},
            frames=[dict(f) for f in frames],
            morph={"type": mtype, "rate": 2.0, "depth": 1.0,
                   "shape": "saw", "duty": duty, "bps": [0.1, 0.9, 0.2]},
            voices=[{"detune": 0.0, "level": 1.0, "pan": 0.0,
                     "morph_offset": 0.0, "attack": 0.01, "decay_tau": 8.0}],
            vector={"type": "fixed", "x0": 0.0, "y0": 0.0, "rate_x": 0.1,
                    "rate_y": 0.13, "depth": 0.0, "shape_x": "sin",
                    "shape_y": "tri"},
            seed=7)

    a = render(mk("lfo"), seconds=2.0)
    b = render(mk("curve"), seconds=2.0)
    c = render(mk("step"), seconds=2.0)
    assert _corr(a, b) < 0.98
    assert _corr(a, c) < 0.95
    assert _corr(b, c) < 0.95
    # the stepped scan must actually step: block-rms variance high
    m = c.mean(axis=1)
    blocks = np.array_split(m, 32)
    br = [float(np.sqrt((bl ** 2).mean())) for bl in blocks]
    assert max(br) / (min(br) + 1e-9) > 1.3


def test_pitch_path_matters():
    """The axis prism froze: fixed vs glide must differ."""
    g = champion("surfer")
    fixed = WtGenome(**{**to_dict(g)["genome"],
                       "pitch": {"type": "fixed", "rate": 0.5,
                                 "depth": 0.0, "end": 1.0}})
    assert _corr(render(g, 2.0), render(fixed, 2.0)) < 0.98


def test_champions_viable_and_distinct():
    auds = {}
    for name in CHAMPIONS:
        g = champion(name)
        x = render(g, seconds=2.0)
        assert viable(x), f"{name} failed the minimal criterion"
        auds[name] = x
    names = list(auds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert _corr(auds[names[i]], auds[names[j]]) < 0.9


# ------------------------------------------------------------------ labels

def test_random_labels_are_words():
    from scan import ARCHETYPES
    for seed in range(12):
        g = _founder(seed)
        assert g.archetype() in ARCHETYPES


def test_the_greedy_trap_is_stable():
    """At the purity optimum (two partials, static morph, one voice,
    fixed pitch) mutation almost never purifies further — the metric's
    fixed point, which is what makes greedy converge and stay boring."""
    pure = WtGenome(
        f0=220.0,
        pitch={"type": "fixed", "rate": 0.5, "depth": 0.0, "end": 1.0},
        frames=[{"partials": [1.0, 0.01], "inharm": 0.0} for _ in range(4)],
        morph={"type": "lfo", "rate": 0.1, "depth": 0.0, "shape": "sin",
               "duty": 0.0, "bps": [0.5, 0.5, 0.5]},
        voices=[{"detune": 0.0, "level": 1.0, "pan": 0.0,
                 "morph_offset": 0.0, "attack": 0.01, "decay_tau": 8.0}],
        vector={"type": "fixed", "x0": 0.0, "y0": 0.0, "rate_x": 0.1,
                "rate_y": 0.13, "depth": 0.0, "shape_x": "sin",
                "shape_y": "tri"},
        seed=1,
    )
    t0 = tonality(pure)
    rng = np.random.default_rng(2)
    purer = 0
    for _ in range(12):
        c = mutate(pure, rng, seed=1)
        if tonality(c) > t0:
            purer += 1
    assert purer <= 1, f"{purer}/12 children purer than the optimum"


# ------------------------------------------------------------------ genetics

def test_structural_ops_respect_bounds():
    g = _founder(6)
    rng = np.random.default_rng(0)
    for _ in range(200):
        op_frame_interp(g, rng)
        op_frame_dup(g, rng)
        op_frame_birth(g, rng)
        op_frame_death(g, rng)
        op_partial_birth(g, rng)
        op_partial_death(g, rng)
        op_voice_birth(g, rng)
        op_voice_death(g, rng)
        assert MIN_FRAMES <= len(g.frames) <= MAX_FRAMES
        assert MIN_VOICES <= len(g.voices) <= MAX_VOICES
        for fr in g.frames:
            assert MIN_PARTIALS <= len(fr["partials"]) <= MAX_PARTIALS


def test_mutation_walk_stays_alive():
    """Phase A acceptance: a 20-mutation walk keeps >= 60% viable."""
    alive = 0
    walks = 0
    for seed in range(5):
        g = _founder(seed + 20)
        rng = np.random.default_rng(seed + 100)
        if not viable(render(g)):
            continue
        parent = g
        for step in range(20):
            child = mutate(parent, rng, seed=int(rng.integers(1 << 30)))
            if viable(render(child)):
                alive += 1
                parent = child
            walks += 1
    assert walks >= 80
    assert alive / walks >= 0.6, f"only {alive}/{walks} viable"


def test_mutation_is_heritable():
    """A tiny mutation must move the child less than a big one
    (echo bench's monotonicity claim, organism-level)."""
    from scan.champions import stats
    g = champion("surfer")
    base = stats(render(g, 2.0))["centroid_hz"]
    small, big = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        c1 = mutate(g, rng, rate=0.05, seed=seed)
        c2 = mutate(g, rng, rate=0.6, seed=seed)
        small.append(abs(stats(render(c1, 2.0))["centroid_hz"] - base))
        big.append(abs(stats(render(c2, 2.0))["centroid_hz"] - base))
    assert np.mean(small) < np.mean(big)


def test_json_roundtrip_bits():
    g = _founder(8)
    d = to_dict(g)
    g2 = from_dict(json.loads(json.dumps(d)))
    assert np.array_equal(render(g), render(g2))


# ------------------------------------------------------------------ speed

def test_render_speed():
    g = _founder(9)
    render(g)                                        # warm the caches
    t0 = time.time()
    for _ in range(5):
        render(g)
    per = (time.time() - t0) / 5 * 1000
    assert per < 150, f"{per:.0f} ms per render — too slow for breeding"


def test_founder_wav_writes(tmp_path):
    g = _founder(10)
    p = write_wav(tmp_path / "spec.wav", render(g, seconds=0.5))
    assert (tmp_path / "spec.wav").stat().st_size > 4000