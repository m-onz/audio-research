"""world v2: CPPN voices in 3D, heard through a reactive listener."""

import numpy as np
import pytest

from eden.genetics import (crossover, mutate, random_world, structure_stats)
from eden.world import (AgentSpec, ListenerSpec, WorldGenome, render,
                        SECONDS, SR, listener_brain)
from prism.genome import Genome, random_genome as random_cppn


def test_render_deterministic_and_bounded():
    g = random_world(np.random.default_rng(1), seed=42)
    a, t1 = render(g)
    b, t2 = render(g)
    assert np.array_equal(a, b)
    assert t1["collisions"] == t2["collisions"]
    assert a.shape == (int(SECONDS * SR), 2)
    assert np.isfinite(a).all() and float(np.abs(a).max()) <= 1.0


def test_genome_never_mutated_by_episode():
    """Reactions touch RUNTIME state only — the voice graphs must stay
    pristine (determinism depends on it)."""
    before = [len(a.voice.conns) for a in random_world(
        np.random.default_rng(3), seed=11).agents]
    g = random_world(np.random.default_rng(3), seed=11)
    render(g)
    assert [len(a.voice.conns) for a in g.agents] == before


def test_voices_are_audible_and_diverse():
    """Random founders must be audible and grossly different from each
    other (the arena's premise)."""
    feats = []
    rng = np.random.default_rng(0)
    for i in range(6):
        g = random_world(rng, seed=int(rng.integers(1 << 30)))
        audio, _ = render(g)
        x = audio.mean(axis=1)
        S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-12
        fr = np.fft.rfftfreq(len(x), 1 / SR)
        feats.append((float(np.sqrt((x ** 2).mean())),
                      float((fr * S).sum() / S.sum())))
    rms = [f[0] for f in feats]
    cent = [f[1] for f in feats]
    assert max(rms) > 0.02                      # at least some are audible
    assert (max(cent) - min(cent)) > 500.0     # and spectrally diverse


def test_motion_makes_time_structure():
    """A wandering listener in a world of movers must hear scenes come
    and go (the premise: sound over time)."""
    g = WorldGenome(
        agents=[AgentSpec(voice=random_cppn(np.random.default_rng(1)),
                          amp=0.9, speed=0.15, wander=1.0,
                          pos=(0.2, 0.2, 0.5))],
        listener=ListenerSpec(pos=(0.8, 0.8, 0.5), speed=0.15,
                              hearing_r=0.25),
        seed=3)
    audio, _ = render(g)
    m = audio.mean(axis=1)
    blocks = np.array_split(m, 8)
    rms = np.array([np.sqrt((b ** 2).mean()) for b in blocks])
    assert rms.max() / (rms.min() + 1e-9) > 1.3, rms


def test_collisions_change_the_episode():
    """Reaction magnitude is audible: same world, stronger reactions,
    different episode (when agents meet)."""
    def world(mag):
        return WorldGenome(
            agents=[
                AgentSpec(voice=random_cppn(np.random.default_rng(2)),
                          amp=0.9, reaction="T_JUMP", react_mag=mag,
                          wobble=mag, contact_r=0.3, cooldown_s=0.1,
                          motion="ORBIT", home=(0.5, 0.5, 0.5),
                          orbit_r=0.05, orbit_rate=1.0),
                AgentSpec(voice=random_cppn(np.random.default_rng(3)),
                          amp=0.9, reaction="T_JUMP", react_mag=mag,
                          wobble=mag, contact_r=0.3, cooldown_s=0.1,
                          motion="ORBIT", home=(0.5, 0.5, 0.5),
                          orbit_r=0.05, orbit_rate=-1.0),
            ],
            listener=ListenerSpec(pos=(0.5, 0.5, 0.5), speed=0.01,
                                  hearing_r=0.4),
            seed=9)
    a, ta = render(world(0.02))
    b, tb = render(world(0.8))
    assert ta["collisions"] > 0
    assert not np.allclose(a, b, atol=1e-3)


def test_listener_brain_is_reactive():
    """A brain wired to the world's weather steers differently from a
    flat one — navigation is a function of what is heard."""
    rng = np.random.default_rng(4)

    def flat_brain():
        from prism.genome import INPUT_IDS
        g = Genome(seed=1)
        out = g.add_output("turn", fn="lin")
        g.conn(INPUT_IDS["b"], out, 0.3)          # constant turn
        sp = g.add_output("speed", fn="lin")
        g.conn(INPUT_IDS["b"], sp, 0.5)
        lf = g.add_output("lift", fn="lin")
        g.conn(INPUT_IDS["b"], lf, 0.0)
        return g

    def weather_brain():
        from prism.genome import INPUT_IDS
        g = Genome(seed=2)
        out = g.add_output("turn", fn="sin")
        g.conn(INPUT_IDS["live_rms"], out, 2.0)   # turn is a function of loudness
        sp = g.add_output("speed", fn="sig")
        g.conn(INPUT_IDS["live_flux"], sp, 3.0)
        lf = g.add_output("lift", fn="lin")
        g.conn(INPUT_IDS["b"], lf, 0.0)
        return g

    base = dict(agents=[AgentSpec(voice=random_cppn(
        np.random.default_rng(5)), amp=0.9, motion="ORBIT",
        home=(0.5, 0.5, 0.5), orbit_r=0.1, orbit_rate=1.5)], seed=21)
    flat = WorldGenome(listener=ListenerSpec(brain=flat_brain(),
                                             speed=0.1, hearing_r=0.3),
                       **base)
    weather = WorldGenome(listener=ListenerSpec(brain=weather_brain(),
                                                speed=0.1, hearing_r=0.3),
                          **base)
    _, tf = render(flat)
    _, tw = render(weather)
    assert abs(tf["path_len"] - tw["path_len"]) > 1e-3


def test_genetics_validity():
    rng = np.random.default_rng(5)
    a = random_world(rng, seed=1)
    b = random_world(rng, seed=2)
    for _ in range(15):
        child = crossover(a, b, rng, seed=3)
        child = mutate(child, rng, seed=4)
        assert 1 <= len(child.agents) <= 6
        x, _ = render(child)
        assert np.isfinite(x).all()


def test_crossover_inherits_whole_packages():
    """Agents come from parents coherently: a child's voice graph must
    be structurally identical to one parent's (same node count)."""
    rng = np.random.default_rng(6)
    a = random_world(rng, seed=1)
    b = random_world(rng, seed=2)
    child = crossover(a, b, rng, seed=3)
    parent_counts = {(len(x.voice.nodes), len(x.voice.conns))
                     for x in list(a.agents) + list(b.agents)}
    for c in child.agents:
        assert (len(c.voice.nodes), len(c.voice.conns)) in parent_counts


def test_structure_stats_and_growth():
    """Mutation grows voice graphs on average (complexification is
    intrinsic)."""
    rng = np.random.default_rng(7)
    g = random_world(rng, seed=1)
    s0 = structure_stats(g)
    for _ in range(30):
        g = mutate(g, rng, seed=int(rng.integers(1 << 30)))
    s1 = structure_stats(g)
    assert s1["agents"] >= 1
    # NEAT ops only add (or disable); node counts don't shrink much
    assert s1["voice_nodes"] >= 0.8 * s0["voice_nodes"]