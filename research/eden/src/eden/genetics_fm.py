"""genetics: mutation, crossover, complexification.

    mutate(world)     parameter drift + structural change (agent
                     birth/death, technique swap, reaction rewire) —
                     the small-step kernel, plus rare leaps
    crossover(a, b)  per-agent uniform recombination: every agent is a
                     coherent gene package, taken whole from one parent
                     or the other; the listener and globals recombine
                     gene by gene (NEAT-style complexification happens
                     by mutation — structure grows when novelty rewards
                     it, and the telemetry watches it happen)
"""

from __future__ import annotations

import copy
import numpy as np

from .world import (AgentSpec, ListenerSpec, MAX_AGENTS, MOTIONS,
                   REACTIONS, TECHNIQUES, WorldGenome)

__all__ = ["mutate", "crossover", "random_world", "structure_stats"]


def _jit(rng, v, lo, hi, rate=0.35):
    return float(np.clip(v * 10 ** rng.normal(0, rate), lo, hi))


def _maybe(rng, p):
    return rng.random() < p


def random_agent(rng):
    tech = str(rng.choice(TECHNIQUES))
    motion = str(rng.choice(MOTIONS))
    return AgentSpec(
        technique=tech,
        freq=float(2 ** rng.uniform(6.0, 11.5)),
        amp=float(10 ** rng.uniform(-0.8, 0.0)),
        c_ratio=float(np.clip(round(rng.uniform(1, 4)), 1, 6)),
        m_ratio=float(2 ** rng.uniform(-1, 3)),
        index=float(10 ** rng.uniform(-1, 1.2)),
        center=float(2 ** rng.uniform(7, 12.5)),
        width=float(np.clip(10 ** rng.uniform(-1.3, 0), 0.05, 1.0)),
        rate=float(10 ** rng.uniform(-0.3, 1.4)),
        jitter=float(rng.uniform(0, 1)),
        grain_ms=float(10 ** rng.uniform(1, 2.3)),
        motion=motion,
        pos=(float(rng.uniform(0, 1)), float(rng.uniform(0, 1))),
        vel=(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))),
        speed=float(10 ** rng.uniform(-1.8, -0.5)),
        turn=float(rng.uniform(0.2, 3.0)),
        wander=float(rng.uniform(0, 1.5)),
        home=(float(rng.uniform(0, 1)), float(rng.uniform(0, 1))),
        orbit_r=float(10 ** rng.uniform(-1.6, -0.4)),
        orbit_rate=float(rng.uniform(0.2, 2.5)),
        coupling=float(rng.normal(0, 0.4)),
        reaction=str(rng.choice(REACTIONS)),
        react_mag=float(10 ** rng.uniform(-1, 0.3)),
        contact_r=float(10 ** rng.uniform(-1.2, -0.4)),
        cooldown_s=float(10 ** rng.uniform(-1, 0.3)),
        gene_wobble=float(rng.choice([0.0, 10 ** rng.uniform(-2, -0.7)])),
        pan_bias=float(rng.normal(0, 0.3)),
    )


def random_world(rng, seed: int) -> WorldGenome:
    n = int(rng.integers(2, 5))
    agents = [random_agent(rng) for _ in range(n)]
    listener = ListenerSpec(
        pos=(float(rng.uniform(0, 1)), float(rng.uniform(0, 1))),
        speed=float(10 ** rng.uniform(-1.7, -0.5)),
        turn=float(rng.uniform(0.2, 3.0)),
        wander=float(rng.uniform(0, 1.5)),
        hearing_r=float(10 ** rng.uniform(-0.9, -0.2)),
        seek=float(rng.choice([0.0, rng.uniform(0, 1.5)])),
    )
    return WorldGenome(agents=agents, listener=listener, seed=int(seed))


def _mutate_agent(rng, a: AgentSpec, rate: float) -> AgentSpec:
    a.technique = str(rng.choice(TECHNIQUES)) if _maybe(rng, 0.08) else a.technique
    if a.technique not in TECHNIQUES:
        a.technique = "TONE"
    a.freq = _jit(rng, a.freq, 30.0, 6000.0, rate)
    a.amp = _jit(rng, a.amp, 0.02, 1.2, rate)
    a.m_ratio = _jit(rng, a.m_ratio, 0.25, 8.0, rate)
    a.index = _jit(rng, a.index, 0.0, 20.0, rate)
    a.center = _jit(rng, a.center, 100.0, 8000.0, rate)
    a.width = _jit(rng, a.width, 0.05, 1.0, rate)
    a.rate = _jit(rng, a.rate, 0.1, 25.0, rate)
    a.jitter = _jit(rng, a.jitter, 0.0, 1.0, rate)
    a.grain_ms = _jit(rng, a.grain_ms, 5.0, 200.0, rate)
    if _maybe(rng, 0.10):
        a.motion = str(rng.choice(MOTIONS))
    a.speed = _jit(rng, a.speed, 0.002, 0.3, rate)
    a.wander = _jit(rng, a.wander, 0.0, 2.0, rate)
    a.home = (float(np.clip(a.home[0] + rng.normal(0, 0.1), 0, 1)),
              float(np.clip(a.home[1] + rng.normal(0, 0.1), 0, 1)))
    a.orbit_r = _jit(rng, a.orbit_r, 0.01, 0.4, rate)
    a.orbit_rate = _jit(rng, a.orbit_rate, 0.05, 4.0, rate)
    a.coupling = float(np.clip(a.coupling + rng.normal(0, 0.15), -1.5, 1.5))
    if _maybe(rng, 0.10):
        a.reaction = str(rng.choice(REACTIONS))
    a.react_mag = _jit(rng, a.react_mag, 0.01, 3.0, rate)
    a.contact_r = _jit(rng, a.contact_r, 0.02, 0.3, rate)
    a.cooldown_s = _jit(rng, a.cooldown_s, 0.1, 5.0, rate)
    a.gene_wobble = _jit(rng, a.gene_wobble, 0.0, 0.3, rate) \
        if a.gene_wobble > 0 else (0.02 if _maybe(rng, 0.05) else 0.0)
    a.pan_bias = float(np.clip(a.pan_bias + rng.normal(0, 0.1), -0.9, 0.9))
    return a


def mutate(g: WorldGenome, rng: np.random.Generator, rate: float = 0.3,
           seed: int = 0) -> WorldGenome:
    child = WorldGenome(
        agents=[copy.deepcopy(a) for a in g.agents],
        listener=copy.deepcopy(g.listener),
        seed=int(seed),
    )
    for a in child.agents:
        _mutate_agent(rng, a, rate)
    l = child.listener
    l.speed = _jit(rng, l.speed, 0.002, 0.3, rate)
    l.wander = _jit(rng, l.wander, 0.0, 2.0, rate)
    l.hearing_r = _jit(rng, l.hearing_r, 0.05, 0.6, rate)
    l.seek = _jit(rng, l.seek, 0.0, 2.0, rate) if l.seek > 0 else \
        (0.4 if _maybe(rng, 0.05) else 0.0)
    l.pos = (float(np.clip(l.pos[0] + rng.normal(0, 0.1), 0, 1)),
             float(np.clip(l.pos[1] + rng.normal(0, 0.1), 0, 1)))
    # structural change: the complexification axis
    r = rng.random()
    if r < 0.12 and len(child.agents) < MAX_AGENTS:       # agent birth
        child.agents.append(random_agent(rng))
    elif r < 0.2 and len(child.agents) > 1:               # agent death
        child.agents.pop(int(rng.integers(len(child.agents))))
    if _maybe(rng, 0.02):                                  # the leap
        axis = int(rng.integers(4))
        if axis == 0 and child.agents:
            child.agents[0].technique = str(rng.choice(TECHNIQUES))
        elif axis == 1:
            while len(child.agents) < MAX_AGENTS:
                child.agents.append(random_agent(rng))
        elif axis == 2:
            child.listener.seek = float(rng.uniform(1.0, 1.8))
        elif axis == 3 and child.agents:
            child.agents[0].rate = float(10 ** rng.uniform(0.8, 1.4))
    return child


def crossover(a: WorldGenome, b: WorldGenome, rng: np.random.Generator,
              seed: int = 0) -> WorldGenome:
    """Per-agent uniform recombination: each agent is a coherent gene
    package taken whole from one parent (keeps voices/behaviors
    together); the listener recombines gene by gene."""
    n = max(len(a.agents), len(b.agents))
    agents = []
    for i in range(min(n, MAX_AGENTS)):
        src = a if (i < len(a.agents)) and \
            ((i >= len(b.agents)) or _maybe(rng, 0.5)) else b
        if i < len(src.agents):
            agents.append(copy.deepcopy(src.agents[i]))
    if not agents:
        agents = [random_agent(rng)]
    la, lb = a.listener, b.listener
    listener = ListenerSpec(
        pos=la.pos if _maybe(rng, 0.5) else lb.pos,
        speed=la.speed if _maybe(rng, 0.5) else lb.speed,
        turn=la.turn if _maybe(rng, 0.5) else lb.turn,
        wander=la.wander if _maybe(rng, 0.5) else lb.wander,
        hearing_r=la.hearing_r if _maybe(rng, 0.5) else lb.hearing_r,
        seek=la.seek if _maybe(rng, 0.5) else lb.seek,
    )
    return WorldGenome(agents=agents, listener=listener, seed=int(seed))


def structure_stats(g: WorldGenome) -> dict:
    """The complexification telemetry: genome structure, measurable
    without listening (agents, techniques, interactions)."""
    techs = [a.technique for a in g.agents]
    return {
        "agents": len(g.agents),
        "techniques": {t: techs.count(t) for t in set(techs)},
        "n_kinds": len(set(techs)),
        "coupled": sum(1 for a in g.agents if abs(a.coupling) > 0.15),
        "wobbling": sum(1 for a in g.agents if a.gene_wobble > 0),
        "seeking": g.listener.seek > 0.3,
    }