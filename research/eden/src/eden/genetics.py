"""genetics: mutation, crossover, complexification — v2 (CPPN voices).

Mutation has two layers now:
    numeric     the world's physics genes (motion, coupling, contact,
                reactions) drift as before
    structural  the VOICE GRAPHS mutate with prism's full NEAT op set
                (drift, add_conn, add_node, add_output, swap_fn,
                toggle, duplicate) — complexification is intrinsic to
                the organism; the telemetry counts nodes/connections
                and watches it grow

Crossover stays whole-package: every agent (voice graph + physics) is
taken coherently from one parent — CPPNs recombine at the SOCIETY
level (which voices share a world), not by graph splicing (which
needs innovation numbers we deliberately don't carry).
"""

from __future__ import annotations

import numpy as np

from prism.genome import random_genome as random_cppn

from .world import (AgentSpec, ListenerSpec, MAX_AGENTS, MOTIONS,
                    REACT, WorldGenome, listener_brain)

__all__ = ["mutate", "crossover", "random_world", "structure_stats"]


def _jit(rng, v, lo, hi, rate=0.3):
    return float(np.clip(v * 10 ** rng.normal(0, rate), lo, hi))


def random_agent(rng):
    return AgentSpec(
        voice=random_cppn(rng, n_hidden=int(rng.integers(2, 4))),
        amp=float(10 ** rng.uniform(-0.8, 0.0)),
        pan_bias=float(rng.normal(0, 0.3)),
        f_center=float(2 ** rng.uniform(np.log2(80), np.log2(6000))),
        span_oct=float(np.clip(rng.normal(4.0, 2.0), 1.0, 9.0)),
        sharp=float(10 ** rng.uniform(-0.15, 0.35)),
        motion=str(rng.choice(MOTIONS)),
        pos=(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)),
             float(rng.uniform(0, 1))),
        speed=float(10 ** rng.uniform(-1.8, -0.5)),
        wander=float(rng.uniform(0, 1.5)),
        home=(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)),
              float(rng.uniform(0, 1))),
        orbit_r=float(10 ** rng.uniform(-1.6, -0.4)),
        orbit_rate=float(rng.uniform(0.2, 2.5)),
        orbit_axis=(float(rng.normal(0, 1)), float(rng.normal(0, 1)),
                    float(rng.normal(0, 1))),
        coupling=float(rng.normal(0, 0.4)),
        reaction=str(rng.choice(REACT)),
        react_mag=float(10 ** rng.uniform(-1, 0.3)),
        contact_r=float(10 ** rng.uniform(-1.2, -0.4)),
        cooldown_s=float(10 ** rng.uniform(-1, 0.3)),
        wobble=float(10 ** rng.uniform(-1.5, -0.3)),
    )


def random_world(rng, seed: int) -> WorldGenome:
    n = int(rng.integers(2, 5))
    agents = [random_agent(rng) for _ in range(n)]
    listener = ListenerSpec(
        brain=listener_brain(rng),
        pos=(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)),
             float(rng.uniform(0, 1))),
        speed=float(10 ** rng.uniform(-1.7, -0.5)),
        hearing_r=float(10 ** rng.uniform(-0.9, -0.2)),
    )
    return WorldGenome(agents=agents, listener=listener, seed=int(seed))


def _copy_agent(a: AgentSpec) -> AgentSpec:
    return AgentSpec(
        voice=a.voice.copy(), amp=a.amp, pan_bias=a.pan_bias,
        f_center=a.f_center, span_oct=a.span_oct, sharp=a.sharp,
        motion=a.motion, pos=a.pos, speed=a.speed, wander=a.wander,
        home=a.home, orbit_r=a.orbit_r, orbit_rate=a.orbit_rate,
        orbit_axis=a.orbit_axis, coupling=a.coupling, reaction=a.reaction,
        react_mag=a.react_mag, contact_r=a.contact_r,
        cooldown_s=a.cooldown_s, wobble=a.wobble,
    )


def _mutate_agent(rng, a: AgentSpec, rate: float) -> AgentSpec:
    a.amp = _jit(rng, a.amp, 0.05, 1.2, rate)
    a.pan_bias = float(np.clip(a.pan_bias + rng.normal(0, 0.1), -0.9, 0.9))
    a.f_center = _jit(rng, a.f_center, 60.0, 8000.0, rate)
    a.span_oct = _jit(rng, a.span_oct, 1.0, 9.0, rate)
    a.sharp = _jit(rng, a.sharp, 0.6, 4.0, rate)
    if rng.random() < 0.10:
        a.motion = str(rng.choice(MOTIONS))
    a.speed = _jit(rng, a.speed, 0.004, 0.3, rate)
    a.wander = _jit(rng, a.wander, 0.0, 2.0, rate)
    a.home = tuple(float(np.clip(h + rng.normal(0, 0.1), 0, 1))
                   for h in a.home)
    a.orbit_r = _jit(rng, a.orbit_r, 0.02, 0.4, rate)
    a.orbit_rate = _jit(rng, a.orbit_rate, 0.05, 4.0, rate)
    a.orbit_axis = tuple(float(v + rng.normal(0, 0.2)) for v in a.orbit_axis)
    a.coupling = float(np.clip(a.coupling + rng.normal(0, 0.15), -1.5, 1.5))
    if rng.random() < 0.10:
        a.reaction = str(rng.choice(REACT))
    a.react_mag = _jit(rng, a.react_mag, 0.01, 3.0, rate)
    a.contact_r = _jit(rng, a.contact_r, 0.02, 0.3, rate)
    a.cooldown_s = _jit(rng, a.cooldown_s, 0.1, 5.0, rate)
    a.wobble = _jit(rng, a.wobble, 0.005, 0.5, rate)
    a.voice.mutate(rng)                       # the NEAT layer, full op set
    return a


def mutate(g: WorldGenome, rng: np.random.Generator, rate: float = 0.3,
           seed: int = 0) -> WorldGenome:
    child = WorldGenome(
        agents=[_copy_agent(a) for a in g.agents],
        listener=ListenerSpec(
            brain=g.listener.brain.copy(),
            pos=g.listener.pos, speed=g.listener.speed,
            hearing_r=g.listener.hearing_r, seek=g.listener.seek),
        seed=int(seed),
    )
    for a in child.agents:
        _mutate_agent(rng, a, rate)
    l = child.listener
    l.speed = _jit(rng, l.speed, 0.004, 0.3, rate)
    l.hearing_r = _jit(rng, l.hearing_r, 0.05, 0.6, rate)
    l.pos = tuple(float(np.clip(p + rng.normal(0, 0.1), 0, 1))
                  for p in l.pos)
    l.brain.mutate(rng)                       # the navigator grows too
    r = rng.random()
    if r < 0.12 and len(child.agents) < MAX_AGENTS:       # agent birth
        child.agents.append(random_agent(rng))
    elif r < 0.20 and len(child.agents) > 2:              # agent death
        child.agents.pop(int(rng.integers(len(child.agents))))
    if rng.random() < 0.02:                                # the leap
        axis = int(rng.integers(3))
        if axis == 0 and child.agents:
            child.agents[0].voice.mutate(rng)             # double strike
            child.agents[0].voice.mutate(rng)
        elif axis == 1:
            child.listener.brain.mutate(rng)
            child.listener.brain.mutate(rng)
        elif axis == 2:
            child.agents[int(rng.integers(len(child.agents)))].coupling = \
                float(rng.uniform(0.6, 1.2))
    return child


def crossover(a: WorldGenome, b: WorldGenome, rng: np.random.Generator,
              seed: int = 0) -> WorldGenome:
    """Per-agent uniform recombination: every agent is a coherent
    package (voice graph + physics); the listener is a package too.
    Society-level mixing — no graph splicing."""
    n = max(len(a.agents), len(b.agents))
    agents = []
    for i in range(min(n, MAX_AGENTS)):
        src = a if (i < len(a.agents)) and \
            ((i >= len(b.agents)) or rng.random() < 0.5) else b
        if i < len(src.agents):
            agents.append(_copy_agent(src.agents[i]))
    if not agents:
        agents = [random_agent(rng)]
    src_l = a if rng.random() < 0.5 else b
    listener = ListenerSpec(
        brain=src_l.listener.brain.copy(),
        pos=src_l.listener.pos, speed=src_l.listener.speed,
        hearing_r=src_l.listener.hearing_r, seek=src_l.listener.seek)
    return WorldGenome(agents=agents, listener=listener, seed=int(seed))


def structure_stats(g: WorldGenome) -> dict:
    """The complexification ledger: genomes and societies, measured
    without listening."""
    voices = [a.voice for a in g.agents]
    return {
        "agents": len(g.agents),
        "voice_nodes": float(np.mean([len(v.nodes) for v in voices]))
        if voices else 0.0,
        "voice_conns": float(np.mean([len(v.conns) for v in voices]))
        if voices else 0.0,
        "brain_nodes": len(g.listener.brain.nodes),
        "coupled": sum(1 for a in g.agents if abs(a.coupling) > 0.15),
        "orbiting": sum(1 for a in g.agents if a.motion == "ORBIT"),
        "outputs": float(np.mean(
            [len(v.outputs) for v in voices])) if voices else 0.0,
    }