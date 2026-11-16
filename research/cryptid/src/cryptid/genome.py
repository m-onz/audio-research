"""genome: the process network — cryptid's growable phenotype.

A genome is a directed network of heterogeneous dynamical units:
    nodes: each has a law (jerk, excitable, fire, osc, sand, map,
           bool, delay, clamp, bytefeed), per-node params, per-node dt
    edges: signed weighted couplings between nodes

The genome is unbounded (can always grow — brant_gecco19). The phenotype
is the time-series of node states, stepped simultaneously. The ear
genome decides which slice of that state becomes the sound you hear.

Mutations: param drift (log-scale), law swap, node birth/death, edge
add/remove/rewire, subgraph duplication, graft from another lineage,
homeostatic param drift (RAAHN: params move online to keep own state
alive).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .process import LAWS, LAW_NAMES, default_params, n_state_vars

__all__ = ["ProcessGenome", "Edge", "random_genome", "mutate_genome"]


@dataclass
class Edge:
    src: int
    dst: int
    gain: float


@dataclass
class ProcessGenome:
    genome_id: int = 0
    # node data (parallel lists)
    laws: list = field(default_factory=list)     # ["jerk", "osc", ...]
    params: list = field(default_factory=list)   # [dict, dict, ...]
    dts: list = field(default_factory=list)       # [0.001, 0.01, ...]
    states: list = field(default_factory=list)   # initial state vectors
    edges: list = field(default_factory=list)     # [Edge, Edge, ...]
    # lineage
    birth_gen: int = 0
    parent_id: int | None = None
    mutation: str = ""

    @property
    def n_nodes(self) -> int:
        return len(self.laws)

    def total_state_vars(self) -> int:
        return sum(n_state_vars(l) for l in self.laws)

    def step(self, n_steps: int, rng: np.random.Generator) -> np.ndarray | None:
        """Advance the network n_steps; return (n_steps, total_state_vars)
        state matrix, or None on death (NaN)."""
        if self.n_nodes == 0:
            return None
        total = self.total_state_vars()
        out = np.zeros((n_steps, total), dtype=np.float32)
        states = [s.copy() for s in self.states]

        # precompute edge adjacency: dst -> [(src, gain), ...]
        incoming = [[] for _ in range(self.n_nodes)]
        for e in self.edges:
            if e.src < self.n_nodes and e.dst < self.n_nodes:
                incoming[e.dst].append((e.src, e.gain))

        for t in range(n_steps):
            # gather inputs for each node
            all_states = np.concatenate(states) if states else np.zeros(0)
            for i in range(self.n_nodes):
                if not incoming[i]:
                    inp = None
                else:
                    inp = np.array([all_states[e_src:e_src + 1][0] * g
                                    for e_src, g in incoming[i]], dtype=np.float32)
                new = self._step_node(i, states[i], inp, rng)
                if new is None:
                    return None  # death
                states[i] = new
            out[t] = np.concatenate(states)

        # save evolved states back for next call
        self.states = [s.copy() for s in states]
        return out

    def _step_node(self, i: int, state: np.ndarray, inp, rng) -> np.ndarray | None:
        from .process import step_unit

        # homeostatic param drift (RAAHN): params drift toward keeping
        # their own state alive (bounded, non-zero variance)
        if abs(float(state[0])) < 1e-6:
            # state dead: nudge a param to revive
            p = self.params[i]
            keys = list(p.keys())
            if keys:
                k = keys[int(rng.integers(len(keys)))]
                p[k] = float(p[k]) * 1.05 + 0.001

        return step_unit(self.laws[i], state, self.params[i], self.dts[i], inp)


def random_genome(genome_id: int, rng: np.random.Generator,
                  n_nodes: int = 6, n_edges: int = 8) -> ProcessGenome:
    """Generate a random network: random laws, random couplings."""
    laws, params, dts, states = [], [], [], []
    for _ in range(n_nodes):
        law = LAW_NAMES[int(rng.integers(len(LAW_NAMES)))]
        p = default_params(law)
        # jitter params
        for k in p:
            if isinstance(p[k], (int, float)) and not isinstance(p[k], bool):
                p[k] = float(p[k]) * float(10 ** rng.uniform(-0.5, 0.5))
        laws.append(law)
        params.append(p)
        dt = float(10 ** rng.uniform(-4, -2))  # 0.1ms to 10ms
        dts.append(dt)
        nv = n_state_vars(law)
        states.append(rng.uniform(-0.5, 0.5, nv).astype(np.float32))

    edges = []
    for _ in range(n_edges):
        src = int(rng.integers(n_nodes))
        dst = int(rng.integers(n_nodes))
        gain = float(rng.uniform(-2, 2))
        edges.append(Edge(src=src, dst=dst, gain=gain))

    return ProcessGenome(genome_id=genome_id, laws=laws, params=params,
                         dts=dts, states=states, edges=edges)


def mutate_genome(g: ProcessGenome, rng: np.random.Generator,
                  rate: float = 0.3) -> ProcessGenome:
    """Asexual mutation: one or two ops from the mutation menu."""
    child = copy.deepcopy(g)
    child.parent_id = g.genome_id
    child.genome_id = 0  # assigned by the queue
    child.mutation = ""

    n_mut = 1 + int(rng.random() < 0.3)
    for _ in range(n_mut):
        op = rng.choice(["param", "law", "edge", "node_add", "node_del", "dup"], p=[0.3, 0.15, 0.2, 0.15, 0.1, 0.1])

        if op == "param" and child.n_nodes > 0:
            i = int(rng.integers(child.n_nodes))
            keys = [k for k, v in child.params[i].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if keys:
                k = keys[int(rng.integers(len(keys)))]
                child.params[i][k] = float(child.params[i][k]) * float(10 ** rng.normal(0, rate))
                child.mutation += f"param[{i}].{k} "

        elif op == "law" and child.n_nodes > 0:
            i = int(rng.integers(child.n_nodes))
            new_law = LAW_NAMES[int(rng.integers(len(LAW_NAMES)))]
            child.laws[i] = new_law
            child.params[i] = default_params(new_law)
            nv = n_state_vars(new_law)
            child.states[i] = rng.uniform(-0.5, 0.5, nv).astype(np.float32)
            child.mutation += f"law[{i}]->{new_law} "

        elif op == "edge" and child.n_nodes > 1:
            if child.edges and rng.random() < 0.5:
                # rewire
                e = child.edges[int(rng.integers(len(child.edges)))]
                e.src = int(rng.integers(child.n_nodes))
                e.dst = int(rng.integers(child.n_nodes))
                e.gain = float(rng.uniform(-2, 2))
            else:
                # add edge
                child.edges.append(Edge(
                    src=int(rng.integers(child.n_nodes)),
                    dst=int(rng.integers(child.n_nodes)),
                    gain=float(rng.uniform(-2, 2))))
            child.mutation += "edge "

        elif op == "node_add":
            law = LAW_NAMES[int(rng.integers(len(LAW_NAMES)))]
            p = default_params(law)
            for k in p:
                if isinstance(p[k], (int, float)) and not isinstance(p[k], bool):
                    p[k] = float(p[k]) * float(10 ** rng.uniform(-0.3, 0.3))
            child.laws.append(law)
            child.params.append(p)
            child.dts.append(float(10 ** rng.uniform(-4, -2)))
            nv = n_state_vars(law)
            child.states.append(rng.uniform(-0.5, 0.5, nv).astype(np.float32))
            # connect it
            if child.n_nodes > 1:
                child.edges.append(Edge(
                    src=int(rng.integers(child.n_nodes)),
                    dst=child.n_nodes - 1,
                    gain=float(rng.uniform(-2, 2))))
                child.edges.append(Edge(
                    src=child.n_nodes - 1,
                    dst=int(rng.integers(child.n_nodes - 1)),
                    gain=float(rng.uniform(-2, 2))))
            child.mutation += "node_add "

        elif op == "node_del" and child.n_nodes > 2:
            i = int(rng.integers(child.n_nodes))
            child.laws.pop(i)
            child.params.pop(i)
            child.dts.pop(i)
            child.states.pop(i)
            child.edges = [e for e in child.edges if e.src != i and e.dst != i]
            # remap indices
            for e in child.edges:
                if e.src > i: e.src -= 1
                if e.dst > i: e.dst -= 1
            child.mutation += "node_del "

        elif op == "dup" and child.n_nodes >= 2:
            # subgraph duplication: copy a connected pair, copies diverge later
            src = int(rng.integers(child.n_nodes - 1))
            dst = int(rng.integers(child.n_nodes - 1))
            law = child.laws[src]
            p = dict(child.params[src])
            dt = child.dts[src]
            st = child.states[src].copy()
            old_n = child.n_nodes
            child.laws.append(law)
            child.params.append(p)
            child.dts.append(dt)
            child.states.append(st)
            # wire the copy to the old destination
            child.edges.append(Edge(src=old_n, dst=dst, gain=1.0))
            child.edges.append(Edge(src=dst, dst=old_n, gain=0.5))
            child.mutation += "dup "

    return child