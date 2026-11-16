"""NEAT-lite genomes: small feed-forward ANNs that drive particle systems.

Faithful to Hastings, Guha & Stanley (2009), "Interactive Evolution of
Particle Systems for Computer Graphics and Animation" (NEAT Particles):

  * inputs        : particle position (x, y), distance from system centre, bias
  * outputs       : velocity (vx, vy), colour (r, g, b) and size
  * node activation drawn from 8 functions (sine, cosine, tangent,
    bipolar sigmoid, tanh, gaussian, ramp, step)
  * mutation      : weight jitter U(-0.5, 0.5) on a random subset of connections,
    add-connection p=0.30, add-node p=0.20 (section III-F of the paper)
  * complexification: start with no hidden nodes, grow structure over time

Novelty-search extensions beyond the paper:
  * an evolvable "generation shape" gene (point / circle / line / disc emitter)
  * an evolvable "trail" gene (drop fading stationary trail particles)
  * asexual reproduction only (no crossover), so no speciation is needed
"""

from __future__ import annotations

import random
import uuid

import numpy as np

N_IN = 4                                  # x, y, dist-from-centre, bias
IN_NAMES = ("x", "y", "d", "bias")
N_OUT = 6                                 # vx, vy, r, g, b, size
OUT_NAMES = ("vx", "vy", "r", "g", "b", "size")
OUT_FIRST = N_IN                          # output node ids: [4, 10)
SHAPES = ("point", "circle", "line", "disc")

# --- the paper's eight activation functions (fig. 8) ------------------------


def _tan(x):
    return np.clip(np.tan(x), -8.0, 8.0)


def _sigmoid(x):                            # bipolar
    return 2.0 / (1.0 + np.exp(-x)) - 1.0


def _gauss(x):
    return 0.3989 * np.exp(-0.5 * x * x)


def _ramp(x):
    return np.clip(x, -1.0, 1.0)


def _step(x):
    return np.where(x < 0.0, -1.0, 1.0)


ACTS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": _tan,
    "sigmoid": _sigmoid,
    "tanh": np.tanh,
    "gauss": _gauss,
    "ramp": _ramp,
    "step": _step,
}
ACT_NAMES = tuple(ACTS)


class Genome:
    """A feed-forward ANN genome. Node ids: inputs 0..3, outputs 4..9, hidden 10+."""

    def __init__(self):
        self.nodes: dict[int, str] = {}    # node id -> activation name (hidden + outputs)
        self.conns: list[list] = []         # [src, dst, weight, innov, enabled]
        self.conn_keys: set[tuple[int, int]] = set()
        self.shape = "point"
        self.trail = False
        self.innov_counter = 0
        self.uid = uuid.uuid4().hex[:8]
        self.parent: str | None = None
        self._order: list[int] | None = None       # cached topological order
        self._incoming: dict[int, list] | None = None
        self._hidden: list[int] = []

    # ------------------------------------------------------------------ build

    @staticmethod
    def initial(rng: random.Random) -> "Genome":
        """Minimal network: every input connected to every output (fig. 7a)."""
        g = Genome()
        for oid in range(OUT_FIRST, OUT_FIRST + N_OUT):
            g.nodes[oid] = rng.choice(ACT_NAMES)
        for i in range(N_IN):
            for o in range(N_OUT):
                g._add_conn(i, OUT_FIRST + o, rng.uniform(-1.0, 1.0))
        g.shape = rng.choice(SHAPES)
        g.trail = rng.random() < 0.35
        return g

    def _add_conn(self, src, dst, w):
        self.innov_counter += 1
        self.conns.append([src, dst, w, self.innov_counter, True])
        self.conn_keys.add((src, dst))
        self._order = None

    def _rebuild(self):
        """Topological order (inputs -> hidden via Kahn -> outputs) + incoming lists."""
        self._hidden = sorted(n for n in self.nodes if n >= OUT_FIRST + N_OUT)
        hid = set(self._hidden)
        indeg = {h: 0 for h in self._hidden}
        succ = {h: [] for h in self._hidden}
        for c in self.conns:
            if c[4] and c[0] in hid and c[1] in hid:
                indeg[c[1]] += 1
                succ[c[0]].append(c[1])
        queue = [h for h in self._hidden if indeg[h] == 0]
        order_h = []
        while queue:
            h = queue.pop(0)
            order_h.append(h)
            for s in succ[h]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    queue.append(s)
        if len(order_h) != len(self._hidden):
            raise RuntimeError("cycle in genome (should be impossible)")
        self._order = list(range(N_IN)) + order_h + list(range(OUT_FIRST, OUT_FIRST + N_OUT))
        self._incoming = {n: [] for n in self._order}
        for c in self.conns:
            if c[4]:
                self._incoming[c[1]].append((c[0], c[2]))

    # ---------------------------------------------------------------- evaluate

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Batch-activate the ANN. X: (B, N_IN) -> (B, N_OUT)."""
        if self._order is None:
            self._rebuild()
        B = X.shape[0]
        vals = {i: X[:, i] for i in range(N_IN)}
        out = np.empty((B, N_OUT))
        for n in self._order[N_IN:]:
            acc = None
            for src, w in self._incoming[n]:
                v = vals[src] * w
                acc = v if acc is None else acc + v
            if acc is None:
                acc = np.zeros(B)
            vals[n] = ACTS[self.nodes[n]](acc)
            if OUT_FIRST <= n < OUT_FIRST + N_OUT:
                out[:, n - OUT_FIRST] = vals[n]
        return out

    # ------------------------------------------------------------------ mutate

    def mutate(self, rng: random.Random) -> None:
        """Paper section III-F parameters, plus shape/trail/activation genes."""
        enabled = [c for c in self.conns if c[4]]
        if enabled:
            k = rng.randint(1, len(enabled))
            for c in rng.sample(enabled, k):
                c[2] += rng.uniform(-0.5, 0.5)
        if rng.random() < 0.30:
            self._try_add_conn(rng)
        if rng.random() < 0.20:
            self._try_add_node(rng)
        if rng.random() < 0.10 and self.nodes:
            n = rng.choice(list(self.nodes))
            self.nodes[n] = rng.choice(ACT_NAMES)
        if rng.random() < 0.10:
            self.shape = rng.choice(SHAPES)
        if rng.random() < 0.08:
            self.trail = not self.trail

    def _reaches(self, start, target) -> bool:
        """DFS over enabled conns: is `target` reachable from `start`?"""
        stack, seen = [start], set()
        while stack:
            n = stack.pop()
            if n == target:
                return True
            if n in seen:
                continue
            seen.add(n)
            for c in self.conns:
                if c[4] and c[0] == n:
                    stack.append(c[1])
        return False

    def _try_add_conn(self, rng) -> None:
        src_pool = list(range(N_IN)) + self._hidden
        dst_pool = self._hidden + list(range(OUT_FIRST, OUT_FIRST + N_OUT))
        for _ in range(25):
            src, dst = rng.choice(src_pool), rng.choice(dst_pool)
            if src == dst or (src, dst) in self.conn_keys:
                continue
            if src >= OUT_FIRST + N_OUT and self._reaches(dst, src):
                continue                     # would create a cycle
            self._add_conn(src, dst, rng.uniform(-1.0, 1.0))
            return

    def _try_add_node(self, rng) -> None:
        enabled = [c for c in self.conns if c[4]]
        if not enabled:
            return
        c = rng.choice(enabled)
        c[4] = False                         # split the existing connection
        nid = max(self.nodes) + 1
        self.nodes[nid] = rng.choice(ACT_NAMES)
        self._add_conn(c[0], nid, c[2])      # keep old weight into the new node
        self._add_conn(nid, c[1], 1.0)

    # ------------------------------------------------------------- housekeeping

    def clone(self) -> "Genome":
        g = Genome.__new__(Genome)
        g.nodes = dict(self.nodes)
        g.conns = [list(c) for c in self.conns]
        g.conn_keys = set(self.conn_keys)
        g.shape, g.trail = self.shape, self.trail
        g.innov_counter = self.innov_counter
        g.uid = uuid.uuid4().hex[:8]
        g.parent = self.uid
        g._order = None
        g._incoming = None
        g._hidden = list(self._hidden)
        return g

    def size(self) -> tuple[int, int]:
        return len(self._hidden), sum(1 for c in self.conns if c[4])

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "parent": self.parent,
            "nodes": {str(k): v for k, v in self.nodes.items()},
            "conns": [list(c) for c in self.conns],
            "shape": self.shape,
            "trail": self.trail,
            "innov_counter": self.innov_counter,
        }

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        g = Genome()
        g.nodes = {int(k): v for k, v in d["nodes"].items()}
        g.conns = [list(c) for c in d["conns"]]
        g.conn_keys = {(c[0], c[1]) for c in g.conns}
        g.shape = d.get("shape", "point")
        g.trail = bool(d.get("trail", False))
        g.innov_counter = int(d.get("innov_counter", 0))
        g.uid = d.get("uid", uuid.uuid4().hex[:8])
        g.parent = d.get("parent")
        g._order = None
        return g