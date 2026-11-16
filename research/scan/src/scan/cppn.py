"""cppn: the pattern organism, torch-backed — HyperNEAT's query engine.

A compositional pattern producing network (prism's genome, the house
organism, adapted): nodes hold one function from the zoo, connections
carry weights, an explicit forward order keeps evaluation one linear
pass. The senses are DeepHyperNEAT's query convention — the CPPN is
asked for a connection weight given a SOURCE node's coordinates
(x1, y1), a TARGET's (x2, y2), their distance d, and bias b — so the
same graph paints geometric motifs into substrate weight matrices.

torch, not numpy: the substrate forward and the decode queries are
batched matmul workloads, which is what makes the sub organism a GPU
citizen (device pluggable: cpu or mps). No torch RNG anywhere —
mutation randomness comes from the numpy generator the engine passes
in; evaluation is a pure function of the graph, deterministic per
device and dtype (float32).

Genomes are plain data (dicts of floats/strings) — json round-trips;
torch only ever touches evaluate().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["Cppn", "Node", "Conn", "FUNCTIONS", "INPUTS", "INPUT_IDS",
           "random_cppn"]

# ------------------------------------------------------------------ senses

INPUTS = ("x1", "y1", "x2", "y2", "d", "b")
INPUT_IDS = {name: i for i, name in enumerate(INPUTS)}
_FIRST_FREE = 10                     # hidden/output ids start here

# ------------------------------------------------------------------ zoo


def _sin(u):
    return torch.sin(u)


def _cos(u):
    return torch.cos(u)


def _tanh(u):
    return torch.tanh(torch.clamp(u, -8.0, 8.0))


def _gauss(u):
    return torch.exp(-(u * u))


def _sig(u):
    return torch.sigmoid(torch.clamp(u, -30.0, 30.0))


def _abs(u):
    return torch.abs(u)


def _sq(u):
    return u * torch.abs(u)          # signed square: keeps the sign


def _sqrt(u):
    return torch.sign(u) * torch.sqrt(torch.abs(u) + 1e-12)


def _lin(u):
    return u


def _step(u):
    return torch.where(u > 0, torch.ones_like(u),
                      torch.where(u < 0, -torch.ones_like(u),
                                  torch.zeros_like(u)))


FUNCTIONS = {
    "sin": _sin, "cos": _cos, "tanh": _tanh, "gauss": _gauss,
    "sig": _sig, "abs": _abs, "sq": _sq, "sqrt": _sqrt,
    "lin": _lin, "step": _step,
}

# ------------------------------------------------------------------ data


@dataclass
class Node:
    fn: str = "lin"
    bias: float = 0.0
    kind: str = "hid"                 # "in" | "hid" | "out"


@dataclass
class Conn:
    src: int
    dst: int
    w: float
    on: bool = True


def _rng_ok(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) \
        else np.random.default_rng(rng)


# ------------------------------------------------------------------ genome


class Cppn:
    """A growable pattern graph. Build by hand (champions) or mutate."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.nodes: dict[int, Node] = {}
        self.conns: list[Conn] = []
        self.order: list[int] = []
        self.outputs: dict[str, int] = {}
        for name, nid in INPUT_IDS.items():
            self.nodes[nid] = Node("lin", 0.0, "in")
            self.order.append(nid)

    # ------------------------------------------------------------ build

    def _new_id(self) -> int:
        nid = _FIRST_FREE
        while nid in self.nodes:
            nid += 1
        return nid

    def add_hidden(self, fn: str = "lin", bias: float = 0.0) -> int:
        nid = self._new_id()
        self.nodes[nid] = Node(fn, bias, "hid")
        pos = len(self.order)
        for i, k in enumerate(self.order):
            if self.nodes[k].kind == "out":
                pos = i
                break
        self.order.insert(pos, nid)
        return nid

    def add_output(self, name: str, fn: str = "lin",
                   bias: float = 0.0) -> int:
        if name in self.outputs:
            raise ValueError(f"output '{name}' already exists")
        nid = self._new_id()
        self.nodes[nid] = Node(fn, bias, "out")
        self.order.append(nid)        # outputs always last
        self.outputs[name] = nid
        return nid

    def conn(self, src: int, dst: int, w: float, on: bool = True) -> None:
        assert self.order.index(src) < self.order.index(dst), \
            "connections must point strictly forward in node order"
        self.conns.append(Conn(int(src), int(dst), float(w), on))

    def pos(self, nid: int) -> int:
        return self.order.index(nid)

    def copy(self) -> "Cppn":
        g = Cppn(self.seed)
        g.nodes = {k: Node(n.fn, n.bias, n.kind)
                   for k, n in self.nodes.items()}
        g.conns = [Conn(c.src, c.dst, c.w, c.on) for c in self.conns]
        g.order = list(self.order)
        g.outputs = dict(self.outputs)
        return g

    def complexity(self) -> int:
        return len(self.nodes) + len(self.conns)

    # ------------------------------------------------------------ evaluate

    def evaluate(self, feeds: dict, names: list[str] | None = None,
                 device=None, dtype=torch.float32) -> dict:
        """One forward pass. All feeds broadcast to a common shape;
        missing feeds default to zero (bias defaults to one). Pure
        function of the graph — no RNG, deterministic per device."""
        ref = None
        vals: dict[int, torch.Tensor] = {}
        for name, nid in INPUT_IDS.items():
            v = feeds.get(name, 1.0 if name == "b" else 0.0)
            a = v if isinstance(v, torch.Tensor) \
                else torch.tensor(np.asarray(v, np.float32), dtype=dtype)
            if device is not None:
                a = a.to(device)
            vals[nid] = a
            if ref is None:
                ref = a
        dev, dt = ref.device, ref.dtype

        by_dst: dict[int, list[tuple[int, float]]] = {}
        for c in self.conns:
            if c.on:
                by_dst.setdefault(c.dst, []).append((c.src, c.w))

        zero = torch.zeros((), device=dev, dtype=dt)
        for nid in self.order:
            if nid in vals:           # input
                continue
            node = self.nodes[nid]
            acc = None
            for src, w in by_dst.get(nid, ()):
                term = vals[src] * w            # python scalar: keeps dt
                acc = term if acc is None else acc + term
            if acc is None:
                acc = zero.broadcast_to(ref.shape)
            acc = acc + node.bias
            v = FUNCTIONS[node.fn](acc)
            # evolution robustness: one NaN poisons everything
            # downstream — keep the graph finite, always (prism's law)
            v = torch.nan_to_num(v, nan=0.0, posinf=3.0e4, neginf=-3.0e4)
            vals[nid] = v
        want = names if names is not None else list(self.outputs)
        return {name: vals[self.outputs[name]].expand(ref.shape)
                for name in want if name in self.outputs}

    # ------------------------------------------------------------ mutations

    def mut_drift(self, rng) -> str:
        rng = _rng_ok(rng)
        for c in self.conns:
            if rng.random() < 0.8:
                c.w *= float(np.exp(0.15 * rng.standard_normal()))
        for n in self.nodes.values():
            if n.kind != "in" and rng.random() < 0.5:
                n.bias *= float(np.exp(0.10 * rng.standard_normal()))
        return "drift"

    def mut_add_conn(self, rng, sigma: float = 0.7) -> str:
        rng = _rng_ok(rng)
        disabled = [c for c in self.conns if not c.on]
        if disabled and rng.random() < 0.3:      # heal an old wound first
            c = disabled[int(rng.integers(len(disabled)))]
            c.on = True
            c.w = float(sigma * rng.standard_normal())
            return "add_conn"
        n = len(self.order)
        for _ in range(40):
            i = int(rng.integers(0, n - 1))
            j = int(rng.integers(i + 1, n))
            src, dst = self.order[i], self.order[j]
            if any(c.src == src and c.dst == dst for c in self.conns):
                continue
            self.conns.append(
                Conn(src, dst, float(sigma * rng.standard_normal())))
            return "add_conn"
        return self.mut_drift(rng)               # dense graph: drift

    def mut_add_node(self, rng) -> str:
        rng = _rng_ok(rng)
        live = [c for c in self.conns if c.on]
        if not live:
            return self.mut_add_conn(rng)
        c = live[int(rng.integers(len(live)))]
        c.on = False                             # NEAT split
        nid = self._new_id()
        fn = list(FUNCTIONS)[int(rng.integers(len(FUNCTIONS)))]
        self.nodes[nid] = Node(fn, 0.0, "hid")
        self.order.insert(self.pos(c.dst), nid)
        self.conns.append(Conn(c.src, nid, 1.0))
        self.conns.append(Conn(nid, c.dst, c.w))
        return "add_node"

    def mut_add_output(self, rng, sigma: float = 0.7) -> str:
        """The unbounded alphabet (brant_gecco19). For the substrate
        organism extra outputs are inert-but-growable: the decode reads
        'w'; future reads (inharm, phase channels) are born here."""
        rng = _rng_ok(rng)
        name = f"out{len(self.outputs)}"
        while name in self.outputs:
            name += "x"
        nid = self.add_output(name, fn=list(FUNCTIONS)[
            int(rng.integers(len(FUNCTIONS)))])
        src = self.order[int(rng.integers(0, len(self.order) - 1))]
        self.conns.append(
            Conn(src, nid, float(sigma * rng.standard_normal())))
        return "add_output"

    def mut_swap_fn(self, rng) -> str:
        rng = _rng_ok(rng)
        hids = [nid for nid in self.order
                if self.nodes[nid].kind == "hid"]
        if not hids:
            return self.mut_drift(rng)
        nid = hids[int(rng.integers(len(hids)))]
        fns = [f for f in FUNCTIONS if f != self.nodes[nid].fn]
        self.nodes[nid].fn = fns[int(rng.integers(len(fns)))]
        return "swap_fn"

    def mut_toggle(self, rng) -> str:
        rng = _rng_ok(rng)
        if not self.conns:
            return self.mut_drift(rng)
        c = self.conns[int(rng.integers(len(self.conns)))]
        c.on = not c.on
        return "toggle"

    def mut_duplicate(self, rng, max_nodes: int = 5) -> str:
        """Copy an upstream motif; rewire the copy to the same sources
        and hook its root in beside the original's outgoing signals —
        a parallel voice that will diverge (prism's duplicate)."""
        rng = _rng_ok(rng)
        hids = [nid for nid in self.order
                if self.nodes[nid].kind == "hid"]
        if not hids:
            return self.mut_add_node(rng)
        root = hids[int(rng.integers(len(hids)))]
        motif = {root}
        frontier = [root]
        while frontier and len(motif) < max_nodes:
            nxt = []
            for m in frontier:
                for c in self.conns:
                    if (c.on and c.dst == m and c.src not in motif
                            and self.nodes[c.src].kind == "hid"):
                        motif.add(c.src)
                        nxt.append(c.src)
            frontier = nxt
        id_map: dict[int, int] = {}
        for o in sorted(motif, key=self.pos):
            nid = self._new_id()
            src_node = self.nodes[o]
            self.nodes[nid] = Node(src_node.fn, src_node.bias, "hid")
            id_map[o] = nid
            self.order.insert(self.pos(o), nid)  # just before original
        snapshot = list(self.conns)
        for c in snapshot:                       # internal wiring
            if c.on and c.src in id_map and c.dst in id_map:
                self.conns.append(
                    Conn(id_map[c.src], id_map[c.dst], c.w))
        for c in snapshot:                       # share external inputs
            if c.on and c.dst in id_map and c.src not in id_map:
                self.conns.append(Conn(c.src, id_map[c.dst], c.w))
        for c in snapshot:                       # hook: parallel voice
            if c.on and c.src == root:
                self.conns.append(
                    Conn(id_map[root], c.dst, 0.5 * c.w))
        return "duplicate"

    # dispatch — prism's weights
    OPS = (("drift", 3), ("add_conn", 2), ("add_node", 2), ("add_output", 1),
           ("swap_fn", 1), ("toggle", 1), ("duplicate", 1))

    def mutate(self, rng) -> str:
        rng = _rng_ok(rng)
        names = [n for n, _ in self.OPS]
        p = np.array([w for _, w in self.OPS], float)
        p = p / p.sum()
        op = names[int(rng.choice(len(names), p=p))]
        return getattr(self, "mut_" + op)(rng)

    # ------------------------------------------------------------ json

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "nodes": [[int(k), n.fn, float(n.bias), n.kind]
                      for k, n in sorted(self.nodes.items())],
            "conns": [[c.src, c.dst, c.w, c.on] for c in self.conns],
            "order": list(self.order),
            "outputs": {k: int(v) for k, v in self.outputs.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Cppn":
        g = cls(int(d.get("seed", 0)))
        g.nodes = {int(k): Node(fn, bias, kind)
                   for k, fn, bias, kind in d["nodes"]}
        g.conns = [Conn(int(s), int(t), float(w), bool(on))
                   for s, t, w, on in d["conns"]]
        g.order = [int(k) for k in d["order"]]
        g.outputs = {k: int(v) for k, v in d["outputs"].items()}
        return g


# ------------------------------------------------------------------ genesis


def random_cppn(rng, n_hidden: int = 3) -> Cppn:
    """A small random founder: a loose chain from the query senses to
    one 'w' output. Everything after this is mutation."""
    rng = _rng_ok(rng)
    g = Cppn(seed=int(rng.integers(2 ** 31)))
    fns = list(FUNCTIONS)
    prev = None
    for _ in range(n_hidden):
        nid = g.add_hidden(fns[int(rng.integers(len(fns)))])
        for name in ("x1", "y1", "x2", "y2", "d"):
            if rng.random() < 0.6:
                g.conn(INPUT_IDS[name], nid, 0.7 * rng.standard_normal())
        if prev is not None and rng.random() < 0.7:
            g.conn(prev, nid, 0.7 * rng.standard_normal())
        prev = nid
    out = g.add_output("w", fn="tanh")
    g.conn(prev if prev is not None else INPUT_IDS["x1"], out,
           0.8 * rng.standard_normal() or 0.5)
    return g