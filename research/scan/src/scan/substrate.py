"""substrate: DeepHyperNEAT repurposed — the medium grows a brain.

DeepHyperNEAT (the repo next door, neat-python derived) evolves a CPPN
that is DECODED into a substrate: a deep network whose connection
weights are painted by querying the CPPN at geometric coordinates —
HyperNEAT's regularity/repetition/symmetry, plus the Deep extension:
mutations that grow the substrate's DEPTH (insert a sheet) and BREADTH
(widen one). Here the substrate is the frame-content generator:

    CPPN(x1,y1, x2,y2, d, b) --decode--> substrate weights (geometry)
    substrate(partial_pos, morph_pos)  --forward-->  amp(p, m)
    amp matrix (P x K) --materialise--> scan frames of partials
    scan's integer-mip readout, paths, viability, echo loop: unchanged

The substrate maps a point on the (partial, morph) plane to a partial
amplitude, so its output lands EXACTLY in scan's harmonic-domain
genome format — the whole rig (band-limiting, phases, readout, vector
mix, pitch paths) is inherited, not rebuilt. What changes is the
country: the wt organism's frames are ~50 direct genes; the sub
organism's frames are the phenotype of a growing graph — the
unboundedness prism argues for ("the encoding is the environment"),
now in scan's arena.

Its Python-loop, global-RNG inspiration was not directly reusable (it
evaluates per point, unseeded) — what is repurposed is the SEMANTICS:
the query convention, the coordinate geometry, the depth/breadth
mutations, the clamp-to-maxweight decode. Vectorised, seeded, torch.

GPU: decode queries and the substrate forward are batched matmul —
the device is pluggable (cpu | mps). Determinism is per device
(float32; MPS matmul reduction order is fixed for a shape, so
re-renders are bit-identical — held by test on this machine).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from . import organism
from .organism import (WtGenome, drift_params, viable, write_wav,
                       SECONDS, F0_LO, F0_HI, _jit)
from .cppn import Cppn, random_cppn, FUNCTIONS, INPUT_IDS

__all__ = ["SubGenome", "render", "random_genome", "mutate", "tonality",
           "viable", "write_wav", "SECONDS", "SUB_CHAMPIONS", "champion"]

MAX_DEPTH = 8                      # hidden sheets
MAX_WIDTH = 24                     # nodes per sheet
P_LO, P_HI = 8, 64                 # partials per frame (the grid)
K_LO, K_HI = 4, 128                # frames per table (the morph axis)
MAX_WEIGHT = 3.0                   # DeepHyperNEAT's decode clamp convention
SUB_ACTS = ("tanh", "sin", "sq", "lin")   # every one has act(0) = 0:
                                          # padded nodes contribute nothing

_DEVICE = None


def set_device(name: str | None = None) -> str:
    """Pin the evaluation device: 'cpu', 'mps', or None = auto.
    Auto is cpu, measured: at v1 sizes the CPU path is shape-stable
    (~16 ms/child) while MPS pays shader compilation for every new
    tensor shape (~99 ms/child in a drifting population); the GPU
    wins once shapes repeat (batched siblings) or substrates grow —
    opt in with --device mps. Env override: SCAN_DEVICE."""
    global _DEVICE
    if name is None:
        name = os.environ.get("SCAN_DEVICE", "auto")
    if name in ("auto", ""):
        name = "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    _DEVICE = name
    return name


def device() -> str:
    return _DEVICE if _DEVICE else set_device()


# ------------------------------------------------------------------ geometry

def _layer_coords(widths: list[int]) -> list[np.ndarray]:
    """Substrate node coordinates: input sheet at x=-1 (partial axis
    low, morph axis high), hidden sheets marching to +1, one output
    node at (1, 0). The geometry the CPPN is queried against."""
    D = len(widths)
    coords = [np.array([[-1.0, -1.0], [-1.0, 1.0]])]   # partial, morph
    for i, w in enumerate(widths):
        x = -1.0 + 2.0 * (i + 1) / (D + 1)
        y = np.linspace(-1.0, 1.0, w) if w > 1 else np.array([0.0])
        coords.append(np.stack([np.full(w, x), y], axis=1))
    coords.append(np.array([[1.0, 0.0]]))
    return coords


def decode(g: "SubGenome", dev: str, coords=None) -> list:
    """CPPN -> substrate weights. One batched query per layer pair at
    all (source x target) coordinate pairs, clamped to ±MAX_WEIGHT —
    DeepHyperNEAT's decode, vectorised. coords may be shared by a
    batch (same widths = same geometry)."""
    coords = coords if coords is not None else _layer_coords(g.widths)
    weights = []
    for src_xy, dst_xy in zip(coords[:-1], coords[1:]):
        ns, nd = len(src_xy), len(dst_xy)
        s = np.repeat(src_xy, nd, axis=0)               # (ns*nd, 2)
        t = np.tile(dst_xy, (ns, 1))
        d = np.linalg.norm(s - t, axis=1)
        feeds = {
            "x1": torch.tensor(s[:, 0], dtype=torch.float32, device=dev),
            "y1": torch.tensor(s[:, 1], dtype=torch.float32, device=dev),
            "x2": torch.tensor(t[:, 0], dtype=torch.float32, device=dev),
            "y2": torch.tensor(t[:, 1], dtype=torch.float32, device=dev),
            "d": torch.tensor(d, dtype=torch.float32, device=dev),
            "b": torch.ones(ns * nd, dtype=torch.float32, device=dev),
        }
        w = g.cppn.evaluate(feeds, names=["w"], device=dev)["w"]
        w = torch.clamp(w, -1.0, 1.0) * MAX_WEIGHT
        weights.append(w.reshape(ns, nd))
    return weights


def amp_matrix(g: "SubGenome", dev: str | None = None) -> np.ndarray:
    """The medium: (P, K) partial amplitudes — the substrate's forward
    pass over the (partial, morph) plane. Routes through the batched
    path (group of one) so single and batched renders share one code
    path — determinism has exactly one definition."""
    return amp_batch([g], dev)[0]


def amp_batch(gs: list, dev: str | None = None) -> list:
    """The medium for many genomes at once: children that share a
    substrate shape (P, K, widths, act) are evaluated as ONE batched
    matmul chain (torch.bmm) — the GPU's honest path, because a
    breeding generation's siblings mostly share their parent's
    shapes. Grouping is by exact shape; a genome's result depends
    only on its genes up to float low-bits (CPU bmm is bit-stable,
    held by test; MPS is deterministic per run)."""
    dev = dev or device()
    groups: dict[tuple, list[int]] = {}
    for i, g in enumerate(gs):
        groups.setdefault((g.P, g.K, tuple(g.widths), g.sub_act),
                          []).append(i)
    out: list = [None] * len(gs)
    for (P, K, widths, act), idxs in groups.items():
        coords = _layer_coords(list(widths))
        p = np.linspace(-1.0, 1.0, P) if P > 1 else np.array([0.0])
        m = np.linspace(-1.0, 1.0, K) if K > 1 else np.array([0.0])
        pg, mg = np.meshgrid(p, m, indexing="ij")          # (P, K)
        X = torch.tensor(np.stack([pg.ravel(), mg.ravel()], axis=1),
                         dtype=torch.float32, device=dev)    # (P*K, 2)
        per_g = [decode(gs[i], dev, coords=coords) for i in idxs]
        Ws = [torch.stack([pg_[li] for pg_ in per_g])       # (G, n_s, n_d)
              for li in range(len(coords) - 1)]
        h = X.unsqueeze(0).expand(len(idxs), -1, -1)         # (G, pts, 2)
        for W in Ws[:-1]:
            h = FUNCTIONS[act](torch.bmm(h, W))
            h = torch.nan_to_num(h, nan=0.0, posinf=3.0e4, neginf=-3.0e4)
        out_v = torch.bmm(h, Ws[-1])                        # (G, pts, 1)
        out_v = torch.nan_to_num(out_v, nan=0.0, posinf=3.0e4,
                                 neginf=-3.0e4)
        amps = torch.abs(out_v).squeeze(-1).cpu().numpy().astype(np.float64)
        scales = np.array([gs[i].out_scale for i in idxs])[:, None]
        amps = amps * scales
        for j, i in enumerate(idxs):
            out[i] = amps[j].reshape(P, K)
    return out


# ------------------------------------------------------------------ genome

@dataclass
class SubGenome:
    """CPPN + substrate geometry + scan's paths. The frames are never
    genes — they are materialised from the substrate (cached per
    instance; mutate returns fresh objects, so the cache is honest)."""
    cppn: Cppn = field(default_factory=Cppn)
    widths: list = field(default_factory=lambda: [4])  # hidden sheets
    sub_act: str = "tanh"
    out_scale: float = 1.0
    P: int = 24                       # partials per frame
    K: int = 16                       # frames per table
    f0: float = 110.0
    pitch: dict = field(default_factory=lambda: {
        "type": "fixed", "rate": 0.5, "depth": 200.0, "end": 1.0})
    morph: dict = field(default_factory=lambda: {
        "type": "lfo", "rate": 0.2, "depth": 0.8, "shape": "saw",
        "duty": 0.0, "bps": [0.1, 0.5, 0.9]})
    voices: list = field(default_factory=list)
    vector: dict = field(default_factory=lambda: {
        "type": "fixed", "x0": 0.0, "y0": 0.0, "rate_x": 0.1,
        "rate_y": 0.13, "depth": 0.0, "shape_x": "sin", "shape_y": "tri"})
    seed: int = 0
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------ materialise

    def to_wt(self) -> WtGenome:
        """Substrate -> scan's harmonic-domain frames. This is the
        repurpose's hinge: everything downstream is scan's rig."""
        if "wt" in self._cache:
            return self._cache["wt"]
        wt = _wt_from(self, amp_matrix(self))
        self._cache["wt"] = wt
        return wt

    # ------------------------------------------------------------ labels

    def archetype(self) -> str:
        return self.to_wt().archetype()

    def anatomy(self) -> dict:
        a = self.to_wt().anatomy()
        a.update({"cppn_nodes": len(self.cppn.nodes),
                  "cppn_conns": len(self.cppn.conns),
                  "widths": list(self.widths), "P": self.P,
                  "sub_act": self.sub_act})
        return a


def _wt_from(g: SubGenome, amp: np.ndarray) -> WtGenome:
    """Amp matrix -> scan's frames (the repurpose's hinge)."""
    frames = [{"partials": [float(a) for a in amp[:, k]],
               "inharm": 0.0} for k in range(g.K)]
    return WtGenome(f0=g.f0, pitch=dict(g.pitch), frames=frames,
                    morph=dict(g.morph), voices=[dict(v) for v in g.voices],
                    vector=dict(g.vector), seed=g.seed)


def tonality(g: SubGenome) -> float:
    """The greedy objective for the substrate organism: wt purity plus
    structural cost — the optimum is a minimal graph at minimum
    partials, static morph, one voice. Bounded below, so a fixed
    point exists (the trap stays honest)."""
    base = organism.tonality(g.to_wt())
    return base - 0.25 * len(g.cppn.nodes) - 0.25 * len(g.cppn.conns) \
        - 0.5 * sum(g.widths)


# ------------------------------------------------------------------ render

def render(g: SubGenome, seconds: float = SECONDS, sr: int = 44100,
           hi: bool = False) -> np.ndarray:
    """Substrate genome -> stereo (n, 2) float32 via scan's rig."""
    return organism.render(g.to_wt(), seconds=seconds, sr=sr, hi=hi)


def render_batch(gs: list, seconds: float = SECONDS, sr: int = 44100,
                 hi: bool = False) -> list:
    """Many substrate genomes -> audios. Siblings sharing a substrate
    shape ride one batched matmul chain (the GPU's honest path);
    every genome still reads out through scan's rig individually."""
    amps = amp_batch(gs)
    return [organism.render(_wt_from(g, amp), seconds=seconds, sr=sr,
                            hi=hi)
            for g, amp in zip(gs, amps)]


# ------------------------------------------------------------------ genetics

def _rand_widths(rng, n: int) -> list:
    return [int(rng.integers(2, 7)) for _ in range(n)]


def random_genome(rng: np.random.Generator, seed: int) -> SubGenome:
    g = SubGenome(seed=int(seed))
    g.cppn = random_cppn(rng)
    g.widths = _rand_widths(rng, int(rng.integers(1, 3)))
    g.sub_act = str(rng.choice(SUB_ACTS))
    g.out_scale = float(10 ** rng.uniform(-0.5, 0.5))
    g.P = int(rng.integers(P_LO, 41))
    g.K = int(rng.integers(K_LO, 49))
    # ---- the paths: scan's founder recipe, verbatim
    g.f0 = float(2 ** rng.uniform(5.9, 9.97))
    g.pitch = {"type": str(rng.choice(
        ("fixed", "glide", "vib", "walk", "step"),
        p=[0.4, 0.15, 0.15, 0.15, 0.15])),
        "rate": float(10 ** rng.uniform(-1.5, 0.7)),
        "depth": float(rng.uniform(5.0, 600.0)),
        "end": float(2 ** rng.uniform(-2.0, 2.0))}
    g.morph = {"type": str(rng.choice(("lfo", "curve", "step"))),
               "rate": float(10 ** rng.uniform(-1.5, 0.7)),
               "depth": float(rng.choice([rng.uniform(0.0, 0.3),
                                          rng.uniform(0.3, 1.0)])),
               "shape": str(rng.choice(("sin", "tri", "saw"))),
               "duty": float(rng.uniform(0.0, 0.5)),
               "bps": [float(rng.uniform(0.0, 1.0))
                       for _ in range(int(rng.integers(3, 6)))]}
    n_v = int(rng.choice([1, 2, 2, 3]))
    g.voices = []
    for _ in range(n_v):
        g.voices.append({
            "detune": float(rng.choice(
                [0.0, rng.uniform(-30, 30), rng.uniform(-700, 700),
                 rng.uniform(-1200, 1200)])),
            "level": float(10 ** rng.uniform(-0.7, 0.0)),
            "pan": float(rng.uniform(-0.8, 0.8)),
            "morph_offset": float(rng.uniform(0.0, 0.3)),
            "attack": float(10 ** rng.uniform(-2.5, -0.7)),
            "decay_tau": float(10 ** rng.uniform(-0.5, 0.9)),
        })
    g.vector = {"type": str(rng.choice(["fixed", "lfo2d"], p=[0.4, 0.6])),
                "x0": float(rng.uniform(-0.6, 0.6)),
                "y0": float(rng.uniform(-0.6, 0.6)),
                "rate_x": float(10 ** rng.uniform(-2.0, 0.3)),
                "rate_y": float(10 ** rng.uniform(-2.0, 0.3)),
                "depth": float(rng.uniform(0.0, 1.0)),
                "shape_x": str(rng.choice(("sin", "tri"))),
                "shape_y": str(rng.choice(("sin", "tri")))}
    return g


# -- substrate structural ops (DeepHyperNEAT's depth/breadth mutations) --

def op_inc_depth(g: SubGenome, rng) -> bool:
    """Insert a hidden sheet at a random position (the Deep move)."""
    if len(g.widths) >= MAX_DEPTH:
        return False
    g.widths.insert(int(rng.integers(len(g.widths) + 1)),
                    int(rng.integers(2, 7)))
    return True


def op_dec_depth(g: SubGenome, rng) -> bool:
    if len(g.widths) <= 1:
        return False
    g.widths.pop(int(rng.integers(len(g.widths))))
    return True


def op_inc_breadth(g: SubGenome, rng) -> bool:
    i = int(rng.integers(len(g.widths)))
    if g.widths[i] >= MAX_WIDTH:
        return False
    g.widths[i] += 1
    return True


def op_dec_breadth(g: SubGenome, rng) -> bool:
    i = int(rng.integers(len(g.widths)))
    if g.widths[i] <= 1:
        return False
    g.widths[i] -= 1
    return True


def mutate(g: SubGenome, rng: np.random.Generator, rate: float = 0.3,
           seed: int = 0) -> SubGenome:
    """One birth: paths drift (shared kernel), the CPPN takes one
    graph op, the substrate takes one depth/breadth step, the grid
    resolutions drift."""
    child = SubGenome(
        cppn=g.cppn.copy(),
        widths=list(g.widths),
        sub_act=g.sub_act,
        out_scale=g.out_scale,
        P=g.P, K=g.K,
        f0=g.f0,
        pitch=dict(g.pitch), morph=dict(g.morph),
        voices=[dict(v) for v in g.voices], vector=dict(g.vector),
        seed=int(seed),
    )
    drift_params(child, rng, rate)                    # the shared paths
    # ---- the graph: one prism op per birth
    child.cppn.mutate(rng)
    # ---- the substrate: one DeepHyperNEAT step per birth
    r = rng.random()
    if r < 0.10:
        op_inc_depth(child, rng)
    elif r < 0.16:
        op_dec_depth(child, rng)
    elif r < 0.26:
        op_inc_breadth(child, rng)
    elif r < 0.34:
        op_dec_breadth(child, rng)
    # ---- the grid + output drift
    child.P = int(np.clip(round(g.P * 10 ** rng.normal(0, 0.12)),
                          P_LO, P_HI))
    child.K = int(np.clip(round(g.K * 10 ** rng.normal(0, 0.12)),
                          K_LO, K_HI))
    child.out_scale = _jit(rng, child.out_scale, 0.05, 20.0, rate)
    if rng.random() < 0.05:
        child.sub_act = str(rng.choice(SUB_ACTS))

    # ---- the leap
    if rng.random() < 0.02:
        axis = int(rng.integers(4))
        if axis == 0:                                 # redraw the sheets
            child.widths = _rand_widths(rng, int(rng.integers(1, 4)))
        elif axis == 1:
            child.morph["type"] = str(rng.choice(("lfo", "step")))
            child.morph["rate"] = float(10 ** rng.uniform(0.0, 1.3))
            child.morph["depth"] = 1.0
        elif axis == 2:
            child.f0 = float(np.clip(child.f0 * rng.choice([0.5, 2.0]),
                                     F0_LO, F0_HI))
        else:
            child.vector.update({
                "type": "lfo2d",
                "x0": float(rng.uniform(-0.6, 0.6)),
                "y0": float(rng.uniform(-0.6, 0.6)),
                "rate_x": float(10 ** rng.uniform(-1.0, 0.3)),
                "rate_y": float(10 ** rng.uniform(-1.0, 0.3)),
                "depth": float(rng.uniform(0.5, 1.0)),
            })
    return child


# ------------------------------------------------------------------ io

def to_dict(g: SubGenome) -> dict:
    return {"scan": "subgenome", "version": 1,
            "cppn": g.cppn.to_dict(), "widths": list(g.widths),
            "sub_act": g.sub_act, "out_scale": g.out_scale,
            "P": g.P, "K": g.K, "f0": g.f0, "pitch": g.pitch,
            "morph": g.morph, "voices": g.voices, "vector": g.vector,
            "seed": g.seed}


def from_dict(d: dict) -> SubGenome:
    g = d.get("subgenome", d)
    return SubGenome(cppn=Cppn.from_dict(g["cppn"]),
                     widths=[int(w) for w in g["widths"]],
                     sub_act=str(g["sub_act"]),
                     out_scale=float(g["out_scale"]),
                     P=int(g["P"]), K=int(g["K"]),
                     f0=float(g["f0"]), pitch=dict(g["pitch"]),
                     morph=dict(g["morph"]),
                     voices=[dict(v) for v in g["voices"]],
                     vector=dict(g["vector"]), seed=int(g["seed"]))


# ------------------------------------------------------------------ champions

def _paths(f0, pitch_type, morph_type, rate, depth, n_voices=1):
    return {
        "f0": f0,
        "pitch": {"type": pitch_type, "rate": 0.5, "depth": 0.0, "end": 1.0},
        "morph": {"type": morph_type, "rate": rate, "depth": depth,
                  "shape": "saw", "duty": 0.0, "bps": [0.1, 0.9, 0.2]},
        "voices": [{"detune": 0.0, "level": 1.0, "pan": 0.0,
                     "morph_offset": 0.0, "attack": 0.02,
                     "decay_tau": 8.0} for _ in range(n_voices)],
        "vector": {"type": "fixed", "x0": 0.0, "y0": 0.0, "rate_x": 0.1,
                   "rate_y": 0.13, "depth": 0.0, "shape_x": "sin",
                   "shape_y": "tri"},
    }


def _rampbank() -> SubGenome:
    """Known answer 1: a CPPN that ignores geometry and answers a
    constant — every weight clamps to +MAX_WEIGHT, so the substrate is
    an all-ones mixer: amp(p, m) = |tanh(c tanh(p + m))|, a smooth
    diagonal ridge. Spectra brighten along the morph axis: the PPG
    sweep, painted by geometry's absence."""
    cppn = Cppn(seed=41)
    cppn.add_output("w", fn="lin", bias=1.0)          # w = 1, always
    g = SubGenome(seed=41, widths=[2], sub_act="tanh", out_scale=1.0,
                  P=24, K=24)
    p = _paths(110.0, "glide", "lfo", 0.3, 1.0)
    g.f0, g.pitch, g.morph = p["f0"], p["pitch"], p["morph"]
    g.voices, g.vector = p["voices"], p["vector"]
    g.cppn = cppn
    return g


def _comb() -> SubGenome:
    """Known answer 2: w = sin(4d) — weights oscillate with the
    geometric distance between substrate nodes, so the substrate
    computes interference: comb spectra whose peaks slide with the
    morph position."""
    cppn = Cppn(seed=42)
    out = cppn.add_output("w", fn="sin")
    cppn.conn(INPUT_IDS["d"], out, 4.0)               # w = sin(4 d)
    g = SubGenome(seed=42, widths=[6, 4], sub_act="tanh", out_scale=1.0,
                  P=32, K=32)
    p = _paths(220.0, "fixed", "lfo", 0.25, 0.9)
    g.f0, g.pitch, g.morph = p["f0"], p["pitch"], p["morph"]
    g.voices, g.vector = p["voices"], p["vector"]
    g.cppn = cppn
    return g


def _stepper() -> SubGenome:
    """Known answer 3: the comb geometry played as a Wavestation —
    step morph, step pitch, two detuned voices."""
    g = _comb()
    g.seed = 43
    g.cppn.seed = 43
    p = _paths(220.0, "step", "step", 2.0, 1.0, n_voices=2)
    g.f0, g.pitch, g.morph = p["f0"], p["pitch"], p["morph"]
    g.voices = [{"detune": 7.0, "level": 0.8, "pan": 0.3,
                 "morph_offset": 0.0, "attack": 0.005, "decay_tau": 4.0},
                {"detune": -7.0, "level": 0.8, "pan": -0.3,
                 "morph_offset": 0.15, "attack": 0.005, "decay_tau": 4.0}]
    g.morph["duty"] = 0.15
    g.pitch["depth"] = 700.0
    g.vector = {"type": "lfo2d", "x0": 0.0, "y0": 0.0, "rate_x": 0.3,
                "rate_y": 0.21, "depth": 0.25, "shape_x": "sin",
                "shape_y": "tri"}
    return g


_BUILDERS = {"rampbank": _rampbank, "comb": _comb, "stepper": _stepper}
SUB_CHAMPIONS = tuple(_BUILDERS)


def champion(name: str) -> SubGenome:
    if name not in _BUILDERS:
        raise ValueError(f"unknown sub champion '{name}' "
                         f"(have: {', '.join(SUB_CHAMPIONS)})")
    return _BUILDERS[name]()