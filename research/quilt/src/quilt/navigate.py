"""navigate: the walkers — six ways to take a trip through the memory.

echo answers "have we heard this?"; quilt asks the follow-up: "then
take me somewhere". A walker produces a *route* — an ordered list of
pieces — by moving through the atlas one piece at a time. The piece
is then rendered by pure concatenation; all the music is in the order
of selection.

    chain      stay near: every next piece is a near neighbor of the
               last, temperature-biased — a slowly evolving drone of
               relatives. (RAVE's nearest-latent walk, discrete.)
    contrast   flee: every next piece is the farthest from the recent
               past — maximal difference, the corpus's extremes in
               alternation. (novelty search as a listening policy.)
    roam       brownian: a virtual point drifts through the space at a
               step sized to the territory's macro scale; the piece is
               whatever is nearest to the drift. Neither seeking nor
               fleeing — wandering.
    attractor  a Lorenz trajectory mapped onto the atlas's principal
               axes: chaotic but structured roaming — strange beauty
               without a random walk's aimlessness.
    drift      a straight line between two distant poles, walked via
               the pieces nearest to it — a 3-minute dissolve from one
               region of the corpus into its opposite.
    follow     memory's own order: mostly the *temporal successor* of
               the current piece (the corpus's own sequences), with
               occasional near-neighbor jumps elsewhere — passages
               that branch away.

All walkers share one law: the recent past is excluded (no piece —
and no piece's identical twin under another scissor — may recur
within the exclusion window), relaxing only when a tiny bank corners
them. Everything is seeded and deterministic: same bank, same seed,
same route, bit-identical piece.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["walk", "MODES", "Route", "Step"]

MAX_STEPS = 20000          # safety: a route can never loop forever


# ------------------------------------------------------------------ route

@dataclass
class Step:
    i: int                       # piece index in the atlas
    dur: float                    # seconds of audio this step plays
    dist_prev: float | None      # distance to the previous piece
    note: str = ""               # the walker's reason
    t0: float | None = None      # grain start within the piece
    t1: float | None = None      # grain end (None/None = the whole piece)

    def as_dict(self, atlas) -> dict:
        m = dict(atlas.metas[self.i])
        if self.t0 is not None:                       # a grain: sub-span
            m["t0"] = round(float(self.t0), 6)
            m["t1"] = round(float(self.t1), 6)
        m.update({"i": int(self.i), "dur": float(self.dur),
                  "dist_prev": (None if self.dist_prev is None
                                else round(float(self.dist_prev), 4)),
                  "note": self.note})
        return m


@dataclass
class Route:
    mode: str
    seed: int
    seconds: float
    strategies: list
    params: dict
    bank: str
    corpus_root: str
    steps: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def total_dur(self) -> float:
        return float(sum(s.dur for s in self.steps))

    def to_json(self, atlas) -> dict:
        return {
            "quilt": "route", "mode": self.mode, "seed": int(self.seed),
            "seconds": float(self.seconds), "total_dur": self.total_dur,
            "strategies": list(self.strategies), "params": dict(self.params),
            "bank": self.bank, "corpus_root": self.corpus_root,
            "stats": dict(self.stats),
            "steps": [s.as_dict(atlas) for s in self.steps],
        }


# ------------------------------------------------------------------ shared nav

class _Nav:
    """The recent-past exclusion every walker shares."""

    def __init__(self, atlas, window: int):
        self.atlas, self.W = atlas, int(window)

    def recent_mask(self, steps) -> np.ndarray:
        a, m = self.atlas, np.zeros(self.atlas.n, bool)
        w = min(self.W, max(1, a.n // 3))
        for s in steps[-w:]:
            m[s.i] = True
            # audio twins: same instant of the same file, another scissor
            m |= (a.files == a.files[s.i]) & (np.abs(a.t0s - a.t0s[s.i]) < 1e-9)
        return m

    def nearest(self, d: np.ndarray, steps, k: int = 1):
        """The k nearest pieces to query distances d, excluding the
        recent past (relaxing when cornered). Returns (idx, dists)
        sorted by distance."""
        dm = np.where(self.recent_mask(steps), np.inf, d)
        if not np.isfinite(dm).any():
            dm = d                                    # cornered: no past
        kk = int(max(1, min(k, int(np.isfinite(dm).sum()))))
        idx = np.argpartition(dm, kk - 1)[:kk]
        idx = idx[np.argsort(dm[idx])]
        return idx, dm[idx]

    def farthest(self, d: np.ndarray, steps) -> int:
        dm = np.where(self.recent_mask(steps), -np.inf, d)
        if not np.isfinite(dm).any():
            dm = d
        return int(np.argmax(dm))


def _rank_pick(n_cand: int, rng, temp: float) -> int:
    """Choose a rank among sorted candidates: temp→0 greedy (always the
    nearest), larger temp spreads into the neighborhood."""
    if temp <= 1e-9 or n_cand <= 1:
        return 0
    w = np.exp(-np.arange(n_cand) / temp)
    return int(rng.choice(n_cand, p=w / w.sum()))


def _successor_map(atlas) -> dict:
    """row -> row of the temporal successor: same original, same
    scissor, index + 1 — the pile's filenames carry the corpus's own
    continuity (orig__onset_0007.wav follows orig__onset_0006.wav).
    Unmarked pieces (originals kept whole) have no successor."""
    pos, succ = {}, {}
    for j, m in enumerate(atlas.metas):
        if m.get("idx", -1) >= 0:
            pos[(m.get("orig"), m.get("strategy"), m["idx"])] = j
    for j, m in enumerate(atlas.metas):
        if m.get("idx", -1) >= 0:
            nxt = pos.get((m.get("orig"), m.get("strategy"),
                           m["idx"] + 1))
            if nxt is not None:
                succ[j] = nxt
    return succ


# ------------------------------------------------------------------ walkers

class _Chain:
    """nearest-neighbor walk with temperature — 'similar in sequence'."""
    def __init__(self, atlas, rng, nav, p):
        self.a, self.rng, self.nav = atlas, rng, nav
        self.k = int(p["k"])
        self.temp = float(p["temp"])

    def start(self):
        j = int(self.rng.integers(self.a.n))
        return j, "start"

    def step(self, steps, anchor=None):
        last = int(anchor) if anchor is not None else steps[-1].i
        d = self.a.dists_from_row(last)
        idx, dd = self.nav.nearest(d, steps, self.k)
        r = _rank_pick(len(idx), self.rng, self.temp)
        j = int(idx[r])
        return j, f"nn r{r} d={dd[r]:.2f}"


class _Contrast:
    """farthest-from-the-recent-past — 'the most different next'."""
    def __init__(self, atlas, rng, nav, p):
        self.a, self.rng, self.nav = atlas, rng, nav
        self.cw = max(1, int(p["contrast_window"]))

    def start(self):
        j = int(self.rng.integers(self.a.n))
        return j, "start"

    def step(self, steps):
        last = [s.i for s in steps[-self.cw:]]
        ref = self.a.vecs_mean(last)
        d = self.a.dists_from_vec(ref)
        j = self.nav.farthest(d, steps)
        return j, f"far d={d[j]:.2f}"


class _Roam:
    """brownian drift through the space, snapped to real pieces. The
    step is speed x 0.1 x the territory's macro scale (median
    random-pair distance) — a stroll that crosses a typical
    neighborhood in ~10 steps. (Sized against median-NN instead, a
    pile dense with near-duplicates pins the walker to its start: the
    survey measured bit-identical walks for every speed — the step was
    ~2000x smaller than the territory.)"""
    def __init__(self, atlas, rng, nav, p):
        self.a, self.rng, self.nav = atlas, rng, nav
        self.step_len = float(p["speed"]) * 0.1 * atlas.macro_scale()
        self.p = None

    def start(self):
        j = int(self.rng.integers(self.a.n))
        self.p = self.a.vec(j).copy()
        return j, "start"

    def step(self, steps):
        u = self.rng.standard_normal(self.a.dim)
        u /= max(np.linalg.norm(u), 1e-12)
        self.p = self.p + self.step_len * u
        d = self.a.dists_from_vec(self.p)
        idx, dd = self.nav.nearest(d, steps, 1)
        return int(idx[0]), f"roam d={dd[0]:.2f}"


class _Attractor:
    """a Lorenz trajectory steering through the principal axes."""
    def __init__(self, atlas, rng, nav, p):
        self.a, self.rng, self.nav = atlas, rng, nav
        self.amp = float(p["amp"])
        self.traj = _lorenz(n=4000, dt=0.05)          # deterministic land
        self.ki = int(rng.integers(0, self.traj.shape[0] // 2))
        P, _, _, cstd = atlas.pca(3)
        self.P, self.cstd = P, cstd
        self.mu = atlas.mean_vec()

    def start(self):
        j = int(self.rng.integers(self.a.n))
        return j, "start"

    def step(self, steps):
        l = self.traj[self.ki]                         # standardized (3,)
        self.ki = (self.ki + 1) % self.traj.shape[0]
        target = self.mu + self.P @ (self.cstd * self.amp * l)
        d = self.a.dists_from_vec(target)
        idx, dd = self.nav.nearest(d, steps, 1)
        return int(idx[0]), f"lorenz d={dd[0]:.2f}"


class _Drift:
    """a straight dissolve between two distant poles."""
    def __init__(self, atlas, rng, nav, p):
        self.a, self.rng, self.nav = atlas, rng, nav
        self.seconds = float(p["seconds"])

    def start(self):
        A = int(self.rng.integers(self.a.n))
        dA = self.a.dists_from_row(A)
        B = int(np.argmax(dA))
        self.av = self.a.vec(A).copy()
        self.bv = self.a.vec(B).copy()
        return A, f"pole A -> B d={dA[B]:.2f}"

    def step(self, steps):
        played = sum(s.dur for s in steps)
        prog = min(1.0, played / max(self.seconds, 1e-9))
        target = (1.0 - prog) * self.av + prog * self.bv
        d = self.a.dists_from_vec(target)
        idx, dd = self.nav.nearest(d, steps, 1)
        return int(idx[0]), f"drift {prog:.2f} d={dd[0]:.2f}"


class _Follow:
    """the corpus's own sequences, with near-neighbor branches."""
    def __init__(self, atlas, rng, nav, p):
        self.a, self.rng, self.nav = atlas, rng, nav
        self.branch = float(p["branch"])
        self.k = int(p["k"])
        self.temp = float(p["temp"])
        self.succ = _successor_map(atlas)

    def start(self):
        j = int(self.rng.integers(self.a.n))
        return j, "start"

    def step(self, steps):
        last = steps[-1].i
        if self.rng.random() < self.branch:
            d = self.a.dists_from_row(last)
            idx, dd = self.nav.nearest(d, steps, self.k)
            r = _rank_pick(len(idx), self.rng, self.temp)
            return int(idx[r]), f"branch r{r} d={dd[r]:.2f}"
        succ = self.succ.get(last)
        if succ is not None:
            return succ, "continue"
        d = self.a.dists_from_row(last)
        idx, dd = self.nav.nearest(d, steps, self.k)
        return int(idx[0]), f"dead end, nn d={dd[0]:.2f}"


def _lorenz(n: int = 4000, dt: float = 0.05, warm: int = 300) -> np.ndarray:
    """The Lorenz attractor (sigma=10, rho=28, beta=8/3), RK4, transient
    discarded, each axis standardized — a deterministic chaotic path
    with structure: two wings, endless but never repeating."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y,
                         x * y - beta * z])

    def rk4(s):
        k1 = f(s)
        k2 = f(s + dt / 2 * k1)
        k3 = f(s + dt / 2 * k2)
        k4 = f(s + dt * k3)
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    s = np.array([1.0, 1.0, 1.0])
    for _ in range(warm):
        s = rk4(s)
    traj = np.empty((n, 3))
    for i in range(n):
        s = rk4(s)
        traj[i] = s
    traj = (traj - traj.mean(axis=0)) / traj.std(axis=0)
    return traj


_WALKERS = {"chain": _Chain, "contrast": _Contrast, "roam": _Roam,
            "attractor": _Attractor, "drift": _Drift, "follow": _Follow}

MODES = {
    "chain": "near neighbors, temperature-biased — pieces similar in sequence",
    "contrast": "the farthest piece from the recent past — maximal difference",
    "roam": "brownian drift through the space, speed x macro-scale steps",
    "attractor": "a Lorenz trajectory steering through the principal axes",
    "drift": "a straight line between two distant poles, walked piece by piece",
    "follow": "the corpus's own sequences with occasional near-neighbor branches",
    "hocket": "voices interleaved at a breathing grain clock — granular, novel",
}


GRAIN_FLOOR = 0.01          # s — remnants shorter than this end a turn


def _hocket_steps(atlas, rng, seconds: float, p) -> list:
    """The granular composition: V voices, each its own walker stream,
    interleaved one grain at a time on a shared clock whose grain size
    breathes between grain_lo and grain_hi (log-space glides, occasional
    snaps to the extremes — sometimes shimmer, sometimes fragments).

    Each voice plays THROUGH its piece grain by grain (the cursor
    advances), advancing to a new piece via the inner policy when
    exhausted — voices anchor on their own last piece while the
    exclusion window is global, so the voices never echo each other.
    Every grain is a sample-exact span of original audio: still pure
    splicing, just fast.

    dist_prev measures each voice's own stream (its previous grain's
    piece), so the temperature reads the voices' coherence."""
    V = max(2, int(p["voices"]))
    lo = min(max(float(p["grain_lo"]), 0.005), 1.0)
    hi = max(lo * 2.0, float(p["grain_hi"]))
    inner = str(p["inner"])
    if inner not in ("chain", "roam"):
        raise ValueError(f"hocket inner must be chain or roam, got {inner}")
    nav = _Nav(atlas, p["exclude_window"])
    rngs = rng.spawn(V)                             # independent voices

    # voice starts: first random, each next the farthest from the
    # centroid so far — the voices begin spread across the territory
    starts = [int(rng.integers(atlas.n))]
    for _ in range(V - 1):
        d = atlas.dists_from_vec(atlas.vecs_mean(starts))
        d[starts] = -np.inf
        starts.append(int(np.argmax(d)))

    voices = []
    for v in range(V):
        w = _WALKERS[inner](atlas, rngs[v], nav, p)
        w.start()                                      # policy init
        if inner == "roam":                             # start where the
            w.p = atlas.vec(starts[v]).copy()          # voice starts
        voices.append({"walker": w, "piece": starts[v], "pos": 0.0})

    # the breathing grain clock
    g = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    g_start, g_target = g, float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    breath_start, breath_len = 0.0, float(rng.uniform(6.0, 18.0))
    breath_end = breath_start + breath_len

    steps, last_piece = [], {}
    t, slot = 0.0, 0
    cap = int(seconds / max(lo * 0.5, 0.005)) + 1000
    while t < seconds and len(steps) < cap:
        if t >= breath_end:                         # a new breath: new speed
            g_start = g
            g_target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            if rng.random() < 0.25:                 # sometimes snap hard
                g_target = lo if g_target > np.sqrt(lo * hi) else hi
            breath_start, breath_len = t, float(rng.uniform(6.0, 18.0))
            breath_end = t + breath_len
        frac = min((t - breath_start) / breath_len, 1.0)
        g = g_start * (g_target / g_start) ** frac  # log-space glide

        vo = voices[slot % V]
        if atlas.durs[vo["piece"]] - vo["pos"] < GRAIN_FLOOR:
            j, _ = vo["walker"].step(steps, anchor=vo["piece"]) \
                if inner == "chain" else vo["walker"].step(steps)
            vo["piece"], vo["pos"] = j, 0.0
        span = min(g, atlas.durs[vo["piece"]] - vo["pos"])
        prev = last_piece.get(slot % V)
        dp = atlas.dist(prev, vo["piece"]) if prev is not None else None
        steps.append(Step(int(vo["piece"]), float(span), dp,
                          f"v{slot % V} g={g:.3f}",
                          t0=float(vo["pos"]), t1=float(vo["pos"] + span)))
        last_piece[slot % V] = vo["piece"]
        vo["pos"] += span
        t += span
        slot += 1
    return steps


# ------------------------------------------------------------------ driver

def _defaults(mode: str, seconds: float) -> dict:
    d = {"seconds": seconds, "exclude_window": 12, "temp": 0.5, "k": 10,
         "speed": 1.0, "contrast_window": 1, "branch": 0.35, "amp": 2.0,
         "voices": 3, "grain_lo": 0.04, "grain_hi": 0.6, "inner": "chain"}
    return d


def walk(atlas, mode: str = "chain", seconds: float = 180.0, seed: int = 0,
         **params) -> Route:
    """Take a trip; return the route. Deterministic in (bank, seed).

    seconds         target duration of the piece (the walk stops once
                    the played pieces cover it; render trims the rest)
    seed            the walker's dice — same seed, same trip
    exclude_window  how many recent pieces (and their twins) are banned
    temp            chain/follow pick temperature (0 = greedy nearest)
    k               candidate pool size for chain/follow picks
    speed           roam's step size in macro-scale tenths
    contrast_window pieces of recent past the contrast walker flees
    branch          follow's probability of jumping instead of continuing
    amp             attractor amplitude in PCA-coordinate stds
    voices          hocket: how many interleaved voices
    grain_lo/hi     hocket: the grain clock breathes between these (s)
    inner           hocket: each voice's own policy (chain | roam)
    """
    if mode not in _WALKERS and mode != "hocket":
        raise ValueError(f"unknown mode {mode!r} — one of {sorted(MODES)}")
    p = _defaults(mode, seconds)
    p.update({k: v for k, v in params.items() if k in p})
    p["seconds"] = float(seconds)
    rng = np.random.default_rng(seed)
    if mode == "hocket":
        steps = _hocket_steps(atlas, rng, seconds, p)
        route = Route(mode=mode, seed=int(seed), seconds=float(seconds),
                      strategies=sorted(set(atlas.strategies.tolist())),
                      params={k: v for k, v in p.items()},
                      bank=atlas.path, corpus_root=str(atlas.corpus_root),
                      steps=steps)
        route.stats = _route_stats(atlas, steps)
        return route
    nav = _Nav(atlas, p["exclude_window"])
    walker = _WALKERS[mode](atlas, rng, nav, p)

    steps: list[Step] = []
    j, note = walker.start()
    steps.append(Step(int(j), float(atlas.durs[j]), None, note))
    total = float(atlas.durs[j])
    while total < seconds and len(steps) < MAX_STEPS:
        j, note = walker.step(steps)
        dist_prev = atlas.dist(int(j), steps[-1].i)
        steps.append(Step(int(j), float(atlas.durs[j]), dist_prev, note))
        total += float(atlas.durs[j])

    route = Route(mode=mode, seed=int(seed), seconds=float(seconds),
                  strategies=sorted(set(atlas.strategies.tolist())),
                  params={k: v for k, v in p.items()},
                  bank=atlas.path, corpus_root=str(atlas.corpus_root),
                  steps=steps)
    route.stats = _route_stats(atlas, steps)
    return route


def _route_stats(atlas, steps) -> dict:
    """The route's temperature trace: how far each step jumped, against
    the atlas's own random-pair distance — the number that shows a
    chain chaining and a contrast contrasting."""
    dp = np.array([s.dist_prev for s in steps[1:]
                   if s.dist_prev is not None], np.float64)
    if len(dp) == 0:
        dp = np.zeros(1)
    rng = np.random.default_rng(1234)                # fixed yardstick
    if atlas.n >= 2:
        a = rng.integers(0, atlas.n, 512)
        b = rng.integers(0, atlas.n, 512)
        keep = a != b
        rp = atlas.pair_dists(a[keep], b[keep])
        med_rand = float(np.median(rp))
    else:
        med_rand = 0.0
    med_dp = float(np.median(dp)) if len(dp) else 0.0
    return {
        "n_steps": len(steps),
        "total_dur": float(sum(s.dur for s in steps)),
        "dist_prev_mean": round(float(dp.mean()), 3) if len(dp) else None,
        "dist_prev_median": round(med_dp, 3),
        "dist_prev_max": round(float(dp.max()), 3) if len(dp) else None,
        "random_pair_median": round(med_rand, 3),
        "temperature": round(med_dp / med_rand, 3) if med_rand > 0 else None,
    }