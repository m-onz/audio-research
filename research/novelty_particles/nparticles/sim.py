"""Vectorised particle-system simulation and behaviour characterisation.

The simulation follows the paper's "generic" and "trail" particle classes
(section III): particles spawn inside a generation shape, an ANN maps
(position, distance-from-centre) -> (velocity, colour, size) every frame,
particles die when their TTL expires or they leave the system space, and a
continuous emitter keeps the system populated.

Because every particle in a system shares one ANN, the network is activated
once per frame for the whole particle batch -- this makes headless novelty
evaluation cheap enough to run inside an interactive loop.

`evaluate()` characterises a genome's *behaviour* as
  * two normalised 32x32 occupancy fingerprints (early vs. late phase of the
    run -- they capture temporal change, i.e. an expanding-then-collapsing
    system looks different from a steady one), and
  * 15 scalar dynamics statistics (speed, spread, entropy, symmetry,
    angular momentum, radial drift, colour dynamics, ...).
Evolution then measures novelty as distance in this behaviour space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .genome import Genome

SPACE = 1.35           # particles beyond |x| or |y| > SPACE die
GRID = 32              # fingerprint resolution
SPEED = 0.012          # ANN velocity output * SPEED = world units / frame


def _spawn_positions(shape: str, k: int, rng: np.random.Generator) -> np.ndarray:
    if shape == "point":                       # attached-to-object effects
        return rng.uniform(-0.02, 0.02, size=(k, 2))
    if shape == "circle":                      # rings, energy fields
        a = rng.uniform(0.0, 2.0 * np.pi, k)
        r = 0.45 + rng.uniform(-0.02, 0.02, k)
        return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    if shape == "line":                        # weapon trails / slashes
        return np.stack([rng.uniform(-0.7, 0.7, k),
                          rng.uniform(-0.02, 0.02, k)], axis=1)
    # disc                                    # smoke, explosions, fire
    a = rng.uniform(0.0, 2.0 * np.pi, k)
    r = 0.5 * np.sqrt(rng.uniform(0.0, 1.0, k))
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


class Sim:
    """One live particle system (used both headless and on-screen)."""

    def __init__(self, genome: Genome, n=128, ttl=150, speed=SPEED, seed=0):
        self.g = genome
        self.n, self.ttl, self.speed = n, ttl, speed
        self.rng = np.random.default_rng(seed)
        self.px = np.zeros(n)
        self.py = np.zeros(n)
        self.age = np.full(n, ttl)
        self.alive = np.zeros(n, dtype=bool)
        self.acc = 0.0
        self.frame = 0
        # trail gene: a ring-ish pool of stationary fading points (paper's trail class)
        self.trail = bool(genome.trail)
        if self.trail:
            cap = n * 12
            self.tx = np.zeros(cap)
            self.ty = np.zeros(cap)
            self.tc = np.zeros((cap, 3))
            self.tlife = np.zeros(cap)         # <= 0 means the slot is free
            self.trail_ttl = 42

    def step(self) -> dict:
        self.frame += 1
        # -- continuous emission --------------------------------------------
        self.acc += self.n / self.ttl
        k = int(self.acc)
        self.acc -= k
        if k > 0:
            dead = np.flatnonzero(~self.alive)
            if len(dead):
                k = min(k, len(dead))
                idx = dead[self.rng.permutation(len(dead))[:k]]
                pos = _spawn_positions(self.g.shape, k, self.rng)
                self.px[idx], self.py[idx] = pos[:, 0], pos[:, 1]
                self.age[idx], self.alive[idx] = 0, True
        # -- activate the ANN once for all live particles --------------------
        ia = np.flatnonzero(self.alive)
        col = size = spd = vx = vy = np.zeros(0)
        if len(ia):
            d = np.sqrt(self.px[ia] ** 2 + self.py[ia] ** 2)
            X = np.stack([self.px[ia], self.py[ia], d,
                          np.ones(len(ia))], axis=1)
            Y = self.g.evaluate(X)
            vx, vy = Y[:, 0] * self.speed, Y[:, 1] * self.speed
            self.px[ia] += vx
            self.py[ia] += vy
            col = np.clip(Y[:, 2:5] * 0.5 + 0.5, 0.0, 1.0)
            size = np.clip(Y[:, 5] * 0.5 + 0.5, 0.0, 1.0)
            spd = np.sqrt(vx * vx + vy * vy)
            self.age[ia] += 1
            kill = ((self.age[ia] >= self.ttl)
                    | (np.abs(self.px[ia]) > SPACE)
                    | (np.abs(self.py[ia]) > SPACE)
                    | np.isnan(self.px[ia]) | np.isnan(self.py[ia]))
            self.alive[ia[kill]] = False
        # -- trails ----------------------------------------------------------
        tpx = tpy = tcol = np.zeros(0)
        if self.trail:
            self.tlife -= 1
            if len(ia) and self.frame % 4 == 0:
                order = np.argsort(self.tlife)             # oldest/free first
                slots = order[:len(ia)]
                self.tx[slots], self.ty[slots] = self.px[ia], self.py[ia]
                self.tc[slots] = col
                self.tlife[slots] = self.trail_ttl
            tm = self.tlife > 0
            if tm.any():
                tpx, tpy = self.tx[tm], self.ty[tm]
                tcol = self.tc[tm] * (self.tlife[tm] / self.trail_ttl)[:, None] * 0.55
        return {
            "n_alive": int(len(ia)),
            "px": self.px[ia], "py": self.py[ia],
            "vx": vx, "vy": vy,
            "col": col, "size": size, "spd": spd,
            "tpx": tpx, "tpy": tpy, "tcol": tcol,
        }


# --------------------------------------------------------------------------
# behaviour characterisation
# --------------------------------------------------------------------------

STATS = [  # (name, floor) -- floor = minimum plausible std used for z-scaling
    ("mean_speed", 0.004), ("std_speed", 0.004),
    ("mean_dist", 0.50), ("std_dist", 0.30),
    ("radial_drift", 0.0010),
    ("ang_mean", 0.003), ("ang_std", 0.006),
    ("alive_frac", 0.25), ("spread", 0.30),
    ("entropy", 0.10), ("sym_h", 0.15), ("sym_v", 0.15),
    ("temporal_change", 0.06),
    ("color_temporal", 0.05), ("color_spatial", 0.05),
]
STAT_FLOORS = np.array([f for _, f in STATS], dtype=np.float64)


@dataclass
class EvalResult:
    grids: np.ndarray          # (2, GRID, GRID) normalised early/late fingerprints
    stats: np.ndarray           # (len(STATS),) raw statistics
    mc_pass: bool
    reasons: list = field(default_factory=list)
    alive_frac: float = 0.0
    mean_speed: float = 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a - a.mean(), b - b.mean()
    den = np.sqrt((va * va).sum() * (vb * vb).sum())
    return float((va * vb).sum() / den) if den > 1e-12 else 0.0


def evaluate(genome: Genome, n=128, frames=170, ttl=150, seed=0,
             speed=SPEED) -> EvalResult:
    """Run one headless simulation and characterise the behaviour."""
    sim = Sim(genome, n=n, ttl=ttl, speed=speed, seed=seed)
    third = max(1, frames // 3)
    occ_e = np.zeros((GRID, GRID))
    occ_l = np.zeros((GRID, GRID))
    occ = np.zeros((GRID, GRID))               # decaying "live" occupancy
    spd_m, spd_s, d_m, d_s, ang_m, ang_s = [], [], [], [], [], []
    d_t = []                                   # frame indices with live data
    alive_c, spread_c, cm_all, cs_all = [], [], [], []
    for t in range(frames):
        r = sim.step()
        mark = np.zeros((GRID, GRID))
        if r["n_alive"]:
            ix = np.clip((r["px"] + SPACE) / (2 * SPACE) * GRID, 0, GRID - 1.001).astype(int)
            iy = np.clip((r["py"] + SPACE) / (2 * SPACE) * GRID, 0, GRID - 1.001).astype(int)
            np.add.at(mark, (iy, ix), 1.0)
            vx = r["spd"]
            dist = np.sqrt(r["px"] ** 2 + r["py"] ** 2)
            spd_m.append(float(vx.mean()))
            spd_s.append(float(vx.std()))
            d_m.append(float(dist.mean()))
            d_s.append(float(dist.std()))
            d_t.append(t)
            mom = r["px"] * r["vy"] - r["py"] * r["vx"]
            ang_m.append(float(mom.mean()))
            ang_s.append(float(mom.std()))
            spread_c.append(float((r["px"].std() + r["py"].std()) * 0.5))
            cm_all.append(r["col"].mean(axis=0))
            cs_all.append(r["col"].std(axis=0).mean())
        if len(r["tpx"]):
            tix = np.clip((r["tpx"] + SPACE) / (2 * SPACE) * GRID, 0, GRID - 1.001).astype(int)
            tiy = np.clip((r["tpy"] + SPACE) / (2 * SPACE) * GRID, 0, GRID - 1.001).astype(int)
            np.add.at(mark, (tiy, tix), 0.4)
        alive_c.append(r["n_alive"])
        occ = occ * 0.97 + mark
        if t < third:
            occ_e += mark
        elif t >= frames - third:
            occ_l += mark
    # -- normalise fingerprints -------------------------------------------
    m = max(occ_e.max(), occ_l.max(), 1e-9)
    occ_e, occ_l = occ_e / m, occ_l / m
    # -- statistics --------------------------------------------------------
    occ_n = occ / max(occ.max(), 1e-9)
    p = occ_n.flatten()
    p = p / p.sum() if p.sum() > 0 else p
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(GRID * GRID)) if len(nz) else 0.0
    sym_h = _corr(occ_n, occ_n[:, ::-1])
    sym_v = _corr(occ_n, occ_n[::-1, :])
    temporal = float(np.abs(occ_e - occ_l).mean())
    F = np.asarray(d_t, dtype=np.float64)
    drift = float(np.polyfit(F, np.asarray(d_m), 1)[0]) if len(d_m) > 2 else 0.0
    cm = np.asarray(cm_all) if cm_all else np.zeros((0, 3))
    color_temporal = float(cm.std(axis=0).mean()) if len(cm) else 0.0
    color_spatial = float(np.asarray(cs_all).mean()) if cs_all else 0.0
    alive_frac = float(np.mean(alive_c) / n)
    stats = np.array([
        np.mean(spd_m) if spd_m else 0.0,
        np.mean(spd_s) if spd_s else 0.0,
        np.mean(d_m) if d_m else 0.0,
        np.mean(d_s) if d_s else 0.0,
        drift,
        np.mean(ang_m) if ang_m else 0.0,
        np.mean(ang_s) if ang_s else 0.0,
        alive_frac,
        np.mean(spread_c) if spread_c else 0.0,
        entropy, sym_h, sym_v, temporal,
        color_temporal, color_spatial,
    ], dtype=np.float64)
    # -- minimal criterion: reject degenerate behaviours --------------------
    reasons = []
    if alive_frac < 0.10:
        reasons.append(f"alive={alive_frac:.2f}")
    if stats[0] < 0.0015:
        reasons.append(f"speed={stats[0]:.4f}")
    if entropy < 0.06:
        reasons.append(f"entropy={entropy:.2f}")
    if stats[8] < 0.015:
        reasons.append(f"spread={stats[8]:.3f}")
    return EvalResult(
        grids=np.stack([occ_e, occ_l]).astype(np.float32),
        stats=stats,
        mc_pass=not reasons,
        reasons=reasons,
        alive_frac=alive_frac,
        mean_speed=float(stats[0]),
    )