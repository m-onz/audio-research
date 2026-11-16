"""Novelty search + minimal criterion + persistent archive (open-ended core).

The evolution loop is deliberately written as a *time-sliced coroutine*:
`work(budget_seconds)` evaluates as many pending genomes as fit in the
budget and finalises a generation once every member is scored.  The viewer
calls it with a small budget every frame (evolution runs "between" rendered
frames without threads), while headless mode calls it with a huge budget.

Novelty (Lehman & Stanley 2011) is the mean distance to the k nearest
neighbours in behaviour space, where behaviour space = fingerprint pixels
+ z-scored dynamics statistics.  A minimal criterion (Mouret & Clune /
Lehman et al.) filters degenerate systems (instant death, no motion, all
particles collapsed into one spot) so that "novel" cannot be gamed by
pathological behaviours.  There is no objective anywhere; the only stop
condition is you closing the window.
"""

from __future__ import annotations

import json
import random
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from .genome import Genome
from .sim import EvalResult, GRID, STAT_FLOORS, evaluate

KNN_K = 15
DESC_W_GRID = 0.35      # weight of fingerprint pixels in the descriptor
DESC_W_STATS = 0.35     # weight of z-scored dynamics statistics
ADMIT_PER_GEN = 3       # max archive admissions per generation
ARCHIVE_CAP = 1200      # RAM archive is FIFO; every admission is saved to disk


class Normalizer:
    """Online per-dimension mean/std with sensible floors for early runs."""

    def __init__(self):
        n = len(STAT_FLOORS)
        self.mean = np.zeros(n)
        self.m2 = np.zeros(n)
        self.count = 0

    def update(self, x: np.ndarray):
        self.count += 1
        d = x - self.mean
        self.mean += d / self.count
        self.m2 += d * (x - self.mean)

    def transform(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.m2 / max(self.count - 1, 1))
        scale = np.maximum(std, STAT_FLOORS * 0.35)
        return np.clip((x - self.mean) / scale, -4.0, 4.0)

    def save(self, path: Path):
        np.savez(path, mean=self.mean, m2=self.m2, count=np.array([self.count]))

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        z = cls()
        if path.exists():
            d = np.load(path)
            z.mean, z.m2, z.count = d["mean"], d["m2"], float(d["count"][0])
        return z


def make_descriptor(res: EvalResult, norm: Normalizer) -> np.ndarray:
    """Fingerprint pixels + z-scored stats -> one behaviour vector."""
    return np.concatenate([
        res.grids.reshape(-1) * DESC_W_GRID,
        norm.transform(res.stats) * DESC_W_STATS,
    ]).astype(np.float32)


def knn_novelty(desc: np.ndarray, pool: np.ndarray) -> float:
    if len(pool) == 0:
        return float("inf")
    d = np.linalg.norm(pool - desc, axis=1)
    k = min(KNN_K, len(d))
    return float(np.partition(d, k - 1)[:k].mean())


class Archive:
    """FIFO archive of discovered behaviours (the 'fossil record' lives on disk)."""

    def __init__(self, capacity=ARCHIVE_CAP):
        self.capacity = capacity
        self.entries: list[dict] = []    # {"genome","desc","gen","novelty","uid","stats"}

    def __len__(self):
        return len(self.entries)

    def add(self, entry: dict):
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self.entries.pop(0)          # oldest evicted; still safe on disk


class Evolver:
    def __init__(self, out_dir: str, pop_size=40, seed=None,
                 particles=128, frames=170, ttl=150, resume=True):
        self.out = Path(out_dir)
        self.gallery_dir = self.out / "gallery"
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.pop_size = pop_size
        self.eval_kw = dict(n=particles, frames=frames, ttl=ttl)
        self.rng = random.Random(seed)
        self.norm = Normalizer()
        self.archive = Archive()
        self.generation = 0
        self.add_thr = 0.0
        self.last_novelty = 0.0
        self.mc_pass_frac = 0.0
        self.eval_ms = 0.0
        self.net_size = (0, 24)
        self.pending: list[Genome] | None = None
        self.results: list[tuple[Genome, EvalResult, np.ndarray]] = []
        self._load(resume)
        if not self.archive.entries:     # fresh start
            self.population = [Genome.initial(self.rng) for _ in range(pop_size)]
        elif not hasattr(self, "population") or self.population is None:
            self.population = [Genome.initial(self.rng) for _ in range(pop_size)]

    # --------------------------------------------------------------- persistence

    def _save_entry(self, entry: dict):
        d = self.gallery_dir / entry["uid"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "genome.json").write_text(json.dumps(entry["genome"].to_dict()))
        np.save(d / "descriptor.npy", entry["desc"])
        (d / "stats.json").write_text(json.dumps({
            "uid": entry["uid"], "gen": entry["gen"], "novelty": entry["novelty"],
            "parent": entry["genome"].parent,
            "shape": entry["genome"].shape, "trail": entry["genome"].trail,
            "net_size": entry["net_size"],
            "stats_raw": entry["stats"].tolist(),
            "mc": True,
        }, indent=1))
        g = entry["grids"]
        img = np.stack([g[0], g[1], np.maximum(g[0], g[1])], axis=-1)
        img = (np.clip(img / max(img.max(), 1e-9), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).resize((GRID * 10, GRID * 10), Image.NEAREST).save(d / "fingerprint.png")

    def _save_state(self):
        self.norm.save(self.out / "normalizer.npz")
        (self.out / "state.json").write_text(json.dumps({
            "generation": self.generation,
            "add_thr": self.add_thr,
            "population": [g.to_dict() for g in self.population],
        }))

    def _load(self, resume: bool):
        if not resume:
            return
        state_p = self.out / "state.json"
        if state_p.exists():
            s = json.loads(state_p.read_text())
            self.generation = s["generation"]
            self.add_thr = s.get("add_thr", 0.0)
            self.population = [Genome.from_dict(d) for d in s["population"]]
        self.norm = Normalizer.load(self.out / "normalizer.npz")
        # rebuild archive from disk (fossil record -> living archive)
        if self.gallery_dir.exists():
            for d in sorted(self.gallery_dir.iterdir()):
                if not (d / "descriptor.npy").exists():
                    continue
                meta = json.loads((d / "stats.json").read_text())
                self.archive.entries.append({
                    "genome": Genome.from_dict(json.loads((d / "genome.json").read_text())),
                    "desc": np.load(d / "descriptor.npy"),
                    "grids": None,
                    "gen": meta["gen"], "novelty": meta["novelty"],
                    "uid": meta["uid"], "stats": np.array(meta["stats_raw"]),
                    "net_size": meta.get("net_size", [0, 24]),
                })
            if len(self.archive) > self.archive.capacity:
                self.archive.entries = self.archive.entries[-self.archive.capacity:]

    # ------------------------------------------------------------- the evolution

    def work(self, budget: float) -> list[dict]:
        """Evaluate pending genomes for up to `budget` seconds.  Returns the
        list of archive admissions when a generation completes (else [])."""
        if self.pending is None:
            self.pending = list(self.population)
            self.results = []
        t0 = time.perf_counter()
        while self.pending and (time.perf_counter() - t0) < budget:
            g = self.pending.pop()
            te = time.perf_counter()
            res = evaluate(g, seed=zlib.crc32(g.uid.encode()) & 0x7FFFFFFF,
                           **self.eval_kw)
            self.eval_ms = 0.9 * self.eval_ms + 0.1 * (time.perf_counter() - te) * 1000
            self.norm.update(res.stats)
            self.results.append((g, res, make_descriptor(res, self.norm)))
        if self.pending:
            return []
        return self._finalize()

    def _finalize(self) -> list[dict]:
        results, self.pending = self.results, None
        mc_ok = [(g, r, d) for (g, r, d) in results if r.mc_pass]
        self.mc_pass_frac = len(mc_ok) / max(len(results), 1)
        if mc_ok:
            pool = np.stack([e["desc"] for e in self.archive.entries] + [d for _, _, d in mc_ok])
        else:
            pool = None
        novs = []
        for (g, r, d) in results:
            if not r.mc_pass:
                novs.append(-1.0)
                continue
            others = pool[[i for i, (_, _, dd) in enumerate(mc_ok) if dd is not d]]
            n = knn_novelty(d, others) if len(others) else float("inf")
            novs.append(n)
        # -- archive admission -------------------------------------------------
        events = []
        order = sorted(range(len(results)), key=lambda i: novs[i], reverse=True)
        admitted = 0
        for i in order:
            if admitted >= ADMIT_PER_GEN:
                break
            g, r, d = results[i]
            if not r.mc_pass:
                break
            if len(self.archive) == 0 or novs[i] > self.add_thr:
                entry = {
                    "genome": g, "desc": d, "grids": r.grids, "stats": r.stats,
                    "gen": self.generation, "novelty": novs[i], "uid": g.uid,
                    "net_size": g.size(),
                }
                self.archive.add(entry)
                self._save_entry(entry)
                events.append(entry)
                admitted += 1
                self.last_novelty = novs[i]
        viable = [n for n in novs if n >= 0]
        if viable:
            self.add_thr = 0.75 * float(np.median(viable))
        # -- breeding ----------------------------------------------------------
        if mc_ok:
            nov_map = {id(g): n for (g, r, d), n in zip(results, novs)}
            ranked = sorted(mc_ok, key=lambda t: nov_map[id(t[0])], reverse=True)
            parents = ranked[:max(4, self.pop_size // 4)]
        else:
            parents = []
        children: list[Genome] = []
        if parents:
            i = 0
            while len(children) < self.pop_size:
                p = parents[i % len(parents)][0]
                c = p.clone()
                c.mutate(self.rng)
                children.append(c)
                i += 1
        else:
            # nothing viable: mix mutants of survivors with fresh blood
            for g in list(self.population):
                if len(children) >= self.pop_size * 0.6:
                    break
                c = g.clone()
                c.mutate(self.rng)
                children.append(c)
            while len(children) < self.pop_size:
                children.append(Genome.initial(self.rng))
        self.population = children
        h_tot = c_tot = 0
        for g, r, d in results:
            h, c = g.size()
            h_tot, c_tot = h_tot + h, c_tot + c
        self.net_size = (h_tot / max(len(results), 1), c_tot / max(len(results), 1))
        self.generation += 1
        self._save_state()
        return events

    def boost(self, genome: Genome, n=8):
        """Inject mutated clones of a chosen genome (human-in-the-loop shortcut)."""
        slots = self.rng.sample(range(len(self.population)), min(n, len(self.population)))
        for s in slots:
            c = genome.clone()
            c.mutate(self.rng)
            self.population[s] = c

    # ------------------------------------------------------------------- report

    def status_line(self) -> str:
        return (f"gen {self.generation}  archive {len(self.archive)}  "
                f"nov {self.last_novelty:.2f}  thr {self.add_thr:.2f}  "
                f"mc {self.mc_pass_frac * 100:.0f}%  "
                f"net {self.net_size[0]:.1f}h/{self.net_size[1]:.0f}c  "
                f"eval {self.eval_ms:.0f}ms")