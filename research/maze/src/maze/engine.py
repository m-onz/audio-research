"""engine: the maze loop — four memories, one organism.

Everything is shared: the founder, the mutation kernel, the viability
gate, the render budget, the survivor rule, the seeds. The ONLY thing
that differs between arms is the memory policy:

    greedy    parent = argmax tonality(genome)     — the objective trap
    random    uniform parents, all viable children admitted — drift
    cell      grid-cell latest parents (womb's rule), fine 24x24 grid
    novelty   uniform parents, children compete by ARCHIVE novelty;
              the most novel half is admitted (Lehman's ranking) —
              the ONLY difference from random is that echo decides
              which children live

Measured WITHOUT echo (see measure.py): archetype census from the
genome (ground truth), cumulative discovery on a 24x24 independent
grid, coverage/radius in plain features. One number measured WITH
echo, against true lineage (not against echo itself): parent-kNN
recall — when the archive says 'this child sounds like X', how often
is X actually the parent?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from echo import EchoBank, encode

from .measure import (ascii_map, plain_coverage, plain_features,
                      plain_radius)
from . import dx as dx_org
from . import voice as bumps_org

__all__ = ["run_experiment", "report_run", "ARMS", "ORGANISMS"]

ARMS = ("greedy", "random", "cell", "novelty")
ORGANISMS = {"dx": dx_org, "bumps": bumps_org}
GRID = 24                    # the independent discovery grid (24x24)


# ------------------------------------------------------------------ viability

def viable(audio) -> bool:
    """The minimal criterion, identical for every arm: audible and
    alive. No quality, no ranking — the maze walls, not the maze."""
    x = np.asarray(audio, np.float64)
    if not np.isfinite(x).all():
        return False
    rms = float(np.sqrt((x ** 2).mean()))
    if not (0.02 <= rms <= 0.6):
        return False
    blocks = np.array_split(x, 4, axis=0)
    rms_b = [float(np.sqrt((b ** 2).mean())) for b in blocks]
    if max(rms_b) / (min(rms_b) + 1e-9) < 1.05:      # static drone gate
        return False
    return True


def _cell_key(f):
    return (int(np.clip(f[1] * GRID, 0, GRID - 1)),
            int(np.clip(f[2] * GRID, 0, GRID - 1)))


# ------------------------------------------------------------------ one arm

def run_arm(arm: str, seed: int, gens: int, pop: int, births: int,
            out: Path, org, verbose=False) -> dict:
    rng = np.random.default_rng(seed)              # children stream
    sel = np.random.default_rng(seed + 555)        # parent policy stream
    bank = EchoBank(":memory:")
    log = []
    parent_recall = []          # echo diagnostic: did kNN find the parent?
    seconds = org.SECONDS

    # ---- founder (identical across arms for a given seed)
    founder = None
    for _ in range(60):
        cand = org.random_genome(rng, seed=int(rng.integers(1 << 30)))
        audio = org.render(cand, seconds)
        if viable(audio):
            founder = cand
            break
    if founder is None:
        bank.close()
        return {"arm": arm, "seed": seed, "log": [], "kinds_final": 0,
                "kinds_ever": 0, "coverage": {"pairwise": 0.0, "cells": 0},
                "cells_cum_final": 0, "radius": 0.0, "novelty_mean_final":
                None, "parent_recall_pct": None, "census_final": {},
                "elapsed": 0.0, "feats_final": []}
    f_audio = org.render(founder, seconds)
    founder_vec = encode(f_audio, 44100)
    f_feat = plain_features(f_audio)
    founder_id = bank.add(founder_vec, {"gen": 0, "arch": founder.archetype()},
                          role="pop", gen=0)
    population = [{"g": founder, "vec": founder_vec, "audio": f_audio,
                   "gen": 0, "id": founder_id, "feat": f_feat}]
    cells_now: dict = {}                            # the cell arm's grid
    cells_seen = set()                              # cumulative discovery
    kinds_seen = set()                               # cumulative archetype discovery
    eras = []                                        # era samples: the journey
    era_every = max(1, gens // 20)                   # ~20 stops on the tour

    def _pop_nov(m) -> float:
        key = id(m)
        if key not in _nov_cache:
            _nov_cache[key] = bank.novelty(m["vec"], k=min(15, len(bank)))
        return _nov_cache[key]

    def pick_parent():
        if arm == "greedy":
            return max(population, key=lambda m: org.tonality(m["g"]))
        if arm == "cell" and cells_now and sel.random() < 0.65:
            key = list(cells_now)[int(sel.integers(len(cells_now)))]
            return cells_now[key]
        return population[int(sel.integers(len(population)))]

    def admit(child, audio, vec, parent, gen, nov=None):
        near = bank.knn(vec, k=min(5, len(bank)))
        parent_recall.append(any(n["id"] == parent["id"] for n in near))
        member = {"g": child, "vec": vec, "audio": audio, "gen": gen}
        member["feat"] = plain_features(audio)              # cached once
        member["id"] = bank.add(vec, {"gen": gen, "arch": child.archetype()},
                                role="pop", gen=gen)
        population.append(member)
        if len(population) > pop:
            population.pop(int(sel.integers(len(population))))
        key = (int(np.clip(member["feat"][1] * GRID, 0, GRID - 1)),
               int(np.clip(member["feat"][2] * GRID, 0, GRID - 1)))
        cells_now[key] = member
        cells_seen.add(key)
        kinds_seen.add(child.archetype())
        return member

    t0 = time.time()
    for gen in range(1, gens + 1):
        _nov_cache = {}
        admits, rejects_mc, rejects_nov = 0, 0, 0
        pending = []                         # novelty arm: children compete
        for _ in range(births):
            parent = pick_parent()
            child = org.mutate(parent["g"], rng, seed=int(rng.integers(1 << 30)))
            audio = org.render(child, seconds)
            if not viable(audio):
                rejects_mc += 1
                continue
            vec = encode(audio, 44100)
            if arm == "novelty":
                nov = bank.novelty(vec, k=min(15, len(bank)))
                pending.append((nov, child, audio, vec, parent))
                continue
            admit(child, audio, vec, parent, gen)
            admits += 1

        if arm == "novelty" and pending:
            pending.sort(key=lambda c: -c[0])
            keep = max(1, births // 2)
            for nov, child, audio, vec, parent in pending[:keep]:
                admit(child, audio, vec, parent, gen, nov=nov)
                admits += 1
            # the era's most novel specimen rides the journey
            if gen % era_every == 0 and pending:
                nov, child, audio, vec, parent = pending[0]
                eras.append((gen, float(nov), child.archetype(), audio))
            rejects_nov = len(pending) - admits

        archs = {m["g"].archetype() for m in population}
        log.append({
            "gen": gen, "admits": admits, "rej_mc": rejects_mc,
            "rej_nov": rejects_nov, "archetypes": len(archs),
            "kinds_cum": len(kinds_seen),
            "novelty_mean": float(np.mean(
                [_pop_nov(m) for m in population])) if population else None,
            "cells": len(cells_now), "cells_cum": len(cells_seen),
            "elapsed": round(time.time() - t0, 1),
        })
        if verbose and gen % 20 == 0:
            print(f"[maze] {arm} s{seed} g{gen}: +{admits} "
                  f"kinds {log[-1]['archetypes']} "
                  f"cells {log[-1]['cells_cum']}", flush=True)

    # ---- final census (independent) + artifacts
    feats = np.stack([m["feat"] for m in population])
    census = {}
    for m in population:
        census[m["g"].archetype()] = census.get(m["g"].archetype(), 0) + 1
    ever = {e["arch"] for e in bank.metas}
    result = {
        "arm": arm, "seed": seed, "gens": gens,
        "log": log,
        "census_final": census,
        "kinds_final": len(census),
        "kinds_ever": len(ever),
        "kinds_ever_list": sorted(ever),
        "coverage": plain_coverage(feats),
        "cells_cum_final": len(cells_seen),
        "radius": plain_radius(feats, f_feat),
        "feats_final": feats.tolist(),
        "novelty_mean_final": log[-1]["novelty_mean"] if log else None,
        "parent_recall_pct": float(100 * np.mean(parent_recall))
        if parent_recall else None,
        "elapsed": round(time.time() - t0, 1),
    }
    rdir = out / f"{arm}_s{seed}"
    rdir.mkdir(parents=True, exist_ok=True)
    from echo.probes import write_wav
    pieces = []
    for i, m in enumerate(population):
        p = rdir / f"spec_{i:02d}_{m['g'].archetype()}.wav"
        write_wav(p, m["audio"], 44100)
        pieces.append(m["audio"])
    if pieces:
        nfade = int(0.4 * 44100)
        total = sum(len(p) for p in pieces) - nfade * (len(pieces) - 1)
        mix = np.zeros((max(total, 1), 2), np.float32)
        pos = 0
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nfade) / nfade)
        for i, p in enumerate(pieces):
            p = p.copy()
            if i > 0:
                p[:nfade] *= ramp[:, None]
            if i < len(pieces) - 1:
                p[-nfade:] *= ramp[::-1][:, None]
            mix[pos: pos + len(p)] += p
            pos += len(p) - nfade
        peak = float(np.abs(mix).max())
        if peak > 1e-9:
            mix = mix / peak * 0.92
        write_wav(rdir / "mixtape.wav", mix, 44100)
    # the journey: one specimen per era, ordered by generation — the
    # run's exploration heard as a tour through its own history
    if eras:
        eras.sort(key=lambda e: e[0])
        nfade = int(0.4 * 44100)
        total = sum(len(e[3]) for e in eras) - nfade * (len(eras) - 1)
        jmix = np.zeros((max(total, 1), 2), np.float32)
        pos = 0
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nfade) / nfade)
        for i, (gen, nov, arch, audio) in enumerate(eras):
            p = audio.copy().astype(np.float32)
            if i > 0:
                p[:nfade] *= ramp[:, None]
            if i < len(eras) - 1:
                p[-nfade:] *= ramp[::-1][:, None]
            jmix[pos: pos + len(p)] += p
            pos += len(p) - nfade
        peak = float(np.abs(jmix).max())
        if peak > 1e-9:
            jmix = jmix / peak * 0.92
        write_wav(rdir / "journey.wav", jmix, 44100)
        (rdir / "journey.json").write_text(json.dumps(
            [{"gen": g, "novelty": n, "archetype": a} for g, n, a, _ in eras],
            indent=1))
    bank.close()
    return result


# ------------------------------------------------------------------ experiment

def run_experiment(out="maze_out/run1", seeds=(0,), gens=40, pop=16,
                   births=8, arms=ARMS, organism="dx", verbose=True) -> dict:
    org = ORGANISMS[organism]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for arm in arms:
        runs = []
        for seed in seeds:
            runs.append(run_arm(arm, seed, gens, pop, births, out, org,
                               verbose))
        ok = [r for r in runs if r.get("log")]
        if not ok:
            all_results[arm] = {"runs": runs, "feats_final": [],
                                 "kinds_final_mean": 0.0,
                                 "kinds_ever_mean": 0.0,
                                 "coverage_pairwise_mean": 0.0,
                                 "coverage_cells_mean": 0.0,
                                 "cells_cum_mean": 0.0, "radius_mean": 0.0,
                                 "novelty_mean_final": 0.0,
                                 "parent_recall_mean": 0.0,
                                 "elapsed_mean": 0.0, "growth": []}
            continue
        L = ok[0]["log"]
        marks = [int(np.clip(round(q * (len(L) - 1)), 0, len(L) - 1))
                 for q in (0.25, 0.5, 0.75, 1.0)]
        growth = []
        for m in marks:
            vals = [r["log"][m]["cells_cum"] for r in ok
                    if len(r["log"]) > m]
            growth.append(float(np.mean(vals)) if vals else 0.0)
        all_results[arm] = {
            "runs": [{k: v for k, v in r.items() if k != "feats_final"}
                     for r in runs],
            "feats_final": np.mean([r["feats_final"] for r in ok],
                                    axis=0).tolist() if ok else [],
            "kinds_final_mean": float(np.mean([r["kinds_final"] for r in ok])),
            "kinds_ever_mean": float(np.mean([r["kinds_ever"] for r in ok])),
            "coverage_pairwise_mean": float(np.mean(
                [r["coverage"]["pairwise"] for r in ok])),
            "coverage_cells_mean": float(np.mean(
                [r["coverage"]["cells"] for r in ok])),
            "cells_cum_mean": float(np.mean(
                [r["cells_cum_final"] for r in ok])),
            "radius_mean": float(np.mean([r["radius"] for r in ok])),
            "novelty_mean_final": float(np.mean(
                [r["novelty_mean_final"] or 0 for r in ok])),
            "parent_recall_mean": float(np.mean(
                [r["parent_recall_pct"] or 0 for r in ok])),
            "elapsed_mean": float(np.mean([r["elapsed"] for r in ok])),
            "growth": growth,
        }
    summary = {"out": str(out), "gens": gens, "pop": pop,
               "births": births, "seeds": list(seeds), "arms": list(arms),
               "organism": organism, "grid": GRID,
               "results": all_results}
    (out / "experiment.json").write_text(json.dumps(summary, indent=1))
    (out / "report.md").write_text(report_run(summary))
    if verbose:
        print(f"[maze] report -> {out / 'report.md'}")
    return summary


# ------------------------------------------------------------------ report

def report_run(summary: dict) -> str:
    r = summary["results"]
    L = [f"# maze — objective vs drift vs memory ({summary.get('organism', 'dx')} organism)",
         "",
         f"gens {summary['gens']} | pop {summary['pop']} | "
         f"births/gen {summary['births']} | seeds {summary['seeds']} | "
         f"discovery grid {summary.get('grid', 24)}x{summary.get('grid', 24)}",
         "identical founder, mutations, viability, budget; only the ",
         "memory policy differs.", "",
         "All coverage numbers are measured in the INDEPENDENT plain-",
         "feature space (or read off the genome) — echo never judges",
         "echo. The last column is the one echo-measured number: does",
         "the archive's k-NN recall each child's true parent?", "",
         "| arm | kinds ever | cells ever | yield | pairwise | radius | parent-recall |",
         "|---|---|---|---|---|---|---|"]
    for arm in summary["arms"]:
        a = r[arm]
        yields = []
        for run in a["runs"]:
            tot = sum(g["admits"] for g in run.get("log", []))
            if tot > 0:
                yields.append(run["cells_cum_final"] / tot)
        y = float(np.mean(yields)) if yields else 0.0
        L.append(f"| {arm} | {a['kinds_ever_mean']:.2f} | "
                 f"{a['cells_cum_mean']:.1f} | {y:.4f} | "
                 f"{a['coverage_pairwise_mean']:.3f} | "
                 f"{a['radius_mean']:.2f} | "
                 f"{a['parent_recall_mean']:.0f}% |")
    L += ["", "yield = independent-grid cells discovered per admission —",
          "the fair rate: novelty admits only the top half of each",
          "generation's children, so raw totals undercount it.", "",
          "**territory growth** — cumulative independent-grid cells",
          "discovered at 25 / 50 / 75 / 100% of the run (the shape is the",
          "story: diffusion flattens, directed exploration keeps rising):",
          "", "| arm | 25% | 50% | 75% | 100% |", "|---|---|---|---|---|"]
    for arm in summary["arms"]:
        g = r[arm].get("growth", [])
        if g:
            L.append(f"| {arm} | " + " | ".join(f"{v:.0f}" for v in g) + " |")
    feats = {arm: np.array(r[arm]["feats_final"]) for arm in summary["arms"]
             if r[arm]["feats_final"]}
    if feats:
        L += ["", ascii_map(feats), ""]
    for arm in summary["arms"]:
        kinds = {}
        for run in r[arm]["runs"]:
            for k, v in run.get("census_final", {}).items():
                kinds[k] = kinds.get(k, 0) + v
        if kinds:
            L.append(f"**{arm}** final census (all seeds): " +
                     ", ".join(f"{k}×{v}" for k, v in
                               sorted(kinds.items(), key=lambda kv: -kv[1])))
            L.append("")
    L.append(f"listen: `{summary['out']}/<arm>_s<seed>/mixtape.wav` — the "
             "subjective metric (Stanley & Soros: you are the judge).")
    L.append("")
    return "\n".join(L)