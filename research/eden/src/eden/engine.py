"""engine: eden's loop — two memories, one world, crossover everywhere.

Both arms share the identical operator set: two-parent crossover,
mutation, viability gate, render budget, survivor rule. The ONLY
difference is the memory policy:

    random    uniform parents, all viable children admitted — no memory
    novelty   uniform parents, children compete by ARCHIVE novelty;
              the most novel half is admitted (Lehman's ranking) —
              echo decides which children live

Telemetry per generation (all independent of echo): discovery on the
24x24 plain-feature grid, the archetype census, AND the
complexification ledger — mean agent count, interaction counts,
collisions per episode — because 'novelty search grows structure' is a
measurable claim, not a slogan.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from echo import EchoBank, encode

from .genetics import (crossover, mutate, random_world, structure_stats)
from .world import SECONDS, SR, WorldGenome, render

from maze.measure import ascii_map, plain_coverage, plain_features, plain_radius

__all__ = ["run_experiment", "report_run", "ARMS", "GRID", "viable"]

ARMS = ("random", "novelty")
GRID = 24


def viable(audio) -> bool:
    """The minimal criterion, identical for every arm: audible and
    alive. The world must perform — a listener parked in silence or a
    static drone dies."""
    x = np.asarray(audio, np.float64)
    if not np.isfinite(x).all():
        return False
    rms = float(np.sqrt((x ** 2).mean()))
    if not (0.02 <= rms <= 0.6):
        return False
    blocks = np.array_split(x, 8, axis=0)
    rms_b = [float(np.sqrt((b ** 2).mean())) for b in blocks]
    if max(rms_b) / (min(rms_b) + 1e-9) < 1.05:
        return False
    return True


def _eval_job(job):
    """Worker: genome -> (audio, echo-vector, telemetry). Pure and
    deterministic — safe to run anywhere (pool, process, machine)."""
    idx, world = job
    audio, tele = render(world)
    vec = encode(audio, SR)
    return idx, audio, vec, tele


def run_arm(arm: str, seed: int, gens: int, pop: int, births: int,
            out: Path, verbose=False, pool=None) -> dict:
    rng = np.random.default_rng(seed)              # genetics stream
    sel = np.random.default_rng(seed + 555)        # parent/survivor stream
    bank = EchoBank(":memory:")
    log = []
    cells_now: dict = {}
    cells_seen = set()
    kinds_seen = set()
    eras = []
    era_every = max(1, gens // 20)
    collisions_hist, agents_hist, events_hist = [], [], []

    # ---- founders
    founder = None
    for _ in range(80):
        cand = random_world(rng, seed=int(rng.integers(1 << 30)))
        audio, tele = render(cand)
        if viable(audio):
            founder = cand
            break
    if founder is None:
        bank.close()
        return {"arm": arm, "seed": seed, "log": [], "census_final": {},
                "cells_cum_final": 0, "kinds_final": 0, "kinds_ever": 0,
                "coverage": {"pairwise": 0.0, "cells": 0, "n": 0},
                "radius": 0.0, "parent_recall_pct": None,
                "novelty_mean_final": None, "feats_final": [], "elapsed": 0.0}
    f_audio, _ = render(founder)
    f_vec = encode(f_audio, SR)
    f_feat = plain_features(f_audio)
    fid = bank.add(f_vec, {"gen": 0, "arch": founder.archetype()},
                   role="pop", gen=0)
    population = [{"g": founder, "audio": f_audio, "vec": f_vec,
                   "feat": f_feat, "gen": 0, "id": fid, "tele": {}}]

    def admit(world, audio, vec, gen, parents_ids, nov=None):
        near = bank.knn(vec, k=min(5, len(bank)))
        recall = any(n["id"] in parents_ids for n in near)
        m = {"g": world, "audio": audio, "vec": vec,
             "feat": plain_features(audio), "gen": gen}
        m["id"] = bank.add(vec, {"gen": gen, "arch": world.archetype()},
                           role="pop", gen=gen)
        m["recall"] = recall
        population.append(m)
        if len(population) > pop:
            population.pop(int(sel.integers(len(population))))
        key = (int(np.clip(m["feat"][1] * GRID, 0, GRID - 1)),
               int(np.clip(m["feat"][2] * GRID, 0, GRID - 1)))
        cells_now[key] = m
        cells_seen.add(key)
        kinds_seen.add(world.archetype())
        return m

    t0 = time.perf_counter()
    for gen in range(1, gens + 1):
        admits = 0
        pending = []
        # -- breed all children FIRST (rng semantics identical to the
        # serial loop: parents + crossover + mutation consume the same
        # streams, in the same order, regardless of render scheduling)
        jobs = []
        for _ in range(births):
            pa = population[int(sel.integers(len(population)))]
            pb = population[int(sel.integers(len(population)))]
            child = crossover(pa["g"], pb["g"], rng,
                              seed=int(rng.integers(1 << 30)))
            child = mutate(child, rng, seed=int(rng.integers(1 << 30)))
            jobs.append((child, {pa["id"], pb["id"]}))
        # -- render + encode in parallel (10 M4 cores beat any GPU here:
        # the workload is small-kernel and python-loop bound)
        if pool is not None:
            results = pool.map(_eval_job, [(i, j[0]) for i, j in enumerate(jobs)],
                               chunksize=1)
            results = sorted(results, key=lambda r: r[0])
        else:
            results = [_eval_job((i, j[0])) for i, j in enumerate(jobs)]
        for (child, parents_ids), (_, audio, vec, tele) in zip(jobs, results):
            if not viable(audio):
                continue
            if arm == "novelty":
                pending.append((bank.novelty(vec, k=min(15, len(bank))),
                                child, audio, vec, tele, parents_ids))
                continue
            admit(child, audio, vec, gen, parents_ids)
            collisions_hist.append(tele["collisions"])
            agents_hist.append(tele["agents"])
            events_hist.append(tele["events"])
            admits += 1

        if arm == "novelty" and pending:
            pending.sort(key=lambda c: -c[0])
            for nov, child, audio, vec, tele, parents_ids in \
                    pending[:max(1, births // 2)]:
                admit(child, audio, vec, gen, parents_ids, nov)
                collisions_hist.append(tele["collisions"])
                agents_hist.append(tele["agents"])
                events_hist.append(tele["events"])
                admits += 1
            if gen % era_every == 0 and pending:
                nov, child, audio, vec, tele, parents_ids = pending[0]
                eras.append((gen, float(nov), child.archetype(), audio))

        # per-gen ledger (independent measurements only)
        kinds_now = len({mm["g"].archetype() for mm in population})
        structs = [structure_stats(mm["g"]) for mm in population]
        log.append({
            "gen": gen, "admits": admits, "kinds_now": kinds_now,
            "kinds_cum": len(kinds_seen),
            "cells_cum": len(cells_seen),
            "agents_mean": float(np.mean([s["agents"] for s in structs])),
            "coupled_mean": float(np.mean([s["coupled"] for s in structs])),
            "collisions_recent": float(np.mean(collisions_hist[-20:]))
            if collisions_hist else 0.0,
            "events_recent": float(np.mean(events_hist[-20:]))
            if events_hist else 0.0,
            "elapsed": round(time.perf_counter() - t0, 1),
        })
        if verbose and gen % 25 == 0:
            print(f"[eden] {arm} s{seed} g{gen}: +{admits} "
                  f"cells {len(cells_seen)} kinds {len(kinds_seen)} "
                  f"agents {log[-1]['agents_mean']:.1f} "
                  f"colls {log[-1]['collisions_recent']:.1f}", flush=True)

    feats = np.stack([mm["feat"] for mm in population])
    census = {}
    for mm in population:
        census[mm["g"].archetype()] = census.get(mm["g"].archetype(), 0) + 1
    ever = {e["arch"] for e in bank.metas}
    recalls = [mm["recall"] for mm in population if "recall" in mm]
    result = {
        "arm": arm, "seed": seed, "gens": gens, "log": log,
        "census_final": census, "kinds_final": len(census),
        "kinds_ever": len(ever), "kinds_ever_list": sorted(ever),
        "coverage": plain_coverage(feats),
        "cells_cum_final": len(cells_seen),
        "radius": plain_radius(feats, f_feat),
        "feats_final": feats.tolist(),
        "novelty_mean_final": None,
        "parent_recall_pct": float(100 * np.mean(recalls)) if recalls else None,
        "collisions_mean": float(np.mean(collisions_hist)) if collisions_hist else 0,
        "agents_mean_final": log[-1]["agents_mean"] if log else 0,
        "elapsed": round(time.perf_counter() - t0, 1),
    }
    # artifacts
    rdir = out / f"{arm}_s{seed}"
    rdir.mkdir(parents=True, exist_ok=True)
    from echo.probes import write_wav
    pieces = []
    for i, mm in enumerate(population):
        write_wav(rdir / f"spec_{i:02d}_{mm['g'].archetype()}.wav",
                  mm["audio"], SR)
        pieces.append(mm["audio"])
    if pieces:
        _write_mix(rdir / "mixtape.wav", pieces, SR)
    if eras:
        eras.sort(key=lambda e: e[0])
        _write_mix(rdir / "journey.wav", [e[3] for e in eras], SR)
        (rdir / "journey.json").write_text(json.dumps(
            [{"gen": g, "novelty": n, "archetype": a} for g, n, a, _ in eras],
            indent=1))
    bank.close()
    return result


def _write_mix(path, pieces, sr):
    from echo.probes import write_wav
    if not pieces:
        return
    nfade = int(0.5 * sr)
    total = sum(len(p) for p in pieces) - nfade * (len(pieces) - 1)
    mix = np.zeros((max(total, 1), 2), np.float32)
    pos = 0
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nfade) / nfade)
    for i, p in enumerate(pieces):
        p = p.copy().astype(np.float32)
        if i > 0:
            p[:nfade] *= ramp[:, None]
        if i < len(pieces) - 1:
            p[-nfade:] *= ramp[::-1][:, None]
        mix[pos: pos + len(p)] += p
        pos += len(p) - nfade
    peak = float(np.abs(mix).max())
    if peak > 1e-9:
        mix = mix / peak * 0.92
    write_wav(path, mix, sr)


def run_experiment(out="eden_out/run1", seeds=(0,), gens=200, pop=12,
                   births=8, arms=ARMS, verbose=True, workers=None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # render pool: episodes are independent and pure, so they farm out
    # across CPU cores (the M4's 10 cores are the right accelerator for
    # this small-kernel, python-loop-bound workload — not the GPU)
    pool = None
    if workers is None:
        workers = min(6, max(1, (os.cpu_count() or 2) - 2))
    if workers > 1:
        import multiprocessing as mp
        # fork, not spawn: workers are pure numpy (no Obj-C runtime to
        # poison), startup is free, and it works from any caller
        # (CLI, scripts, notebooks). Children inherit the imported
        # modules; _eval_job is pure + deterministic.
        ctx = mp.get_context("fork")
        pool = ctx.Pool(workers)
        if verbose:
            print(f"[eden] render pool: {workers} workers", flush=True)
    try:
        return _run_experiment(out, seeds, gens, pop, births, arms,
                               verbose, pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _run_experiment(out, seeds, gens, pop, births, arms, verbose, pool) -> dict:
    results = {}
    for arm in arms:
        runs = [run_arm(arm, s, gens, pop, births, out, verbose, pool=pool)
                for s in seeds]
        ok = [r for r in runs if r.get("log")]
        if not ok:
            results[arm] = {}
            continue
        L = ok[0]["log"]
        marks = [int(np.clip(round(q * (len(L) - 1)), 0, len(L) - 1))
                 for q in (0.25, 0.5, 0.75, 1.0)]
        growth = [float(np.mean([r["log"][m]["cells_cum"] for r in ok
                                 if len(r["log"]) > m])) for m in marks]
        agents_curve = [float(np.mean([r["log"][m]["agents_mean"] for r in ok
                                       if len(r["log"]) > m])) for m in marks]
        results[arm] = {
            "runs": [{k: v for k, v in r.items() if k != "feats_final"}
                     for r in runs],
            "feats_final": np.mean([r["feats_final"] for r in ok],
                                   axis=0).tolist(),
            "cells_cum_mean": float(np.mean([r["cells_cum_final"] for r in ok])),
            "kinds_ever_mean": float(np.mean([r["kinds_ever"] for r in ok])),
            "coverage_pairwise_mean": float(np.mean(
                [r["coverage"]["pairwise"] for r in ok])),
            "radius_mean": float(np.mean([r["radius"] for r in ok])),
            "parent_recall_mean": float(np.mean(
                [r["parent_recall_pct"] or 0 for r in ok])),
            "collisions_mean": float(np.mean([r["collisions_mean"] for r in ok])),
            "agents_mean_final": float(np.mean(
                [r["agents_mean_final"] for r in ok])),
            "growth": growth, "agents_curve": agents_curve,
        }
    summary = {"out": str(out), "gens": gens, "pop": pop, "births": births,
               "seeds": list(seeds), "arms": list(arms), "grid": GRID,
               "seconds": SECONDS, "results": results}
    (out / "experiment.json").write_text(json.dumps(summary, indent=1))
    (out / "report.md").write_text(report_run(summary))
    if verbose:
        print(f"[eden] report -> {out / 'report.md'}")
    return summary


def report_run(summary: dict) -> str:
    r = summary["results"]
    L = ["# eden — worlds bred by memory", "",
         f"gens {summary['gens']} | pop {summary['pop']} | "
         f"births/gen {summary['births']} | seeds {summary['seeds']} | "
         f"episodes {summary['seconds']}s | grid {summary['grid']}x{summary['grid']}",
         "both arms share the same two-parent crossover, mutation,",
         "viability and budget; only the memory differs.", "",
         "| arm | cells ever | kinds ever | agents (final) | collisions/ep | "
         "pairwise | parent-recall |",
         "|---|---|---|---|---|---|---|"]
    for arm in summary["arms"]:
        a = r.get(arm, {})
        if not a:
            continue
        L.append(f"| {arm} | {a['cells_cum_mean']:.1f} | "
                 f"{a['kinds_ever_mean']:.2f} | "
                 f"{a['agents_mean_final']:.2f} | "
                 f"{a['collisions_mean']:.1f} | "
                 f"{a['coverage_pairwise_mean']:.3f} | "
                 f"{a['parent_recall_mean']:.0f}% |")
    L += ["", "**territory growth** (cells at 25/50/75/100% of the run) and",
          "**complexification** (mean agent count — structure, measured):",
          "", "| arm | cells | | | | agents | | | |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in summary["arms"]:
        a = r.get(arm, {})
        if not a:
            continue
        L.append(f"| {arm} | " + " | ".join(f"{v:.0f}" for v in a["growth"])
                 + " | " + " | ".join(f"{v:.2f}" for v in a["agents_curve"])
                 + " |")
    feats = {arm: np.array(r[arm]["feats_final"]) for arm in summary["arms"]
             if r.get(arm, {}).get("feats_final")}
    if feats:
        L += ["", ascii_map(feats), ""]
    for arm in summary["arms"]:
        kinds = {}
        for run in r.get(arm, {}).get("runs", []):
            for k, v in run.get("census_final", {}).items():
                kinds[k] = kinds.get(k, 0) + v
        if kinds:
            L.append(f"**{arm}** final census: " + ", ".join(
                f"{k}×{v}" for k, v in
                sorted(kinds.items(), key=lambda kv: -kv[1])))
            L.append("")
    L.append(f"listen: `{summary['out']}/<arm>_s<seed>/journey.wav` (the "
             "run's exploration in generation order) and `mixtape.wav` "
             "(the final population). You are the judge.")
    L.append("")
    return "\n".join(L)