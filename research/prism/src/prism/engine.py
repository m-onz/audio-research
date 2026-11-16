"""engine: the breeding loop — three selection policies, one organism.

Everything is shared: the founder, the mutation kernel, the viability
gate (with the strictness dial), the render budget, the survivor rule,
the seeds. The ONLY thing that differs between arms is the memory
policy:

    novelty   parents = top quartile by ROTATING-LENS novelty
              (echo's blocks as multiple lenses — timbre, time, pitch,
              all — the pugh_ppsn16 multi-BC move); children must also
              clear a floor at half the population's current median
              novelty (maze's calibration: a moving target, never
              impossible)
    random    uniform parents, every viable child admitted — drift
    greedy    parent = argmax tonality — the objective trap, kept in
              the study because the literature says it converges to
              the first pure tone and the field should see it happen

Measured WITHOUT echo (measure.py): cumulative discovery on a 24×24
grid of plain features, coverage/radius, and the complexity census
read straight off genomes (nodes, connections, and the OUTPUT
ALPHABET — brant_gecco19's metrics). One number measured WITH echo,
against true lineage: parent-kNN recall.

The strictness dial (soros): scales the minimal criterion bands.
0.1 = narrow (stagnant), 1.0 = wide (junk floods), 0.5 calibrated —
the maze/Chromaria convention.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from echo import LAYOUT, EchoBank, encode

from .genome import random_genome
from .measure import (atlas_png, complexity, plain_features, plain_coverage,
                      plain_radius)
from .render import MODES, SR, render, write_wav

__all__ = ["run_experiment", "report_run", "run_arm", "ARMS", "LENSES",
           "viability", "tonality", "lens_rank", "floor_ok"]

ARMS = ("novelty", "random", "greedy")
LENSES = ("all", "timbre", "time", "pitch")
GRID = 24                    # the independent discovery grid
EVAL_LEN = 4.0               # seconds per child render (echo budget)
MUTS_PER_BIRTH = 2           # leaps, not steps (the maze README's advice)

# ------------------------------------------------------------------ lenses

_BLOCK_BOUNDS: dict[str, tuple[int, int, float]] = {}
_pos = 0
for _name, _dim, _w in LAYOUT:
    _BLOCK_BOUNDS[_name] = (_pos, _pos + _dim, _w)
    _pos += _dim

_LENS_BLOCKS = {
    "all": ["mfcc_mean", "mfcc_std", "scal_mean", "silhouette",
            "dynamics", "chroma", "spatial"],
    "timbre": ["mfcc_mean", "mfcc_std", "silhouette"],
    "time": ["scal_mean", "dynamics"],
    "pitch": ["chroma", "spatial"],
}


def _lens_index(blocks: list[str]):
    idx, w = [], []
    for name in blocks:
        a, b, bw = _BLOCK_BOUNDS[name]
        idx.append(np.arange(a, b))
        w.append(np.full(b - a, bw))
    return np.concatenate(idx), np.sqrt(np.concatenate(w).astype(np.float64))


LENS_INDEX = {lens: _lens_index(blocks)
              for lens, blocks in _LENS_BLOCKS.items()}


def lens_rank(pop_vecs, pop_ids, arch_vecs, arch_ids, lens: str,
              k: int = 15) -> np.ndarray:
    """Novelty of each member under one lens, against the archive,
    self excluded — z-standardized on the archive itself. One matmul;
    the archive is the world, the lens decides what 'like' means."""
    idx, w = LENS_INDEX[lens]
    A = np.atleast_2d(arch_vecs)[:, idx].astype(np.float64)
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Aw = (A - mu) / sd * w
    V = np.atleast_2d(pop_vecs)[:, idx].astype(np.float64)
    V = (V - mu) / sd * w
    d2 = ((V ** 2).sum(1)[:, None] + (Aw ** 2).sum(1)[None, :]
          - 2.0 * (V @ Aw.T))
    d2 = np.maximum(d2, 0.0)
    ids = np.asarray(arch_ids)
    for r, mid in enumerate(np.atleast_1d(pop_ids)):
        d2[r, ids == mid] = np.inf
    d2 = np.sort(d2, axis=1)                    # infs sink to the end
    finite = np.isfinite(d2).sum(axis=1)
    kk = np.minimum(k, np.maximum(finite, 1))  # per-row neighbor budget
    means = np.array([np.sqrt(d2[r, :kk[r]]).mean() for r in range(len(kk))])
    return np.where(np.isfinite(means) & (finite > 0), means, 0.0)


# ------------------------------------------------------------------ gates


def viability(audio, strictness: float = 0.5) -> tuple[bool, dict]:
    """The minimal criterion, identical for every arm: finite, audible,
    alive. The bands widen with strictness — no quality anywhere."""
    x = np.asarray(audio, np.float64)
    if not np.isfinite(x).all():
        return False, {"rms": 0.0, "block_ratio": 0.0}
    rms = float(np.sqrt((x ** 2).mean()))
    blocks = np.array_split(x, 4, axis=0)
    rms_b = [float(np.sqrt((b ** 2).mean())) for b in blocks]
    ratio = max(rms_b) / (min(rms_b) + 1e-9)
    floor = 0.02 * (0.5 / strictness)
    static = 1.0 + 0.05 * (0.5 / strictness)
    ok = (floor <= rms <= 0.6) and (ratio >= static)
    return ok, {"rms": rms, "block_ratio": ratio}


def tonality(audio, sr: int = SR) -> float:
    """Spectral crest — the objective the greedy arm will trap itself
    in. Pure tone → high; wash/click weather → low."""
    x = np.asarray(audio, np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    S = S[f > 30.0]
    if S.sum() <= 1e-12:
        return 0.0
    return float(S.max() / S.sum())


def floor_ok(nov: float, pop_novs) -> bool:
    """The relative admission floor: newer than half the population's
    current median novelty — a moving target that never freezes."""
    if not pop_novs:
        return True
    return bool(nov >= 0.5 * float(np.median(pop_novs)))


def _cell_key(f: np.ndarray):
    return (int(np.clip(f[1] * GRID, 0, GRID - 1)),
            int(np.clip(f[2] * GRID, 0, GRID - 1)))


def _crossfade(audios, sr: int = SR, fade: float = 0.4) -> np.ndarray:
    if not audios:
        return np.zeros((0, 2), np.float32)
    nf = int(fade * sr)
    total = sum(len(a) for a in audios) - nf * (len(audios) - 1)
    mix = np.zeros((max(total, 1), 2), np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
    pos = 0
    for i, a in enumerate(audios):
        p = np.asarray(a, np.float32).copy()
        if i > 0:
            p[:nf] *= ramp[:, None]
        if i < len(audios) - 1:
            p[-nf:] *= ramp[::-1][:, None]
        mix[pos:pos + len(p)] += p
        pos += len(p) - nf
    peak = float(np.abs(mix).max())
    if peak > 1e-9:
        mix = mix / peak * 0.92
    return mix


# ------------------------------------------------------------------ one arm


def run_arm(arm: str, seed: int, gens: int, pop: int, births: int,
            out: Path, organism: str = "spectra",
            strictness: float = 0.5, eval_len: float = EVAL_LEN,
            verbose: bool = False) -> dict:
    assert arm in ARMS and organism in MODES
    rng = np.random.default_rng(seed)          # children stream
    sel = np.random.default_rng(seed + 555)     # policy stream
    bank = EchoBank(":memory:")
    ledger: list[dict] = []
    parent_recall: list[bool] = []
    t0 = time.time()

    # ---- founder: identical across arms for a given seed
    founder = None
    for _ in range(200):
        cand = random_genome(rng, n_hidden=int(rng.integers(1, 4)))
        audio = render(cand, eval_len, mode=organism, seed=0)
        ok, _ = viability(audio, strictness)
        if ok:
            founder = (cand, audio)
            break
    if founder is None:
        bank.close()
        raise SystemExit(f"[prism] no viable founder in 200 tries "
                         f"(arm {arm}, seed {seed}, strictness "
                         f"{strictness}) — loosen the dial")
    f_g, f_audio = founder
    f_vec = encode(f_audio, SR)
    f_feat = plain_features(f_audio)
    f_cx = complexity(f_g)
    f_id = bank.add(f_vec, {"gen": 0, "arm": arm}, role="pop", gen=0)
    population = [{"g": f_g, "vec": f_vec, "audio": f_audio, "feat": f_feat,
                  "gen": 0, "depth": 0, "ops": ["founder"], "id": f_id,
                  "tonality": tonality(f_audio), "nov": None}]
    arch_vecs = [f_vec]
    arch_ids = [f_id]
    cells_seen = {_cell_key(f_feat)}
    eras: list[dict] = []
    era_every = max(1, gens // 20)

    def _pop_novs() -> list[float]:
        k = min(15, len(bank))
        return [bank.novelty(m["vec"], k=k) for m in population]

    def _pick_parent(birth_i: int):
        if arm == "greedy":
            return max(population, key=lambda m: m["tonality"]), None
        if arm == "random":
            return population[int(sel.integers(len(population)))], None
        lens = LENSES[birth_i % len(LENSES)]        # the rotating lens
        scores = lens_rank([m["vec"] for m in population],
                           [m["id"] for m in population],
                           arch_vecs, arch_ids, lens)
        order = np.argsort(scores)[::-1]
        q = max(1, len(order) // 4)
        top = [population[int(i)] for i in order[:q]]
        return top[int(sel.integers(len(top)))], lens

    log = []
    for gen in range(1, gens + 1):
        pop_novs = _pop_novs()
        admits, rej_mc, rej_nov = 0, 0, 0
        best_nov, best_member = -1.0, None
        for birth in range(births):
            parent, lens = _pick_parent(birth)
            child = parent["g"].copy()
            ops = [child.mutate(rng) for _ in range(MUTS_PER_BIRTH)]
            cx = complexity(child)
            audio = render(child, eval_len, mode=organism, seed=0)
            ok, vm = viability(audio, strictness)
            ton = tonality(audio)
            entry = {"gen": gen, "arm": arm, "lens": lens, "ops": ops,
                     "parent_gen": parent["gen"], "depth": parent["depth"] + 1,
                     "viable": bool(ok), "admitted": False, "nov": None,
                     "rms": round(vm["rms"], 5),
                     "block_ratio": round(vm["block_ratio"], 4),
                     "tonality": round(ton, 5),
                     "nodes": cx["nodes"], "conns": cx["conns"],
                     "outputs": cx["outputs"]}
            if not ok:
                rej_mc += 1
                ledger.append(entry)
                continue
            vec = encode(audio, SR)
            nov = bank.novelty(vec, k=min(15, len(bank)))
            entry["nov"] = round(float(nov), 4)
            if arm == "novelty" and not floor_ok(nov, pop_novs):
                rej_nov += 1
                ledger.append(entry)
                continue
            near = bank.knn(vec, k=min(5, len(bank)))
            parent_recall.append(any(n["id"] == parent["id"] for n in near))
            feat = plain_features(audio)
            cid = bank.add(vec, {"gen": gen, "ops": ops}, role="pop", gen=gen)
            member = {"g": child, "vec": vec, "audio": audio, "feat": feat,
                      "gen": gen, "depth": parent["depth"] + 1, "ops": ops,
                      "id": cid, "tonality": ton, "nov": float(nov)}
            population.append(member)
            if len(population) > pop:               # identical survivor rule
                population.pop(int(sel.integers(len(population))))
            arch_vecs.append(vec)
            arch_ids.append(cid)
            cells_seen.add(_cell_key(feat))
            entry["admitted"] = True
            ledger.append(entry)
            admits += 1
            if nov > best_nov:
                best_nov, best_member = float(nov), member
        if gen % era_every == 0 and best_member is not None:
            eras.append(best_member)                 # the era's freshest voice

        cx_pop = [complexity(m["g"]) for m in population]
        log.append({
            "gen": gen, "admits": admits, "rej_mc": rej_mc,
            "rej_nov": rej_nov, "cells_cum": len(cells_seen),
            "pop": len(population),
            "mean_nodes": float(np.mean([c["nodes"] for c in cx_pop])),
            "mean_outputs": float(np.mean([c["outputs"] for c in cx_pop])),
            "max_outputs": int(max(c["outputs"] for c in cx_pop)),
            "mean_conns": float(np.mean([c["conns"] for c in cx_pop])),
            "max_depth": int(max(m["depth"] for m in population)),
            "mean_tonality": float(np.mean([m["tonality"] for m in population])),
            "mean_pop_nov": float(np.mean(_pop_novs())) if population else None,
        })
        if verbose and gen % 20 == 0:
            print(f"[prism] {arm} s{seed} g{gen}: +{admits} "
                  f"cells {log[-1]['cells_cum']} "
                  f"nodes {log[-1]['mean_nodes']:.1f} "
                  f"outs {log[-1]['mean_outputs']:.1f}", flush=True)

    # ---- final census (independent) + artifacts
    feats = np.stack([m["feat"] for m in population])
    rdir = out / f"{arm}_s{seed}"
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "ledger.jsonl", "w") as fh:
        for e in ledger:
            fh.write(json.dumps(e) + "\n")
    for i, m in enumerate(population):              # specimens + biographies
        write_wav(rdir / f"spec_{i:02d}.wav", m["audio"], SR)
        nov_str = f"{m['nov']:.2f}" if m["nov"] is not None else "genesis"
        (rdir / f"spec_{i:02d}.json").write_text(json.dumps({
            "genome": m["g"].to_dict(), "mode": organism,
            "born": m["gen"], "depth": m["depth"], "ops": m["ops"],
            "nov_at_birth": m["nov"],
            "biography": (f"born gen {m['gen']} at depth {m['depth']} "
                          f"by {'+'.join(m['ops'])}; admitted at "
                          f"novelty {nov_str}")}, indent=1))
    write_wav(rdir / "mixtape.wav",
              _crossfade([m["audio"] for m in population]), SR)
    if eras:                                        # the journey
        for m in eras:
            (rdir / f"journey_gen{m['gen']:04d}.json").write_text(
                json.dumps({"genome": m["g"].to_dict(), "mode": organism,
                            "born": m["gen"], "nov": m["nov"]}, indent=1))
        write_wav(rdir / "journey.wav",
                  _crossfade([m["audio"] for m in eras]), SR)
    atlas_png(rdir / "atlas.png", [m["audio"] for m in population])
    result = {
        "arm": arm, "seed": seed, "gens": gens, "organism": organism,
        "strictness": strictness, "log": log,
        "founder_complexity": f_cx,
        "cells_cum_final": len(cells_seen),
        "cells_curve": [l["cells_cum"] for l in log],
        "coverage": plain_coverage(feats),
        "radius": plain_radius(feats, f_feat),
        "complexity_final": {
            "mean_nodes": log[-1]["mean_nodes"] if log else f_cx["nodes"],
            "mean_outputs": log[-1]["mean_outputs"] if log else f_cx["outputs"],
            "max_outputs": log[-1]["max_outputs"] if log else f_cx["outputs"],
            "max_depth": log[-1]["max_depth"] if log else 0,
        },
        "mean_tonality_final": log[-1]["mean_tonality"] if log else None,
        "admits": sum(l["admits"] for l in log),
        "births": gens * births,
        "parent_recall_pct": float(100 * np.mean(parent_recall))
        if parent_recall else None,
        "elapsed": round(time.time() - t0, 1),
    }
    (rdir / "run.json").write_text(json.dumps(result, indent=1))
    bank.close()
    return result


# ------------------------------------------------------------------ the run


def run_experiment(out: str | Path, seeds=(0,), gens: int = 240,
                   pop: int = 24, births: int = 12,
                   arms: tuple = ARMS, organism: str = "spectra",
                   strictness: float = 0.5, eval_len: float = EVAL_LEN,
                   verbose: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"config": {"gens": gens, "pop": pop, "births": births,
                          "arms": list(arms), "organism": organism,
                          "strictness": strictness, "eval_len": eval_len,
                          "grid": GRID, "leaps": MUTS_PER_BIRTH},
               "runs": []}
    for seed in seeds:
        for arm in arms:
            r = run_arm(arm, seed, gens, pop, births, out, organism,
                        strictness, eval_len, verbose)
            summary["runs"].append(r)
    (out / "experiment.json").write_text(json.dumps(summary, indent=1))
    print(report_run(summary))
    return summary


def report_run(summary_or_path) -> str:
    if not isinstance(summary_or_path, dict):
        summary = json.loads((Path(summary_or_path) / "experiment.json")
                             .read_text())
    else:
        summary = summary_or_path
    lines = ["# prism run", ""]
    cfg = summary["config"]
    lines.append(f"organism **{cfg['organism']}** · {cfg['gens']} gens · "
                 f"pop {cfg['pop']} × {cfg['births']} births/gen · "
                 f"strictness {cfg['strictness']} · "
                 f"{cfg['leaps']} mutations/leap · grid {cfg['grid']}×"
                 f"{cfg['grid']}")
    lines += ["", "| arm | seed | cells ever | pairwise | radius | "
              "parent-recall | yield | nodes | outputs | max-outs | depth | "
              "elapsed |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["runs"]:
        y = r["admits"] / max(1, r["births"])
        cf = r["complexity_final"]
        lines.append(
            f"| {r['arm']} | {r['seed']} | {r['cells_cum_final']} "
            f"| {r['coverage']['pairwise']:.3f} | {r['radius']:.2f} "
            f"| {r['parent_recall_pct']:.0f}% "
            f"| {y:.3f} | {cf['mean_nodes']:.1f} | {cf['mean_outputs']:.2f} "
            f"| {cf['max_outputs']} | {cf['max_depth']} "
            f"| {r['elapsed']:.0f}s |")
    lines += ["", "growth (cumulative cells, mean over seeds):", ""]
    for arm in dict.fromkeys(r["arm"] for r in summary["runs"]):
        runs = [r for r in summary["runs"] if r["arm"] == arm]
        L = min(len(r["cells_curve"]) for r in runs)
        curve = np.mean([r["cells_curve"][:L] for r in runs], axis=0)
        stops = [max(0, min(int(np.ceil(L * q)) - 1, L - 1))
                 for q in (0.25, 0.5, 0.75, 0.9999)]
        bar = " ".join(f"{curve[s]:5.1f}" for s in stops)
        lines.append(f"    {arm:8s} 25%/50%/75%/100%:  {bar}")
    fc = next(iter(summary["runs"]))["founder_complexity"]
    lines += ["", f"founder: {fc['nodes']} nodes · {fc['conns']} conns · "
              f"{fc['outputs']} outputs — complexity can only rise if the "
              "alphabet does (brant_gecco19); watch the outputs column.",
              ""]
    return "\n".join(lines)