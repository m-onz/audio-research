"""bench: measuring whether the memory organ actually hears.

Four questions, four experiments, one report:

    1. PAIR DISCRIMINATION — near-duplicates (gain, mp3 rounds, seed
       jitter...) must land close, distinct pairs far. AUC, d', EER —
       echo vs echo-without-chunks vs MFCC-only vs womb's 64-band
       signature vs tomb's 16 scalars vs tomb-style 4-D ear projections.

    2. MUTATION MONOTONICITY — the evolution claim: distance from a
       parent must grow with mutation magnitude (Spearman rho), and a
       different law must be farther than any parameter drift.

    3. CORPUS LOCALITY — on the real corpus: does the nearest neighbor
       of a chunk come from the same take? (the manifest's start_s is
       ground truth we can listen to.)

    4. A/B SELECTION — three breeding arms, identical judge and seeds,
       only the memory differs: drift-floor (MCC alone), cell-parents
       (womb's current pick), novelty-gate (the echo archive). Coverage
       measured in womb's own grid cells — does the archive push the
       run into rooms the others never enter?

Everything lands in echo_out/: bench_report.md + results.json, so any
encoder tweak can be scored against the same yardstick.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from .bank import EchoBank, EchoMetric
from .encode import DIM, LAYOUT, LAYOUT_FLAT, encode, layout_for, N_CHUNK
from .probes import PERTURB_NAMES, contrasts, decode, perturbations

__all__ = ["full_bench", "ab_experiment", "pair_suite", "locality",
           "mutation_monotonicity", "speed_test", "roc_auc"]

SR = 44100


# ------------------------------------------------------------------ tiny stats

def roc_auc(pos, neg) -> float:
    """P(neg_score > pos_score) + 0.5 ties — higher = separates better."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = neg[None, :] - pos[:, None]                  # (npos, nneg)
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def d_prime(pos, neg) -> float:
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    pooled = np.sqrt((pos.var() + neg.var()) / 2.0) + 1e-12
    return float((neg.mean() - pos.mean()) / pooled)


def eer(pos, neg) -> float:
    """Equal error rate: where 'different' threshold misjudges both
    classes at the same rate (lower is better)."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    ths = np.unique(np.concatenate([pos, neg]))
    best = 1.0
    for t in ths:
        far = np.mean(neg <= t)                          # different read as same
        frr = np.mean(pos > t)                           # same read as different
        best = min(best, max(far, frr))
    return float(best)


def spearman(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


# ------------------------------------------------------------------ corpus

def load_corpus(corpus_dir, limit: int | None = None):
    """Decode + encode every corpus chunk once. Returns rows with
    provenance (start_s from the manifest) and 556-D vectors."""
    d = Path(corpus_dir)
    manifest = json.loads((d / "manifest.json").read_text()) \
        if (d / "manifest.json").exists() else []
    rows = []
    files = sorted(d.glob("*.wav"))
    if limit:
        files = files[:limit]
    t0 = time.time()
    for p in files:
        meta = next((m for m in manifest if m["file"] == p.name), {})
        audio = decode(p, SR, stereo=True)
        rows.append({"file": str(p), "name": p.name,
                     "start_s": float(meta.get("start_s", 0.0)),
                     "dur_s": float(meta.get("dur_s", len(audio) / SR)),
                     "vec": encode(audio, SR)})
    return rows, time.time() - t0


# ------------------------------------------------------------------ encoders under test

def build_encoders(rows, extras: bool | None = None):
    """The competitors, each as {name: (vec_fn, dist_fn)}.

    echo       — full chunked vector, corpus-standardized, block weights
    echo_flat  — same, but N_CHUNK=1: the time axis REMOVED (ablation)
    echo_mfcc  — only the mfcc_mean block (ablation: MFCCs alone)
    echo_plus  — core + librosa tier (chroma_cqt, contrast, hpss,
                 rhythm) — analysis-grade, ~10x cost
    womb_sig   — the current drift-floor metric (64-band silhouette)
    tomb16     — tomb's whole-clip scalar features
    tomb_ear   — tomb-style 4-D random projection (what its ears hear)
    """
    X = np.stack([r["vec"] for r in rows])
    enc = {}

    metric = EchoMetric(DIM).fit(X)
    enc["echo"] = (lambda a, sr=SR: encode(a, sr), metric.dist)

    flat = EchoMetric(layout=LAYOUT_FLAT)
    flat.fit(np.stack(_flat_rows(X)))
    enc["echo_flat"] = (lambda a, sr=SR: encode(a, sr, n_chunks=1), flat.dist)

    mfcc_only = EchoMetric(DIM, weights={n: (0.0 if n != "mfcc_mean" else w)
                                         for n, _, w in LAYOUT}).fit(X)
    enc["echo_mfcc"] = (lambda a, sr=SR: encode(a, sr), mfcc_only.dist)

    try:
        from womb.breed import signature, distance
        enc["womb_sig"] = (lambda a, sr=SR: signature(a), distance)
    except Exception:
        pass
    try:
        from tomb.ears import features_vector

        def _euclid(a, b):
            return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        enc["tomb16"] = (lambda a, sr=SR: features_vector(
            a.mean(axis=1) if getattr(a, "ndim", 1) == 2 else a), _euclid)

        rng = np.random.default_rng(3)
        proj = rng.normal(0, 1, (4, 16))
        proj /= np.linalg.norm(proj, axis=1, keepdims=True)

        def _ear_vec(a, sr=SR):
            f = features_vector(a.mean(axis=1) if getattr(a, "ndim", 1) == 2 else a)
            return proj @ f
        enc["tomb_ear"] = (_ear_vec, _euclid)
    except Exception:
        pass

    # --- the librosa tier (only if installed / requested)
    if extras is None:
        try:
            import librosa  # noqa: F401
            extras = True
        except Exception:
            extras = False
    if extras:
        try:
            from .extras import LAYOUT_PLUS, encode_plus
            Xp = np.stack([encode_plus(decode(r["file"], SR, stereo=True), SR)
                           for r in rows])
            metric_p = EchoMetric(layout=LAYOUT_PLUS).fit(Xp)
            enc["echo_plus"] = (lambda a, sr=SR: encode_plus(a, sr), metric_p.dist)
        except Exception as e:                          # noqa: BLE001
            print(f"[echo] echo_plus unavailable: {e}")
    return enc


def _flat_rows(X):
    """Reshape a corpus of chunked vectors into flat (n_chunks=1) ones —
    collapse each chunked block into its global mean. These ARE the flat
    vectors (they are not audio; never re-encode them)."""
    rows = []
    for v in X:
        blocks, pos = {}, 0
        for name, dim, _ in LAYOUT:
            blocks[name] = v[pos: pos + dim]
            pos += dim
        m = blocks["mfcc_mean"].reshape(N_CHUNK, -1).mean(axis=0)
        s = blocks["mfcc_std"].reshape(N_CHUNK, -1).mean(axis=0)
        c = blocks["scal_mean"].reshape(N_CHUNK, -1).mean(axis=0)
        rows.append(np.concatenate([m, s, c, blocks["silhouette"],
                                    blocks["dynamics"], blocks["chroma"],
                                    blocks["spatial"]]))
    return rows


# ------------------------------------------------------------------ pair suite

def pair_suite(corpus_rows, rng, n_perturb: int = 50, n_distant: int = 60,
               seconds: float = 3.0, verbose=False):
    """Known-answer pairs from real material + champions + archetypes."""
    pos, neg = [], []                                   # (a, b, tag)

    # --- near duplicates: corpus chunks + every perturbation flavor
    idx = rng.choice(len(corpus_rows), size=min(n_perturb, len(corpus_rows)),
                     replace=False)
    for i in idx:
        p = Path(corpus_rows[int(i)]["file"])
        audio = decode(p, SR, stereo=True)
        perturbs = perturbations(audio, SR, rng,
                                 names=[PERTURB_NAMES[int(rng.integers(len(PERTURB_NAMES)))]])
        for name, pb in perturbs.items():
            pos.append((audio, pb, f"corpus:{name}"))

    # --- champions: mutation children = same species, drifted (near).
    #    The champions are seed-invariant (bit-identical renders), so
    #    instead of seed jitter we mutate the process with womb's own
    #    operator and keep children that the current drift floor itself
    #    would call 'close to parent' — the exact pairs a novelty metric
    #    must NOT mistake for new discoveries.
    champs = {}
    try:
        from womb.breed import (distance, drift_eps, mutate_process,
                                render_pair, signature)
        from womb.cryptid import build_champion, render
        for name in ("chimera", "clickstorm", "jerk_voice"):
            g = build_champion(name)
            base = render(g, length=seconds)["audio"]
            champs[name] = (g, base)
            proc = {"nodes": g["nodes"], "edges": g.get("edges", []),
                    "sim_rate": g.get("sim_rate", 11025),
                    "seed": int(g.get("seed", 0))}
            rng_m = np.random.default_rng(abs(hash(name)) % (1 << 30))
            eps = drift_eps(0.5)
            kept = 0
            for _ in range(30):
                if kept >= 3:
                    break
                child, _m = mutate_process(rng_m, proc, rate=0.12)
                try:
                    audio = render_pair(child, g.get("ears", []), seconds)
                except Exception:
                    continue
                d = distance(signature(audio), signature(base))
                if eps < d <= 3.5 * eps:
                    pos.append((base, audio, f"champion_child:{name}"))
                    kept += 1
    except Exception as e:                              # noqa: BLE001
        if verbose:
            print(f"[echo] champions unavailable: {e}")

    # --- distant corpus pairs (different takes / regions)
    by_stem = {}
    for r in corpus_rows:
        by_stem.setdefault(r["file"].rsplit("_", 1)[0], []).append(r)
    far_pairs = 0
    guard = 0
    while far_pairs < n_distant and guard < n_distant * 30:
        guard += 1
        s = list(by_stem)[int(rng.integers(len(by_stem)))]
        rows = by_stem[s]
        a, b = rows[int(rng.integers(len(rows)))], rows[int(rng.integers(len(rows)))]
        if abs(a["start_s"] - b["start_s"]) < 120.0 and len(rows) > 4:
            continue                                    # likely same take
        neg.append((decode(Path(a["file"]), SR, stereo=True),
                    decode(Path(b["file"]), SR, stereo=True),
                    "corpus_distant"))
        far_pairs += 1

    # --- archetypes: pairwise distinct by construction
    syn = contrasts(SR, seconds=seconds)
    keys = list(syn)
    for _ in range(12):
        i, j = rng.choice(len(keys), 2, replace=False)
        neg.append((syn[keys[i]], syn[keys[j]], "contrast"))

    # --- champions: cross-law (far) + vs corpus (far)
    if champs:
        names = list(champs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                neg.append((champs[names[i]][1], champs[names[j]][1],
                            "champion_cross"))
        for name, (_, base) in champs.items():
            for _ in range(4):
                r = corpus_rows[int(rng.integers(len(corpus_rows)))]
                neg.append((base, decode(Path(r["file"]), SR, stereo=True),
                            "champion_vs_corpus"))
    if verbose:
        print(f"[echo] pairs: {len(pos)} near / {len(neg)} far")
    return pos, neg, champs


def discrimination(encoders, pos, neg, verbose=False):
    """AUC / d' / EER per encoder, + per-perturbation detail for echo."""
    out = {}
    for name, (vec_fn, dist_fn) in encoders.items():
        dpos = []
        for a, b, tag in pos:
            try:
                dpos.append((dist_fn(vec_fn(a), vec_fn(b)), tag))
            except Exception:
                continue
        dneg = []
        for a, b, tag in neg:
            try:
                dneg.append(dist_fn(vec_fn(a), vec_fn(b)))
            except Exception:
                continue
        ps = [d for d, _ in dpos]
        out[name] = {"auc": roc_auc(ps, dneg), "d_prime": d_prime(ps, dneg),
                     "eer": eer(ps, dneg), "n_pos": len(ps), "n_neg": len(dneg),
                     "pos_median": float(np.median(ps)) if ps else None,
                     "neg_median": float(np.median(dneg)) if dneg else None}
        if verbose:
            print(f"[echo] {name:10s} AUC={out[name]['auc']:.3f} "
                  f"d'={out[name]['d_prime']:.2f} EER={out[name]['eer']:.3f}")
    # per-perturbation view through the echo lens
    if "echo" in encoders:
        vec_fn, dist_fn = encoders["echo"]
        detail = {}
        for a, b, tag in pos:
            fam = tag.split(":")[-1]
            try:
                d = dist_fn(vec_fn(a), vec_fn(b))
            except Exception:
                continue
            detail.setdefault(fam, []).append(d)
        out["_perturbation_detail"] = {
            k: {"median": float(np.median(v)), "n": len(v)}
            for k, v in sorted(detail.items())}
    return out


# ------------------------------------------------------------------ mutation monotonicity

def mutation_monotonicity(metric, seconds: float = 4.0, verbose=False):
    """The evolution claim: bigger mutations = farther children, and a
    different law is farther than any parameter drift."""
    try:
        from womb.cryptid import build_champion, render
    except Exception as e:                               # noqa: BLE001
        if verbose:
            print(f"[echo] monotonicity skipped (no womb): {e}")
        return None
    g = copy.deepcopy(build_champion("jerk_voice"))
    base = render(g, length=seconds)["audio"]
    factors = [1.01, 1.03, 1.08, 1.2, 1.5, 2.5]
    dists, vecs = [], []
    for f in factors:
        g2 = copy.deepcopy(g)
        g2["nodes"][0]["params"]["speed"] = float(g["nodes"][0]["params"]["speed"]) * f
        audio = render(g2, length=seconds)["audio"]
        dists.append(metric.dist(encode(base), encode(audio)))
    cross = {}
    for name in ("chimera", "clickstorm"):
        other = render(build_champion(name), length=seconds)["audio"]
        cross[name] = metric.dist(encode(base), encode(other))
    rho = spearman(np.log(factors), dists)
    if verbose:
        print(f"[echo] monotonicity rho={rho:.2f} max_mut={max(dists):.3f} "
              f"cross={ {k: round(v, 2) for k, v in cross.items()} }")
    return {"factors": factors, "dists": dists, "spearman": rho,
            "cross_law": cross,
            "cross_exceeds_max_mut": bool(min(cross.values()) > max(dists))}


# ------------------------------------------------------------------ corpus locality

def locality(corpus_rows, verbose=False):
    """On real material: is the nearest neighbor of a chunk its neighbor
    in the source recording? (start_s provenance = listenable truth.)"""
    rows = sorted(corpus_rows, key=lambda r: (r["file"].rsplit("_", 1)[0],
                                              r["start_s"]))
    X = np.stack([r["vec"] for r in rows])
    ranks, nn_d, true_d = [], [], []
    for i in range(len(rows)):
        r = rows[i]
        stem = r["file"].rsplit("_", 1)[0]
        # truth: the temporally adjacent chunk of the same take
        truth = None
        for j in (i - 1, i + 1):
            if 0 <= j < len(rows):
                o = rows[j]
                if o["file"].rsplit("_", 1)[0] != stem:
                    continue
                gap = abs(o["start_s"] - r["start_s"])
                if gap < 60.0 and (truth is None or gap < truth[1]):
                    truth = (j, gap)
        if truth is None:
            continue
        d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        order = np.argsort(d)
        rank = int(np.where(order == truth[0])[0][0]) + 1
        ranks.append(rank)
        nn_d.append(float(d[order[0]]))
        true_d.append(float(d[truth[0]]))
    ranks = np.array(ranks)
    # separation: nearest-neighbor distance vs random-pair distance
    rng = np.random.default_rng(11)
    rp = []
    for _ in range(1000):
        i, j = rng.integers(len(X), size=2)
        if i != j:
            rp.append(float(np.linalg.norm(X[i] - X[j])))
    res = {"n_queries": int(len(ranks)),
           "median_rank": float(np.median(ranks)),
           "recall_at_1": float(np.mean(ranks == 1)),
           "recall_at_5": float(np.mean(ranks <= 5)),
           "recall_at_10": float(np.mean(ranks <= 10)),
           "mean_true_neighbor_dist": float(np.mean(true_d)) if true_d else None,
           "separation": float(np.mean(nn_d) / (np.mean(rp) + 1e-12))}
    if verbose:
        print(f"[echo] locality: medrank={res['median_rank']:.0f} "
              f"R@5={res['recall_at_5']:.2f} sep={res['separation']:.3f}")
    return res


# ------------------------------------------------------------------ A/B selection experiment

def _run_arm(arm: str, seed: int, gens: int, pop: int, births: int,
             pairs: int, seconds: float, strictness: float, verbose=False):
    """One breeding run, one memory policy. Everything else identical:
    same laws, same judge, same ear pool, same attempt counts."""
    from womb.breed import (GRIDS, _cells_of, descriptors, drift_eps,
                            distance, mutate_process, random_ear,
                            random_process, render_pair, signature)
    from womb.cryptid import viability

    mut = np.random.default_rng(seed)                   # children (identical stream)
    sel = np.random.default_rng(seed + 555)              # parents + ears (policy stream)
    eps = drift_eps(strictness)

    def judge(audio):
        return viability(audio, SR, strictness=strictness)

    ear_pool = [random_ear(mut) for _ in range(6)]
    bank = EchoBank(":memory:")
    archive = []                                        # every admission ever
    cells_now = {g: {} for g in GRIDS}                   # latest per cell (womb's rule)
    cells_seen = set()                                   # cumulative coverage
    nov_hist = []
    log = []

    def admit(proc, ear, audio, gen, parent_entry):
        desc = descriptors(audio)
        entry = {"proc": proc, "ear": ear, "desc": desc,
                 "sig": signature(audio).tolist(), "vec": encode(audio),
                 "cells": _cells_of(desc), "gen": gen}
        entry["nov"] = bank.novelty(entry["vec"], k=15)
        archive.append(entry)
        bank.add(entry["vec"], {"gen": gen, "arm": arm}, role="archive")
        for g, cell in entry["cells"].items():
            cells_now[g][f"{cell[0]},{cell[1]}"] = entry
            cells_seen.add((g, cell[0], cell[1]))
        return entry

    # -- founders (same for every arm: the rng streams are shared here)
    tries = 0
    while len(archive) < pop and tries < pop * 6:
        tries += 1
        proc = random_process(mut)
        for ear in ear_pool[:2]:
            try:
                audio = render_pair(proc, ear, seconds)
            except Exception:
                continue
            if judge(audio)["viable"]:
                admit(proc, ear, audio, 0, None)
                break
    if not archive:
        return {"admissions": 0}
    bank.fit(np.stack([e["vec"] for e in archive]), save=False)

    live = list(archive)                                # sliding population

    def pick_parent():
        if arm == "cell" and sel.random() < 0.65 and any(cells_now.values()):
            g = [k for k in cells_now if cells_now[k]]
            gname = g[int(sel.integers(len(g)))]
            cm = cells_now[gname]
            return cm[list(cm)[int(sel.integers(len(cm)))]]
        return live[int(sel.integers(len(live)))]

    # -- generations
    for gen in range(1, gens + 1):
        admits, nov_admits, rule_rejects = 0, 0, 0
        for _ in range(births):
            parent = pick_parent()
            child, m = mutate_process(mut, parent["proc"], rate=0.4)
            for _ in range(pairs):
                ear = ear_pool[int(sel.integers(len(ear_pool)))]
                try:
                    audio = render_pair(child, ear, seconds)
                except Exception:
                    continue
                if not judge(audio)["viable"]:
                    continue
                if arm == "novelty":
                    vec = encode(audio)
                    nov = bank.novelty(vec, k=min(15, len(bank)))
                    floor = float(np.quantile(nov_hist, 0.2)) \
                        if len(nov_hist) >= 10 else 0.0
                    if nov < floor:
                        rule_rejects += 1
                        continue
                    nov_hist.append(nov)
                else:
                    d = distance(signature(audio), np.asarray(parent["sig"]))
                    if d <= eps:
                        rule_rejects += 1
                        continue
                admit(child, ear, audio, gen, parent)
                nov_admits = nov_admits + 1
                live.append(archive[-1])
                if len(live) > pop:
                    live.pop(int(sel.integers(len(live))))
                admits += 1
                break
        recent = [e["desc"] for e in archive[-40:]]
        spread = float(np.mean([np.std([d[a] for d in recent])
                                for a in ("flatness", "slope", "events",
                                          "modrate", "asym", "side")])) \
            if len(recent) > 1 else 0.0
        log.append({"gen": gen, "admissions": admits,
                    "rule_rejects": rule_rejects,
                    "cells_cumulative": len(cells_seen),
                    "cells_now": {g: len(cells_now[g]) for g in cells_now},
                    "spread": spread, "nov_mean": nov_admits})
        if verbose:
            print(f"[echo]   {arm} seed{seed} gen{gen}: +{admits} "
                  f"cells {len(cells_seen)} spread {spread:.3f}", flush=True)
    # -- dense coverage: the grid cannot see collisions; echo space can.
    #    pairwise = mean distance between archive members (spread of the
    #    run's discovered territory); nn = mean nearest-neighbor distance
    #    (how distinguishable the run's own specimens are from each other).
    V = np.stack([e["vec"] for e in archive])
    echo_pairwise = echo_nn = 0.0
    if len(V) > 2:
        Vs = (V - bank.metric.mu) * bank.metric.scale()
        sq = (Vs * Vs).sum(axis=1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (Vs @ Vs.T)
        np.fill_diagonal(D2, np.inf)
        D = np.sqrt(np.maximum(D2, 0.0))
        echo_pairwise = float(D.sum() / (len(V) * (len(V) - 1)))
        echo_nn = float(np.sqrt(np.maximum(np.min(D2, axis=1), 0.0)).mean())
    bank.close()
    return {"log": log, "admissions": sum(r["admissions"] for r in log),
            "cells_cumulative": len(cells_seen),
            "cells_per_grid": {g: len(cells_now[g]) for g in cells_now},
            "final_spread": log[-1]["spread"] if log else 0.0,
            "echo_pairwise": echo_pairwise, "echo_nn": echo_nn}


def ab_experiment(seeds=(0, 1), gens=10, pop=12, births=6, pairs=2,
                  seconds=3.0, strictness=0.5, verbose=False):
    """Three memories, one question: which breeding run finds more
    character-space? drift = MCC alone (drift floor), cell = womb's
    grid-cell parent pull, novelty = the echo archive gate."""
    arms = {}
    for arm in ("drift", "cell", "novelty"):
        runs = []
        for seed in seeds:
            t0 = time.time()
            runs.append(_run_arm(arm, seed, gens, pop, births, pairs,
                                 seconds, strictness, verbose))
            if verbose:
                print(f"[echo] arm {arm} seed {seed}: "
                      f"{runs[-1].get('cells_cumulative', 0)} cells "
                      f"({time.time() - t0:.0f}s)", flush=True)
        ok = [r for r in runs if r.get("log")]
        arms[arm] = {
            "runs": [r for r in runs],
            "cells_cumulative_mean": float(np.mean(
                [r["cells_cumulative"] for r in ok])) if ok else 0.0,
            "admissions_mean": float(np.mean(
                [r["admissions"] for r in ok])) if ok else 0.0,
            "final_spread_mean": float(np.mean(
                [r["final_spread"] for r in ok])) if ok else 0.0,
            "echo_pairwise_mean": float(np.mean(
                [r["echo_pairwise"] for r in ok])) if ok else 0.0,
            "echo_nn_mean": float(np.mean(
                [r["echo_nn"] for r in ok])) if ok else 0.0,
        }
    return arms


# ------------------------------------------------------------------ speed

def speed_test(corpus_rows, verbose=False):
    import timeit

    res = {}
    # encode
    t = []
    for r in corpus_rows[:60]:
        p = Path(r["file"])
        try:
            audio = decode(p, SR, stereo=True)
        except Exception:
            continue
        t0 = time.perf_counter()
        encode(audio, SR)
        t.append(time.perf_counter() - t0)
    res["encode_ms_median"] = float(1000 * np.median(t)) if t else None
    res["encode_ms_p90"] = float(1000 * np.percentile(t, 90)) if t else None
    # knn at growing archive sizes (synthetic archive from real vectors)
    X = np.stack([r["vec"] for r in corpus_rows])
    for n in (1000, 5000, 10000):
        reps = int(np.ceil(n / len(X)))
        rng = np.random.default_rng(2)
        M = np.tile(X, (reps, 1))[:n] + rng.normal(0, 0.01, (n, X.shape[1]))
        bank = EchoBank(":memory:")
        bank.fit(X, save=False)
        bank.add_many(M)
        t = []
        for i in range(20):
            v = X[int(rng.integers(len(X)))] + rng.normal(0, 0.01, X.shape[1])
            t0 = time.perf_counter()
            bank.knn(v, k=15)
            t.append(time.perf_counter() - t0)
        res[f"knn_ms_at_{n}"] = float(1000 * np.median(t))
        bank.close()
    if verbose:
        print(f"[echo] speed: encode {res['encode_ms_median']:.1f} ms, "
              f"knn@10k {res['knn_ms_at_10000']:.2f} ms")
    return res


# ------------------------------------------------------------------ the full suite

def full_bench(corpus_dir, out_dir="echo_out", n_perturb=50, n_distant=60,
               seeds=(0, 1), gens=10, pop=12, births=6, pairs=2,
               seconds=3.0, strictness=0.5, extras=None, verbose=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {"date": str(date.today()), "corpus": str(corpus_dir)}

    if verbose:
        print(f"[echo] encoding corpus: {corpus_dir}")
    rows, enc_time = load_corpus(corpus_dir)
    if not rows:
        raise RuntimeError("no wavs found in corpus dir")
    results["corpus"] = {"chunks": len(rows),
                         "encode_seconds_total": round(enc_time, 2)}

    if verbose:
        print(f"[echo] building competitor encoders")
    encoders = build_encoders(rows, extras=extras)
    rng = np.random.default_rng(0)

    if verbose:
        print(f"[echo] pair suite")
    pos, neg, champs = pair_suite(rows, rng, n_perturb, n_distant, seconds, verbose)
    results["discrimination"] = discrimination(encoders, pos, neg, verbose)

    metric = EchoMetric(DIM).fit(np.stack([r["vec"] for r in rows]))
    results["mutation_monotonicity"] = mutation_monotonicity(metric, verbose=verbose)
    results["locality"] = locality(rows, verbose=verbose)
    if verbose:
        print(f"[echo] A/B selection experiment "
              f"(3 arms x {len(seeds)} seeds x {gens} gens)")
    results["ab"] = ab_experiment(seeds, gens, pop, births, pairs, seconds,
                                  strictness, verbose)
    results["speed"] = speed_test(rows, verbose=verbose)

    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (out / "bench_report.md").write_text(render_report(results))
    if verbose:
        print(f"[echo] report -> {out / 'bench_report.md'}")
    return results


# ------------------------------------------------------------------ report

def render_report(r) -> str:
    L = [f"# echo bench report — {r['date']}", "",
         f"corpus: `{r['corpus']}` — {r['corpus']['chunks']} chunks, "
         f"{r['corpus']['encode_seconds_total']}s to encode", ""]

    L += ["## 1. pair discrimination — have we heard this before?", "",
          "near-duplicates must land close, distinct pairs far.",
          "AUC 1.0 = perfect ordering; d' = separation in sigmas; EER = error rate.",
          "", "| encoder | AUC | d' | EER | pos med | neg med |", "|---|---|---|---|---|---|"]
    for name, m in r["discrimination"].items():
        if name.startswith("_"):
            continue
        L.append(f"| {name} | {m['auc']:.3f} | {m['d_prime']:.2f} | "
                 f"{m['eer']:.3f} | {m['pos_median']:.3f} | {m['neg_median']:.3f} |")
    det = r["discrimination"].get("_perturbation_detail", {})
    if det:
        L += ["", "echo's per-perturbation median distance (the near pairs):", ""]
        for k, v in det.items():
            L.append(f"- `{k}`: {v['median']:.3f} (n={v['n']})")

    m = r.get("mutation_monotonicity")
    if m:
        L += ["", "## 2. mutation monotonicity — does drift have a gradient?", "",
              f"Spearman rho (mutation size vs distance): **{m['spearman']:.2f}**", ""]
        for f, d in zip(m["factors"], m["dists"]):
            L.append(f"- speed x{f}: dist {d:.3f}")
        L.append("")
        for k, v in m["cross_law"].items():
            L.append(f"- vs {k} (different law): dist {v:.3f}")
        L.append("")
        L.append(f"different law farther than the biggest parameter mutation: "
                  f"**{'yes' if m['cross_exceeds_max_mut'] else 'no'}**")

    loc = r.get("locality", {})
    if loc:
        L += ["", "## 3. corpus locality — nearest neighbors on real material", "",
              f"- median rank of the true (adjacent-take) neighbor: "
              f"**{loc['median_rank']:.0f}** of {loc['n_queries']} chunks",
              f"- recall@1 / @5 / @10: **{loc['recall_at_1']:.2f} / "
              f"{loc['recall_at_5']:.2f} / {loc['recall_at_10']:.2f}**",
              f"- separation (nn dist / random-pair dist): "
              f"**{loc['separation']:.3f}** (lower = tighter neighborhoods)"]

    ab = r.get("ab", {})
    if ab:
        L += ["", "## 4. A/B selection — what does the memory buy evolution?", "",
              "identical laws, judge, seeds; only the memory differs.",
              "cells = womb grid cells discovered (cumulative, 3 grids of 64);",
              "spread = descriptor-space diversity of the last 40 admissions;",
              "echo pairwise / nn = dense echo-space spread and",
              "nearest-neighbor distance of the final archive (what the grid cannot see).",
              "", "| arm | cells | admissions | spread | echo pairwise | echo nn |", "|---|---|---|---|---|---|"]
        for arm, a in ab.items():
            L.append(f"| {arm} | {a['cells_cumulative_mean']:.1f} | "
                     f"{a['admissions_mean']:.1f} | {a['final_spread_mean']:.4f} | "
                     f"{a['echo_pairwise_mean']:.3f} | {a['echo_nn_mean']:.3f} |")
        L += ["", "per-generation cumulative cells (mean over seeds):", ""]
        gens_ = len(next(iter(ab.values()))["runs"][0].get("log", [])) \
            if ab else 0
        if gens_:
            L.append("| gen | " + " | ".join(ab) + " |")
            L.append("|---" * (len(ab) + 1) + "|")
            for g in range(gens_):
                vals = []
                for arm, a in ab.items():
                    logs = [run["log"][g] for run in a["runs"] if run.get("log")]
                    vals.append(f"{np.mean([l['cells_cumulative'] for l in logs]):.0f}"
                                if logs else "-")
                L.append(f"| {g + 1} | " + " | ".join(vals) + " |")

    sp = r.get("speed", {})
    if sp:
        L += ["", "## 5. speed", "",
              f"- encode: {sp.get('encode_ms_median', float('nan')):.1f} ms/clip "
              f"(p90 {sp.get('encode_ms_p90', float('nan')):.1f})",
              f"- knn@1k {sp.get('knn_ms_at_1000', float('nan')):.2f} ms, "
              f"@5k {sp.get('knn_ms_at_5000', float('nan')):.2f} ms, "
              f"@10k {sp.get('knn_ms_at_10000', float('nan')):.2f} ms"]
    L.append("")
    return "\n".join(L)