"""validate: echo at scale — the full-corpus proof.

After `echo bench` (calibration on a designed corpus), `echo validate`
answers the production question on whatever pile of audio you point it
at: 2000+ files, mixed rates and lengths, folder-provenance as the only
ground truth. Four measurements:

    1. INDEX        — every wav encoded into a bank (robustness +
                      speed at scale; decode failures logged, not fatal)
    2. FAMILY       — leave-one-out retrieval: is a file's nearest
                      neighbor its sibling from the same source folder?
                      recall@k, micro and macro, per-family league table
    3. DISCRIMINATION — perturbations of corpus files (near) vs
                      cross-family pairs (far): AUC / d' / EER through
                      the fitted metric
    4. NOVELTY SCAN — every entry's novelty against the rest of the
                      corpus: the distribution of "have we heard this",
                      the most embedded clichés and the strangest
                      outliers the memory has ever seen

    python -m echo validate corpus/ -o echo_out/validate
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from .bank import EchoBank
from .encode import DIM, encode
from .probes import PERTURB_NAMES, decode, perturbations

from .bench import d_prime, eer, roc_auc

__all__ = ["validate_corpus", "render_validate_report"]

SR = 44100


# ------------------------------------------------------------------ manifest

def load_manifest(corpus_dir) -> dict[str, str]:
    """{flat filename: family} from corpus/manifest.json (written by the
    move). Without a manifest every file is its own family — the family
    measurements simply go quiet, the rest still runs."""
    p = Path(corpus_dir) / "manifest.json"
    if p.exists():
        rows = json.loads(p.read_text())
        return {r["file"]: r.get("family", "root") for r in rows}
    return {}


# ------------------------------------------------------------------ indexing

def index_corpus(corpus_dir, bank_path, limit=None,
                 probe_s: float = 30.0, verbose=True):
    """Encode every wav into the bank. Returns (bank, timings, failures).
    `probe_s`: files longer than this also yield a steady-state probe
    vector (role 'probe') for the discrimination pass — long takes are
    indexed whole AND sampled, so probes stay cheap."""
    files = sorted(p for p in Path(corpus_dir).glob("*.wav"))
    if limit:
        files = files[:limit]
    bank = EchoBank(bank_path)
    timings, failures = [], []
    t_all = time.time()
    for i, p in enumerate(files):
        try:
            audio = decode(p, SR, stereo=True)
        except Exception as e:                          # noqa: BLE001
            failures.append({"file": p.name, "error": str(e)[:120]})
            continue
        t0 = time.perf_counter()
        v = encode(audio, SR)
        dt = time.perf_counter() - t0
        timings.append(dt)
        bank.add(v, {"file": p.name}, role="corpus", kind="wav")
        if len(audio) > probe_s * SR and len(audio) // 2 > probe_s * SR:
            half = len(audio) // 2
            seg = audio[half: half + int(probe_s * SR)]
            bank.add(encode(seg, SR), {"file": p.name, "part": "mid"},
                     role="probe", kind="wav")
        if verbose and (i + 1) % 100 == 0:
            print(f"[echo] indexed {i + 1}/{len(files)} "
                  f"({np.median(timings) * 1000:.0f} ms/clip median)",
                  flush=True)
    # fit the metric on everything just heard, then rescale the cache
    X = bank.matrix()
    bank.fit(X, save=True)
    if verbose:
        print(f"[echo] index: {len(bank)} vectors in {time.time() - t_all:.0f}s, "
              f"metric fitted on {len(X)}", flush=True)
    return bank, timings, failures


# ------------------------------------------------------------------ family retrieval

def family_retrieval(bank: EchoBank, fam_of: dict, k: int = 10):
    """Leave-one-out: for every corpus entry whose family has siblings,
    does its nearest neighbor come from home? R@k + per-family table."""
    entries = [(i, bank.metas[i].get("file", ""), bank.ids[i])
               for i in range(len(bank)) if bank.roles[i] == "corpus"]
    fam_count: dict[str, int] = {}
    for _, name, _ in entries:
        f = fam_of.get(name, "")
        if f:
            fam_count[f] = fam_count.get(f, 0) + 1
    ranks = []
    per_family: dict[str, list[int]] = {}
    t0 = time.time()
    for i, name, eid in entries:
        f = fam_of.get(name, "")
        if not f or fam_count[f] < 2:
            continue
        near = bank.knn(bank._M[i], k=k, roles="corpus", exclude=eid)
        r = None
        for j, n in enumerate(near):
            if fam_of.get(Path(n["meta"]["file"]).name, "") == f:
                r = j + 1
                break
        if r is not None:
            ranks.append(r)
            per_family.setdefault(f, []).append(r)
        else:
            ranks.append(k + 1)                        # miss = worst rank
            per_family.setdefault(f, []).append(k + 1)
    ranks = np.array(ranks)
    table = []
    for f, rs in per_family.items():
        if fam_count[f] >= 3:
            table.append({"family": f, "n": fam_count[f],
                          "r1": float(np.mean([r == 1 for r in rs])),
                          "r5": float(np.mean([r <= 5 for r in rs]))})
    table.sort(key=lambda t: -t["r1"])
    res = {"n_queries": int(len(ranks)),
           "recall_at_1": float(np.mean(ranks == 1)) if len(ranks) else None,
           "recall_at_5": float(np.mean(ranks <= 5)) if len(ranks) else None,
           "recall_at_10": float(np.mean(ranks <= 10)) if len(ranks) else None,
           "median_rank": float(np.median(ranks)) if len(ranks) else None,
           "n_families": len(per_family),
           "knn_seconds": round(time.time() - t0, 1),
           "per_family": table}
    if table:
        res["macro_r1"] = float(np.mean([t["r1"] for t in table]))
        res["macro_r5"] = float(np.mean([t["r5"] for t in table]))
    return res


# ------------------------------------------------------------------ discrimination

def discrimination_at_scale(bank: EchoBank, corpus_dir, fam_of: dict,
                             n_probes: int = 60, n_far: int = 100,
                             probe_s: float = 12.0, n_mp3: int = 10,
                             rng=None, verbose=True):
    """Near = perturbed corpus sounds (must stay close); far =
    cross-family pairs (must separate). Through the fitted metric."""
    rng = rng or np.random.default_rng(0)
    corpus_files = [Path(corpus_dir) / m for m in fam_of] or \
        sorted(Path(corpus_dir).glob("*.wav"))
    sample = [corpus_files[int(rng.integers(len(corpus_files)))]
              for _ in range(min(n_probes, len(corpus_files)))]
    pos, pos_tags = [], []
    for pi, p in enumerate(sample):
        try:
            audio = decode(p, SR, stereo=True)
        except Exception:
            continue
        if len(audio) > probe_s * SR:
            half = len(audio) // 2
            audio = audio[half: half + int(probe_s * SR)]
        names = ["gain", "shift", "noise_floor", "eq_tilt"]
        if pi < n_mp3:
            names.append("mp3_round")
        perts = perturbations(audio, SR, rng,
                              names=[names[int(rng.integers(len(names)))]])
        base_vec = encode(audio, SR)
        for tag, pb in perts.items():
            pos.append((base_vec, encode(pb, SR)))
            pos_tags.append(tag)
    # far: cross-family pairs
    fams = {}
    for name, f in fam_of.items():
        fams.setdefault(f, []).append(name)
    multi = [f for f in fams if f != "root" and len(fams[f]) >= 2]
    neg = []
    guard = 0
    while len(neg) < n_far and guard < n_far * 20 and len(multi) >= 2:
        guard += 1
        fa, fb = [multi[int(rng.integers(len(multi)))] for _ in range(2)]
        if fa == fb:
            continue
        va = _vec_of(bank, fams[fa][int(rng.integers(len(fams[fa])))])
        vb = _vec_of(bank, fams[fb][int(rng.integers(len(fams[fb])))])
        if va is not None and vb is not None:
            neg.append((va, vb))
    dpos = [bank.metric.dist(a, b) for a, b in pos]
    dneg = [bank.metric.dist(a, b) for a, b in neg]
    res = {"auc": roc_auc(dpos, dneg), "d_prime": d_prime(dpos, dneg),
           "eer": eer(dpos, dneg), "n_pos": len(dpos), "n_neg": len(dneg),
           "pos_median": float(np.median(dpos)) if dpos else None,
           "neg_median": float(np.median(dneg)) if dneg else None}
    detail = {}
    for d, tag in zip(dpos, pos_tags):
        detail.setdefault(tag, []).append(d)
    res["per_perturbation"] = {k: {"median": float(np.median(v)), "n": len(v)}
                               for k, v in sorted(detail.items())}
    if verbose:
        print(f"[echo] discrimination: AUC {res['auc']:.3f} d' {res['d_prime']:.2f} "
              f"({len(dpos)} near / {len(dneg)} far)", flush=True)
    return res


def _vec_of(bank: EchoBank, name: str):
    for i in range(len(bank)):
        if bank.roles[i] == "corpus" and bank.metas[i].get("file") == name:
            return bank._M[i]
    return None


# ------------------------------------------------------------------ novelty scan

def novelty_scan(bank: EchoBank, fam_of: dict, k: int = 15, verbose=True):
    """Every corpus entry vs the rest of the corpus: the distribution of
    'have we heard this', the embedded clichés, the strangers."""
    t0 = time.time()
    novs, names = [], []
    for i in range(len(bank)):
        if bank.roles[i] != "corpus":
            continue
        novs.append(bank.novelty(bank._M[i], k=k, roles="corpus"))
        names.append(bank.metas[i].get("file", ""))
    novs = np.array(novs)
    order_lo = np.argsort(novs)[:8]                 # most heard
    order_hi = np.argsort(novs)[-8:][::-1]         # strangest
    res = {"n": int(len(novs)), "k": k,
           "mean": float(novs.mean()), "median": float(np.median(novs)),
           "p10": float(np.percentile(novs, 10)),
           "p90": float(np.percentile(novs, 90)),
           "seconds": round(time.time() - t0, 1),
           "most_embedded": [{"file": names[i], "novelty": float(novs[i]),
                              "family": fam_of.get(names[i], "")}
                             for i in order_lo],
           "strangest": [{"file": names[i], "novelty": float(novs[i]),
                          "family": fam_of.get(names[i], "")}
                         for i in order_hi]}
    if verbose:
        print(f"[echo] novelty scan: median {res['median']:.2f}, "
              f"range {novs.min():.2f}..{novs.max():.2f}", flush=True)
    return res


# ------------------------------------------------------------------ spot checks

def spot_checks(bank: EchoBank, fam_of: dict, n: int = 8, seed: int = 5):
    """Human-readable proof: a few files and their nearest neighbors."""
    rng = np.random.default_rng(seed)
    idx = [i for i in range(len(bank)) if bank.roles[i] == "corpus"]
    picks = [idx[int(rng.integers(len(idx)))] for _ in range(min(n, len(idx)))]
    out = []
    for i in picks:
        name = bank.metas[i].get("file", "")
        near = bank.knn(bank._M[i], k=5, roles="corpus", exclude=bank.ids[i])
        out.append({"query": name, "family": fam_of.get(name, ""),
                    "neighbors": [{"file": n["meta"]["file"], "dist": n["dist"],
                                   "family": fam_of.get(n["meta"]["file"], "")}
                                  for n in near]})
    return out


# ------------------------------------------------------------------ the suite

def validate_corpus(corpus_dir, out_dir="echo_out/validate", limit=None,
                    n_probes=60, n_far=100, n_mp3=10, bank_path=None,
                    verbose=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank_path = bank_path or str(out / "bank.sqlite")
    fam_of = load_manifest(corpus_dir)
    results = {"date": str(date.today()), "corpus": str(corpus_dir)}

    bank, timings, failures = index_corpus(corpus_dir, bank_path, limit=limit,
                                           verbose=verbose)
    if len(bank) < 2:
        raise RuntimeError("bank too small to validate")
    results["index"] = {
        "vectors": len(bank), "files": len(timings) + len(failures),
        "failures": failures,
        "encode_ms_median": float(1000 * np.median(timings)) if timings else None,
        "encode_ms_p90": float(1000 * np.percentile(timings, 90)) if timings else None,
        "encode_total_s": round(float(np.sum(timings)), 1),
        "audio_total_s": None, "bank": bank_path}

    results["family"] = family_retrieval(bank, fam_of)
    results["discrimination"] = discrimination_at_scale(
        bank, corpus_dir, fam_of, n_probes=n_probes, n_far=n_far,
        n_mp3=n_mp3, verbose=verbose)
    results["novelty"] = novelty_scan(bank, fam_of)
    results["spot_checks"] = spot_checks(bank, fam_of)

    # knn speed at this scale
    t = []
    rng = np.random.default_rng(3)
    idx = [i for i in range(len(bank)) if bank.roles[i] == "corpus"]
    for _ in range(20):
        v = bank._M[idx[int(rng.integers(len(idx)))]]
        t0 = time.perf_counter()
        bank.knn(v, k=15)
        t.append(time.perf_counter() - t0)
    results["speed"] = {"knn_ms_at_scale": float(1000 * np.median(t)),
                       "entries": len(bank)}
    bank.close()

    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (out / "validate_report.md").write_text(render_validate_report(results))
    if verbose:
        print(f"[echo] report -> {out / 'validate_report.md'}")
    return results


# ------------------------------------------------------------------ report

def render_validate_report(r) -> str:
    L = [f"# echo validation at scale — {r['date']}", "",
         f"corpus: `{r['corpus']}` — {r['index']['files']} files, "
         f"{r['index']['vectors']} vectors "
         f"({r['index']['encode_ms_median']:.0f} ms median encode, "
         f"{r['index']['encode_total_s']}s total)", ""]
    if r["index"]["failures"]:
        L.append(f"decode failures (skipped, logged): "
                 f"{len(r['index']['failures'])}")
        L.append("")

    fam = r["family"]
    if fam.get("n_queries"):
        L += ["## 1. family retrieval — does home sound like home?", "",
              "leave-one-out: a file's nearest neighbors vs its source folder.",
              "", f"- queries: {fam['n_queries']} across "
              f"{fam['n_families']} families ({fam['knn_seconds']}s)",
              f"- recall@1 / @5 / @10 (micro): "
              f"**{fam['recall_at_1']:.2f} / {fam['recall_at_5']:.2f} / "
              f"{fam['recall_at_10']:.2f}**, median rank "
              f"**{fam['median_rank']:.0f}**",
              f"- recall@1 / @5 (macro, per family): "
              f"**{fam.get('macro_r1', 0):.2f} / {fam.get('macro_r5', 0):.2f}**",
              "", "the tightest families (most self-similar material):", ""]
        for t in fam["per_family"][:6]:
            L.append(f"- `{t['family']}` (n={t['n']}): R@1 {t['r1']:.2f}, "
                     f"R@5 {t['r5']:.2f}")
        L += ["", "the loosest:", ""]
        for t in fam["per_family"][-4:]:
            L.append(f"- `{t['family']}` (n={t['n']}): R@1 {t['r1']:.2f}, "
                     f"R@5 {t['r5']:.2f}")

    d = r["discrimination"]
    L += ["", "## 2. discrimination at scale — near vs far, fitted metric", "",
          f"AUC **{d['auc']:.3f}** | d' {d['d_prime']:.2f} | EER {d['eer']:.3f} "
          f"({d['n_pos']} near / {d['n_neg']} far)",
          f"near median {d['pos_median']:.3f} vs far median {d['neg_median']:.3f}",
          "", "per-perturbation near distances:"]
    for k, v in d["per_perturbation"].items():
        L.append(f"- `{k}`: {v['median']:.3f} (n={v['n']})")

    nv = r["novelty"]
    L += ["", "## 3. the novelty landscape of the whole corpus", "",
          f"median novelty {nv['median']:.2f} "
          f"(p10 {nv['p10']:.2f} .. p90 {nv['p90']:.2f})",
          "", "most embedded (the corpus's clichés — everything sounds like these):", ""]
    for t in nv["most_embedded"][:5]:
        L.append(f"- {t['novelty']:.2f}  `{t['file']}`  ({t['family']})")
    L += ["", "strangest (the memory has never heard anything like these):", ""]
    for t in nv["strangest"][:5]:
        L.append(f"- {t['novelty']:.2f}  `{t['file']}`  ({t['family']})")

    L += ["", "## 4. spot checks — read these like a listener", ""]
    for s in r["spot_checks"]:
        L.append(f"**`{s['query']}`** ({s['family']})")
        for n in s["neighbors"]:
            same = "← home" if n["family"] == s["family"] and n["family"] else ""
            L.append(f"- {n['dist']:8.3f}  `{n['file']}` ({n['family']}) {same}")
        L.append("")

    L += ["## 5. speed at scale", "",
          f"- knn at {r['speed']['entries']} entries: "
          f"**{r['speed']['knn_ms_at_scale']:.2f} ms** per query (k=15)",
          f"- encode: {r['index']['encode_ms_median']:.0f} ms median "
          f"(p90 {r['index']['encode_ms_p90']:.0f})", ""]
    return "\n".join(L)