"""Engine: batch a folder of sounds through the transform registry into a
structured output bank (oneshot_out-style), with provenance manifest,
deterministic per-(file, transform) seeds, optional random mutation chains,
and a multiprocessing pool."""
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .core import clamp_dur, find_audio, read_audio, safe, write_audio
from .registry import FAMILIES, TRANSFORMS, resolve

MAX_NAME = 44


def _tag(params):
    """Compact parameter signature for the output filename."""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}{v:.3g}")
        elif isinstance(v, bool):
            parts.append(f"{k}{int(v)}")
        elif isinstance(v, (list, tuple)):
            parts.append(k + "x".join(f"{i:g}" if isinstance(i, float) else str(i)
                                     for i in v))
        else:
            parts.append(f"{k}{v}")
    s = "_".join(parts)
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return "".join(c for c in s if c in keep)[:64] or "def"


def seed_for(master_seed, stem, name, variant=0):
    """Deterministic seed per (master, file, transform, variant)."""
    h = zlib.crc32(f"{master_seed}|{stem}|{name}|{variant}".encode())
    return h


def random_params(name, rng):
    """Random parameters for a transform (falls back to defaults)."""
    _, _, defaults, _, _, rand = TRANSFORMS[name]
    if rand is None:
        return dict(defaults)
    try:
        p = dict(rand(rng))
        for k, v in defaults.items():          # fill any gaps with defaults
            p.setdefault(k, v)
        return p
    except Exception:
        return dict(defaults)


# ----------------------------------------------------------------- worker


def run_job(job):
    """One (file × transform) job. Returns a manifest record."""
    name = job["name"]
    rng = np.random.default_rng(seed_for(job["master_seed"], job["stem"],
                                         name, job["variant"]))
    rec = {"source": job["src"], "stem": job["stem"], "transform": name,
           "params": job["params"], "variant": job["variant"],
           "ok": False}
    try:
        x, sr = read_audio(job["src"], max_in_s=job["max_in_s"], rng=rng)
        params = dict(job["params"])
        stages = None
        if job["chain"]:
            # a mutation chain: apply each stage to the previous output
            y = x
            stages = params["stages"]
            for s_name in stages:
                fn, _, defaults, needs_bytes, _, _ = TRANSFORMS[s_name]
                s_params = params["stage_params"].get(s_name, defaults)
                s_rng = np.random.default_rng(
                    seed_for(job["master_seed"], job["stem"],
                             name + "/" + s_name, job["variant"]))
                try:
                    if needs_bytes:
                        y = fn(y, sr, s_rng, path=job["src"], **s_params)
                    else:
                        y = fn(y, sr, s_rng, **s_params)
                except Exception:
                    continue
                y = clamp_dur(safe(y), sr, job["max_out_s"])
        else:
            fn, fam, defaults, needs_bytes, _, _ = TRANSFORMS[name]
            if needs_bytes:
                y = fn(x, sr, rng, path=job["src"], **params)
            else:
                y = fn(x, sr, rng, **params)
        if stages is not None:
            rec["params"] = {"stages": stages}
            ptag = "stages+" + "-".join(stages)
        else:
            ptag = _tag(params)
        y = clamp_dur(safe(y), sr, job["max_out_s"])
        if y.shape[0] < 256:
            rec["error"] = "output too short"
            return rec
        vtag = f"_v{job['variant']}" if job["variant"] else ""
        fname = (f"{job['stem'][:MAX_NAME]}__{name}__"
                 f"{ptag}{vtag}.wav")
        outp = Path(job["out_dir"]) / job["subdir"] / fname
        rec["out"] = str(outp)
        rec["dur_s"] = write_audio(outp, y, sr, subtype=job["subtype"])
        rec["ok"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# ----------------------------------------------------------------- planning


def select_files(files, seed, pick=None, pick_per_folder=None):
    """Seeded selection of inputs: `pick` total, or `pick_per_folder` from
    each immediate subfolder (balanced sampling across a transform bank)."""
    files = list(files)
    if not pick and not pick_per_folder:
        return files
    rng = np.random.default_rng(seed_for(seed, "select", "files", 0))
    if pick_per_folder:
        groups = {}
        for f in files:
            groups.setdefault(str(Path(f).parent.name), []).append(f)
        picked = []
        for g in sorted(groups):
            pool = sorted(groups[g])
            idx = rng.permutation(len(pool))[: pick_per_folder]
            picked.extend(pool[i] for i in sorted(idx))
        return sorted(picked)
    pool = sorted(files)
    idx = rng.permutation(len(pool))[: min(pick, len(pool))]
    return sorted(pool[i] for i in sorted(idx))


def plan(input_dir, out_dir="oneshot_out", transforms=("all",),
         random_chains=0, seed=42, max_in_s=25.0, max_out_s=8.0,
         per_transform_variants=1, subtype="PCM_16",
         recursive=False, pick=None, pick_per_folder=None):
    """Build the job list without running anything."""
    files = find_audio(input_dir, recursive=recursive)
    if not files:
        raise SystemExit(f"no audio files found in {input_dir}")
    files = select_files(files, seed, pick, pick_per_folder)
    if not files:
        raise SystemExit("selection empty")
    if "all" in transforms:
        names = list(TRANSFORMS.keys())
    else:
        names = resolve(list(transforms))
    jobs = []
    for f in files:
        stem = f.stem
        for name in names:
            for v in range(per_transform_variants):
                rng = np.random.default_rng(seed_for(seed, stem, name, v))
                if v == 0:
                    params = dict(TRANSFORMS[name][2])   # canonical defaults
                else:
                    params = random_params(name, rng)     # extra variants: random
                jobs.append(dict(src=str(f), stem=stem, name=name,
                                  variant=v,
                                  params=params, chain=False,
                                  subdir=name, out_dir=out_dir,
                                  max_in_s=max_in_s, max_out_s=max_out_s,
                                  master_seed=seed, subtype=subtype))
        # random mutation chains
        for ci in range(random_chains):
            rng = np.random.default_rng(seed_for(seed, stem, "chain", ci))
            n_stages = int(rng.integers(2, 4))
            pool = list(TRANSFORMS.keys())
            stages = list(rng.choice(pool, size=n_stages, replace=False))
            stage_params = {s: random_params(s, np.random.default_rng(
                seed_for(seed, stem, "chain/" + s, ci))) for s in stages}
            cname = f"chain{ci + 1}"
            jobs.append(dict(src=str(f), stem=stem, name=cname, variant=0,
                             params={"stages": list(stages),
                                     "stage_params": stage_params},
                             chain=True, subdir=cname, out_dir=out_dir,
                             max_in_s=max_in_s, max_out_s=max_out_s,
                             master_seed=seed, subtype=subtype))
    return files, jobs


def run_folder(input_dir, out_dir="oneshot_out", transforms=("all",),
               random_chains=0, seed=42, max_in_s=25.0, max_out_s=8.0,
               per_transform_variants=1, subtype="PCM_16", workers=None,
               verbose=True, recursive=False, pick=None, pick_per_folder=None):
    """The main entry: mutate a folder of sounds into a bank of variations.

    Output: out_dir/<transform_name>/<source>__<transform>__<params>.wav
    plus out_dir/_manifest.json with full provenance.
    """
    files, jobs = plan(input_dir, out_dir, transforms, random_chains, seed,
                       max_in_s, max_out_s, per_transform_variants, subtype,
                       recursive, pick, pick_per_folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_manifest.jsonl"
    with open(manifest_path, "w") as fh:
        fh.write(json.dumps({"run": {"input": str(input_dir),
                                     "seed": seed,
                                     "n_files": len(files),
                                     "n_jobs": len(jobs),
                                     "transforms": sorted({j["name"] for j in jobs})}}) + "\n")
    if workers is None:
        import multiprocessing as mp
        workers = max(1, mp.cpu_count() - 1)
    t0 = time.time()
    done = failed = 0
    records = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            with open(manifest_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if rec["ok"]:
                done += 1
            else:
                failed += 1
                print(f"  ! {rec['stem']} {rec['transform']}: "
                      f"{rec.get('error')}", file=sys.stderr)
            if verbose and (done + failed) % 100 == 0:
                el = time.time() - t0
                print(f"  {done + failed}/{len(jobs)} jobs, "
                      f"{done} ok, {failed} failed, {el:.0f}s", flush=True)
    summary = {"input": str(input_dir), "out": str(out_dir),
               "n_files": len(files), "n_jobs": len(jobs),
               "ok": done, "failed": failed,
               "seed": seed, "seconds": round(time.time() - t0, 1)}
    with open(out_dir / "_manifest.json", "w") as fh:
        json.dump({"summary": summary, "records": records}, fh, indent=1)
    if verbose:
        print(f"done: {done} outputs, {failed} failed, "
              f"{summary['seconds']}s -> {out_dir}")
    return summary


# ----------------------------------------------------------------- in-process API


def mutate(x, sr, rng, names=None, n_stages=(2, 3), max_s=8.0):
    """Mutate an in-memory sound through a random chain of transforms.

    Returns (audio, [stage names used]). Library-style single-sound use:
        y, chain = oneshot_lab.mutate(my_hit, 44100, rng)
    """
    if names is None:
        names = list(TRANSFORMS.keys())
    n = int(rng.integers(n_stages[0], n_stages[1] + 1)) if n_stages[0] < n_stages[1] \
        else n_stages[0]
    stages = list(rng.choice(names, size=n, replace=False))
    y = np.asarray(x, np.float32)
    used = []
    for s in stages:
        fn, _, defaults, _, _, rand = TRANSFORMS[s]
        params = random_params(s, rng)
        try:
            y = fn(y, sr, rng, **params)
            y = clamp_dur(safe(y), sr, max_s)
            used.append(s)
        except Exception:
            continue
    return y, used


def describe():
    """Pretty table of every registered transform."""
    lines = []
    for fam, names in FAMILIES.items():
        lines.append(f"[{fam}]")
        for n in names:
            fn, _, defaults, needs_bytes, blurb, _ = TRANSFORMS[n]
            d = ",".join(f"{k}={v}" for k, v in defaults.items())
            lines.append(f"  {n:<24} {blurb}")
            if d:
                lines.append(f"  {'':24} defaults: {d}")
        lines.append("")
    return "\n".join(lines)