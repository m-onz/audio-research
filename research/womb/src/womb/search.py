"""search: novelty search for audiovisual organisms (MAP-Elites).

Instead of hand-picking organ parameters, let a quality-diversity loop
find the extremes: candidate genomes (an organ sequence with parameters,
plus the lowpoly reprocessing lens) are rendered as short in-memory
samples, measured (audio brightness x visual activity), and placed in a
behavior archive. Mutations breed from whatever the archive already holds
— the space fills outward toward configurations nobody would have tried.

    womb search --iters 150 -o womb_out/search1          # fills the archive
    womb walk  --archive womb_out/search1 -l 45           # piece from elites

Deliverables per search: `atlas.json` (genomes + features + cells),
`atlas.png` (contact sheet of every occupied cell) — a diversity map of
womb's own possibility space.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .lowpoly import LOWPOLY_DEFAULTS, lowpoly_frame
from .organs import ORGANS, make_organ
from .piece import normalize_plan

__all__ = ["search", "walk", "mutate_genome", "random_genome", "measure_sample"]

SR = 44100


# ------------------------------------------------------------------ genome

PALETTES = ["inferno", "ps2_dusk", "vhs_night", "black_bronze", "verdigris",
            "blood", "toxic", "ice", "amber", "mono", "gameboy"]


def _organs_for(corpus) -> list[str]:
    names = sorted(ORGANS)
    if not corpus:
        names = [n for n in names if n != "territory"]  # needs a corpus
    return names


def random_genome(rng, corpus=None) -> dict:
    names = _organs_for(corpus)
    n_seg = int(rng.integers(1, 4))
    segs = []
    for _ in range(n_seg):
        segs.append(_random_seg(names[int(rng.integers(len(names)))], rng))
    return {
        "organs": segs,
        "lowpoly": _random_lowpoly(rng),
        "seed": int(rng.integers(1 << 30)),
    }


def _random_seg(name, rng):
    defaults = dict(
        waveplate=dict(grid=int(rng.integers(48, 144)), damp=float(rng.uniform(0.995, 0.9999)),
                      strike_every=float(rng.choice([2, 4, 8, 12, 24]))),
        automata=dict(grid=int(rng.integers(64, 160))),
        swarm=dict(agents=int(rng.integers(40, 320)), cols=int(rng.integers(3, 10)),
                   rows=int(rng.integers(2, 7))),
        kuramoto=dict(grid=int(rng.integers(32, 96)), coupling=float(rng.uniform(0.2, 4.0)),
                      noise=float(10 ** rng.uniform(-3, -1))),
        territory=dict(window=float(rng.uniform(0.8, 4.0)), hop_len=float(rng.uniform(0.1, 0.6))),
    )
    p = dict(defaults.get(name, {}))
    p["palette"] = PALETTES[int(rng.integers(len(PALETTES)))]
    return {"organ": name, "params": p}


def _random_lowpoly(rng):
    return dict(
        grid=(int(rng.integers(10, 48)), int(rng.integers(16, 84))),
        jitter=float(10 ** rng.uniform(-1, 1)),
        tear=float(rng.uniform(0.0, 0.6)),
        levels=int(rng.choice([4, 8, 16, 24, 32])),
        edges=float(rng.uniform(0.0, 0.7)),
    )


_MUTATE_CORPUS = None  # set by search() so swaps respect corpus availability


def mutate_genome(g: dict, rng, rate: float = 0.35) -> dict:
    """Mutate a genome: jitter numbers, swap palettes/organs, resize the
    sequence — children stay close to their parent (the archive drifts)."""
    import copy

    child = copy.deepcopy(g)
    hit = False

    def jit(x, lo, hi):
        nonlocal hit
        hit = True
        x = float(x)
        f = 10.0 ** float(np.clip(rng.normal(0, rate), -4.0, 4.0))
        nv = x * f
        if not np.isfinite(nv):   # 0 x inf guard
            nv = x
        return float(np.clip(nv, lo, hi))

    for seg in child["organs"]:
        p = seg["params"]
        for k, v in list(p.items()):
            if k == "palette":
                if rng.random() < rate:
                    p[k] = PALETTES[int(rng.integers(len(PALETTES)))]
                    hit = True
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                lo, hi = _bounds(k, v)
                nv = jit(v, lo, hi)
                p[k] = int(round(nv)) if isinstance(v, int) else float(nv)
                hit = True
        if rng.random() < rate * 0.5:  # organ swap: fresh params wholesale
            swap_names = _organs_for(_MUTATE_CORPUS)
            new_seg = _random_seg(swap_names[int(rng.integers(len(swap_names)))], rng)
            seg["organ"] = new_seg["organ"]
            seg["params"] = new_seg["params"]
            hit = True
    lp = child["lowpoly"]
    for k, v in list(lp.items()):
        if k == "grid":
            lp["grid"] = (int(np.clip(lp["grid"][0] * 10 ** rng.normal(0, rate), 8, 64)),
                          int(np.clip(lp["grid"][1] * 10 ** rng.normal(0, rate), 12, 110)))
            hit = True
        elif isinstance(v, (int, float)):
            lo, hi = _bounds(k, v)
            nv = jit(v, lo, hi)
            lp[k] = int(round(nv)) if isinstance(v, int) else float(nv)
            hit = True
    if rng.random() < rate * 0.4:  # grow/prune a segment
        if len(child["organs"]) > 1 and rng.random() < 0.5:
            child["organs"].pop(int(rng.integers(len(child["organs"]))))
        else:
            name = _organs_for(_MUTATE_CORPUS)[int(rng.integers(len(_organs_for(_MUTATE_CORPUS))))]
            child["organs"].insert(int(rng.integers(len(child["organs"]) + 1)),
                                   _random_seg(name, rng))
        hit = True
    child["seed"] = int(rng.integers(1 << 30))
    return child if hit else random_genome(rng)


def _bounds(key, v):
    B = dict(jitter=(0.05, 12.0), tear=(0.0, 0.9), levels=(2, 32), edges=(0.0, 0.9),
             grid=(24, 180), damp=(0.99, 0.99995), strike_every=(1.0, 60.0),
             agents=(16, 500), cols=(2, 14), rows=(2, 9), coupling=(0.05, 6.0),
             noise=(0.0005, 0.5), window=(0.4, 8.0), hop_len=(0.05, 1.0))
    return B.get(key, (v * 0.25, v * 4.0 + 1e-6))


# ------------------------------------------------------------------ evaluation

def render_sample(genome: dict, seconds: float, size: tuple, fps: float,
                  corpus=None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Render a genome in memory: (audio, frames). No encode — fast eval."""
    w, h = size
    n_frames = int(round(seconds * fps))
    plan = normalize_plan(genome["organs"], seconds)
    lp = dict(LOWPOLY_DEFAULTS)
    lp.update(genome["lowpoly"] or {})
    hop_n = int(round(SR / fps))
    audio = np.zeros(n_frames * hop_n, np.float32)
    frames = []
    seg_starts = np.array([s["start"] for s in plan])
    seg_i, current = -1, None
    for i in range(n_frames):
        t = i / fps
        k = int(np.searchsorted(seg_starts, t, side="right") - 1)
        k = max(0, min(k, len(plan) - 1))
        if k != seg_i:
            seg = plan[k]
            seg_i = k
            current = make_organ(seg["organ"], w, h, fps,
                                 seed=genome["seed"] + 977 * k,
                                 corpus=corpus, **seg.get("params", {}))
        current.step(i)
        a = current.emit()
        if not np.isfinite(a).all():
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        frame = current.render()
        frame = lowpoly_frame(frame, energy=float(np.sqrt((a ** 2).mean()) * 4),
                              rng=np.random.default_rng(genome["seed"] + i), **lp)
        frames.append(frame)
        audio[i * hop_n: (i + 1) * hop_n] = a[:hop_n]
    return audio, frames


def measure_sample(audio: np.ndarray, frames: list[np.ndarray]) -> dict:
    """Behavior description + viability fitness of one sample."""
    x = audio.astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
    nfft = 2048
    seg = x[: len(x) // nfft * nfft]
    if len(seg) == 0:
        seg = np.pad(x, (0, nfft))
    S = np.abs(np.fft.rfft(seg.reshape(-1, nfft) * np.hanning(nfft), axis=1)).mean(axis=0)
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    centroid = float((fr * S).sum() / (S.sum() + 1e-9))
    flux = float(np.abs(np.diff(S)).mean())

    fv = np.stack([f.mean(axis=(0, 1), dtype=np.float64) for f in frames]) if frames \
        else np.zeros((1, 3))
    luma = fv.mean()
    luma_std = fv.std()
    if len(frames) > 1:
        motion = float(np.abs(np.diff(np.stack(
            [f.mean(axis=(0, 1), dtype=np.float64) for f in frames]))).mean())
    else:
        motion = 0.0

    # BC space: audio brightness x visual activity
    bx = float(np.clip((np.log2(max(60.0, centroid)) - np.log2(60)) / (np.log2(16000) - np.log2(60)), 0, 1))
    by = float(np.clip(motion / 6.0 + luma_std / 80.0, 0, 1))
    # fitness: viability of both modalities (rewards healthy signal, punishes
    # silence/black-out and clipping/white-out)
    rms = float(np.sqrt(np.mean(x * x)))          # plain, no epsilon tricks
    luma_std = float(luma_std) if np.isfinite(luma_std) else 0.0
    motion = float(motion) if np.isfinite(motion) else 0.0
    rms = float(rms) if np.isfinite(rms) else 0.0
    fit = (min(rms / 0.2, 1.0) * (1.0 if luma_std > 2.0 else 0.3)
           + min(luma_std / 40.0, 1.0) * (0.6 if rms > 0.02 else 0.15))
    out = {"bx": bx, "by": by, "fitness": fit, "rms": rms,
           "centroid": centroid, "flux": flux, "luma": luma,
           "luma_std": luma_std, "motion": motion}
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in out.items()}


# ------------------------------------------------------------------ MAP-Elites

def search(out_dir, iters: int = 120, seconds: float = 4.0, size: tuple = (180, 320),
           fps: float = 12.0, cells: tuple = (12, 12), corpus=None, seed: int = 0,
           init_frac: float = 0.3, verbose: bool = False) -> dict:
    """Fill a MAP-Elites archive of diverse audiovisual organisms."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive: dict[tuple, dict] = {}
    global _MUTATE_CORPUS
    _MUTATE_CORPUS = corpus
    t0 = time.time()
    n_init = max(2, int(iters * init_frac))

    for it in range(iters):
        if it < n_init or not archive:
            g = random_genome(rng, corpus=corpus)
        else:
            parent = archive[list(archive)[int(rng.integers(len(archive)))]]
            g = mutate_genome(parent["genome"], rng, rate=float(rng.uniform(0.1, 0.6)))
        try:
            audio, frames = render_sample(g, seconds, size, fps, corpus=corpus)
            m = measure_sample(audio, frames)
        except Exception as e:  # degenerate genome (e.g. territory w/o corpus)
            if verbose:
                print(f"[womb] iter {it}: rejected ({e})")
            continue
        cell = (int(m["bx"] * cells[0]), int(m["by"] * cells[1]))
        cell = (min(cell[0], cells[0] - 1), min(cell[1], cells[1] - 1))
        cur = archive.get(cell)
        if cur is None or m["fitness"] > cur["fitness"]:
            thumb = frames[len(frames) // 2] if frames else None
            archive[cell] = {"genome": g, "fitness": m["fitness"],
                             "features": {k: v for k, v in m.items() if k != "fitness"},
                             "thumb": thumb}
        if verbose and (it + 1) % max(1, iters // 10) == 0:
            print(f"[womb] iter {it + 1}/{iters}: archive {len(archive)} cells "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # persist: atlas.json + atlas.png contact sheet
    atlas = {"cells": list(cells), "corpus": str(corpus) if corpus else None,
             "seed": seed, "seconds": seconds, "elites": []}
    thumbs = []
    for (cx, cy), e in sorted(archive.items()):
        atlas["elites"].append({"cell": [cx, cy], "fitness": e["fitness"],
                                "features": e["features"], "genome": e["genome"]})
        if e["thumb"] is not None:
            thumbs.append((cx, cy, e["thumb"]))
    (out / "atlas.json").write_text(json.dumps(atlas, indent=1))
    if thumbs:
        from PIL import Image

        tw, th = thumbs[0][2].shape[1], thumbs[0][2].shape[0]
        sheet = Image.new("RGB", (cells[0] * tw, cells[1] * th), "black")
        for cx, cy, f in thumbs:
            sheet.paste(Image.fromarray(f), (cx * tw, cy * th))
        sheet.save(out / "atlas.png")
    result = {"archive": str(out / "atlas.json"), "cells_filled": len(archive),
              "grid": list(cells), "elapsed": round(time.time() - t0, 1)}
    if verbose:
        print(f"[womb] search done: {len(archive)}/{cells[0] * cells[1]} cells "
              f"in {result['elapsed']}s -> {out}")
    return result


# ------------------------------------------------------------------ walk (piece from elites)

def walk(archive_path, length: float, out, size: tuple = (720, 1280),
         fps: float = 24.0, seed: int = 0, corpus=None,
         n_segments: int | None = None, verbose: bool = False) -> dict:
    """Compose a full piece from archive elites: a tour through the
    discovered diversity, each segment rendered with its winning genome
    (organs AND lowpoly lens included)."""
    atlas = json.loads(Path(archive_path).read_text())
    elites = atlas["elites"]
    if not elites:
        raise RuntimeError("empty archive — run womb search first")
    rng = np.random.default_rng(seed)
    # order elites into a journey: bright->dark audio sweep with jitter
    elites = sorted(elites, key=lambda e: e["features"]["bx"] + rng.uniform(-0.2, 0.2))
    # pacing: enough elites that every organ gets >= ~2.5 s of real time
    n_elites = int(n_segments) if n_segments else max(4, min(len(elites), int(length / 8)))
    idx = np.linspace(0, len(elites) - 1, n_elites).astype(int)
    elites = [elites[i] for i in idx]
    # explicit timing: each elite owns an equal slice, split across its organs
    plan = []
    for k, e in enumerate(elites):
        e_start = k * length / n_elites
        e_end = (k + 1) * length / n_elites
        organs = e["genome"]["organs"]
        for j, seg in enumerate(organs):
            plan.append(dict(seg, params=dict(seg["params"]),
                             lowpoly=e["genome"].get("lowpoly"),
                             start=e_start + (e_end - e_start) * j / len(organs),
                             end=e_start + (e_end - e_start) * (j + 1) / len(organs)))
    if verbose:
        print(f"[womb] walking {len(plan)} segments from {len(elites)} elites "
              f"({length / n_elites:.1f}s per elite)")
    return _gen_with_lenses(plan, length, out, size, fps, seed, corpus, verbose)


def _gen_with_lenses(plan, length, out, size, fps, seed, corpus, verbose):
    """generate(), but every segment carries its own lowpoly lens."""
    from .organs import make_organ as _mk
    from .writer import AVWriter
    from .lowpoly import LOWPOLY_DEFAULTS
    import time as _t

    plan = normalize_plan(plan, float(length))
    n_frames = int(round(length * fps))
    writer = AVWriter(out, size[0], size[1], fps, sr=SR)
    seg_starts = np.array([s["start"] for s in plan])
    seg_i, current, lens = -1, None, None
    t0 = _t.time()
    for i in range(n_frames):
        t = i / fps
        k = int(np.searchsorted(seg_starts, t, side="right") - 1)
        k = max(0, min(k, len(plan) - 1))
        if k != seg_i:
            seg = plan[k]
            seg_i = k
            current = _mk(seg["organ"], size[0], size[1], fps,
                          seed=seed + 977 * k, corpus=corpus, **seg.get("params", {}))
            lp = dict(LOWPOLY_DEFAULTS)
            lp.update(seg.get("lowpoly") or {})
            lens = (lp, np.random.default_rng(seed + 31 * k))
            if verbose:
                print(f"[womb] {t:6.1f}s  {seg['organ']}  lens grid={lp['grid']}", flush=True)
        current.step(i)
        a = current.emit()
        frame = current.render()
        lp, lrng = lens
        frame = lowpoly_frame(frame, energy=float(np.sqrt((a ** 2).mean()) * 4), rng=lrng, **lp)
        writer.write(frame, a)
        if verbose and i % max(1, n_frames // 10) == 0:
            print(f"[womb]   frame {i}/{n_frames}", flush=True)
    peak = float(np.abs(writer.audio).max())
    if peak > 1.0:
        writer.audio = writer.audio / peak * 0.95
    stats = writer.close()
    stats["elapsed"] = round(_t.time() - t0, 1)
    return stats