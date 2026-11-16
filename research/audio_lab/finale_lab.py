#!/usr/bin/env python3
"""
finale_lab.py — final 3-minute review pieces.

Every stage of the new stack in one instrument:
  curated corpus units  (curate_lab  - the spread set, onset-chunked)
  grammar dreams        (grammar_lab - codebook-LM phrases of the corpus)
  flywheel dreams       (alien-python 39 run on OUR curated corpus)
  binary chunks         (binary_lab  - sonified/stamped binaries)
  cloned structures     (structure_lab - selfsim structure of reference pieces)
  terrain walks         (terrain_lab - the archive as a landscape)

Six recipes, each ~3 minutes, written to audio_lab/final_review/ with a
listening manifest. Every piece is documented in manifest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import granular_lab as gl
import alien_rhythms as ar
import rhythm_lab as rl
import structure_lab as sl
import terrain_lab as tl

SR = 44100
EPS = 1e-10


# ------------------------------------------------------------------ pools

def _prep(y, rng):
    """Normalize to a consistent rms; force stereo."""
    y = np.asarray(y, np.float32)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    elif y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    rms = float(np.sqrt(np.mean(y ** 2) + EPS))
    if rms > EPS:
        y = y * min(12.0, 0.15 / rms)
    return y


def pool_corpus(corpus_dir, rng, per_file=6, cap=900):
    files = sorted(Path(corpus_dir).glob("*.wav"))
    _, chunks = rl.extract_pool(files, rng, pool_cap=cap, per_file=per_file)
    units = [dict(data=_prep(c.data, rng), rms=c.rms, cent=c.centroid)
             for c in chunks if len(c.data) > 4096]
    return units


def pool_folder(folder, rng, min_len=4096, rms_target=0.15):
    units = []
    for f in sorted(Path(folder).glob("*.wav")):
        try:
            y, sr = sf.read(str(f), always_2d=True)
        except Exception:
            continue
        if sr != SR:
            n = int(len(y) * SR / sr)
            y = np.stack([np.interp(np.linspace(0, len(y) - 1, n),
                                    np.arange(len(y)), y[:, c])
                          for c in range(y.shape[1])], axis=1)
        if len(y) < min_len:
            continue
        units.append(dict(data=_prep(y, rng), rms=rms_target, cent=0.0))
    return units


def chop_pool(units, rng, lo=0.4, hi=2.5):
    """Long units -> shorter takes for tighter interleave."""
    out = []
    for u in units:
        y = u["data"]
        if len(y) <= hi * SR:
            out.append(u)
            continue
        pos = 0
        while pos < len(y) - SR // 4:
            m = int(rng.uniform(lo, hi) * SR)
            seg = y[pos:pos + m]
            if len(seg) > SR // 8:
                out.append(dict(data=seg, rms=u["rms"], cent=u["cent"]))
            pos += m
    return out


# ------------------------------------------------------------------ render

def stretch_energy(secs):
    es = np.array([s["energy"] for s in secs])
    lo, hi = float(es.min()), float(es.max())
    out = []
    for s in secs:
        e = 0.12 + 0.88 * ((s["energy"] - lo) / (hi - lo + EPS))
        out.append(dict(frac=s["frac"], energy=float(e),
                        chaos=s["chaos"]))
    return out


def render_sections(secs, pools, target_s, rng, pan_spread=0.5):
    """pools: list of dict(units, weight, gain, name). Structure drives all."""
    secs = stretch_energy(secs)
    events = []
    t0 = 0.0
    for sec in secs:
        dur = sec["frac"] * target_s
        slot = float(np.exp(np.interp(sec["chaos"], [0, 1],
                                       [np.log(1.6), np.log(0.22)])))
        for pi, pool in enumerate(pools):
            w = pool["weight"]
            if w <= 0 or not pool["units"]:
                continue
            spacing = max(0.08, slot * 0.35 / max(0.1, w))
            prof = dict(k=(2, 3 + int(9 * sec["energy"] * w)), val="mid",
                         poly=0.3)
            try:
                pat = ar.normalize_velocities(ar.ev(ar.gen_pattern(rng, prof), rng))
            except Exception:
                continue
            L = len(pat) or 1
            n_slots = max(4, int(dur / spacing))
            take_cap = max(0.25, spacing * rng.uniform(1.0, 2.2))
            for i in range(n_slots):
                val = pat[i % L]
                if val is None or val <= 0.06:
                    continue
                u = pool["units"][int(rng.integers(len(pool["units"])))]
                y = u["data"]
                if len(y) > take_cap * SR:
                    take = int(take_cap * SR)
                    st = int(rng.integers(0, max(1, len(y) - take)))
                    y = y[st: st + take].copy()
                else:
                    y = y.copy()
                if len(y) < 512:
                    continue
                f = min(len(y) // 4, int(0.02 * SR))
                y[:f] *= np.linspace(0, 1, f)[:, None]
                y[-f:] *= np.linspace(1, 0, f)[:, None]
                amp = val * sec["energy"] * pool["gain"]
                pan = np.clip(rng.uniform(-1, 1) * pan_spread, -1, 1)
                th = (pan + 1) * np.pi / 4
                y[:, 0] *= np.cos(th) * amp
                y[:, 1] *= np.sin(th) * amp
                t = t0 + i * spacing + rng.uniform(-1, 1) * 0.01
                events.append((max(0, int(t * SR)), np.ascontiguousarray(y, np.float32)))
        t0 += dur
    return events


def assemble(events, target_s, normalize=True):
    L = max(t + len(y) for t, y in events) if events else 0
    audio = np.zeros((int(target_s * SR) + 2 * SR, 2), np.float32)
    for t, y in events:
        e = min(len(audio), t + len(y))
        audio[t:e] += y[: e - t]
    audio = audio[: max(L, SR)]
    f = int(0.05 * SR)
    audio[:f] *= np.linspace(0, 1, f)[:, None]
    audio[-f:] *= np.linspace(1, 0, f)[:, None]
    if normalize:
        pk = float(np.abs(audio).max())
        if pk > 1e-4:
            audio *= 0.97 / pk
    return audio


# ------------------------------------------------------------------ recipes

def recipe_clone(name, ref_path, pools_spec, target_s, rng, ctx):
    secs = sl.extract_structure(ref_path)
    pools = []
    for key, weight, gain in pools_spec:
        pools.append(dict(name=key, units=ctx[key], weight=weight, gain=gain))
    events = render_sections(secs, pools, target_s, rng)
    desc = (f"structure cloned from {Path(ref_path).name} "
            f"({len(secs)} sections); "
            + ", ".join(f"{p['name']} x{p['weight']:.2f}" for p in pools))
    return assemble(events, target_s), desc


def recipe_walk(name, target_s, rng, ctx):
    emb, names = tl.load_map("audio_lab/curated_corpus", rng)
    files = [Path("audio_lab/curated_corpus") / n for n in names]
    kind = ["lissajous", "spiral", "walk"][int(rng.integers(3))]
    audio = tl.render_walk(emb, files, kind, target_s, rng, None)
    if audio is None:
        return None, "walk failed"
    # dream interjections at path transitions
    extra = []
    for _ in range(int(rng.integers(4, 8))):
        t = float(rng.uniform(5, target_s - 12))
        u = ctx["dreams"][int(rng.integers(len(ctx["dreams"])))]
        y = u["data"]
        take = min(len(y), int(rng.uniform(2, 6) * SR))
        st = int(rng.integers(0, max(1, len(y) - take)))
        seg = y[st:st + take].copy()
        f = min(len(seg) // 4, int(0.1 * SR))
        seg[:f] *= np.linspace(0, 1, f)[:, None]
        seg[-f:] *= np.linspace(1, 0, f)[:, None]
        extra.append((int(t * SR), (seg * 0.5).astype(np.float32)))
    bed = assemble(extra, target_s, normalize=False)
    n = max(len(bed), len(audio))
    if len(bed) < n:
        bed = np.pad(bed, ((0, n - len(bed)), (0, 0)))
    if len(audio) < n:
        audio = np.pad(audio, ((0, n - len(audio)), (0, 0)))
    audio = bed + audio
    pk = np.abs(audio).max()
    if pk > 1e-4:
        audio = audio * (0.97 / pk)
    return audio, f"terrain walk ({kind}) over the curated map + dream interjections"


def recipe_dream_led(name, target_s, rng, ctx):
    """Grammar dreams as the foreground sequence; corpus underlay; accents."""
    dreams = list(ctx["dreams"])
    order = rng.permutation(len(dreams))
    events = []
    t = 0.0
    arc = lambda p: 0.55 + 0.45 * np.sin(np.pi * min(1, p) * 1.5 - 0.25)
    while t < target_s - 4:
        u = dreams[int(order[int(rng.integers(len(order)))])]
        y = u["data"]
        if len(y) > 12 * SR:
            y = y[: int(rng.uniform(6, 12) * SR)]
        f = min(len(y) // 4, int(0.25 * SR))
        y = y.copy()
        y[:f] *= np.linspace(0, 1, f)[:, None]
        y[-f:] *= np.linspace(1, 0, f)[:, None]
        amp = float(arc(t / target_s)) * 0.9
        events.append((int(t * SR), (y * amp).astype(np.float32)))
        # curated underlay between dreams
        n_gap = int(rng.integers(2, 5))
        for _ in range(n_gap):
            v = ctx["curated"][int(rng.integers(len(ctx["curated"])))]
            z = v["data"]
            take = min(len(z), int(rng.uniform(0.5, 2.0) * SR))
            st = int(rng.integers(0, max(1, len(z) - take)))
            z = z[st:st + take].copy() * 0.35
            f2 = min(len(z) // 4, 512)
            z[:f2] *= np.linspace(0, 1, f2)[:, None]
            z[-f2:] *= np.linspace(1, 0, f2)[:, None]
            events.append((int((t + rng.uniform(0.2, len(y)/SR)) * SR),
                           z.astype(np.float32)))
        # flywheel accents
        if rng.random() < 0.5:
            w = ctx["flywheel"][int(rng.integers(len(ctx["flywheel"])))]
            events.append((int((t + rng.uniform(0, len(y)/SR)) * SR),
                          (w["data"] * 0.45).astype(np.float32)))
        t += len(y) / SR + rng.uniform(0.1, 0.6)
    return assemble(events, target_s), \
        "grammar-dream sequence (corpus n-gram LM), curated underlay, flywheel accents"


def recipe_flywheel_lead(name, ref_path, target_s, rng, ctx):
    secs = sl.extract_structure(ref_path)
    pools = [
        dict(name="flywheel", units=ctx["flywheel"], weight=1.0, gain=1.0),
        dict(name="binary", units=ctx["binary"], weight=0.55, gain=0.8),
        dict(name="curated", units=ctx["curated"], weight=0.7, gain=0.7),
    ]
    events = render_sections(secs, pools, target_s, rng, pan_spread=0.7)
    desc = (f"flywheel-dream lead over binary chunks + curated corpus, "
            f"structure cloned from {Path(ref_path).name}")
    return assemble(events, target_s), desc


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="audio_lab")
    ap.add_argument("-o", "--out", default="audio_lab/final_review")
    ap.add_argument("--len", type=float, default=182.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    base = Path(args.base)
    rng = np.random.default_rng(args.seed)
    print("building pools ...")
    ctx = {}
    ctx["curated"] = chop_pool(pool_corpus(base / "curated_corpus", rng,
                                           per_file=6, cap=700), rng,
                               lo=0.3, hi=2.0)
    print(f"  curated units: {len(ctx['curated'])}")
    ctx["dreams"] = pool_folder(base / "grammar_out", rng)
    print(f"  grammar dreams: {len(ctx['dreams'])}")
    # optional extra "flywheel" corpus — set AUDIO_RESEARCH_FLYWHEEL to a dir of
    # out_seed*/ or out/ wav folders to fold it in; skipped if unset/missing.
    fw_dir = Path(os.environ.get("AUDIO_RESEARCH_FLYWHEEL", base / "_flywheel"))
    fw = sorted(fw_dir.glob("out_seed*/*.wav")) + sorted(fw_dir.glob("out/*.wav"))
    fw_units = []
    for f in fw:
        try:
            y, sr = sf.read(str(f), always_2d=True)
        except Exception:
            continue
        if sr != SR:
            n = int(len(y) * SR / sr)
            y = np.stack([np.interp(np.linspace(0, len(y) - 1, n),
                                    np.arange(len(y)), y[:, c])
                          for c in range(y.shape[1])], axis=1)
        fw_units.append(dict(data=_prep(y, rng), rms=0.15, cent=0.0))
    ctx["flywheel"] = fw_units
    print(f"  flywheel dreams: {len(ctx['flywheel'])}")
    bin_files = sorted((base / "binary_pieces").glob("*.wav"))[:6]
    _, bchunks = rl.extract_pool(bin_files, rng, pool_cap=300, per_file=20)
    ctx["binary"] = [dict(data=_prep(c.data, rng), rms=c.rms, cent=c.centroid)
                     for c in bchunks if len(c.data) > 4096]
    print(f"  binary chunks: {len(ctx['binary'])}")

    edm = sorted((base / "edm_pieces").glob("*.wav"))
    storm = sorted((base / "storm_pieces").glob("*.wav"))
    bins = sorted((base / "binary_pieces").glob("*.wav"))
    ref_dnb = str(edm[4]) if len(edm) > 4 else str(edm[0])
    ref_storm = str(storm[3]) if len(storm) > 3 else str(storm[0])
    ref_bin = str(bins[10]) if len(bins) > 10 else str(bins[0])

    recipes = [
        ("final01_clone_dnb", lambda r: recipe_clone(
            "x", ref_dnb, [("curated", 0.60, 0.85), ("dreams", 0.32, 0.9),
                           ("flywheel", 0.14, 0.7)], args.len, r, ctx)),
        ("final02_clone_storm", lambda r: recipe_clone(
            "x", ref_storm, [("curated", 0.80, 0.9), ("dreams", 0.18, 0.8),
                             ("flywheel", 0.10, 0.6)], args.len, r, ctx)),
        ("final03_walk_dreams", lambda r: recipe_walk("x", args.len, r, ctx)),
        ("final04_dream_led", lambda r: recipe_dream_led("x", args.len, r, ctx)),
        ("final05_flywheel_lead", lambda r: recipe_flywheel_lead(
            "x", ref_bin, args.len, r, ctx)),
        ("final06_binary_hybrid", lambda r: recipe_clone(
            "x", ref_bin, [("binary", 0.50, 0.9), ("dreams", 0.25, 0.85),
                           ("curated", 0.35, 0.7)], args.len, r, ctx)),
    ]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i, (name, fn) in enumerate(recipes):
        prng = np.random.default_rng(args.seed + 7919 * i + 5)
        audio, desc = fn(prng)
        if audio is None or len(audio) < SR * 30:
            print(f"  {name}: skipped ({desc})")
            continue
        wav = outdir / f"{name}.wav"
        gl.write_wav(wav, SR, audio)
        print(f"  {name} ({len(audio)/SR/60:.2f} min) -> {wav.name}")
        print(f"     {desc}")
        manifest[str(wav)] = dict(name=name, dur_s=len(audio) / SR,
                                  description=desc, seed=args.seed + 7919 * i + 5)
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())