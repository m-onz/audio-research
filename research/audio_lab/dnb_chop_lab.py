#!/usr/bin/env python3
"""
dnb_chop_lab.py — event-chopped corpus -> drum & bass.

THE CHOP. Every wav ever generated under audio_lab is onset-sliced AT ITS
EVENTS: a slice runs from one transient to the next, so every slice begins
on an attack. Placed on the grid, corpus events land ON the beat — the
lab's own hits become the drum kit. Slices are role-classified (kick /
snare / clap / hat / tom / stab) and cached to a slice library for reuse.

THE TRACKS. 170-176 bpm two-step DnB: kick 1 + 3.5-ish, snares 2 & 4
(cycling several snare slices per bar, amen-style), rolling 16th hats,
ghost chops. Mutations: slice swaps, reverses, pitch ladders, snare rolls
at section ends, sub drops. Sections: intro -> groove -> switch -> break
(halftime, pitched-down dub) -> drop -> outro.

THE NOVELTY MACHINERY (this is where the corpus is used efficiently):
  - per-track KITS are selected by farthest-point selection in whitened
    slice-feature space, with a GLOBAL USAGE PENALTY: slices already used
    by earlier tracks are pushed away, so each new track mines fresh
    material. Across the batch the library gets COVERED, not repeated.
  - after rendering, every track is featurized (84-d) and the batch is
    reported: pairwise spread, coverage, listening order.

Usage:
    python3 dnb_chop_lab.py                     # library + 12 x 3min tracks
    python3 dnb_chop_lab.py --pieces 12 --seed 7
    python3 dnb_chop_lab.py --rebuild           # re-slice the whole corpus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import granular_lab as gl
import rhythm_lab as rl
import alien_rhythms as ar
import curate_lab as cl

SR = 44100
EPS = 1e-10
ROLES = ("kick", "tom", "snare", "clap", "hat", "stab")
# per-folder file caps keep the scan efficient and every generation present
FOLDER_CAPS = {"sounds": 60, "bank": 60, "harmonic_corpus": 80,
               "fusion_harmonic_corpus": 80, "curated_corpus": 80,
               "dnb_slice_library": 0}
DEFAULT_CAP = 90
KIT_SIZES = dict(kick=7, tom=8, snare=11, clap=6, hat=16, stab=10)


# ------------------------------------------------------------------ library

def gather_files(base, exclude):
    files = []
    folders = {}
    for p in sorted(Path(base).rglob("*.wav")):
        if any(part in exclude or part.startswith("dnb_slice_library")
               for part in p.parts):
            continue
        folder = p.parent.name
        folders.setdefault(folder, []).append(p)
    for folder, lst in folders.items():
        cap = FOLDER_CAPS.get(folder, DEFAULT_CAP)
        if cap == 0:
            continue
        if len(lst) > cap:
            stride = int(np.ceil(len(lst) / cap))
            lst = lst[::stride]
        files += lst
    return files


def extract_slices(files, rng, min_grain, max_grain, pool_cap=3000,
                   per_file=200):
    """Event-chop with configurable grain sizes (mirrors rhythm_lab's
    extract_pool, parametrised)."""
    chunks = []
    for path in files:
        try:
            sr, a = gl.read_wav(path)
        except Exception:
            continue
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        elif a.shape[1] > 2:
            a = a[:, :2]
        a = np.ascontiguousarray(a, np.float32)
        segs = gl._onset_segments(a, sr,
                                  dict(min_grain=min_grain, max_grain=max_grain))
        if len(segs) < 16:
            sz = int(max(min_grain * 2, 0.5) * sr)
            hop = int(min_grain * sr)
            segs = [(st, sz) for st in range(0, max(1, len(a) - sz), hop)]
        if len(segs) > per_file:
            segs = segs[:: int(np.ceil(len(segs) / per_file))]
        for st, sz in segs:
            d = a[st:st + sz]
            if d.shape[0] < 256:
                continue
            f = gl.compute_features(d, sr)
            if f["rms"] < 0.008:
                continue
            chunks.append(rl.Chunk(d, f["rms"], d.shape[0] / sr, f["centroid"],
                                   f["flatness"], f["transient"],
                                   Path(path).stem, st / sr))
    if len(chunks) > pool_cap:
        chunks = chunks[:: int(np.ceil(len(chunks) / pool_cap))]
    for attr, key in (("centroid", "bright"), ("flat", "noisy"),
                      ("trans", "trans_rank")):
        vals = np.array([getattr(c, attr) for c in chunks], float)
        order = np.argsort(vals, kind="stable")
        r = np.empty(len(vals))
        r[order] = np.arange(len(vals)) / max(1, len(vals) - 1)
        for c, v in zip(chunks, r):
            setattr(c, key, float(v))
    return chunks


def build_library(files, rng, cache_dir, min_grain=0.07, max_grain=0.5,
                   per_file=140, pool_cap=9000):
    """Onset-chop everything; role-classify; cache as npz."""
    print(f"chopping {len(files)} files at their events "
          f"(grains {min_grain}-{max_grain}s) ...")
    chunks = extract_slices(files, rng, min_grain, max_grain,
                            pool_cap=pool_cap, per_file=per_file)
    pools = rl.assign_roles(chunks)
    slices = []
    for role in ROLES:
        for c in pools.get(role, []):
            slices.append(dict(data=np.ascontiguousarray(c.data, np.float32),
                               role=role, rms=c.rms, cent=c.centroid,
                               flat=c.flat, trans=c.trans, dur=c.dur,
                               src=c.src))
    if not slices:
        return None
    # whiten features for kit selection
    X = np.array([[np.log10(s["cent"] + 1.0), np.log10(s["rms"] + EPS),
                   s["flat"], s["trans"], s["dur"]] for s in slices],
                 np.float32)
    med = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    Xw = (X - med) / iqr
    cache_dir.mkdir(parents=True, exist_ok=True)
    lens = np.array([len(s["data"]) for s in slices], np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)])
    flat = np.concatenate([s["data"] for s in slices])
    np.savez_compressed(cache_dir / "library.npz",
                        data=flat.astype(np.float32), lens=lens, offs=offs,
                        Xw=Xw,
                        role=np.array([ROLES.index(s["role"])
                                       for s in slices]),
                        cent=np.array([s["cent"] for s in slices]),
                        rms=np.array([s["rms"] for s in slices]),
                        dur=np.array([s["dur"] for s in slices]))
    with open(cache_dir / "sources.json", "w") as f:
        json.dump([s["src"] for s in slices], f)
    return dict(data=flat.astype(np.float32), lens=lens, offs=offs, Xw=Xw,
                role=np.array([ROLES.index(s["role"]) for s in slices]),
                cent=np.array([s["cent"] for s in slices]),
                dur=np.array([s["dur"] for s in slices]),
                src=[s["src"] for s in slices])


def load_library(cache_dir):
    cache_dir = Path(cache_dir)
    z = np.load(cache_dir / "library.npz")
    with open(cache_dir / "sources.json") as f:
        src = json.load(f)
    return dict(data=z["data"], lens=z["lens"], offs=z["offs"], Xw=z["Xw"],
                role=z["role"], cent=z["cent"], dur=z["dur"], src=src)


def slice_data(lib, i):
    return lib["data"][lib["offs"][i]: lib["offs"][i] + lib["lens"][i]]


# ------------------------------------------------------------------ kits

def pick_kit(lib, rng, usage, sizes=KIT_SIZES):
    """Farthest-point per role with a global usage penalty -> fresh kits."""
    kit = {}
    Xw = lib["Xw"]
    roles = lib["role"]
    for role, n in sizes.items():
        idx = np.flatnonzero(roles == ROLES.index(role))
        if len(idx) == 0:
            kit[role] = []
            continue
        use = np.array([usage.get(int(i), 0) for i in idx], np.float32)
        seed = int(idx[int(np.argmin(use + rng.random(len(idx)) * 0.5))])
        selected = [seed]
        usage[int(seed)] = usage.get(int(seed), 0) + 1
        mind = np.linalg.norm(Xw[idx] - Xw[seed], axis=1)
        while len(selected) < min(n, len(idx)):
            adj = mind / (1.0 + 2.0 * use)
            j = int(np.argmax(adj))
            pick = int(idx[j])
            selected.append(pick)
            usage[pick] = usage.get(pick, 0) + 1
            d = np.linalg.norm(Xw[idx] - Xw[j], axis=1)
            mind = np.minimum(mind, d)
            use[j] += 1.0
        kit[role] = selected
    return kit


# ------------------------------------------------------------------ track

SECTIONS = [("intro", 10, 0.35), ("groove", 26, 0.65), ("switch", 22, 0.8),
            ("break", 12, 0.35), ("drop", 26, 1.0), ("outro", 8, 0.35)]

KICK_VARS = [[0, 10], [0, 10], [0, 6, 10], [0, 10, 13], [0, 4, 10],
             [0, 10, 11]]


def fade(y, s=0.003):
    f = min(len(y) // 4, int(s * SR))
    if f > 1:
        y[:f] *= np.linspace(0, 1, f)[:, None]
        y[-f:] *= np.linspace(1, 0, f)[:, None]
    return y


def hit(lib, i, rng, vel, semis=0.0, rev=False, gain=1.0, pan=0.0):
    y = slice_data(lib, i).copy()
    if semis != 0.0:
        y = gl.resample_grain(y, float(2 ** (semis / 12.0)), "hermite")
    if rev:
        y = y[::-1].copy()
    y = fade(y)
    th = ((pan + 1) * np.pi / 4)
    return np.ascontiguousarray(
        y * np.array([np.cos(th), np.sin(th)], np.float32) * vel * gain,
        np.float32)


def render_track(lib, kit, rng, target_s, clean=False):
    bpm = float(rng.uniform(170, 176))
    beat = 60.0 / bpm
    step = beat / 4
    bar_dur = 16 * step
    root = float(rng.choice(np.array([36.71, 41.20, 43.65, 49.00, 55.00])))
    # adapt to big slices: thin hats, short rolls
    def med_dur(role):
        return float(np.median([lib["dur"][i] for i in kit[role]])) \
            if kit[role] else 0.0
    hat_stride = 2 if (kit["hat"] and med_dur("hat") > 0.3) else 1
    sn_big = bool(kit["snare"] and med_dur("snare") > 1.2)
    bars_total = int(round(target_s / bar_dur))
    scale = max(1.0, target_s / (104 * bar_dur))
    secs, used_bars = [], 0
    for name, bars, e in SECTIONS:
        bars = max(2, int(round(bars * scale)))
        bars = min(bars, bars_total - used_bars)
        if bars <= 0:
            break
        secs.append((name, bars, e))
        used_bars += bars

    kick_bus, main_bus = [], []
    kick_times = []
    block = dict(kv=[0, 10], hatk=12, hatrot=0, ghk=3, ghrot=0,
                 snare_set=rng.permutation(kit["snare"])[:3].tolist(),
                 pitch_lad=rng.uniform(-4, 4))
    bar_i = 0
    desc = []
    for sname, bars, e in secs:
        for b in range(bars):
            bar_t = bar_i * bar_dur
            if b % 4 == 0:                              # re-roll the block
                block["kv"] = KICK_VARS[int(rng.integers(len(KICK_VARS)))]
                block["hatk"] = int(rng.integers(9, 15))
                block["hatrot"] = int(rng.integers(16))
                block["ghk"] = int(rng.integers(2, 6))
                block["ghrot"] = int(rng.integers(16))
                if len(kit["snare"]) > 2:
                    block["snare_set"] = rng.permutation(
                        kit["snare"])[:3].tolist()
            halftime = sname == "break"
            # ---- kick (event slices from the corpus)
            if not halftime:
                for s in block["kv"]:
                    if not kit["kick"]:
                        break
                    ks = kit["kick"][int(rng.integers(len(kit["kick"])))]
                    semis = 0.0 if clean else (
                        float(rng.choice(np.array(
                            [0, 0, 0, block["pitch_lad"] * 0.5])))
                        if e > 0.7 else 0.0)
                    kick_bus.append((max(0, int((bar_t + s * step) * SR)),
                                     hit(lib, ks, rng, 1.0, semis=semis)))
                    kick_times.append(bar_t + s * step)
                    if not clean and rng.random() < 0.85:
                        kick_bus.append((int((bar_t + s * step) * SR),
                                         ar.sub_thump(SR, 0.8, rng)))
            else:
                if kit["kick"]:
                    ks = kit["kick"][0]
                    kick_bus.append((int(bar_t * SR),
                                     hit(lib, ks, rng, 0.9,
                                         semis=0.0 if clean else -5.0)))
                    kick_times.append(bar_t)
                if not clean:
                    kick_bus.append((int(bar_t * SR),
                                     ar.sub_thump(SR, 0.9, rng)))
            # ---- snare two-step (cycling slices, amen style)
            snare_steps = [8] if halftime else [4, 12]
            for si, s in enumerate(snare_steps):
                if not kit["snare"]:
                    break
                ks = block["snare_set"][si % len(block["snare_set"])]
                rev = False if clean else rng.random() < 0.08
                semis = 0.0 if clean else (
                    float(rng.choice(np.array(
                        [0, 0, 0, 2, -2, block["pitch_lad"]])))
                    if e > 0.6 and rng.random() < 0.3 else 0.0)
                main_bus.append((int((bar_t + s * step) * SR),
                                 hit(lib, ks, rng, 0.95, semis=semis, rev=rev,
                                     pan=rng.uniform(-0.1, 0.1))))
                if not halftime and rng.random() < 0.35:   # ghost snare
                    gs = int(rng.choice([7, 15]))
                    main_bus.append((int((bar_t + gs * step) * SR),
                                     hit(lib, ks, rng, 0.3)))
            # ---- rolling hats (8ths if slices are big)
            if kit["hat"] and not halftime:
                if hat_stride == 2:
                    bits8 = gl._euclid_bits(min(8, max(2, block["hatk"] // 2)), 8)
                    r8 = block["hatrot"] % 8
                    bits8 = bits8[r8:] + bits8[:r8]
                    for i8, on in enumerate(bits8):
                        if not on:
                            continue
                        hs = kit["hat"][int(rng.integers(len(kit["hat"])))]
                        main_bus.append((int((bar_t + i8 * 2 * step) * SR),
                                         hit(lib, hs, rng, 0.5,
                                             pan=(0.5 if i8 % 2 else -0.5))))
                else:
                    bits = gl._euclid_bits(block["hatk"], 16)
                    bits = bits[block["hatrot"] % 16:] + \
                        bits[: block["hatrot"] % 16]
                    for i16, on in enumerate(bits):
                        if not on:
                            continue
                        hs = kit["hat"][int(rng.integers(len(kit["hat"])))]
                        vel = 0.55 if i16 % 4 == 0 else 0.3
                        main_bus.append((int((bar_t + i16 * step) * SR),
                                         hit(lib, hs, rng, vel,
                                             pan=(0.5 if i16 % 2 else -0.5))))
            # ---- ghost chops (tom/stab layer — corpus melodic events)
            if (kit["tom"] or kit["stab"]) and not halftime:
                bits = gl._euclid_bits(block["ghk"], 16)
                bits = bits[block["ghrot"] % 16:] + \
                    bits[: block["ghrot"] % 16]
                for i16, on in enumerate(bits):
                    if not on or rng.random() > 0.4 + 0.4 * e:
                        continue
                    pool = kit["tom"] if (kit["tom"] and rng.random() < 0.5) \
                        else kit["stab"]
                    if not pool:
                        continue
                    gs = pool[int(rng.integers(len(pool)))]
                    main_bus.append((int((bar_t + i16 * step) * SR),
                                     hit(lib, gs, rng, 0.4,
                                         semis=0.0 if clean else
                                         float(rng.choice(np.array(
                                             [0, 0, 5, -5, 7]))),
                                         rev=False if clean else
                                         rng.random() < 0.12,
                                         pan=rng.uniform(-0.4, 0.4))))
            # ---- section-end snare roll (the amen moment; shorter if big)
            if b == bars - 1 and sname in ("groove", "switch", "drop") \
                    and kit["snare"] and rng.random() < 0.6:
                roll_steps = [12, 14] if sn_big else [10, 11, 12, 13]
                for j, s in enumerate(roll_steps):
                    ks = block["snare_set"][j % len(block["snare_set"])]
                    main_bus.append((int((bar_t + s * step) * SR),
                                     hit(lib, ks, rng, 0.5 + 0.12 * j)))
                    desc.append(f"roll@{bar_i}")
            # ---- drop-section sub drops
            if not clean and sname == "drop" and rng.random() < 0.18 \
                    and kit["kick"]:
                ks = kit["kick"][int(rng.integers(len(kit["kick"])))]
                main_bus.append((int(bar_t * SR),
                                 hit(lib, ks, rng, 0.8, semis=-12.0)))
            bar_i += 1
    # ---- assemble (clean mode: pure additive interleave, no ducking)
    L = int((bar_i + 2) * bar_dur * SR)
    audio = np.zeros((L, 2), np.float32)
    if not clean:
        env = np.ones(L, np.float32)
        t_arr = np.arange(L) / SR
        for tk in kick_times:
            i0 = int(tk * SR)
            i1 = min(L, i0 + int(0.25 * SR))
            if i0 >= L:
                continue
            seg = np.exp(-(t_arr[i0:i1] - tk) / 0.10)
            env[i0:i1] = np.minimum(env[i0:i1], 1.0 - 0.25 * seg)
    for bus in (kick_bus, main_bus):
        for t, y in bus:
            i0 = t if isinstance(t, (int, np.integer)) else int(t * SR)
            if i0 >= L:
                continue
            e2 = min(L, i0 + len(y))
            if clean:
                audio[i0:e2] += y[: e2 - i0]
            else:
                audio[i0:e2] += y[: e2 - i0] * env[i0:e2, None]
    f = int(0.05 * SR)
    audio[:f] *= np.linspace(0, 1, f)[:, None]
    audio[-f:] *= np.linspace(1, 0, f)[:, None]
    audio = audio[: int(target_s * SR)]
    pk = np.abs(audio).max()
    if pk > 1e-4:
        audio *= 0.97 / pk
    d = (f"{bpm:.0f}bpm root {root:.0f}Hz, {len(secs)} sections, "
         f"{bar_i} bars, kit " +
         "/".join(f"{r}{len(kit[r])}" for r in ROLES if kit[r]) +
         (f", {len(desc)} rolls" if desc else "") +
         (", CLEAN interleave (no effects)" if clean else ""))
    return audio, d


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="audio_lab")
    ap.add_argument("--corpus", default=None,
                    help="chop this folder's wavs instead of scanning --base")
    ap.add_argument("--min-grain", type=float, default=0.07,
                    help="minimum slice length (s)")
    ap.add_argument("--max-grain", type=float, default=0.5,
                    help="maximum slice length (s)")
    ap.add_argument("--cache", default="audio_lab/dnb_slice_library")
    ap.add_argument("-o", "--out", default="audio_lab/dnb_chops")
    ap.add_argument("--pieces", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clean", action="store_true",
                    help="no effects: natural pitches, no reverses, no sub "
                         "synth, no ducking - pure additive interleave")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-chop the whole corpus even if cached")
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    rng = np.random.default_rng(args.seed)
    cache = Path(args.cache)
    lib = None
    if (cache / "library.npz").exists() and not args.rebuild:
        try:
            lib = load_library(cache)
            print(f"slice library (cached): {len(lib['lens'])} slices")
        except Exception:
            lib = None
    if lib is None:
        if args.corpus:
            files = sorted(Path(args.corpus).glob("*.wav"))
            per_file, pool_cap = 300, 4000
        else:
            files = gather_files(args.base, {Path(args.out).name})
            per_file, pool_cap = 140, 9000
        if not files:
            ap.error("no corpus found")
        lib = build_library(files, rng, cache,
                            min_grain=args.min_grain, max_grain=args.max_grain,
                            per_file=per_file, pool_cap=pool_cap)
    n = len(lib["lens"])
    srcs = lib["src"]
    from collections import Counter
    top = Counter(srcs).most_common(5)
    print(f"library: {n} event slices from {len(set(srcs))} files "
          f"(busiest: {[t[0] for t in top]})")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    mpath = outdir / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.load(open(mpath))
        except Exception:
            manifest = {}
    usage = {}
    kit_union = set()
    for i in range(args.pieces):
        pseed = args.seed + 7919 * i + 37
        prng = np.random.default_rng(pseed)
        kit = pick_kit(lib, prng, usage)
        kit_union.update(sum(kit.values(), []))
        audio, desc = render_track(lib, kit, prng,
                                   float(prng.uniform(178, 192)),
                                   clean=args.clean)
        wav = outdir / f"dnb{i:02d}_{pseed}.wav"
        gl.write_wav(wav, SR, audio)
        print(f"  dnb{i:02d} ({len(audio)/SR/60:.2f} min, kit "
              f"{sum(len(v) for v in kit.values())} slices) -> {wav.name}")
        print(f"     {desc}")
        manifest[str(wav)] = dict(seed=pseed, description=desc,
                                  kit={r: kit[r] for r in ROLES})
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- batch novelty report
    import warnings
    warnings.filterwarnings("ignore")
    outs = sorted(outdir.glob("dnb*.wav"))
    X = []
    for f in outs:
        y, sr = sf.read(str(f), always_2d=True)
        v = cl.views(y, sr)
        X.append(v if v is not None else np.zeros(84))
    X = np.stack(X)
    med = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, axis=0) -
           np.percentile(X, 25, axis=0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    Xw = (X - med) / iqr
    d = np.linalg.norm(Xw[:, None, :] - Xw[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    order = np.argsort(-nn)
    with open(outdir / "listening_order.json", "w") as f:
        json.dump([outs[int(j)].name for j in order], f, indent=1)
    print(f"\nbatch novelty: pairwise spread mean {d[np.isfinite(d)].mean():.2f}, "
          f"NN med {np.median(nn):.2f}, min {nn.min():.2f}")
    print(f"corpus coverage: {len(kit_union)}/{n} slices used by kits "
          f"({100 * len(kit_union) / n:.0f}%)")
    print(f"listening order -> listening_order.json")
    print(f"done -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())