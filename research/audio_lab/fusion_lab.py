#!/usr/bin/env python3
"""
fusion_lab.py — the clubbed tracks conduct everything (50-piece edition).

The clubbed pieces (audio_lab/clubbed/) are the rhythm spine — their
manifests log the exact beat grid (slot times, BPMs, genres) and stutter
moments. fusion_lab layers material from grammar_out / terrain_out /
structure_out / curated_corpus ON THAT GRID, in TEN recipes — and every
piece draws its own randomised settings (grids, gains, ducks, densities,
folder orders, path shapes), so 50 pieces explore the combination space:

  beatlock        16th/8th/quarter-locked interleave, euclid-gated
  dream_dj        grammar phrases stretched to exact slot bar counts
  map_dialog      beat-locked t-SNE walk, cross-folder call-and-response
  stutter_melt    every clubbed stutter doubles into a foreign layer
  structure_cover clubbed structure re-covered; original groove reveals
  soup            everything beat-locked, drifting layer weights
  reverse_cover   a TERRAIN walk's structure refilled with clubbed slices
  duet            two grammar dreams beat-matched to the same slot, L/R
  stutter_lead    dense foreign stutters as the lead, clubbed underneath
  map_dj          the dream sequence ordered by position on the landscape

Usage:
    python3 fusion_lab.py -o audio_lab/fusion_pieces --pieces 50 --seed 21
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.manifold import TSNE

import granular_lab as gl
import structure_lab as sl
import curate_lab as cl

SR = 44100
EPS = 1e-10
FOLDERS = dict(
    grammar="audio_lab/grammar_out",
    terrain="audio_lab/terrain_out",
    structure="audio_lab/structure_out",
)
MAP_CACHE = {}


# ------------------------------------------------------------------ helpers

def _stereo(y):
    y = np.asarray(y, np.float32)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    elif y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return np.ascontiguousarray(y[:, :2], np.float32)


def load_units(folder, rng, chop=(3.0, 12.0), per_cap=40):
    units = []
    for f in sorted(Path(folder).glob("*.wav")):
        try:
            y, sr = sf.read(str(f), always_2d=True)
        except Exception:
            continue
        y = _stereo(y)
        if sr != SR:
            n = int(len(y) * SR / sr)
            y = np.stack([np.interp(np.linspace(0, len(y) - 1, n),
                                     np.arange(len(y)), y[:, c])
                          for c in range(2)], axis=1)
        rms = float(np.sqrt(np.mean(y ** 2) + EPS))
        y = y * min(12.0, 0.15 / max(rms, EPS))
        if len(y) <= chop[1] * SR:
            units.append(dict(data=y, tag=folder))
            continue
        pos, cnt = 0, 0
        while pos < len(y) - SR // 2 and cnt < per_cap:
            m = int(rng.uniform(*chop) * SR)
            seg = y[pos:pos + m]
            if len(seg) > SR // 2:
                units.append(dict(data=seg, tag=folder))
            pos += m
            cnt += 1
    return units


def curated_unit(path, cap_s=40.0):
    try:
        y, sr = sf.read(str(path), frames=int(cap_s * SR), always_2d=True)
    except Exception:
        return None
    y = _stereo(y)
    rms = float(np.sqrt(np.mean(y ** 2) + EPS))
    return dict(data=y * min(12.0, 0.15 / max(rms, EPS)))


def fade(y, s=0.03):
    f = min(len(y) // 4, int(s * SR))
    if f > 1:
        y[:f] *= np.linspace(0, 1, f)[:, None]
        y[-f:] *= np.linspace(1, 0, f)[:, None]
    return y


def place(events, unit, t, gain, rng, pan=None, stretch_dur=None):
    y = unit["data"]
    if stretch_dur is not None and len(y) > 0.2 * SR:
        speed = float(np.clip(len(y) / (stretch_dur * SR), 0.4, 2.5))
        y2 = gl.resample_grain(y, speed, "hermite")
        y = y2[: int(stretch_dur * SR)] if len(y2) > stretch_dur * SR else \
            np.pad(y2, ((0, int(stretch_dur * SR) - len(y2)), (0, 0)))
    y = fade(y.copy())
    pan = rng.uniform(-0.7, 0.7) if pan is None else pan
    th = (pan + 1) * np.pi / 4
    y = y * np.array([np.cos(th), np.sin(th)], np.float32) * gain
    events.append((max(0, int(t * SR)), np.ascontiguousarray(y, np.float32)))


def mixdown(track, events, track_gain=1.0, duck_depth=0.3, duck_tau=0.15,
            duck_times=None, normalize=True):
    L = len(track)
    out = track.astype(np.float32) * track_gain
    if duck_depth > 0 and duck_times:
        env = np.ones(L, np.float32)
        t = np.arange(L) / SR
        for tk in duck_times:
            i0 = int(tk * SR)
            i1 = min(L, i0 + int(0.35 * SR))
            if i0 >= L:
                continue
            seg = np.exp(-(t[i0:i1] - tk) / duck_tau)
            env[i0:i1] = np.minimum(env[i0:i1], 1.0 - duck_depth * seg)
        out = out * env[:, None]
    for t, y in events:
        i0 = int(t * SR) if not isinstance(t, (int, np.integer)) else t
        if i0 >= L:
            continue
        e = min(L, i0 + len(y))
        out[i0:e] += y[: e - i0]
    f = int(0.02 * SR)
    out[:f] *= np.linspace(0, 1, f)[:, None]
    out[-f:] *= np.linspace(1, 0, f)[:, None]
    if normalize:
        pk = np.abs(out).max()
        if pk > 1e-4:
            out *= 0.97 / pk
    return out


def load_clubbed(clubbed_dir, index):
    files = sorted(Path(clubbed_dir).glob("*.wav"))
    man = json.load(open(Path(clubbed_dir) / "manifest.json"))
    wav = files[index % len(files)]
    entry = man[[k for k in man if k.endswith(wav.name)][0]]
    track, sr = sf.read(str(wav), always_2d=True)
    track = _stereo(track)
    if sr != SR:
        n = int(len(track) * SR / sr)
        track = np.stack([np.interp(np.linspace(0, len(track) - 1, n),
                                     np.arange(len(track)), track[:, c])
                          for c in range(2)], axis=1).astype(np.float32)
    return track, entry, wav


def grid_bars(entry, track_dur):
    bars = []
    for s in entry["slots"]:
        bar_dur = 16 * 60.0 / s["bpm"] / 4.0
        for b in range(s["bars"]):
            t0 = s["t"] + b * bar_dur
            if t0 >= track_dur - 0.05:
                break
            bars.append(dict(t=t0, t1=t0 + bar_dur, bpm=s["bpm"],
                             genre=s["genre"]))
    return bars


def build_map(pools, corpus_dir):
    """One shared t-SNE landscape: curated + grammar + terrain + structure."""
    if MAP_CACHE.get("emb") is not None:
        return MAP_CACHE
    pts, tags = [], []
    for f in sorted(Path(corpus_dir).glob("*.wav")):
        y, sr = sf.read(str(f), frames=int(45 * 44100), always_2d=True)
        v = cl.views(_stereo(y), sr)
        if v is not None and np.isfinite(v).all():
            pts.append(v)
            tags.append(("curated", str(f)))
    for tag, pool in pools.items():
        for i, u in enumerate(pool):
            v = cl.views(u["data"], SR)
            if v is not None and np.isfinite(v).all():
                pts.append(v)
                tags.append((tag, i))
    X = np.stack(pts)
    emb = TSNE(n_components=2, perplexity=12, init="pca",
               random_state=17).fit_transform(X)
    MAP_CACHE.update(emb=emb, tags=tags)
    return MAP_CACHE


def unit_of(tag_key, mp, pools):
    """Material for a map tag (pools index or curated path)."""
    tag, val = tag_key
    if tag == "curated":
        u = curated_unit(val)
        return u if u is not None else pools["grammar"][0]
    return pools[tag][val]


# ------------------------------------------------------------------ recipes
# every recipe draws its own settings and returns (audio, desc, settings)

def r_beatlock(track, entry, ctx, rng):
    s = dict(grid=int(rng.choice(np.array([4, 8, 16]))),
             k=int(rng.integers(3, 11)), skip=float(rng.uniform(0.2, 0.7)),
             gain=float(rng.uniform(0.25, 0.6)),
             track_gain=float(rng.uniform(0.7, 1.0)),
             duck=float(rng.uniform(0.2, 0.45)),
             rotate=bool(rng.random() < 0.5))
    keys = [k for k in ctx["pools"]]
    order = [str(x) for x in rng.permutation(keys)]
    bars = grid_bars(entry, len(track) / SR)
    events, kicks = [], []
    bi = 0
    step = 16 // s["grid"]
    for b in bars:
        pool = ctx["pools"][order[bi % len(order)]]
        bi += 1
        gate = gl._euclid_bits(s["k"], 16)
        if s["rotate"]:
            r = int(rng.integers(16))
            gate = gate[r:] + gate[:r]
        for s16 in range(0, 16, step):
            if not gate[(s16 + bi) % 16] or rng.random() < s["skip"]:
                continue
            t = b["t"] + s16 * (b["t1"] - b["t"]) / 16.0
            u = pool[int(rng.integers(len(pool)))]
            place(events, u, t, s["gain"] * rng.uniform(0.7, 1.3), rng)
            if s16 == 0:
                kicks.append(t)
    audio = mixdown(track, events, track_gain=s["track_gain"],
                   duck_depth=s["duck"], duck_times=kicks)
    return audio, f"beatlock: {s['grid']}-grid, euclid k={s['k']}, pools {'/'.join(order)}", s


def r_dream_dj(track, entry, ctx, rng):
    s = dict(take_bars=int(rng.integers(3, 12)),
             gain=float(rng.uniform(0.5, 0.95)),
             accent_p=float(rng.uniform(0.3, 0.9)),
             track_gain=float(rng.uniform(0.85, 1.0)),
             duck=float(rng.uniform(0.35, 0.55)),
             mix=bool(rng.random() < 0.4))
    bars = grid_bars(entry, len(track) / SR)
    events, kicks = [], []
    dream_pool = ctx["pools"]["grammar"] + \
        (ctx["pools"]["structure"] if s["mix"] else [])
    accents = ctx["pools"]["terrain"] + ctx["pools"]["structure"]
    bar_i = 0
    while bar_i < len(bars) - 1:
        take = s["take_bars"]
        grp = bars[bar_i: bar_i + take]
        if not grp:
            break
        bar_i += take
        t0, dur = grp[0]["t"], grp[-1]["t1"] - grp[0]["t"]
        u = dream_pool[int(rng.integers(len(dream_pool)))]
        place(events, u, t0, s["gain"], rng, pan=rng.uniform(-0.2, 0.2),
              stretch_dur=dur)
        kicks.append(t0)
        if rng.random() < s["accent_p"] and accents:
            acc = accents[int(rng.integers(len(accents)))]
            place(events, acc, grp[min(1, len(grp) - 1)]["t"],
                  s["gain"] * 0.6, rng)
    return mixdown(track, events, track_gain=s["track_gain"],
                   duck_depth=s["duck"], duck_times=kicks), \
        f"dream_dj: {s['take_bars']}-bar phrases{' +structure' if s['mix'] else ''}, gain {s['gain']:.2f}", s


def r_map_dialog(track, entry, ctx, rng):
    s = dict(kind=str(rng.choice(np.array(["lissajous", "spiral", "walk"]))),
             step=int(rng.choice(np.array([1, 2]))),
             gain=float(rng.uniform(0.45, 0.7)),
             resp_gain=float(rng.uniform(0.3, 0.55)),
             freq=float(rng.uniform(1.0, 3.0)),
             track_gain=float(rng.uniform(0.8, 1.0)))
    mp = build_map(ctx["pools"], ctx["corpus"])
    emb, tags = mp["emb"], mp["tags"]
    bars = grid_bars(entry, len(track) / SR)
    stops = bars[::s["step"]]
    t = np.linspace(0, 1, len(stops))
    if s["kind"] == "lissajous":
        a, b = int(rng.integers(1, 4)), int(rng.integers(2, 5))
        p = np.stack([np.sin(a * t * 2 * np.pi + 0.7),
                      np.sin(b * t * 2 * np.pi)], 1)
    elif s["kind"] == "spiral":
        r = 1 - 0.8 * t
        f = s["freq"]
        p = np.stack([r * np.cos(t * 2 * np.pi * f), r * np.sin(t * 2 * np.pi * f)], 1)
    else:
        v = np.cumsum(rng.standard_normal((len(t), 2)) * 0.02, axis=0)
        v -= v.min(axis=0)
        v /= (v.max(axis=0) + EPS)
        p = 2 * v - 1
    p = p * (emb.max(axis=0) * 0.45) + emb.mean(axis=0)
    events, kicks = [], []
    for i, b in enumerate(stops):
        d = np.linalg.norm(emb - p[i], axis=1)
        j = int(np.argmin(d))
        u = unit_of(tags[j], mp, ctx["pools"])
        dur = b["t1"] - b["t"]
        place(events, u, b["t"], s["gain"], rng, stretch_dur=dur)
        if rng.random() < 0.7 and i + 1 < len(bars):
            other = [q for q in range(len(tags)) if tags[q][0] != tags[j][0]]
            cand = sorted(other, key=lambda q: d[q])[:80]
            q = cand[int(rng.integers(min(6, len(cand))))]
            u2 = unit_of(tags[q], mp, ctx["pools"])
            place(events, u2, bars[i + 1]["t"], s["resp_gain"], rng,
                  stretch_dur=bars[i + 1]["t1"] - bars[i + 1]["t"])
        kicks.append(b["t"])
    return mixdown(track, events, track_gain=s["track_gain"],
                   duck_depth=0.3, duck_times=kicks), \
        f"map_dialog: {s['kind']} walk, every {s['step']} bar(s), cross-folder responses", s


def r_stutter_melt(track, entry, ctx, rng):
    s = dict(gain=float(rng.uniform(0.5, 0.85)),
             every=int(rng.choice(np.array([2, 4, 6, 8]))),
             inter_gain=float(rng.uniform(0.25, 0.5)),
             track_gain=float(rng.uniform(0.9, 1.0)),
             hard_pan=bool(rng.random() < 0.5))
    events, kicks = [], []
    pnames = list(ctx["pools"])
    for st in entry["stutters"]:
        pool = ctx["pools"][pnames[int(rng.integers(len(pnames)))]]
        u = pool[int(rng.integers(len(pool)))]
        sl = int(0.05 * SR)
        if len(u["data"]) < sl * 2 or st["t"] * SR >= len(track):
            continue
        seg = u["data"][:sl].copy()
        want = int(st["dur"] * SR)
        reps, total, rep = [seg], len(seg), seg
        mode = st.get("mode", "const")
        while total < want:
            if mode == "ladder":
                rep = gl.resample_grain(rep, 1.059, "hermite")
            elif mode == "ramp" and rng.random() < 0.3:
                rep = seg[int(np.clip(len(seg) * 0.85, 256, len(seg) - 1)):]
            reps.append(rep)
            total += len(rep)
        block = fade(np.concatenate(reps)[:want], 0.01)
        pan = rng.choice([-0.75, 0.75]) if s["hard_pan"] else None
        place(events, dict(data=block), st["t"], s["gain"], rng, pan=pan)
        kicks.append(st["t"])
    bars = grid_bars(entry, len(track) / SR)
    for b in bars[::s["every"]]:
        pool = ctx["pools"][pnames[int(rng.integers(len(pnames)))]]
        u = pool[int(rng.integers(len(pool)))]
        place(events, u, b["t"], s["inter_gain"], rng)
        kicks.append(b["t"])
    return mixdown(track, events, track_gain=s["track_gain"],
                   duck_depth=0.35, duck_times=kicks), \
        f"stutter_melt: every {s['every']} bars interleave, foreign gain {s['gain']:.2f}", s


def r_structure_cover(track, entry, ctx, rng):
    s = dict(n_sections=int(rng.integers(4, 8)),
             cover_gain=float(rng.uniform(0.35, 0.65)),
             reveal_at=float(rng.uniform(0.6, 0.85)),
             reveal_gain=float(rng.uniform(0.4, 0.8)),
             order=[str(x) for x in rng.permutation(list(ctx["pools"]))])
    secs = sl.extract_structure(entry["_path"], n_sections=s["n_sections"])
    dur = len(track) / SR
    events, kicks = [], []
    keys = s["order"]
    t0 = 0.0
    for si, sec in enumerate(secs):
        s_dur = sec["frac"] * dur
        pool = ctx["pools"][keys[si % len(keys)]]
        step = max(2.0, s_dur / max(1, int(rng.integers(3, 7))))
        tt = t0
        while tt < t0 + s_dur - 0.5:
            u = pool[int(rng.integers(len(pool)))]
            place(events, u, tt, s["cover_gain"] * (0.4 + 0.6 * sec["energy"]),
                  rng, stretch_dur=min(step, t0 + s_dur - tt))
            kicks.append(tt)
            tt += step
        t0 += s_dur
    out = mixdown(track, events, track_gain=0.35, duck_depth=0.2,
                  duck_times=kicks, normalize=False)
    w = np.zeros(len(out), np.float32)
    rs = int(dur * s["reveal_at"] * SR)
    ramp = max(1, min(int(3 * SR), len(out) - rs))
    w[rs:rs + ramp] = np.linspace(0, 1, ramp)
    w[rs + ramp:] = 1.0
    out = out + track[:len(out)] * (s["reveal_gain"] * w)[:, None]
    pk = np.abs(out).max()
    if pk > 1e-4:
        out *= 0.97 / pk
    return out, f"structure_cover: {len(secs)} sections from {s['n_sections']}-way, reveal at {s['reveal_at']:.0%}", s


def r_soup(track, entry, ctx, rng):
    s = dict(freq=[float(rng.uniform(0.02, 0.5)) for _ in ctx["pools"]],
             gain=float(rng.uniform(0.25, 0.6)),
             grid=int(rng.choice(np.array([8, 16]))),
             track_gain=float(rng.uniform(0.8, 1.0)))
    bars = grid_bars(entry, len(track) / SR)
    events, kicks = [], []
    keys = list(ctx["pools"])
    phases = {k: float(rng.uniform(0, 2 * np.pi)) for k in keys}
    for bi, b in enumerate(bars):
        p = bi / max(1, len(bars))
        for ki, k in enumerate(keys):
            w = 0.5 + 0.5 * np.sin(2 * np.pi * s["freq"][ki] * p + phases[k])
            if w < 0.35 or rng.random() > w:
                continue
            step = 16 // s["grid"]
            s16 = int(rng.integers(0, 16, )) * 1
            s16 = int(rng.choice(np.arange(0, 16, step)))
            t = b["t"] + s16 * (b["t1"] - b["t"]) / 16
            u = ctx["pools"][k][int(rng.integers(len(ctx["pools"][k])))]
            place(events, u, t, s["gain"] * (0.5 + 0.5 * w), rng)
            if s16 == 0:
                kicks.append(t)
    return mixdown(track, events, track_gain=s["track_gain"],
                   duck_depth=0.3, duck_times=kicks), \
        f"soup: {s['grid']}-grid, weights drifting at {[round(f,2) for f in s['freq']]}", s


def r_reverse_cover(track, entry, ctx, rng):
    """A TERRAIN walk's (or structure clone's) structure, refilled with
    slices of the clubbed track itself + folder accents. No bed: the beat
    appears only through its own sliced material."""
    src = str(rng.choice(ctx["terrain_files"] + ctx["structure_files"]))
    s = dict(source=Path(src).stem, n_sections=int(rng.integers(4, 7)),
             slice_lo=float(rng.uniform(1.0, 2.5)),
             slice_hi=float(rng.uniform(3.0, 6.0)),
             accent_p=float(rng.uniform(0.2, 0.6)))
    secs = sl.extract_structure(src, n_sections=s["n_sections"])
    dur = len(track) / SR
    events, kicks = [], []
    pool_names = list(ctx["pools"])
    t0 = 0.0
    for si, sec in enumerate(secs):
        s_dur = sec["frac"] * dur
        acc_pool = ctx["pools"][pool_names[si % len(pool_names)]]
        tt = t0
        while tt < t0 + s_dur - 0.5:
            m = int(rng.uniform(s["slice_lo"], s["slice_hi"]) * SR)
            st = int(rng.integers(0, max(1, len(track) - m)))
            u = dict(data=fade(track[st:st + m].copy()))
            place(events, u, tt, 0.55 * (0.4 + 0.6 * sec["energy"]), rng)
            kicks.append(tt)
            if rng.random() < s["accent_p"]:
                acc = acc_pool[int(rng.integers(len(acc_pool)))]
                place(events, acc, tt + rng.uniform(0, m / SR), 0.35, rng)
            tt += m / SR * rng.uniform(0.7, 0.95)
        t0 += s_dur
    audio = mixdown(np.zeros_like(track), events, track_gain=0.0,
                   duck_depth=0, duck_times=None)
    return audio, f"reverse_cover: structure of {s['source']}, refilled with clubbed slices + accents", s


def r_duet(track, entry, ctx, rng):
    s = dict(pan=float(rng.uniform(0.6, 0.9)),
             gain=float(rng.uniform(0.5, 0.8)),
             offset=bool(rng.random() < 0.5),
             track_gain=float(rng.uniform(0.7, 0.95)))
    bars = grid_bars(entry, len(track) / SR)
    events, kicks = [], []
    pool = ctx["pools"]["grammar"]
    bar_i = 0
    while bar_i < len(bars) - 1:
        take = int(rng.integers(4, 9))
        grp = bars[bar_i:bar_i + take]
        if len(grp) < 2:
            break
        bar_i += take
        t0, dur = grp[0]["t"], grp[-1]["t1"] - grp[0]["t"]
        a = pool[int(rng.integers(len(pool)))]
        b = pool[int(rng.integers(len(pool)))]
        place(events, a, t0, s["gain"], rng, pan=-s["pan"], stretch_dur=dur)
        t1 = grp[min(1, len(grp) - 1)]["t"] if s["offset"] else t0
        place(events, b, t1, s["gain"] * 0.9, rng, pan=s["pan"],
              stretch_dur=dur)
        kicks.append(t0)
    return mixdown(track, events, track_gain=s["track_gain"],
                   duck_depth=0.3, duck_times=kicks), \
        f"duet: two dreams per slot, pans ±{s['pan']:.2f}, {'offset' if s['offset'] else 'aligned'}", s


def r_stutter_lead(track, entry, ctx, rng):
    """Dense foreign stutters as the LEAD; the clubbed track ducks under."""
    s = dict(every=int(rng.choice(np.array([2, 3, 4]))),
             gain=float(rng.uniform(0.65, 0.9)),
             track_gain=float(rng.uniform(0.5, 0.7)),
             dur_lo=float(rng.uniform(0.25, 0.5)),
             dur_hi=float(rng.uniform(0.8, 1.6)))
    bars = grid_bars(entry, len(track) / SR)
    events, kicks = [], []
    pnames = list(ctx["pools"])
    for bi, b in enumerate(bars):
        if bi % s["every"] != 0 or rng.random() < 0.25:
            continue
        pool = ctx["pools"][pnames[int(rng.integers(len(pnames)))]]
        u = pool[int(rng.integers(len(pool)))]
        sl = float(rng.choice(np.array([0.04, 0.06, 0.1, 0.15])))
        i0 = int(u["data"].shape[0] * rng.uniform(0, 0.8))
        seg = u["data"][i0: i0 + int(sl * SR)]
        if len(seg) < 256:
            continue
        want = int(rng.uniform(s["dur_lo"], s["dur_hi"]) * SR)
        reps, total, rep = [seg], len(seg), seg
        while total < want:
            rep = gl.resample_grain(rep, float(rng.choice(
                np.array([1.059, 1.0, 0.944]))), "hermite")
            reps.append(rep)
            total += len(rep)
        block = fade(np.concatenate(reps)[:want], 0.008)
        place(events, dict(data=block), b["t"], s["gain"], rng,
              pan=rng.uniform(-0.6, 0.6))
        kicks.append(b["t"])
    return mixdown(track, events, track_gain=s["track_gain"],
                   duck_depth=0.4, duck_times=kicks), \
        f"stutter_lead: foreign stutter every {s['every']} bars, lead gain {s['gain']:.2f}, clubbed ducked to {s['track_gain']:.2f}", s


def r_map_dj(track, entry, ctx, rng):
    """Dream sequence ordered by position on the shared landscape."""
    s = dict(clock=bool(rng.random() < 0.5),
             take_bars=int(rng.integers(3, 8)),
             gain=float(rng.uniform(0.55, 0.85)),
             track_gain=float(rng.uniform(0.85, 1.0)))
    mp = build_map(ctx["pools"], ctx["corpus"])
    emb, tags = mp["emb"], mp["tags"]
    gi = [i for i, t in enumerate(tags) if t[0] == "grammar"]
    cen = emb.mean(axis=0)
    if s["clock"]:
        ang = np.arctan2(emb[gi][:, 1] - cen[1], emb[gi][:, 0] - cen[0])
        order = [gi[i] for i in np.argsort(ang)]
    else:
        order = sorted(gi, key=lambda i: emb[i, 0])
    order = order + order[:4]
    bars = grid_bars(entry, len(track) / SR)
    events, kicks = [], []
    bar_i, oi = 0, 0
    while bar_i < len(bars) - 1:
        grp = bars[bar_i: bar_i + s["take_bars"]]
        if not grp:
            break
        bar_i += s["take_bars"]
        tag = tags[order[oi % len(order)]]
        oi += 1
        u = unit_of(tag, mp, ctx["pools"])
        t0, dur = grp[0]["t"], grp[-1]["t1"] - grp[0]["t"]
        place(events, u, t0, s["gain"], rng, stretch_dur=dur)
        kicks.append(t0)
    return mixdown(track, events, track_gain=s["track_gain"],
                   duck_depth=0.4, duck_times=kicks), \
        f"map_dj: dreams sequenced {'around' if s['clock'] else 'across'} the landscape in order", s


RECIPES = [
    ("beatlock", r_beatlock), ("dream_dj", r_dream_dj),
    ("map_dialog", r_map_dialog), ("stutter_melt", r_stutter_melt),
    ("structure_cover", r_structure_cover), ("soup", r_soup),
    ("reverse_cover", r_reverse_cover), ("duet", r_duet),
    ("stutter_lead", r_stutter_lead), ("map_dj", r_map_dj),
]


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clubbed", default="audio_lab/clubbed")
    ap.add_argument("--corpus", default="audio_lab/curated_corpus")
    ap.add_argument("-o", "--out", default="audio_lab/fusion_pieces")
    ap.add_argument("--pieces", type=int, default=50)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    rng = np.random.default_rng(args.seed)
    pools = {}
    for tag, folder in FOLDERS.items():
        pools[tag] = load_units(folder, rng)
        print(f"  {tag}: {len(pools[tag])} units")
    ctx = dict(pools=pools, corpus=args.corpus,
               terrain_files=sorted(str(p) for p in
                                    Path(FOLDERS["terrain"]).glob("*.wav")),
               structure_files=sorted(str(p) for p in
                                     Path(FOLDERS["structure"]).glob("*.wav")))
    n_spines = len(list(Path(args.clubbed).glob("*.wav")))
    if n_spines == 0:
        ap.error("no clubbed tracks found")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    mpath = outdir / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.load(open(mpath))
        except Exception:
            manifest = {}

    for i in range(args.pieces):
        name, fn = RECIPES[i % len(RECIPES)]
        pseed = args.seed + 7919 * i + 9
        prng = np.random.default_rng(pseed)
        spine_i = int((i * 7) % n_spines)
        track, entry, wav = load_clubbed(args.clubbed, spine_i)
        entry["_path"] = str(Path(args.clubbed) / wav.name)
        audio, desc, s = fn(track, entry, ctx, prng)
        out = outdir / f"fusion{i:02d}_{name}_{pseed}.wav"
        gl.write_wav(out, SR, audio)
        manifest[str(out)] = dict(name=name, seed=pseed, spine=str(wav),
                                  settings=s, description=desc)
        if (i + 1) % 10 == 0 or i == args.pieces - 1:
            print(f"  {i + 1}/{args.pieces} done")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- diversity report + listening order
    import warnings
    warnings.filterwarnings("ignore")
    outs = sorted(outdir.glob("fusion*.wav"))
    X = []
    for f in outs:
        y, sr = sf.read(str(f), always_2d=True)
        v = cl.views(y, sr)
        X.append(v if v is not None else np.zeros(84))
    X = np.stack(X)
    med = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    Xw = (X - med) / iqr
    d = np.linalg.norm(Xw[:, None, :] - Xw[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    order = np.argsort(-nn)          # most unusual first
    recipes_used = {}
    for f in outs:
        e = manifest.get(str(f), {})
        recipes_used[e.get("name", "?")] = recipes_used.get(e.get("name", "?"), 0) + 1
    print(f"\n{len(outs)} pieces | {len(RECIPES)} recipes | {n_spines} spines")
    print(f"recipes: {recipes_used}")
    print(f"pairwise spread (whitened): mean {d[np.isfinite(d)].mean():.2f}, "
          f"NN distance med {np.median(nn):.2f}, min {nn.min():.2f}")
    with open(outdir / "listening_order.json", "w") as f:
        json.dump([outs[int(j)].name for j in order], f, indent=1)
    print(f"listening order (most unusual first) -> listening_order.json")
    print(f"done -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())