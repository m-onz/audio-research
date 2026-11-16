#!/usr/bin/env python3
"""bomb — the corpus-anchored experiment.

The premise (born from an honest complaint: womb's bred output was
"just crackles and quite boring"): the minimal criterion decides what
evolves, so let YOUR MATERIAL decide it. bomb trims sounds.wav /
sounds2.wav by silence into a corpus, measures the corpus's spectral
character window by window, and uses that character envelope as the
MINIMAL CRITERION for breeding cryptid sound-objects: a candidate is
viable iff it sounds like it could have come from your material —
inside the envelope on enough axes. No fitness, still MCC: the corpus
is the environment (POET's environments, made of your own sounds).

    python3 bomb.py corpus  ../sounds.wav ../sounds2.wav -o corpus
    python3 bomb.py analyze corpus -o analysis.json           # the why
    python3 bomb.py calibrate corpus                          # self-kinship
    python3 bomb.py breed     corpus -o runs/bomb1 --gens 20 -v
    python3 bomb.py mixtape   runs/bomb1 --grid texture --seg 8

Why these sounds are interesting (measured, then written up in
ANALYSIS.md): the corpus is loud, spectrally full, harmonic-but-noisy,
and it KEEPS MOVING — centroid drift and spectral flux that the old
crackle gate could not see and did not require.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 44100
WIN = 2048
HOP = 1024
AXES = ("rms", "flatness", "centroid", "slope", "tonal", "flux", "crest", "side", "cdrift")


# ------------------------------------------------------------------ audio io

def decode(path, sr: int = SR, stereo: bool = False):
    """Decode any audio to float32 mono-mid (n,) or stereo (n, 2)."""
    ac = 2 if stereo else 1
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", str(ac),
         "-ar", str(sr), "pipe:1"], capture_output=True, check=True).stdout
    x = np.frombuffer(out, np.float32)
    return x.reshape(-1, ac) if stereo else x


def write_wav(path, audio, sr: int = SR):
    x = np.asarray(audio, np.float32)
    if x.ndim == 2 and x.shape[1] == 2:
        inter = np.empty(len(x) * 2, np.float32)
        inter[0::2], inter[1::2] = x[:, 0], x[:, 1]
        x, ch = inter, 2
    else:
        ch = 1
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(sr),
         "-ac", str(ch), "-i", "pipe:0", str(path)],
        input=pcm.tobytes(), check=True)
    return str(path)


# ------------------------------------------------------------------ corpus

def trim_silence(path, out_dir, prefix, sr: int = SR, pad_ms: int = 60,
                 merge_ms: int = 250, min_ms: int = 200, cap_s: float = 10.0,
                 verbose: bool = True) -> list[dict]:
    """Silence-trim one file into corpus chunks. Threshold adapts:
    2% of the loud (p95) window rms, or 3x the quiet floor, whichever
    is larger. Gaps shorter than merge_ms are kept; chunks shorter than
    min_ms dropped; longer than cap_s split."""
    x = decode(path, sr, stereo=True)
    mid = x.mean(axis=1).astype(np.float64)
    n = len(mid)
    win, hop = 2048, 512
    k = max(1, (n - win) // hop + 1)
    idx = np.arange(k) * hop
    rms = np.array([np.sqrt((mid[i:i + win] ** 2).mean() + 1e-12) for i in idx])
    thr = max(float(np.percentile(rms, 95)) * 0.02,
              float(np.percentile(rms, 10)) * 3.0, 1e-4)
    active = rms > thr
    pad = int(pad_ms / 1000 * sr)
    merge = int(merge_ms / 1000 / (hop / sr))       # in windows
    # runs of active windows -> merged segments in samples
    segs = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j + 1 < len(active) and (active[j + 1] or gap < merge):
            gap = 0 if active[j + 1] else gap + 1
            j += 1
        s = max(0, int(idx[i]) - pad)
        e = min(n, int(idx[j]) + win + pad)
        segs.append((s, e))
        i = j + 1 + gap
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    counter = 0
    for (s, e) in segs:
        dur = (e - s) / sr
        if dur < min_ms / 1000:
            continue
        pieces = [(s, e)]
        if dur > cap_s:
            pieces = [(s + int(q * cap_s * sr),
                       min(e, s + int((q + 1) * cap_s * sr)))
                      for q in range(int(np.ceil(dur / cap_s)))]
        for (a, b) in pieces:
            if (b - a) / sr < min_ms / 1000:
                continue
            name = f"{prefix}_{counter:04d}.wav"
            write_wav(out_dir / name, x[a:b], sr)
            manifest.append({"file": name, "start_s": round(a / sr, 3),
                             "dur_s": round((b - a) / sr, 3),
                             "rms": round(float(np.sqrt((mid[a:b] ** 2).mean())), 4)})
            counter += 1
    if verbose:
        total = sum(m["dur_s"] for m in manifest)
        print(f"[bomb] {Path(path).name}: {len(manifest)} chunks, "
              f"{total:.0f}s of {n / sr:.0f}s kept (thr {thr:.5f})")
    return manifest


# ------------------------------------------------------------------ spectral profile

def _windows(mid: np.ndarray, sr: int = SR):
    n = len(mid)
    if n < WIN:
        mid = np.pad(mid, (0, WIN - n))
        n = len(mid)
    k = max(2, (n - WIN) // HOP + 1)
    idx = np.arange(k) * HOP
    frames = np.stack([mid[i:i + WIN] * np.hanning(WIN) for i in idx])
    S = np.abs(np.fft.rfft(frames, axis=1)) + 1e-12
    fr = np.fft.rfftfreq(WIN, 1 / sr)
    return S, fr, idx


def profile(audio, sr: int = SR) -> dict:
    """Window-level spectral character of stereo audio. Aggregates +
    temporal stats (the moving parts crackle never had)."""
    x = np.asarray(audio, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    mid = x.mean(axis=1)
    S, fr, idx = _windows(mid, sr)
    spec = S.mean(axis=0)
    flat_w = np.exp(np.log(S).mean(axis=1)) / S.mean(axis=1)
    cent_w = (S * fr).sum(axis=1) / S.sum(axis=1)
    band = (fr >= 80) & (fr <= 8000)
    slope_w = np.array([np.polyfit(np.log2(fr[band]), 20 * np.log10(s[band]), 1)[0]
                        for s in S]) if band.sum() > 4 else np.zeros(len(S))
    # tonal fraction: energy in top-8 spectral peaks of each window
    tonal_w = []
    for s in S:
        pk = s[1:-1]
        strong = pk > (pk.mean() + 3.0 * pk.std())
        top = np.argsort(strong)[::-1][:8]
        energy = float(pk[top].sum() / (s.sum() + 1e-12))
        tonal_w.append(energy)
    tonal_w = np.array(tonal_w)
    # flux: mean positive log-magnitude difference between windows
    logS = np.log1p(S)
    flux_w = np.maximum(np.diff(logS, axis=0), 0).mean(axis=1)
    flux_w = np.concatenate([[0.0], flux_w])
    # per-window rms, crest, stereo side
    rms_w = np.array([np.sqrt((mid[i:i + WIN] ** 2).mean() + 1e-12) for i in idx])
    peak_w = np.array([np.abs(mid[i:i + WIN]).max() for i in idx])
    crest_w = peak_w / (rms_w + 1e-9)
    side_w = np.array([np.abs(x[i:i + WIN, 0] - x[i:i + WIN, 1]).mean() /
                       (np.abs(x[i:i + WIN, 0] + x[i:i + WIN, 1]).mean() + 1e-9)
                       for i in idx])
    cum = np.cumsum(spec) / spec.sum()
    roll_i = int(np.searchsorted(cum, 0.85))
    rolloff = float(fr[min(roll_i, len(fr) - 1)])
    return {
        "rms": float(rms_w.mean()), "rms_std": float(rms_w.std()),
        "flatness": float(flat_w.mean()), "flatness_min": float(flat_w.min()),
        "centroid": float(cent_w.mean()), "centroid_std": float(cent_w.std()),
        "rolloff": rolloff,
        "slope": float(slope_w.mean()),
        "tonal": float(tonal_w.mean()),
        "flux": float(flux_w.mean()), "flux_max": float(flux_w.max()),
        "crest": float(crest_w.mean()),
        "side": float(side_w.mean()),
        "seconds": round(len(mid) / sr, 2),
    }


def _window_axis_table(audio, sr: int = SR) -> dict:
    """Per-WINDOW axis values (the kinship envelope is measured on these,
    not on chunk means — window-level keeps the envelope tight)."""
    x = np.asarray(audio, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    mid = x.mean(axis=1)
    S, fr, idx = _windows(mid, sr)
    band = (fr >= 80) & (fr <= 8000)
    tab = {}
    tab["rms"] = np.array([np.sqrt((mid[i:i + WIN] ** 2).mean() + 1e-12) for i in idx])
    tab["flatness"] = np.exp(np.log(S).mean(axis=1)) / S.mean(axis=1)
    cent_w = (S * fr).sum(axis=1) / S.sum(axis=1)
    tab["centroid"] = np.log2(cent_w + 1.0) / 14.0          # normalized
    tab["slope"] = np.array([np.polyfit(np.log2(fr[band]), 20 * np.log10(s[band]), 1)[0]
                             for s in S]) / 24.0 + 0.5
    tonal = []
    for s in S:
        pk = s[1:-1]
        strong = pk > (pk.mean() + 3.0 * pk.std())
        top = np.argsort(strong)[::-1][:8]
        tonal.append(float(pk[top].sum() / (s.sum() + 1e-12)))
    tab["tonal"] = np.array(tonal)
    logS = np.log1p(S)
    flux = np.maximum(np.diff(logS, axis=0), 0).mean(axis=1)
    tab["flux"] = np.concatenate([[0.0], flux]) * 20.0
    peak_w = np.array([np.abs(mid[i:i + WIN]).max() for i in idx])
    tab["crest"] = np.clip(peak_w / (tab["rms"] + 1e-9), 0, 24) / 24.0
    side = np.array([np.abs(x[i:i + WIN, 0] - x[i:i + WIN, 1]).mean() /
                     (np.abs(x[i:i + WIN, 0] + x[i:i + WIN, 1]).mean() + 1e-9)
                     for i in idx])
    tab["side"] = side
    # temporal axes: smoothed drift of centroid + rms dynamics
    n = len(cent_w)
    if n >= 4:
        klen = min(31, n if n % 2 else n - 1)
        ker = np.hanning(klen)
        sm = np.convolve(cent_w, ker / ker.sum(), mode="same") if klen >= 3 else cent_w
    else:
        sm = cent_w
    tab["cdrift"] = np.abs(cent_w - sm) / 2000.0
    return tab


# ------------------------------------------------------------------ analysis

def analyze(corpus_dir, out_json=None, extra_dirs=(), verbose=True):
    """Profile every corpus chunk (+ optional extra dirs) and the boring
    baseline (late-gen womb specimens) for contrast."""
    rows = []
    corpus = Path(corpus_dir)
    for p in sorted(corpus.glob("*.wav")):
        rows.append({"set": "corpus", "file": p.name, **profile(decode(p, stereo=True))})
    for d in extra_dirs:
        for p in sorted(Path(d).glob("*.wav"))[:40]:
            rows.append({"set": Path(d).name, "file": p.name,
                         **profile(decode(p, stereo=True))})
    if verbose:
        by_set = {}
        for r in rows:
            by_set.setdefault(r["set"], []).append(r)
        keys = ("rms", "flatness", "centroid", "tonal", "flux", "slope", "crest", "side")
        hdr = "set".ljust(14) + "".join(k.rjust(10) for k in keys) + "   centroid_std"
        print(hdr)
        print("-" * len(hdr))
        for s, rs in by_set.items():
            m = {k: float(np.median([r[k] for r in rs])) for k in keys}
            cs = float(np.median([r["centroid_std"] for r in rs]))
            print(s.ljust(14) + "".join(f"{m[k]:>10.3f}" for k in keys) + f"{cs:>15.1f}")
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=1))
    return rows


# ------------------------------------------------------------------ kinship judge

def corpus_envelope(corpus_dir, sr: int = SR) -> dict:
    """Percentile envelope of per-window axis values over the WHOLE corpus."""
    tables = []
    for p in sorted(Path(corpus_dir).glob("*.wav")):
        tables.append(_window_axis_table(decode(p, sr, stereo=True), sr))
    env = {}
    for axis in AXES:
        vals = np.concatenate([t[axis] for t in tables])
        lo, hi = np.percentile(vals, [4, 96])
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        env[axis] = {"lo": float(lo), "hi": float(hi), "iqr": max(iqr, 1e-6),
                     "med": float(np.median(vals))}
    return env


def make_judge(env: dict, strictness: float = 0.5):
    """The corpus as minimal criterion: a candidate is viable iff its
    steady-state windows sit inside the corpus envelope on EVERY axis
    (per-axis floor) and inside on average. The mean alone was too lax:
    it let the rejected crackle in on 8-of-9 axes. Strictness scales the
    envelope margin (Soros' dial on the environment)."""
    m = 1.5 * (1.0 - strictness)                            # IQR expansion
    mean_need = float(np.clip(0.5 + 0.3 * strictness, 0.5, 0.85))
    min_need = float(np.clip(0.2 + 0.3 * strictness, 0.15, 0.55))

    def judge(audio, sr):
        x = np.asarray(audio, np.float64)
        if not np.isfinite(x).all():
            return {"viable": False, "rms": 0.0, "kinship": 0.0, "inside": {}}
        cut = int(len(x) * 0.25)                            # skip cold-start transient
        tab = _window_axis_table(x[cut:], sr)
        inside = {}
        for axis in AXES:
            e = env[axis]
            lo = e["lo"] - m * e["iqr"]
            hi = e["hi"] + m * e["iqr"]
            if axis == "side":
                # the corpus is never mono (window p4 = 0.055): expansion
                # may not push the side floor to zero — stereo is family.
                lo = max(lo, e["lo"] * 0.5)
            v = tab[axis]
            inside[axis] = float(np.mean((v >= lo) & (v <= hi)))
        kinship = float(np.mean(list(inside.values())))
        rms = float(np.sqrt((x ** 2).mean()))
        viable = bool(kinship >= mean_need and min(inside.values()) >= min_need
                      and rms > 0.01)
        return {"viable": viable, "rms": round(rms, 4), "kinship": round(kinship, 3),
                "inside": {k: round(v, 3) for k, v in inside.items()}}

    return judge


# ------------------------------------------------------------------ genesis

# analysis-informed founders: the corpus is dense harmonic-noise that
# keeps moving — single-law voices are too pure to pass (measured:
# flatness 0.0, crest 0.0 for 60/60 random pairs). Three layered
# families, hand-tuned against the corpus MC (tune_founders.py).

def _floor_founder(freq=550.0, osc_g=0.40, sand_g=0.30):
    """dense-driven sandpile = smooth avalanche bed; tonal ring above."""
    proc = {"nodes": [
        {"law": "sand", "params": dict(w=32, h=32, interval_s=0.004, jitter=0.9,
                                       warmup=1200, act_tau=0.05)},
        {"law": "osc_pop", "params": dict(n=64, windows=8, radius=18, k=1.1,
                                         alpha=1.3, freq=freq, spread=3.0,
                                         kernel="ring", ic="uniform")},
        {"law": "map", "params": dict(r=3.93, period_s=3.5)},
    ], "edges": [], "sim_rate": 11025, "seed": 9}
    ear = [
        {"op": "tap", "node": 0, "ch": 0, "gain": sand_g, "pan": -0.3},
        {"op": "events", "node": 0, "gain": 0.22, "pan_spread": 0.9},
        {"op": "tap", "node": 1, "ch": 0, "gain": osc_g, "pan": 0.45},
        {"op": "tap", "node": 1, "ch": 1, "gain": 0.5 * osc_g, "pan": -0.65},
        {"op": "embed", "node": 1, "ch": 0, "delay": 220, "gain": 0.3 * osc_g, "pan": 0.1},
    ]
    return proc, ear


def _buzz_founder(buzz_g=0.12, osc_g=0.6):
    """fast fire pulse train (bright buzz) + tonal ring + quantizer shadow."""
    proc = {"nodes": [
        {"law": "fire", "params": dict(bias=2.0, theta=1.0, refr_s=0.004)},
        {"law": "clamp", "params": dict(gain=1.4, steps=6)},
        {"law": "osc_pop", "params": dict(n=48, windows=8, radius=14, k=1.2,
                                         alpha=1.25, freq=740.0, spread=5.0,
                                         kernel="global", ic="uniform")},
    ], "edges": [{"src": 0, "ch": 0, "dst": 1, "gain": 1.0}],
    "sim_rate": 11025, "seed": 3}
    ear = [
        {"op": "tap", "node": 0, "ch": 0, "gain": buzz_g, "pan": -0.4},
        {"op": "tap", "node": 1, "ch": 0, "gain": buzz_g * 0.8, "pan": 0.5},
        {"op": "tap", "node": 2, "ch": 0, "gain": osc_g, "pan": 0.15},
        {"op": "tap", "node": 2, "ch": 1, "gain": 0.4 * osc_g, "pan": -0.7},
        {"op": "embed", "node": 2, "ch": 0, "delay": 150, "gain": 0.25 * osc_g, "pan": 0.35},
    ]
    return proc, ear


def _choir_founder():
    """fast excitable pulse train under a bright inharmonic ring."""
    proc = {"nodes": [
        {"law": "osc_pop", "params": dict(n=96, windows=12, radius=24, k=0.9,
                                         alpha=1.2, freq=1400.0, spread=7.0,
                                         kernel="global", ic="uniform")},
        {"law": "excitable", "params": dict(a=0.7, b=0.8, eps=0.02, z0=0.42)},
        {"law": "map", "params": dict(r=3.97, period_s=2.7)},
    ], "edges": [{"src": 2, "ch": 0, "dst": 1, "gain": 1.0}],
    "sim_rate": 4410, "seed": 5}
    ear = [
        {"op": "tap", "node": 0, "ch": 0, "gain": 0.45, "pan": -0.5},
        {"op": "tap", "node": 0, "ch": 1, "gain": 0.27, "pan": 0.6},
        {"op": "tap", "node": 1, "ch": 0, "gain": 0.12, "pan": 0.2},
        {"op": "events", "node": 1, "gain": 0.06, "pan_spread": 0.8},
        {"op": "embed", "node": 0, "ch": 0, "delay": 300, "gain": 0.11, "pan": -0.1},
    ]
    return proc, ear


def bomb_founders():
    return [_floor_founder(), _floor_founder(freq=1200.0, osc_g=0.5, sand_g=0.28),
            _buzz_founder(), _choir_founder()]


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bomb", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("corpus", help="silence-trim source files into corpus chunks")
    c.add_argument("files", nargs="+")
    c.add_argument("-o", "--out", default="corpus")

    an = sub.add_parser("analyze", help="spectral profiles: corpus + extra dirs")
    an.add_argument("corpus")
    an.add_argument("-o", "--out", default=None)
    an.add_argument("--extra", nargs="*", default=[])

    cal = sub.add_parser("calibrate", help="corpus self-kinship + baseline rejection")
    cal.add_argument("corpus")
    cal.add_argument("--strictness", type=float, default=0.5)

    br = sub.add_parser("breed", help="breed cryptids under the corpus MC")
    br.add_argument("corpus")
    br.add_argument("-o", "--out", default="runs/bomb1")
    br.add_argument("--pop", type=int, default=24)
    br.add_argument("--ears", type=int, default=10)
    br.add_argument("--gens", type=int, default=20)
    br.add_argument("--strictness", type=float, default=0.5)
    br.add_argument("--pairs", type=int, default=3)
    br.add_argument("--seconds", type=float, default=3.0)
    br.add_argument("--champion-seeds", type=int, default=0)
    br.add_argument("--founders", type=int, default=4,
                    help="how many analysis-informed genesis families to inject")
    br.add_argument("--seed", type=int, default=5)
    br.add_argument("--resume", action="store_true")
    br.add_argument("-v", "--verbose", action="store_true")

    mx = sub.add_parser("mixtape", help="the run's album (via womb.breed)")
    mx.add_argument("run_dir")
    mx.add_argument("--grid", default="texture", choices=["texture", "time", "space"])
    mx.add_argument("--seg", type=float, default=8.0)
    mx.add_argument("-o", "--out", default=None)
    mx.add_argument("-v", "--verbose", action="store_true")

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1

    if a.cmd == "corpus":
        manifest = []
        for i, f in enumerate(a.files):
            manifest += trim_silence(f, a.out, Path(f).stem)
        Path(a.out, "manifest.json").write_text(json.dumps(manifest, indent=1))
        total = sum(m["dur_s"] for m in manifest)
        print(f"[bomb] corpus: {len(manifest)} chunks, {total:.0f}s -> {a.out}/")
        return 0

    if a.cmd == "analyze":
        analyze(a.corpus, out_json=a.out, extra_dirs=a.extra)
        return 0

    if a.cmd == "calibrate":
        env = corpus_envelope(a.corpus)
        judge = make_judge(env, a.strictness)
        files = sorted(Path(a.corpus).glob("*.wav"))
        kins = [judge(decode(f, stereo=True), SR) for f in files]
        print(f"corpus self-kinship: n={len(kins)} median={np.median([k['kinship'] for k in kins]):.3f} "
              f"pass={100 * np.mean([k['viable'] for k in kins]):.0f}%")
        # the boring baseline: what the artist rejected
        from womb.cryptid import build_champion, render
        for name in ("chimera", "clickstorm", "jerk_voice"):
            st = render(build_champion(name), length=6.0)
            j = judge(st["audio"], SR)
            worst = sorted(j["inside"].items(), key=lambda kv: kv[1])[:3]
            print(f"champion {name}: kinship={j['kinship']} viable={j['viable']} "
                  + " ".join(f"{k}={v}" for k, v in worst))
        Path("corpus_envelope.json").write_text(json.dumps(env, indent=1))
        print("[bomb] envelope saved to corpus_envelope.json")
        return 0

    if a.cmd == "breed":
        from womb.breed import seed_run

        env = corpus_envelope(a.corpus)
        judge = make_judge(env, a.strictness)
        founders = bomb_founders()[:max(0, int(a.founders))]
        r = seed_run(out=a.out, pop=a.pop, ears=a.ears, gens=a.gens,
                     strictness=a.strictness, pairs=a.pairs, seconds=a.seconds,
                     champion_seeds=a.champion_seeds, seed=a.seed,
                     resume=a.resume, verbose=a.verbose, judge=judge,
                     founders=founders)
        print(f"{r['out']}: {r['specimens']} specimens, cells "
              f"{r['cells']['texture']}/{r['cells']['time']}/{r['cells']['space']} "
              f"in {r['elapsed']}s")
        return 0

    if a.cmd == "mixtape":
        from womb.breed import mixtape
        outp = mixtape(a.run_dir, out=a.out, grid=a.grid, seg=a.seg)
        print(outp)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())