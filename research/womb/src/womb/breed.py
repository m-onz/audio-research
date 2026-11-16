"""breed: the MCC loop — minimal criterion coevolution of processes x ears.

Two populations, paired by a MUTUAL minimal criterion (pap140s3 /
brant_gecco19): a process is viable iff at least one ear hears it
alive; an ear is viable iff it hears at least one process alive.
Nothing is ever ranked — no fitness, no novelty metric, no cell
competition. Offspring additionally differ from their parent by more
than the drift floor (divergence by drift, never novelty
maximization). Soros' strictness dial scales every band; the run logs
population, admissions, occupied cells and descriptor spread per
generation — Chromaria's experiment, run inside womb:

    womb cryptid seed --strictness 0.5 --gens 16 -o womb_out/crypt1
    womb cryptid preview womb_out/crypt1 -l 60 -o crypt1_previews.mp4
    womb cryptid info womb_out/crypt1

The character archive (pugh_ppsn16): every admitted specimen is
measured once, inserted into every grid (texture / time / space);
parents are chosen by picking a grid, then a cell — search is pulled
by whichever lens the offspring lands in. Cells hold the LATEST
arrival (recency, not quality). Curation (the beholder) is phase C.

Structure: genomes are pure JSON; the run is resumable (state +
archive checkpointed every generation); renders are deterministic.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from .cryptid import SR, law_channels, simulate, _finish, viability

__all__ = ["seed_run", "report_run", "montage", "mixtape", "random_process",
           "mutate_process", "random_ear", "mutate_ear", "descriptors",
           "signature", "distance", "drift_eps", "GRIDS", "GRID_CELLS"]

# ------------------------------------------------------------------ constants

BREEDABLE = ("jerk", "osc_pop", "map", "sand", "excitable", "fire",
             "bool", "delay", "clamp")           # bytefeed needs data
SIM_RATES = (2205, 4410, 11025)
GRID_CELLS = 8
GRIDS = {                                          # axis = character, never quality
    "texture": ("flatness", "slope"),              # aperiodicity x spectral tilt
    "time": ("events", "modrate"),                 # onset weather x envelope speed
    "space": ("side", "asym"),                     # stereo incoherence x rise/fall
}
_INT_KEYS = {"w", "h", "steps", "k"}

_BOUNDS = {
    "jerk": dict(a=(-3.0, -0.1), b=(-4.0, 4.0), c=(-4.0, 4.0), d=(-4.0, 4.0),
                 e=(-4.0, 4.0), f=(-4.0, 4.0), g=(-4.0, 4.0), mod=(0.0, 0.4),
                 speed=(30.0, 2500.0), ode_rate=(20.0, 120.0), x0=(-2.0, 2.0)),
    "map": dict(r=(3.4, 4.0), period_s=(0.01, 30.0)),
    "sand": dict(w=(12, 64), h=(12, 64), interval_s=(0.004, 0.3),
                 jitter=(0.0, 1.0), act_tau=(0.05, 0.5)),
    "excitable": dict(a=(0.2, 1.5), b=(0.2, 2.0), eps=(0.005, 0.1), z0=(0.1, 1.2)),
    "fire": dict(bias=(0.05, 3.0), theta=(0.3, 3.0), refr_s=(0.002, 0.1)),
    "delay": dict(length_s=(0.01, 1.5), gain=(0.1, 3.0), nl=(0.0, 4.0)),
    "clamp": dict(gain=(0.2, 5.0), steps=(2, 16)),
}

_OSC_N = (32, 64, 96, 128, 160, 192)
_OSC_W = (4, 8, 16, 32)
_OSC_FLOAT = dict(k=(0.02, 3.0), alpha=(0.8, 1.8), freq=(25.0, 420.0),
                  spread=(0.0, 5.0), ext_gain=(0.0, 2.0))


def drift_eps(strictness: float) -> float:
    """Drift floor: child must differ from parent by more than this
    (never maximized). Calibrated so mid-strictness rejects only
    *silent* mutations (measured drift of real mutations: 0.01-0.04);
    scales with Soros' dial."""
    return float(np.clip(0.01 * (strictness / 0.5) ** 2, 2e-4, 0.06))


# ------------------------------------------------------------------ law params

def _jit(rng, v, lo, hi, rate=0.4):
    f = 10.0 ** float(np.clip(rng.normal(0.0, rate), -3.0, 3.0))
    nv = float(v) * f
    if not np.isfinite(nv):
        nv = float(v)
    return float(np.clip(nv, lo, hi))


def _rand_in(rng, lo, hi, int_key=False):
    v = rng.uniform(lo, hi)
    return int(round(v)) if int_key else float(v)


def _rand_params(rng, law):
    if law == "jerk":
        return dict(
            a=rng.uniform(-3.0, -0.1), b=rng.uniform(-4.0, 4.0), c=rng.uniform(-4.0, 4.0),
            d=rng.uniform(-4.0, 4.0) if rng.random() < 0.45 else 0.0,
            e=rng.uniform(-4.0, 4.0) if rng.random() < 0.45 else 0.0,
            f=rng.uniform(-4.0, 4.0) if rng.random() < 0.45 else 0.0,
            g=rng.uniform(-4.0, 4.0) if rng.random() < 0.45 else 0.0,
            mod=rng.uniform(0.0, 0.4) if rng.random() < 0.3 else 0.0,
            speed=rng.uniform(30.0, 2000.0), ode_rate=rng.uniform(20.0, 90.0),
            x0=rng.uniform(-2.0, 2.0))
    if law == "osc_pop":
        n = int(rng.choice(_OSC_N))
        w = int(rng.choice(_OSC_W))
        while n % w:
            w //= 2
        return dict(n=n, windows=w, radius=int(rng.uniform(4, n // 2)),
                    k=rng.uniform(0.02, 3.0), alpha=rng.uniform(0.8, 1.8),
                    freq=rng.uniform(25.0, 420.0), spread=rng.uniform(0.0, 5.0),
                    kernel=str(rng.choice(["ring", "global"])),
                    ext_gain=rng.uniform(0.0, 2.0),
                    ic=str(rng.choice(["half", "uniform"])))
    if law == "bool":
        k = int(rng.integers(1, 5))
        return dict(k=k, table=[int(rng.random() < 0.5) for _ in range(2 ** k)],
                    period_s=rng.uniform(0.02, 3.0))
    if law == "sand":
        return dict(w=int(rng.integers(12, 65)), h=int(rng.integers(12, 65)),
                    interval_s=rng.uniform(0.004, 0.3), jitter=rng.uniform(0.0, 1.0),
                    act_tau=rng.uniform(0.05, 0.5), warmup=int(rng.integers(200, 1500)))
    p = {}
    for k, (lo, hi) in _BOUNDS[law].items():
        p[k] = _rand_in(rng, lo, hi, int_key=k in _INT_KEYS)
    return p


def _mut_params(rng, law, p, rate=0.4):
    if law == "jerk":
        q = dict(p)
        for k, (lo, hi) in _BOUNDS["jerk"].items():
            v = q.get(k, 0.0)
            if k in ("d", "e", "f", "g", "mod") and not v:   # sparse activation
                if rng.random() < 0.12:
                    q[k] = rng.uniform(0.0, 0.4) if k == "mod" else rng.uniform(-4.0, 4.0)
                continue
            q[k] = _jit(rng, v, lo, hi, rate)
        return q
    if law == "osc_pop":
        q = dict(p)
        if rng.random() < 0.3:
            n = int(rng.choice(_OSC_N))
            w = min(int(q.get("windows", 16)), n)
            while n % w:
                w //= 2
            q["n"], q["windows"] = n, w
        for k, (lo, hi) in _OSC_FLOAT.items():
            q[k] = _jit(rng, float(q.get(k, (lo + hi) / 2)), lo, hi, rate)
        if rng.random() < 0.1:
            q["kernel"] = str(rng.choice(["ring", "global"]))
        if rng.random() < 0.1:
            q["ic"] = str(rng.choice(["half", "uniform"]))
        q["radius"] = int(np.clip(q.get("radius", q["n"] // 4), 4, q["n"] // 2))
        return q
    if law == "bool":
        q = dict(p)
        table = list(p.get("table", [0, 1]))
        if rng.random() < 0.4 and table:
            table[int(rng.integers(len(table)))] ^= 1
        q["table"] = table
        q["period_s"] = _jit(rng, float(p.get("period_s", 0.25)), 0.02, 3.0, rate)
        return q
    if law == "sand":
        q = dict(p)
        for k, (lo, hi) in _BOUNDS["sand"].items():
            v = float(q.get(k, (lo + hi) / 2))
            q[k] = _jit(rng, v, lo, hi, rate)
            if k in ("w", "h"):
                q[k] = int(round(q[k]))
        q.setdefault("warmup", 400)
        return q
    q = dict(p)
    for k, (lo, hi) in _BOUNDS[law].items():
        v = float(q.get(k, (lo + hi) / 2))
        q[k] = _jit(rng, v, lo, hi, rate)
        if k in _INT_KEYS:
            q[k] = int(round(q[k]))
    return q


# ------------------------------------------------------------------ process genome

def _add_edge(rng, proc):
    n = len(proc["nodes"])
    if n < 1:
        return
    s = int(rng.integers(n))
    d = int(rng.integers(n))
    law = proc["nodes"][s]["law"]
    ch = int(rng.integers(max(1, law_channels(law))))
    gain = float(10 ** rng.uniform(-1.3, 0.5)) * (1.0 if rng.random() < 0.8 else -1.0)
    proc["edges"].append({"src": s, "ch": ch, "dst": d, "gain": gain,
                          "slot": int(rng.integers(0, 3))})


def _sanitize(proc):
    """Edges stay valid against the current nodes (ears are tolerant)."""
    n = len(proc["nodes"])
    keep = []
    for e in proc["edges"]:
        if not (0 <= e["src"] < n and 0 <= e["dst"] < n):
            continue
        e["ch"] = min(int(e.get("ch", 0)), law_channels(proc["nodes"][e["src"]]["law"]) - 1)
        e["slot"] = min(int(e.get("slot", 0)), 2)
        keep.append(e)
    proc["edges"] = keep
    return proc


def random_process(rng) -> dict:
    n_nodes = int(rng.integers(1, 5))
    nodes = [{"law": str(l), "params": _rand_params(rng, str(l))}
             for l in rng.choice(BREEDABLE, n_nodes)]
    proc = {"nodes": nodes, "edges": [], "sim_rate": int(rng.choice(SIM_RATES)),
            "seed": int(rng.integers(1 << 30))}
    for _ in range(int(rng.integers(0, 5))):
        _add_edge(rng, proc)
    return _sanitize(proc)


def mutate_process(rng, proc, rate: float = 0.4):
    """Drift a process genome. Returns (child, mutation description)."""
    child = copy.deepcopy(proc)
    muts = []
    picks = min(len(child["nodes"]), max(1, int(rng.integers(1, 3))))
    for idx in rng.choice(len(child["nodes"]), size=picks, replace=False):
        node = child["nodes"][int(idx)]
        node["params"] = _mut_params(rng, node["law"], node["params"], rate)
        muts.append(f"drift {node['law']}[{int(idx)}]")
    r = rng.random()
    if r < 0.22 and len(child["nodes"]) < 6:                    # node birth
        law = str(rng.choice(BREEDABLE))
        child["nodes"].append({"law": law, "params": _rand_params(rng, law)})
        muts.append(f"birth {law}")
        for _ in range(int(rng.integers(0, 2))):
            _add_edge(rng, child)
    elif r < 0.34 and len(child["nodes"]) > 2:                 # node death
        i = int(rng.integers(len(child["nodes"])))
        law = child["nodes"][i]["law"]
        child["nodes"].pop(i)
        child["edges"] = [e for e in child["edges"]
                         if e["src"] != i and e["dst"] != i]
        for e in child["edges"]:
            if e["src"] > i:
                e["src"] -= 1
            if e["dst"] > i:
                e["dst"] -= 1
        muts.append(f"death {law}[{i}]")
    r2 = rng.random()
    if r2 < 0.28 and len(child["nodes"]) > 1:
        _add_edge(rng, child)
        muts.append("edge birth")
    elif r2 < 0.45 and child["edges"]:
        child["edges"].pop(int(rng.integers(len(child["edges"]))))
        muts.append("edge death")
    elif r2 < 0.6 and child["edges"]:
        e = child["edges"][int(rng.integers(len(child["edges"])))]
        if rng.random() < 0.5:
            e["src"] = int(rng.integers(len(child["nodes"])))
            e["ch"] = int(rng.integers(law_channels(child["nodes"][e["src"]]["law"])))
        else:
            e["dst"] = int(rng.integers(len(child["nodes"])))
            e["slot"] = int(rng.integers(0, 3))
        muts.append("edge rewire")
    if rng.random() < 0.18:
        child["sim_rate"] = int(rng.choice(SIM_RATES))
        muts.append(f"sim {child['sim_rate']}")
    if rng.random() < 0.3:
        child["seed"] = int(rng.integers(1 << 30))
        muts.append("seed")
    return _sanitize(child), "; ".join(muts) if muts else "silent"


# ------------------------------------------------------------------ ear genome

def _rand_op(rng):
    kind = str(rng.choice(["tap", "diff", "embed", "events"]))
    op = {"op": kind, "node": int(rng.integers(0, 5)), "ch": int(rng.integers(0, 3)),
          "gain": float(10 ** rng.uniform(-1.5, 0.2))}
    if kind == "embed":
        op["delay"] = int(rng.integers(8, 900))
    if kind == "events":
        op["pan_spread"] = float(rng.uniform(0.0, 1.0))
        op["gain"] = float(10 ** rng.uniform(-0.8, 0.3))
    if rng.random() < 0.4:
        op["pan"] = float(rng.uniform(-0.9, 0.9))
    return op


def random_ear(rng) -> list:
    return [_rand_op(rng) for _ in range(int(rng.integers(1, 5)))]


def mutate_ear(rng, ear):
    child = copy.deepcopy(ear)
    muts = []
    for op in child:
        if "gain" in op and rng.random() < 0.6:
            op["gain"] = float(np.clip(op["gain"] * 10 ** rng.normal(0, 0.35), 0.005, 2.5))
        if "pan" in op and rng.random() < 0.4:
            op["pan"] = float(np.clip(op["pan"] + rng.normal(0, 0.25), -1.0, 1.0))
        if "delay" in op and rng.random() < 0.4:
            op["delay"] = int(np.clip(round(op["delay"] * 10 ** rng.normal(0, 0.3)), 4, 1500))
        if "pan_spread" in op and rng.random() < 0.4:
            op["pan_spread"] = float(np.clip(op["pan_spread"] + rng.normal(0, 0.15), 0.0, 1.0))
        if rng.random() < 0.25:
            op["node"] = int(rng.integers(0, 6))
        if rng.random() < 0.2:
            op["ch"] = int(rng.integers(0, 3))
    r = rng.random()
    if r < 0.35 and len(child) < 8:
        child.append(_rand_op(rng))
        muts.append("op birth")
    elif r < 0.55 and len(child) > 1:
        child.pop(int(rng.integers(len(child))))
        muts.append("op death")
    elif r < 0.7 and len(child) < 8:
        op = dict(child[int(rng.integers(len(child)))])
        op["gain"] = op.get("gain", 0.4) * 0.7
        child.append(op)
        muts.append("op dup")
    return child, "; ".join(muts) if muts else "silent"


# ------------------------------------------------------------------ measurement

def _mono(audio):
    x = np.asarray(audio, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    return x


def descriptors(audio, sr: int = SR) -> dict:
    """The character axes (multi-BC): each is a character, none is quality.
    All bounded 0..1 for gridding."""
    x = _mono(audio)
    L, R = x[:, 0], x[:, 1]
    mono = x.mean(axis=1)
    nfft = 4096
    seg = mono[: len(mono) // nfft * nfft]
    if len(seg) < nfft:
        seg = np.pad(seg, (0, nfft - len(seg)))
    S = np.abs(np.fft.rfft(seg.reshape(-1, nfft) * np.hanning(nfft), axis=1)).mean(axis=0) + 1e-12
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    flatness = float(np.exp(np.log(S).mean()) / S.mean())
    band = (fr >= 80) & (fr <= 8000)
    if band.sum() > 4:
        sl = np.polyfit(np.log2(fr[band]), 20 * np.log10(S[band]), 1)[0]   # dB/octave
    else:
        sl = 0.0
    slope = float(np.clip((sl + 20.0) / 40.0, 0.0, 1.0))
    # envelope (50 ms boxcar — slow enough to ignore tonal ripple,
    # fast enough to keep clicks) -> events, modrate, asym
    k = max(512, int(0.05 * sr))
    cs = np.cumsum(np.abs(mono))
    env = (cs[k:] - cs[:-k]) / k if len(mono) > k else np.abs(mono)
    d_env = np.diff(env)
    pos = float(d_env[d_env > 0].mean()) if np.any(d_env > 0) else 0.0
    neg = float(-d_env[d_env < 0].mean()) if np.any(d_env < 0) else 0.0
    asym = float(pos / (pos + neg + 1e-12))
    med = float(np.median(env)) + 1e-12
    above = env > 1.6 * med                # median-relative: drones stay under, spikes cross
    events = float(np.clip(np.count_nonzero(above[1:] & ~above[:-1]) /
                           max(1e-9, len(mono) / sr) / 8.0, 0.0, 1.0))
    e = env - env.mean()
    if len(e) > nfft:
        e = e[: len(e) // nfft * nfft]
        ES = np.abs(np.fft.rfft(e.reshape(-1, nfft) * np.hanning(nfft), axis=1)).mean(axis=0) + 1e-12
        efr = np.fft.rfftfreq(nfft, 1 / sr)
        cent = float((efr * ES).sum() / ES.sum())
    else:
        cent = 0.0
    modrate = float(np.clip(np.log2(cent + 1.0) / 10.0, 0.0, 1.0))
    if len(L) > 8 and L.std() > 1e-9 and R.std() > 1e-9:
        side = float(np.clip(1.0 - np.corrcoef(L, R)[0, 1], 0.0, 1.0))
    else:
        side = 0.0
    return {"flatness": float(np.clip(flatness, 0.0, 1.0)), "slope": slope,
            "events": events, "modrate": modrate, "asym": float(np.clip(asym, 0.0, 1.0)),
            "side": side}


def signature(audio, sr: int = SR) -> np.ndarray:
    """Perceptual signature for the drift floor: 64-band log-magnitude
    (L2-normalized) + loudness term."""
    x = _mono(audio)
    mono = x.mean(axis=1)
    nfft = 4096
    seg = mono[len(mono) // 4: 3 * len(mono) // 4]
    seg = seg[: len(seg) // nfft * nfft]
    if len(seg) < nfft:
        seg = np.pad(seg, (0, nfft - len(seg)))
    S = np.abs(np.fft.rfft(seg.reshape(-1, nfft) * np.hanning(nfft), axis=1)).mean(axis=0)
    edges = np.geomspace(1, len(S), 65).astype(int)
    sig = np.array([np.log1p(S[edges[i]: edges[i + 1]].mean())
                    if edges[i + 1] > edges[i] else 0.0 for i in range(64)])
    nrm = np.linalg.norm(sig) + 1e-9
    rms = float(np.sqrt((mono ** 2).mean()))
    return np.concatenate([sig / nrm, [np.log1p(rms) * 0.5]])


def distance(sig_a, sig_b) -> float:
    return float(np.abs(np.asarray(sig_a) - np.asarray(sig_b)).mean())


def _cells_of(desc: dict) -> dict:
    cells = {}
    for gname, (a, b) in GRIDS.items():
        x = int(np.clip(desc.get(a, 0.0), 0.0, 0.9999) * GRID_CELLS)
        y = int(np.clip(desc.get(b, 0.0), 0.0, 0.9999) * GRID_CELLS)
        cells[gname] = [x, y]
    return cells


# ------------------------------------------------------------------ evaluation

def render_pair(proc: dict, ear: list, seconds: float, sr: int = SR):
    """Render a (process, ear) pairing candidate. Returns stereo audio
    or raises; breeding wraps this in try/except (failure = death)."""
    genome = {"name": "candidate", "nodes": proc["nodes"],
              "edges": proc.get("edges", []),
              "sim_rate": int(proc.get("sim_rate", 11025)),
              "seed": int(proc.get("seed", 0)), "ears": ear}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        L, R, rate, _ = simulate(genome, seconds)
        return _finish(L, R, rate, sr)


def _thumb(audio, sr: int = SR, w: int = 96, h: int = 64) -> np.ndarray:
    """Tiny log-spectrogram thumbnail (uint8 grayscale) for the atlas."""
    from PIL import Image

    mono = _mono(audio).mean(axis=1)
    hop = max(1, len(mono) // max(1, w))
    win = 1024
    cols = []
    for i in range(w):
        s0 = i * hop
        seg = mono[s0: s0 + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        S = np.abs(np.fft.rfft(seg * np.hanning(win))) + 1e-9
        idx = np.geomspace(1, len(S), h).astype(int) - 1
        cols.append(np.log1p(S[idx]))
    img = np.stack(cols, axis=1)
    img = img / (img.max() + 1e-9)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------ the run

def _specimen(entry) -> dict:
    """A renderable genome snapshot (proc + ear + biography)."""
    proc, ear = entry["proc"], entry["ear"]
    return {"name": f"cryptid-{entry['id']:04d}", "lineage": entry.get("lineage", {}),
            "born": entry.get("born", {}), "nodes": proc["nodes"],
            "edges": proc.get("edges", []), "sim_rate": proc.get("sim_rate", 11025),
            "seed": proc.get("seed", 0), "ears": ear,
            "notes": f"bred specimen {entry['id']:04d}, gen {entry.get('gen')}"}


def _champion_parents():
    """Bootstrap specimens from the phase-A champions (mutually viable)."""
    from .cryptid import CHAMPIONS

    out = []
    for name, builder in CHAMPIONS.items():
        g = builder()
        out.append(({"nodes": g["nodes"], "edges": g.get("edges", []),
                     "sim_rate": g.get("sim_rate", 11025), "seed": g.get("seed", 0)},
                    g.get("ears", [])))
    return out


def seed_run(out="womb_out/crypt1", pop: int = 20, ears: int = 8, gens: int = 16,
             strictness: float = 0.5, pairs: int = 2, seconds: float = 3.0,
             champion_seeds: int = 2, births_frac: float = 0.5, rate: float = 0.4,
             seed: int = 0, resume: bool = False, verbose: bool = False,
             judge=None, founders=None) -> dict:
    """Run the MCC loop. Checkpoints every generation; resumable.
    `judge(audio, sr) -> dict` overrides the viability gate (must return
    at least 'viable' and 'rms'; the whole dict is archived with each
    specimen) — bomb injects a corpus-kinship judge here. `founders` is
    a list of (proc, ear) genesis pairings evaluated before random
    founders (bomb's analysis-informed genesis)."""
    out = Path(out)
    (out / "specimens").mkdir(parents=True, exist_ok=True)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    eps = drift_eps(strictness)
    if judge is None:
        def judge(audio, sr):
            return viability(audio, sr, strictness=strictness)

    # ---- load or init state ------------------------------------------------
    if resume and (out / "state.json").exists():
        state = json.loads((out / "state.json").read_text())
        procs, ear_pop = state["procs"], state["ears"]
        archive = json.loads((out / "archive.json").read_text())
        gen0 = state["gen"]
        next_id = state["next_id"]
        log = state["log"]
        founder_records = []
        # resume must NOT replay the original mutation stream: advance the rng
        rng = np.random.default_rng(seed + 7919 * gen0)
        if verbose:
            print(f"[cryptid] resuming {out} at gen {gen0} "
                  f"({len(procs)} procs, {len(ear_pop)} ears)")
    else:
        procs, ear_pop, archive, log, gen0, next_id = [], [], {"entries": [], "cells": {}}, [], 0, 1
        founder_records = []
        genesis = list(founders or []) + _champion_parents()[:max(0, int(champion_seeds))]
        for k, (proc, ear) in enumerate(genesis):
            try:
                audio = render_pair(proc, ear, seconds)
                viab = judge(audio, SR)
                if not viab["viable"]:
                    continue
                sig = signature(audio).tolist()
                procs.append({"id": 0, "proc": proc, "sig": sig, "gen": 0,
                              "lineage": f"champion-seed", "mutation": "none"})
                founder_records.append((proc, ear, sig, audio, viab, "champion-seed"))
                if len(ear_pop) < ears:
                    ear_pop.append({"id": next_id, "ear": ear,
                                    "lineage": "genesis", "heard": 1})
                    next_id += 1
            except Exception:
                continue
        if not ear_pop:
            ear_pop.append({"id": next_id, "ear": random_ear(rng),
                            "lineage": "founder", "heard": 0})
            next_id += 1
        tries = 0
        while len(procs) < pop and tries < pop * 4:               # random founders
            tries += 1
            proc = random_process(rng)
            viable_pair = None
            for _ in range(pairs):
                if not ear_pop:
                    ear = random_ear(rng)
                else:
                    ear = ear_pop[int(rng.integers(len(ear_pop)))]["ear"]
                try:
                    audio = render_pair(proc, ear, seconds)
                    viab = judge(audio, SR)
                    if viab["viable"]:
                        viable_pair = (ear, audio, viab, signature(audio).tolist())
                        break
                except Exception:
                    continue
            if viable_pair:
                ear_v, audio_v, viab_v, sig_v = viable_pair
                procs.append({"id": next_id, "proc": proc, "sig": sig_v,
                              "gen": 0, "lineage": "founder", "mutation": "random"})
                founder_records.append((proc, ear_v, sig_v, audio_v, viab_v, "founder"))
                next_id += 1
            if len(ear_pop) < ears and rng.random() < 0.5:
                ear_pop.append({"id": next_id, "ear": random_ear(rng),
                                "lineage": "founder", "heard": 0})
                next_id += 1
        while len(ear_pop) < max(2, ears // 2):
            ear_pop.append({"id": next_id, "ear": random_ear(rng),
                            "lineage": "founder", "heard": 0})
            next_id += 1
        if verbose:
            print(f"[cryptid] founders: {len(procs)} procs, {len(ear_pop)} ears "
                  f"(strictness {strictness}, eps {eps:.4f})")
        if not procs:
            raise RuntimeError("no founders passed the minimal criterion — "
                               "loosen strictness or widen the judge")

    entries_by_id = {e["id"]: e for e in archive["entries"]}

    def archive_insert(entry, audio):
        entry["cells"] = _cells_of(entry["desc"])
        archive["entries"].append(entry)
        entries_by_id[entry["id"]] = entry
        for gname, cell in entry["cells"].items():
            archive["cells"].setdefault(gname, {})[f"{cell[0]},{cell[1]}"] = entry["id"]
        # specimen + thumbnail
        sid = f"s{entry['id']:04d}"
        (out / "specimens" / f"{sid}.json").write_text(json.dumps(_specimen(entry), indent=1))
        try:
            from PIL import Image

            Image.fromarray(_thumb(audio)).save(out / "thumbs" / f"{sid}.png")
        except Exception:
            pass

    for proc, ear, sig, audio, viab, tag in founder_records:
        entry = {"id": next_id, "gen": 0, "proc": proc, "ear": ear,
                 "desc": descriptors(audio), "sig": sig, "viab": viab,
                 "lineage": {"parent": None, "mutation": tag},
                 "born": {"gen": 0}}
        archive_insert(entry, audio)
        next_id += 1

    # ---- generations --------------------------------------------------------
    t_start = time.time()
    for gen in range(gen0, gen0 + int(gens)):
        t0 = time.time()
        proc_admits, ear_admits, attempts = 0, 0, 0
        heard = {int(e["id"]): 0 for e in ear_pop}
        distances = []

        def pick_proc_parent():
            cellmaps = [g for g in archive["cells"] if archive["cells"][g]]
            if cellmaps and rng.random() < 0.65:
                gname = cellmaps[int(rng.integers(len(cellmaps)))]
                cm = archive["cells"][gname]
                eid = cm[list(cm)[int(rng.integers(len(cm)))]]
                e = entries_by_id[eid]
                return e["proc"], e["sig"], e["id"]
            m = procs[int(rng.integers(len(procs)))]
            return m["proc"], m["sig"], m["id"]

        # -- process children (admitted iff >= 1 ear hears them alive)
        n_births = max(2, int(pop * births_frac))
        for _ in range(n_births):
            if not ear_pop:
                break
            parent_proc, parent_sig, parent_id = pick_proc_parent()
            child, mut = mutate_process(rng, parent_proc, rate=rate)
            passes = []
            for _ in range(pairs):
                ear_member = ear_pop[int(rng.integers(len(ear_pop)))]
                attempts += 1
                try:
                    audio = render_pair(child, ear_member["ear"], seconds)
                    viab = judge(audio, SR)
                except Exception:
                    continue
                if viab["viable"]:
                    sig = signature(audio)
                    dist = distance(sig, parent_sig)
                    passes.append((ear_member, audio, viab, sig, dist))
            passes = [p for p in passes if p[4] > eps]        # the drift floor
            if not passes:
                continue
            ear_member, audio, viab, sig, dist = passes[int(rng.integers(len(passes)))]
            heard[int(ear_member["id"])] += 1
            distances.append(dist)
            entry = {"id": next_id, "gen": gen, "proc": child, "ear": ear_member["ear"],
                     "desc": descriptors(audio), "sig": sig.tolist(), "viab": viab,
                     "lineage": {"parent": parent_id, "mutation": mut},
                     "born": {"gen": gen}}
            archive_insert(entry, audio)
            procs.append({"id": next_id, "proc": child, "sig": sig.tolist(), "gen": gen,
                          "lineage": f"#{parent_id}", "mutation": mut})
            next_id += 1
            if len(procs) > pop:                              # random removal, no ranking
                procs.pop(int(rng.integers(len(procs))))
            proc_admits += 1

        # -- ear children (admitted iff they hear >= 1 process alive)
        for _ in range(max(1, len(ear_pop) // 2)):
            viable_ears = [e for e in ear_pop if heard.get(int(e["id"]), 0) > 0]
            parent = (viable_ears or ear_pop)[int(rng.integers(len(viable_ears or ear_pop)))]
            child_ear, mut = mutate_ear(rng, parent["ear"])
            heard_any = False
            for _ in range(pairs):
                pm = procs[int(rng.integers(len(procs)))]
                attempts += 1
                try:
                    audio = render_pair(pm["proc"], child_ear, seconds)
                    viab = judge(audio, SR)
                except Exception:
                    continue
                if viab["viable"]:
                    heard_any = True
                    break
            if heard_any:
                ear_pop.append({"id": next_id, "ear": child_ear,
                                "lineage": f"#{parent['id']}", "heard": 1})
                heard[next_id] = 1
                next_id += 1
                if len(ear_pop) > ears:
                    ear_pop.pop(int(rng.integers(len(ear_pop))))
                ear_admits += 1

        # -- the generation's ledger entry
        cells = {g: len(archive["cells"].get(g, {})) for g in GRIDS}
        # diversity is measured on RECENT admissions (last 40) — the whole-archive
        # std just dilutes as the archive grows dense in its occupied cells
        recent = [e["desc"] for e in archive["entries"][-40:]]
        spread = {}
        for axis in ("flatness", "slope", "events", "modrate", "asym", "side"):
            vals = [d[axis] for d in recent]
            spread[axis] = round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0
        pop_rms = [e["viab"]["rms"] for e in archive["entries"] if e["gen"] == gen]
        log.append({"gen": gen, "pop": len(procs), "ears": len(ear_pop),
                    "proc_admissions": proc_admits, "ear_admissions": ear_admits,
                    "attempts": attempts, "cells": cells, "spread": spread,
                    "mean_dist": round(float(np.mean(distances)), 4) if distances else None,
                    "mean_rms": round(float(np.mean(pop_rms)), 4) if pop_rms else None,
                    "elapsed": round(time.time() - t0, 1)})
        # checkpoint
        state = {"procs": procs, "ears": ear_pop, "gen": gen + 1, "next_id": next_id,
                 "log": log, "config": dict(pop=pop, ears=ears, gens=gens,
                                            strictness=strictness, pairs=pairs,
                                            seconds=seconds, seed=seed)}
        (out / "state.json").write_text(json.dumps(state))
        (out / "archive.json").write_text(json.dumps(archive))
        if verbose:
            print(f"[cryptid] gen {gen}: +{proc_admits}/{n_births} procs, "
                  f"+{ear_admits} ears, cells {cells['texture']}/{cells['time']}/{cells['space']}, "
                  f"{log[-1]['elapsed']}s", flush=True)

    _draw_atlases(out, archive)
    elapsed = round(time.time() - t_start, 1)
    if verbose:
        print(f"[cryptid] done: {len(archive['entries'])} specimens in {elapsed}s -> {out}")
    return {"out": str(out), "specimens": len(archive["entries"]),
            "cells": {g: len(archive["cells"].get(g, {})) for g in GRIDS},
            "log": log, "elapsed": elapsed}


# ------------------------------------------------------------------ atlases

def _draw_atlases(out: Path, archive: dict):
    try:
        from PIL import Image
    except Exception:
        return
    for gname in GRIDS:
        sheet = Image.new("RGB", (GRID_CELLS * 98, GRID_CELLS * 66), "black")
        for cell_key, eid in archive["cells"].get(gname, {}).items():
            x, y = (int(v) for v in cell_key.split(","))
            p = out / "thumbs" / f"s{eid:04d}.png"
            if p.exists():
                sheet.paste(Image.fromarray(np.asarray(Image.open(p))), (x * 98 + 1, y * 66 + 1))
        sheet.save(out / f"atlas-{gname}.png")


# ------------------------------------------------------------------ report

def report_run(run_dir) -> str:
    out = Path(run_dir)
    state = json.loads((out / "state.json").read_text())
    archive = json.loads((out / "archive.json").read_text())
    cfg = state["config"]
    lines = [f"cryptid run {out}",
             f"  strictness {cfg.get('strictness')} | pop {cfg.get('pop')} | "
             f"ears {cfg.get('ears')} | eval {cfg.get('seconds')}s | seed {cfg.get('seed')}",
             f"  specimens: {len(archive['entries'])} | lineages alive: {len(state['procs'])}",
             "",
             "gen  pop ears  admP admE  cells(t/x/s)   spread(flat/slope/ev/mod/side/asym)   meanRMS"]
    for row in state["log"]:
        c = row["cells"]
        sp = row["spread"]
        lines.append(f"{row['gen']:>3}  {row['pop']:>3} {row['ears']:>4}  "
                     f"{row['proc_admissions']:>4} {row['ear_admissions']:>4}  "
                     f"{c['texture']:>2}/{c['time']}/{c['space']:<2}        "
                     f"{sp['flatness']:.2f}/{sp['slope']:.2f}/{sp['events']:.2f}/"
                     f"{sp['modrate']:.2f}/{sp['side']:.2f}/{sp['asym']:.2f}   "
                     f"{row['mean_rms'] if row['mean_rms'] is not None else '-'}")
    lines.append("")
    lines.append("regime hint: run the same seed at strictness 0.1 / 0.5 / 1.0 —")
    lines.append("mid should dominate cells + spread (soros' orderly divergence).")
    return "\n".join(lines)


# ------------------------------------------------------------------ montage

def montage(run_dir, out=None, total: float = 60.0, grid: str = "texture",
            seed: int = 0, verbose: bool = False) -> str:
    """The atlas you can LISTEN to: one segment per occupied cell (latest
    specimen), ordered along the grid's x axis — a journey through the
    discovered character space. Panels: scrolling spectrogram, waveform,
    pan trace, Takens portrait, and the six character bars."""
    from .writer import AVWriter
    from doom.palettes import apply_palette, get_palette

    out_dir = Path(run_dir)
    archive = json.loads((out_dir / "archive.json").read_text())
    by_id = {e["id"]: e for e in archive["entries"]}
    rng = np.random.default_rng(seed)
    cellmap = archive["cells"].get(grid, {})
    entries = [by_id[eid] for eid in cellmap.values()]
    if not entries:
        raise RuntimeError(f"no occupied cells in grid {grid!r}")
    entries.sort(key=lambda e: e["cells"][grid][0] + rng.uniform(-0.4, 0.4))
    entries = entries[:40]
    n_seg = len(entries)
    seg = float(np.clip(total / n_seg, 1.5, 8.0))

    out = out or str(out_dir / "previews.mp4")
    W, H, fps = 960, 540, 24.0
    sr = SR
    hop = int(sr / fps)
    writer = AVWriter(out, W, H, fps, sr=sr)
    lut_spec = get_palette("inferno")
    lut_att = get_palette("ice")
    D = 97                                   # Takens delay

    for k, entry in enumerate(entries):
        if verbose:
            print(f"[cryptid] preview {k + 1}/{n_seg}: specimen "
                  f"s{entry['id']:04d} (cell {entry['cells'][grid]})", flush=True)
        try:
            audio = render_pair(entry["proc"], entry["ear"], seg)
        except Exception:
            continue
        mono = audio.mean(axis=1)
        spec = np.zeros((296, W))
        wfm = np.full((80, W), 128, np.uint8)
        pan = np.full((36, W), 24, np.uint8)
        att = np.zeros((116, 116))
        peak = 1e-9
        desc = entry["desc"]
        n_frames = int(seg * fps)
        for i in range(n_frames):
            a0, a1 = i * hop, min(len(mono), (i + 1) * hop)
            seg_audio = audio[a0:a1]
            mono_i = mono[a0:a1]
            if len(mono_i) == 0:
                break
            # spectrogram column
            win = mono[max(0, a1 - 2048): a1]
            if len(win) < 2048:
                win = np.pad(win, (0, 2048 - len(win)))
            S = np.abs(np.fft.rfft(win * np.hanning(2048))) + 1e-9
            idx = np.geomspace(2, len(S), 296).astype(int) - 1
            col = np.log1p(S[idx])
            peak = max(float(col.max()), peak * 0.995)
            spec[:, :-1] = spec[:, 1:]
            spec[:, -1] = col / peak
            # waveform columns (L upper, R lower)
            wfm[:, :-1] = wfm[:, 1:]
            wfm[:, -1] = 128
            if len(seg_audio):
                for ch, mid in ((0, 20), (1, 60)):
                    x = seg_audio[:, ch]
                    lo = int(np.clip(20 * x.min(), -19, 0))
                    hi = int(np.clip(20 * x.max(), 0, 19))
                    wfm[mid + lo: mid + hi + 1, -1] = 255
            # pan trace
            pan[:, :-1] = np.maximum((pan[:, 1:].astype(int) * 9) // 10, 24).astype(np.uint8)
            if len(seg_audio):
                rl = float(np.sqrt((seg_audio[:, 0] ** 2).mean()))
                rr = float(np.sqrt((seg_audio[:, 1] ** 2).mean()))
                pv = float(np.clip((rl - rr) / (rl + rr + 1e-9), -1, 1))
                px = int(np.clip(pv * 0.5 + 0.5, 0, 1) * (W - 1))
            else:
                px = W // 2
            pan[:, max(0, px - 1): px + 2] = 255
            # Takens portrait
            att *= 0.93
            if a1 - D > 0:
                xs = mono[a0: a1 - D]
                ys = mono[a0 + D: a1]
                xi = np.clip((xs * 40 + 58), 0, 115).astype(int)
                yi = np.clip((ys * 40 + 58), 0, 115).astype(int)
                np.add.at(att, (yi, xi), 1.0)
            # assemble the frame
            frame = np.zeros((H, W, 3), np.uint8)
            frame[0:296] = apply_palette(np.clip(spec, 0, 1), lut_spec)
            frame[300:380] = np.stack([wfm] * 3, axis=2)
            frame[384:420] = np.stack([pan] * 3, axis=2)
            amax = att.max() + 1e-9
            frame[424:540, 0:116] = apply_palette(np.clip(att / amax, 0, 1), lut_att)
            # character bars
            for bi, axis in enumerate(("flatness", "slope", "events", "modrate", "asym", "side")):
                x0, y0 = 140, 430 + bi * 18
                frame[y0: y0 + 12, x0: x0 + 120] = 45
                w = int(desc[axis] * 120)
                if w:
                    frame[y0: y0 + 12, x0: x0 + w] = 210
            frame[296:300] = 60
            frame[380:384] = 60
            frame[420:424] = 60
            frame[424:540, 116:124] = 60
            writer.write(frame, seg_audio if len(seg_audio) else None)
        # specimen gap
        gap = np.zeros((H, W, 3), np.uint8)
        for _ in range(4):
            writer.write(gap)
        writer.write(np.full((H, W, 3), 235, np.uint8))
    stats = writer.close()
    return out


# ------------------------------------------------------------------ mixtape

def mixtape(run_dir, out=None, grid: str = "texture", seg: float = 8.0,
             fade: float = 0.75, seed: int = 0, max_segments: int = 48,
             verbose: bool = False) -> str:
    """The run's album — the usable artifact of a long run: every
    occupied cell's latest specimen, rendered `seg` seconds, ordered
    along the grid's x axis (tonal -> noisy on `texture`), equal-power
    crossfaded into one finished stereo piece."""
    from .sonify import write_wav

    out_dir = Path(run_dir)
    archive = json.loads((out_dir / "archive.json").read_text())
    by_id = {e["id"]: e for e in archive["entries"]}
    rng = np.random.default_rng(seed)
    cellmap = archive["cells"].get(grid, {})
    entries = [by_id[eid] for eid in cellmap.values()]
    if not entries:
        raise RuntimeError(f"no occupied cells in grid {grid!r}")
    entries.sort(key=lambda e: e["cells"][grid][0] + rng.uniform(-0.5, 0.5))
    entries = entries[:max_segments]
    nfade = max(1, min(int(fade * SR), int(seg * SR) // 3))

    pieces = []
    for k, entry in enumerate(entries):
        try:
            audio = render_pair(entry["proc"], entry["ear"], seg)
        except Exception:
            continue
        pieces.append(audio.astype(np.float32))
        if verbose:
            print(f"[cryptid] mixtape {k + 1}/{len(entries)}: specimen "
                  f"s{entry['id']:04d} (cell {entry['cells'][grid]})", flush=True)
    if not pieces:
        raise RuntimeError("no renderable specimens in this grid")

    total = sum(len(p) for p in pieces) - nfade * (len(pieces) - 1)
    mix = np.zeros((total, 2), np.float32)
    pos = 0
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nfade) / nfade)   # equal-power-ish
    for i, p in enumerate(pieces):
        p = p.copy()
        if i > 0:
            p[:nfade] *= ramp[:, None]
        if i < len(pieces) - 1:
            p[-nfade:] *= ramp[::-1][:, None]
        mix[pos: pos + len(p)] += p
        pos += len(p) - nfade
    peak = float(np.abs(mix).max())
    if peak > 1e-9:
        mix = mix / peak * 0.92
    out = out or str(out_dir / f"mixtape-{grid}.wav")
    write_wav(out, mix, SR)
    return out