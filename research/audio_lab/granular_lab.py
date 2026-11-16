#!/usr/bin/env python3
"""
granular_lab.py — a bench for algorithmic-audio experiments.

Pipeline:
    load wavs  ->  chop into grains  ->  classify each grain by acoustic features
    (loudness, brightness, pitch, noisiness, transient-ness, ...)  ->
    arrange the grains in an "order" (sorting / bucketing / interleaving /
    patterns / generative sequences: Fibonacci, exponential, cycling, Euclidean)
    ->  render that order at time-varying speed with selectable interpolation
    ->  normalized .wav into the output folder.

Any numeric parameter can be a *curve*: a constant, or a list like [0.5, 2.0]
or [3, 12, 3], interpreted piecewise-linearly across the output (0 -> 1).
That's how you make things "speed up and slow down" over the piece.

Examples:
    python3 granular_lab.py trimmed/ -o out/ --recipe cloud_drift
    python3 granular_lab.py sounds.wav sounds2.wav -o out/ --recipe all --seed 7
    python3 granular_lab.py sounds.wav -o out/ --recipe crescendo --set speed=[0.5,3]
    python3 granular_lab.py trimmed/ -o out/ --recipe none --random 12      # 12 experiments
    python3 granular_lab.py trimmed/ -o out/ --recipe all --random 8       # recipes + randoms
    python3 granular_lab.py --list

Recipes are plain dicts in RECIPES below — copy one, tweak it, add your own.
Every render writes its resolved parameters to manifest.json next to the
output wavs, so every experiment is reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS = 1e-10
INTERP_METHODS = ("hold", "linear", "hermite", "sinc")


# --------------------------------------------------------------------------- wav IO

def read_wav(path):
    """Read a wav -> (sr, float32 array shaped (n, ch))."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return int(sr), np.ascontiguousarray(data, dtype=np.float32)
    except ImportError:
        pass
    try:
        from scipy.io import wavfile
        sr, arr = wavfile.read(str(path))
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.dtype == np.int16:
            data = arr.astype(np.float32) / 32768.0
        elif arr.dtype == np.int32:
            data = arr.astype(np.float32) / 2147483648.0
        elif arr.dtype == np.uint8:
            data = (arr.astype(np.float32) - 128.0) / 128.0
        else:
            data = arr.astype(np.float32)
        return int(sr), data
    except ImportError:
        pass
    return _read_wav_stdlib(path)


def _read_wav_stdlib(path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        u = np.frombuffer(raw, np.uint8).reshape(-1, ch, 3).astype(np.uint32)
        v = (u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16)).astype(np.int32)
        v = np.where(v & 0x800000, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    return sr, x.reshape(-1, ch)


def write_wav(path, sr, data):
    """Write wav (16-bit PCM - outputs are peak-normalized, half the disk)."""
    data = np.asarray(data, np.float32)
    if data.ndim == 1:
        data = data[:, None]
    try:
        import soundfile as sf
        sf.write(str(path), data, sr, subtype="PCM_16")
        return
    except ImportError:
        pass
    y16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(y16.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y16.tobytes())


# --------------------------------------------------------------------------- curves

def as_curve(v):
    """Normalize a parameter to a callable f(t), t in [0,1]. Accepts a number,
    a numeric string, or a list (piecewise-linear)."""
    if callable(v):
        return v
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        c = float(v)
        return lambda t: c
    if isinstance(v, str):
        c = float(v)  # raises if not numeric
        return lambda t: c
    seq = [float(a) for a in v]
    if not seq:
        raise ValueError("empty curve")
    if len(seq) == 1:
        return lambda t: seq[0]
    xs = np.linspace(0.0, 1.0, len(seq))
    ys = np.asarray(seq, float)
    return lambda t: float(np.interp(float(t), xs, ys))


# --------------------------------------------------------------------------- envelopes

def _tukey(n, alpha=0.5):
    w = np.ones(n)
    r = max(1, int(alpha * (n - 1) / 2))
    ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(r, 0, -1) / r))
    w[:r] = ramp
    w[-r:] = ramp[::-1]
    return w


def make_env(name, n):
    """Per-grain amplitude envelope, length n, peak 1."""
    if n <= 2:
        return np.ones(n, np.float32)
    if name == "hann":
        e = np.hanning(n + 2)[1:-1]
    elif name == "tri":
        e = 1.0 - np.abs(np.linspace(-1.0, 1.0, n))
    elif name == "tukey":
        e = _tukey(n, 0.5)
    elif name == "perc":
        e = np.exp(-4.5 * np.linspace(0.0, 1.0, n))
        atk = max(8, int(0.05 * n))
        e[:atk] *= np.linspace(0.0, 1.0, atk)
    elif name == "const":
        e = np.ones(n)
        f = min(64, n // 3)
        e[:f] = np.linspace(0, 1, f)
        e[-f:] = np.linspace(1, 0, f)
    else:
        raise ValueError(f"unknown envelope {name!r} (hann|tri|tukey|perc|const)")
    return (e / e.max()).astype(np.float32)


# --------------------------------------------------------------------------- resampling / interpolation

def _resample_1d(x, speed, method):
    n = len(x)
    if n == 0:
        return x.astype(np.float32)
    if speed == 1.0:
        return x.astype(np.float32).copy()
    m = max(1, int(round(n / speed)))
    pos = np.arange(m, dtype=np.float64) * speed
    np.clip(pos, 0.0, n - 1, out=pos)
    if method == "hold":
        return x[np.floor(pos).astype(np.int64)].astype(np.float32)
    if method == "linear":
        return np.interp(pos, np.arange(n, dtype=np.float64), x).astype(np.float32)
    if method == "hermite":
        i = np.floor(pos).astype(np.int64)
        t = pos - i
        x0 = x[i]
        xm1 = x[np.clip(i - 1, 0, n - 1)]
        x1 = x[np.clip(i + 1, 0, n - 1)]
        x2 = x[np.clip(i + 2, 0, n - 1)]
        c1 = 0.5 * (x1 - xm1)
        c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
        c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
        return (((c3 * t + c2) * t + c1) * t + x0).astype(np.float32)
    if method == "sinc":
        taps = 16
        half = taps // 2
        i = np.floor(pos).astype(np.int64)
        f = pos - i
        y = np.zeros(m, np.float64)
        for k in range(-half, half):
            d = f - k
            w = np.where(np.abs(d) < half,
                         0.5 * (1.0 + np.cos(np.pi * d / half)), 0.0)
            y += x[np.clip(i + k, 0, n - 1)] * np.sinc(d) * w
        return y.astype(np.float32)
    raise ValueError(f"unknown interp {method!r} ({'|'.join(INTERP_METHODS)})")


def resample_grain(x, speed, method):
    """Play grain x (n, ch) at `speed` (2.0 = double pitch+tempo, 0.5 = half)."""
    if x.ndim == 1:
        return _resample_1d(x, speed, method)
    return np.stack([_resample_1d(x[:, c], speed, method)
                     for c in range(x.shape[1])], axis=1)


# --------------------------------------------------------------------------- grain features

def _pitch(m, sr):
    """Crude autocorrelation pitch (Hz), 0.0 if not confident."""
    if len(m) < 512:
        return 0.0
    dec = max(1, sr // 12000)
    xs = m[::dec].astype(np.float64)
    srs = sr / dec
    n = len(xs)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(xs, nfft)
    ac = np.fft.irfft(np.abs(X) ** 2, nfft)[: n // 2]
    if ac[0] <= EPS:
        return 0.0
    ac = ac / ac[0]
    lagmin = max(2, int(srs / 800))
    lagmax = min(int(srs / 50), n - 1)
    if lagmax <= lagmin + 2:
        return 0.0
    k = lagmin + int(np.argmax(ac[lagmin:lagmax]))
    if ac[k] < 0.35:
        return 0.0
    d = 0.0
    if 0 < k < n - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if abs(den) > EPS else 0.0
    return float(srs / (k + d))


def compute_features(x, sr):
    """x: (n, ch) float32 -> dict of scalar features used for sorting/bucketing."""
    m = x[:, 0] if x.shape[1] == 1 else x.mean(axis=1)
    n = len(m)
    rms = float(np.sqrt(np.mean(m * m) + EPS))
    peak = float(np.max(np.abs(m)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(m))))) if n > 1 else 0.0
    nfft = min(1 << 16, max(256, 1 << int(np.ceil(np.log2(max(2, n))))))
    w = np.hanning(n) if n > 1 else np.ones(1)
    spec = np.abs(np.fft.rfft(m * w, nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    s = spec.sum() + EPS
    centroid = float((freqs * spec).sum() / s)
    i = int(np.searchsorted(np.cumsum(spec), 0.85 * spec.sum()))
    rolloff = float(freqs[min(i, len(freqs) - 1)])
    flatness = float(np.exp(np.mean(np.log(spec + EPS))) / (np.mean(spec) + EPS))
    q = max(1, n // 4)
    transient = float(np.sqrt(np.mean(m[:q] ** 2)) / (rms + EPS))
    return dict(rms=rms, peak=peak, zcr=zcr, centroid=centroid, rolloff=rolloff,
                flatness=flatness, pitch=_pitch(m, sr),
                crest=peak / (rms + EPS), transient=transient, dur=n / sr)


# --------------------------------------------------------------------------- grains

@dataclass
class Grain:
    data: np.ndarray      # (n, 2) float32
    sr: int
    source: int
    start: float          # seconds into the source
    features: dict


@dataclass
class Src:
    name: str
    sr: int
    audio: np.ndarray     # (n, 2) float32


def load_sources(paths):
    srcs = []
    for p in paths:
        sr, a = read_wav(p)
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        elif a.shape[1] > 2:
            a = a[:, :2]
        srcs.append(Src(Path(p).stem, int(sr), np.ascontiguousarray(a, np.float32)))
    if not srcs:
        raise SystemExit("no input files")
    sr0 = srcs[0].sr
    for s in srcs:                      # cheap linear resample to a common rate
        if s.sr != sr0:
            n_new = int(round(s.audio.shape[0] * sr0 / s.sr))
            pos = np.linspace(0, s.audio.shape[0] - 1, n_new)
            base = np.arange(s.audio.shape[0], dtype=np.float64)
            s.audio = np.stack(
                [np.interp(pos, base, s.audio[:, c]) for c in range(2)], axis=1
            ).astype(np.float32)
            s.sr = sr0
    return srcs


def _onset_segments(audio, sr, p):
    """Slice audio at energy onsets; returns [(start_sample, size), ...]."""
    n = audio.shape[0]
    hop = max(64, int(sr * 0.01))
    nf = n // hop
    if nf < 8:
        return []
    e = np.abs(audio[: nf * hop]).reshape(nf, hop, audio.shape[1]).mean(axis=(1, 2))
    base = float(np.median(e[e > 0])) if np.any(e > 0) else 0.0
    thr = max(base * 2.5, 1e-4)
    peaks, min_gap = [], max(1, int(p["min_grain"] * sr / hop))
    for i in range(1, nf - 1):
        if e[i] > thr and e[i] >= e[i - 1] and e[i] > e[i + 1]:
            if peaks and i - peaks[-1] < min_gap:
                continue
            peaks.append(i)
    if not peaks:
        return []
    bounds = [0] + [i * hop for i in peaks] + [n]
    segs, minsz, maxsz = [], max(64, int(p["min_grain"] * sr)), 0
    maxsz = max(minsz + 1, int(p["max_grain"] * sr))
    for a, b in zip(bounds[:-1], bounds[1:]):
        sz = b - a
        if sz < minsz:
            continue
        if sz > maxsz:
            parts = int(np.ceil(sz / maxsz))
            step = sz // parts
            segs += [(a + j * step, step) for j in range(parts)]
        else:
            segs.append((a, sz))
    return segs


def extract_grains(srcs, p, rng):
    grains = []
    for si, s in enumerate(srcs):
        n, sr = s.audio.shape[0], s.sr
        segs = []
        if p["segment"] == "onset":
            segs = _onset_segments(s.audio, sr, p)
        elif p["segment"] == "random":
            lo = max(64, int(p["min_grain"] * sr))
            hi = max(lo + 1, int(p["max_grain"] * sr))
            if n > hi:
                st = rng.integers(0, n - hi, size=int(p["count"]))
                sz = rng.integers(lo, hi + 1, size=int(p["count"]))
                segs = [(int(a), int(b)) for a, b in zip(st, sz)]
        else:  # uniform
            sz = max(64, int(p["grain_size"] * sr))
            hop = max(1, int(p["hop"] * sr))
            segs = [(st, sz) for st in range(0, max(1, n - sz), hop)]
        cap = int(p.get("max_grains") or 0)
        if cap and len(segs) > cap:                     # thin out huge extractions
            stride = int(np.ceil(len(segs) / cap))
            segs = segs[::stride]
        src_grains = []
        for st, sz in segs:
            a = s.audio[st: st + sz]
            if a.shape[0] < 32:
                continue
            src_grains.append(Grain(a, sr, si, st / sr, compute_features(a, sr)))
        floor = 10.0 ** (float(p.get("min_grain_db", -60.0)) / 20.0)
        audible = [g for g in src_grains if g.features["rms"] >= floor]
        grains += audible if audible else src_grains   # never drop a whole source
    return grains


# --------------------------------------------------------------------------- order building

def _parse_pattern(s):
    """'0,0,1,r,-' -> [0, 0, 1, RANDOM, REST]; RANDOM=-2, REST=-1."""
    toks = []
    for tk in s.split(","):
        tk = tk.strip().lower()
        if tk in ("-", "rest"):
            toks.append(-1)
        elif tk in ("r", "rand", "random"):
            toks.append(-2)
        else:
            toks.append(int(tk))
    return toks


# ----------------------------------------------------- interleave generators
# Each returns a token stream of bucket indices (and -1 rests) that the order
# builder consumes cyclically, so all of these are repeatable cycles.

def _euclid_bits(pulses, steps):
    """Euclidean rhythm E(pulses, steps) as a 0/1 list (even distribution)."""
    steps = max(1, steps)
    pulses = max(0, min(pulses, steps))
    return [1 if (i * pulses) % steps < pulses else 0 for i in range(steps)]


def _gen_fibo(k, args, rng, max_tokens=2048):
    """Fibonacci interleave.

    fibo          runs of length 1,1,2,3,5,8,13,... traded between buckets
    fibo:mod      bucket = fib(i) % k each step (quasirandom, self-similar)
    fibo:rest:P   insert a rest between runs with probability P
    fibo:long     extend the run series (55, 89, 144)
    """
    if "mod" in args:
        a, b, toks = 1, 1, []
        for _ in range(377):          # Pisano-period flavoured sequence
            toks.append(a % k)
            a, b = b, a + b
        return toks
    runs = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    if "long" in args:
        runs += [55, 89, 144]
    rest_p = 0.0
    if "rest" in args:
        i = args.index("rest")
        rest_p = float(args[i + 1]) if i + 1 < len(args) else 0.35
    toks, b = [], 0
    for r in runs:
        toks += [b] * r
        if rest_p > 0 and rng.random() < rest_p:
            toks.append(-1)
        b = (b + 1) % k
        if len(toks) > max_tokens:
            break
    return toks


def _gen_expo(k, args, max_tokens=2048):
    """Exponential interleave: run lengths grow geometrically (1,2,4,8,...).
    expo:R sets the growth ratio R (clamped 1.25..8)."""
    ratio = 2.0
    for a in args:
        try:
            ratio = float(a)
            break
        except ValueError:
            pass
    ratio = min(max(ratio, 1.25), 8.0)
    toks, r, b = [], 1.0, 0
    while len(toks) < max_tokens:
        n = min(max(1, int(round(r))), max_tokens - len(toks))
        toks += [b] * n
        b = (b + 1) % k
        r *= ratio
    return toks


def _gen_cycling(k, args):
    """Cycling interleave: visit buckets as 0, step, 2*step, ... mod k.
    cycling:2 with k=5 -> 0,2,4,1,3 ... ; negative steps run backwards."""
    step = 1
    for a in args:
        try:
            s = int(a)
            if s != 0:
                step = s
            break
        except ValueError:
            pass
    return [(i * step) % k for i in range(k)]


def _gen_euclid(k, args):
    """Euclidean interleave 'euclid:pulses:steps': pulses distributed over
    steps (rests fill the gaps), each pulse takes from the next bucket."""
    p, s = k, 2 * k
    if len(args) >= 1:
        p = int(args[0])
    if len(args) >= 2:
        s = int(args[1])
    bits = _euclid_bits(p, s)
    toks, b = [], 0
    for bit in bits:
        if bit:
            toks.append(b)
            b = (b + 1) % k
        else:
            toks.append(-1)
    return toks


def build_order(grains, spec, rng):
    """Return a list of grain indices (an 'order'), possibly containing -1 rests.

    spec keys:
      axis:       None | "source" | feature name (rms, centroid, pitch, flatness,
                  zcr, rolloff, transient, crest, peak, dur)
      buckets:    k quantile classes along axis (axis="source" is its own axis)
      within:     "as_is" | "reverse" | "random" | "sort:FEATURE:asc|desc"
      interleave: "sequential" | "round_robin" | "zip" | "random"
                  | "pattern:0,0,1,r,-" | "fibo[:mod|rest:P|long]"
                  | "expo[:ratio]" | "cycling[:step]" | "euclid[:pulses:steps]"
    """
    n = len(grains)
    if n == 0:
        return []
    axis = spec.get("axis")
    if axis in (None, "", "none", "as_is"):
        groups = [np.arange(n, dtype=np.int64)]
    elif axis == "source":
        ns = 1 + max(g.source for g in grains)
        groups = [np.array([i for i, g in enumerate(grains) if g.source == s],
                           dtype=np.int64) for s in range(ns)]
        groups = [g for g in groups if len(g)]
    else:
        vals = np.array([float(g.features.get(axis, 0.0)) for g in grains])
        k = int(spec.get("buckets", 3))
        if k <= 1:
            groups = [np.arange(n, dtype=np.int64)]
        else:
            edges = np.quantile(vals, np.linspace(0, 1, k + 1)[1:-1])
            b = np.digitize(vals, edges)
            groups = [np.where(b == j)[0].astype(np.int64) for j in range(k)]
            groups = [g for g in groups if len(g)]

    within = spec.get("within", "as_is")
    ordered = []
    for arr in groups:
        if within == "reverse":
            a = arr[::-1]
        elif within == "random":
            a = rng.permutation(arr)
        elif isinstance(within, str) and within.startswith("sort"):
            parts = within.split(":", 2)
            feat = parts[1] if len(parts) > 1 else (axis or "rms")
            d = parts[2] if len(parts) > 2 else "asc"
            vals = np.array([float(grains[i].features.get(feat, 0.0)) for i in arr])
            o = np.argsort(vals, kind="stable")
            if d.lower().startswith("d"):
                o = o[::-1]
            a = arr[o]
        else:
            a = arr
        ordered.append([int(i) for i in a])

    iw = spec.get("interleave", "sequential")
    if iw == "sequential":
        return [i for grp in ordered for i in grp]
    if iw == "zip":
        flat = [i for grp in ordered for i in grp]
        h = len(flat) // 2
        a, b = flat[:h], flat[h:]
        return [x for i in range(max(len(a), len(b)))
                for x in ((a[i],) if i < len(a) else ()) + ((b[i],) if i < len(b) else ())]
    if iw == "round_robin":
        tokens = list(range(len(ordered)))
    elif iw == "random":
        tokens = None
    elif iw.startswith("pattern:"):
        tokens = _parse_pattern(iw.split(":", 1)[1])
    elif iw.startswith("fibo"):
        tokens = _gen_fibo(len(ordered), iw.split(":")[1:], rng)
    elif iw.startswith("expo"):
        tokens = _gen_expo(len(ordered), iw.split(":")[1:])
    elif iw.startswith("cycling"):
        tokens = _gen_cycling(len(ordered), iw.split(":")[1:])
    elif iw.startswith("euclid"):
        tokens = _gen_euclid(len(ordered), iw.split(":")[1:])
    else:
        raise ValueError(f"bad interleave {iw!r}")

    ptrs = [0] * len(ordered)
    remain = [len(g) for g in ordered]
    total = sum(remain)
    seq, step, taken = [], 0, 0
    max_steps = 8 * total + 64
    while taken < total and step < max_steps:
        step += 1
        if tokens is None:
            tk = int(rng.integers(len(ordered)))
        else:
            tk = tokens[step % len(tokens)]
        if tk == -2:
            tk = int(rng.integers(len(ordered)))
        if tk == -1:
            seq.append(-1)
            continue
        if tk >= len(ordered) or remain[tk] <= 0:
            continue
        seq.append(ordered[tk][ptrs[tk]])
        ptrs[tk] += 1
        remain[tk] -= 1
        taken += 1
    return seq


# --------------------------------------------------------------------------- rendering

def _pan_gain(rng, spread):
    ppan = rng.uniform(-1.0, 1.0) * spread
    th = (ppan + 1.0) * (np.pi / 4.0)
    return np.cos(th), np.sin(th)


def _prep_grain(g, speed, method, p, rng):
    sp = max(0.02, speed)
    if p["speed_jitter"] > 0:
        sp *= 1.0 + rng.uniform(-p["speed_jitter"], p["speed_jitter"])
    y = resample_grain(g.data, sp, method)
    if p["reverse_prob"] > 0 and rng.random() < p["reverse_prob"]:
        y = y[::-1]
    y = y * make_env(p["envelope"], len(y))[:, None] * p["gain"]
    if p["gain_jitter_db"] > 0:
        y *= 10.0 ** (rng.uniform(-p["gain_jitter_db"], p["gain_jitter_db"]) / 20.0)
    if p["pan_spread"] > 0:
        gl, gr = _pan_gain(rng, p["pan_spread"])
        y[:, 0] *= gl
        y[:, 1] *= gr
    return y


def render_sequence(grains, order_seq, p, sr, rng):
    """Grains are placed one after another (with overlap); output length follows."""
    speed_c, ospeed_c = as_curve(p["speed"]), as_curve(p["order_speed"])
    methods = [p["interp"]] if isinstance(p["interp"], str) else list(p["interp"])
    overlap=float(np.clip(p["overlap"], 0.0, 0.9))
    cycles = max(0.01, float(p["cycles"]))
    L = len(order_seq)
    rest = max(1, int(p["rest_s"] * sr))
    max_out = int(p["max_output_s"] * sr)
    n_ev = max(1, int(round(L * cycles)))
    events, cursor, pos = [], 0, 0.0
    for ei in range(n_ev):
        if cursor >= max_out:
            break
        t = ei / max(1, n_ev - 1)
        idx = order_seq[int(pos) % L]
        pos += ospeed_c(t)
        if idx < 0:
            cursor += rest
            continue
        method = methods[min(len(methods) - 1, int(t * len(methods)))]
        y = _prep_grain(grains[idx], speed_c(t), method, p, rng)
        events.append((cursor, y))
        cursor += max(1, int(len(y) * (1.0 - overlap)))
    return events


def render_cloud(grains, order_seq, p, sr, rng):
    """Grains rain onto a fixed-duration timeline at a (variable) grains/sec rate."""
    dur = float(p["duration"])
    rate_c, speed_c, ospeed_c = as_curve(p["grain_rate"]), as_curve(p["speed"]), as_curve(p["order_speed"])
    methods = [p["interp"]] if isinstance(p["interp"], str) else list(p["interp"])
    tj = float(p["timing_jitter"])
    # schedule events
    times, t = [0.0], 0.0
    while t < dur:
        r = max(0.05, rate_c(min(1.0, t / dur)))
        t += 1.0 / r
        if tj > 0:
            t += rng.uniform(-tj, tj) / r
        if 0.0 <= t < dur:
            times.append(t)
    grid = p.get("grid")
    if grid:
        g = float(grid)
        seen = {0.0}
        for x in times:
            q = round(x / g) * g
            if 0.0 <= q < dur:
                seen.add(q)
        times = sorted(seen)
    L = len(order_seq)
    pos, events = 0.0, []
    for t in times:
        tn = min(1.0, max(0.0, t / dur))
        idx = order_seq[int(pos) % L]
        pos += ospeed_c(tn)
        if idx < 0:
            continue
        method = methods[min(len(methods) - 1, int(tn * len(methods)))]
        y = _prep_grain(grains[idx], speed_c(tn), method, p, rng)
        off = int(t * sr)
        events.append((off, y))
    return events


def finalize(events, sr, peak_target, fade_s=0.01):
    if not events:
        return np.zeros((0, 2), np.float32)
    L = max(off + y.shape[0] for off, y in events)
    out = np.zeros((L, 2), np.float32)
    for off, y in events:
        out[off: off + y.shape[0]] += y
    pk = float(np.abs(out).max())
    if pk > 1e-4:
        out *= peak_target / pk
    f = min(len(out) // 2, max(1, int(sr * fade_s)))
    if f > 1:
        out[:f] *= np.linspace(0.0, 1.0, f)[:, None]
        out[-f:] *= np.linspace(1.0, 0.0, f)[:, None]
    return out


def render(grains, order_seq, p, sr, rng):
    if p["mode"] == "sequence":
        events = render_sequence(grains, order_seq, p, sr, rng)
    elif p["mode"] == "cloud":
        events = render_cloud(grains, order_seq, p, sr, rng)
    else:
        raise ValueError(f"bad mode {p['mode']!r} (sequence|cloud)")
    return finalize(events, sr, float(p["peak_target"]))


def auto_cycles(p, grains, order_seq):
    """Sequence mode: if target_s is set, scale cycles so the piece lands near
    that duration, instead of truncating structure with max_output_s."""
    if p["mode"] != "sequence" or not p.get("target_s") or not grains:
        return
    avg_len = float(np.mean([g.data.shape[0] for g in grains])) / grains[0].sr
    sp = as_curve(p["speed"])
    avg_sp = float(np.mean([sp(t) for t in (0.25, 0.5, 0.75)]))
    overlap = float(np.clip(p["overlap"], 0.0, 0.9))
    est_per_event = max(1e-6, avg_len * (1.0 - overlap) / max(0.05, avg_sp))
    L = max(1, len(order_seq))
    p["cycles"] = float(np.clip(float(p["target_s"]) / (L * est_per_event), 0.002, 12.0))


# --------------------------------------------------------------------------- random experiments


def random_params(rng):
    """Sample one experiment configuration: structure-heavy, duration-bounded.
    Everything is drawn from musically sensible ranges so every draw is at
    least *interesting*, even before you curate."""
    def logu(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    p = {}
    p["mode"] = "cloud" if rng.random() < 0.5 else "sequence"
    p["segment"] = str(rng.choice(["uniform", "uniform", "uniform", "onset", "random"]))
    p["grain_size"] = logu(0.06, 0.7)
    p["hop"] = max(0.1, p["grain_size"] * float(rng.uniform(0.3, 1.2)))
    p["min_grain"] = logu(0.04, 0.25)
    p["max_grain"] = min(1.5, p["min_grain"] * float(rng.uniform(2.5, 8.0)))
    p["envelope"] = str(rng.choice(["hann", "hann", "tukey", "tri", "perc", "const"]))
    if rng.random() < 0.45:                             # morphing interp chains
        p["interp"] = [str(rng.choice(INTERP_METHODS)) for _ in range(int(rng.integers(2, 5)))]
    else:
        p["interp"] = str(rng.choice(INTERP_METHODS))

    def speed():
        r = rng.random()
        if r < 0.35:
            return logu(0.4, 2.2)
        if r < 0.7:
            return [logu(0.3, 2.2) for _ in range(2)]
        return [logu(0.25, 2.5) for _ in range(3)]

    p["speed"] = speed()
    if rng.random() < 0.3:
        p["speed_jitter"] = float(rng.uniform(0.05, 0.4))

    # --- order spec (the structural heart)
    spec = {}
    axes = ["rms", "centroid", "pitch", "flatness", "zcr", "transient",
            "crest", "rolloff", "dur", "source"]
    if rng.random() < 0.85:
        spec["axis"] = str(rng.choice(axes))
        spec["buckets"] = 3 if spec["axis"] == "source" else int(rng.integers(2, 7))
    else:
        spec["buckets"] = 1
    k = max(1, int(spec["buckets"]))
    spec["within"] = str(rng.choice([
        "random", "random", "as_is", "reverse", "sort:rms:asc", "sort:rms:desc",
        "sort:centroid:asc", "sort:centroid:desc", "sort:pitch:asc",
        "sort:pitch:desc", "sort:transient:desc", "sort:zcr:asc"]))
    r = rng.random()
    if r < 0.14:
        spec["interleave"] = "sequential"
    elif r < 0.30:
        spec["interleave"] = "round_robin"
    elif r < 0.38:
        spec["interleave"] = "zip"
    elif r < 0.46:
        spec["interleave"] = "random"
    elif r < 0.60:                                      # fibonacci family
        spec["interleave"] = str(rng.choice(["fibo", "fibo:rest:0.3", "fibo:mod", "fibo:long"]))
    elif r < 0.72:                                      # exponential family
        spec["interleave"] = str(rng.choice(["expo", "expo:1.5", "expo:3", "expo:4"]))
    elif r < 0.82:                                      # cycling family
        spec["interleave"] = str(rng.choice(["cycling", "cycling:2", "cycling:3", "cycling:-1"]))
    elif r < 0.94:                                      # euclidean family
        pe = int(rng.integers(max(2, k - 1), 2 * k + 1))
        st = pe + int(rng.integers(1, 2 * k + 1))
        spec["interleave"] = f"euclid:{pe}:{st}"
    else:                                               # hand pattern with holes
        toks = []
        for _ in range(int(rng.integers(4, 10))):
            q = rng.random()
            if q < 0.15:
                toks.append("-")
            elif q < 0.25:
                toks.append("r")
            else:
                toks.append(str(int(rng.integers(0, max(1, k)))))
        spec["interleave"] = "pattern:" + ",".join(toks)
    p["order"] = spec

    r = rng.random()
    if r < 0.55:
        p["order_speed"] = 1.0
    elif r < 0.78:                                       # stutters / un-jams
        p["order_speed"] = [1.0, float(rng.uniform(0.15, 0.6))]
    else:                                                # forward then rewind
        p["order_speed"] = [float(rng.uniform(0.6, 1.4)), -float(rng.uniform(0.3, 0.9))]

    p["reverse_prob"] = 0.0 if rng.random() < 0.75 else float(rng.uniform(0.1, 0.6))
    p["pan_spread"] = 0.0 if rng.random() < 0.4 else float(rng.uniform(0.2, 1.0))
    p["gain_jitter_db"] = 0.0 if rng.random() < 0.6 else float(rng.uniform(1.0, 6.0))
    p["rest_s"] = float(rng.uniform(0.05, 0.6))
    p["max_output_s"] = float(rng.choice([60, 120, 180]))

    if p["mode"] == "cloud":
        p["duration"] = logu(15, 45)
        if rng.random() < 0.5:
            p["grain_rate"] = logu(1.5, 16)
        else:
            p["grain_rate"] = [logu(1.5, 16) for _ in range(int(rng.integers(2, 4)))]
        p["timing_jitter"] = float(rng.uniform(0.0, 0.6))
        p["grid"] = None if rng.random() < 0.65 else float(
            rng.choice([0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.5]))
    else:
        p["overlap"] = float(rng.uniform(0.15, 0.75))
        p["target_s"] = logu(25, 150)
    return p


def _rand_tag(p):
    """Short descriptive tag for a random experiment, e.g. cloud_fibo_pitch."""
    spec = p.get("order") or {}
    iw = str(spec.get("interleave", "sequential")).split(":")[0]
    if iw == "pattern":
        iw = "pat"
    axis = str(spec.get("axis") or "none")
    return f"{p['mode']}_{iw}_{axis}"


# --------------------------------------------------------------------------- recipes

DEFAULTS = dict(
    mode="cloud",           # sequence | cloud
    segment="uniform",      # uniform | onset | random
    grain_size=0.25,        # seconds (uniform base size)
    hop=None,               # seconds between uniform grains (default grain_size/2)
    min_grain=0.06, max_grain=0.8,   # clamps for onset/random
    count=100,              # grains for segment="random"
    # sequence mode
    cycles=1.0,             # passes through the order; fractions (0.2) play only
                            # the first 20% - use this to bound sequence-mode length
    overlap=0.5,            # grain overlap on the output timeline
    rest_s=0.3,             # length of pattern rests
    max_output_s=600,       # safety cap
    target_s=None,          # sequence mode: desired seconds -> auto-scales cycles
    max_grains=20000,       # thin out extractions bigger than this (speed)
    min_grain_db=-60.0,     # drop grains quieter than this (padding residue)
    # cloud mode
    duration=20.0,          # output length in seconds
    grain_rate=4.0,         # grains/sec (curve)
    grid=None,              # quantize event times to this many seconds (rhythm)
    timing_jitter=0.0,      # 0..1, randomize event timing
    # both
    order=dict(axis=None, buckets=3, within="as_is", interleave="sequential"),
    order_speed=1.0,         # advance through the order per grain (curve! <1 stutters,
                            # >1 skips, negative rewinds)
    speed=1.0,              # grain playback rate (curve)
    speed_jitter=0.0,       # random per-grain speed variation
    interp="linear",        # hold | linear | hermite | sinc, or a list to morph
    envelope="hann",        # hann | tri | tukey | perc | const
    reverse_prob=0.0,       # chance a grain plays backwards
    gain=1.0, gain_jitter_db=0.0,
    pan_spread=0.0,         # random stereo placement 0..1
    peak_target=0.97,
    seed=None,
)

RECIPES = {
    "crescendo": dict(
        description="grains sorted quiet->loud, slowly accelerating, hi-fi",
        mode="sequence", segment="uniform", grain_size=0.35, hop=2.4,
        order=dict(axis="rms", buckets=1, within="sort:rms:asc"),
        speed=[0.8, 1.3], interp="sinc", envelope="tukey", overlap=0.5,
    ),
    "spectral_pulse": dict(
        description="4 brightness classes round-robin -> dark/bright alternation",
        mode="sequence", segment="uniform", grain_size=0.25, hop=1.8,
        order=dict(axis="centroid", buckets=4, within="random", interleave="round_robin"),
        speed=1.0, interp="hermite", envelope="hann", overlap=0.5,
    ),
    "pitch_mosaic": dict(
        description="onset-sliced chunks arranged as an up/down pitch arc",
        mode="sequence", segment="onset", grain_size=0.5, min_grain=0.12, max_grain=0.8,
        order=dict(axis="pitch", buckets=5, within="random",
                   interleave="pattern:0,1,2,3,4,3,2,1"),
        speed=[1.0, 0.8, 1.2], interp="hermite", envelope="hann", overlap=0.4,
        cycles=0.35,
    ),
    "cloud_drift": dict(
        description="stereo granular cloud; density and texture morph over time",
        mode="cloud", duration=30.0, segment="uniform", grain_size=0.15, hop=0.08,
        grain_rate=[3, 12, 3], order=dict(within="random"),
        speed=[0.5, 1.5, 0.9], interp=["hold", "hermite", "sinc"], envelope="hann",
        pan_spread=0.8, gain_jitter_db=3, timing_jitter=0.3,
    ),
    "rewind": dict(
        description="plays the sorted order forward, stalls, then rewinds it",
        mode="sequence", segment="uniform", grain_size=0.2, hop=0.12,
        order=dict(axis="centroid", buckets=1, within="sort:centroid:asc"),
        order_speed=[1, -1], speed=[1.5, 0.4], interp="linear",
        envelope="tukey", overlap=0.6, target_s=80,
    ),
    "stutter": dict(
        description="fractional order speed repeats grains; gradually un-jams",
        mode="sequence", segment="uniform", grain_size=0.12, hop=0.07,
        order=dict(axis="rms", buckets=3, within="sort:rms:desc", interleave="round_robin"),
        order_speed=[0.2, 1.0], speed=[1.0, 1.6], interp=["hold", "linear", "hermite"],
        envelope="tukey", overlap=0.5, target_s=70,
    ),
    "rhythm_rests": dict(
        description="loudness classes in a pattern with holes, quantized to a grid",
        mode="cloud", duration=24.0, grid=0.25, segment="uniform", grain_size=0.2, hop=0.1,
        grain_rate=6,
        order=dict(axis="rms", buckets=3, within="random",
                   interleave="pattern:0,0,-,1,-,0,1,1,-,2"),
        speed=[0.9, 1.1], interp="hermite", envelope="perc", pan_spread=0.4,
    ),
    "sources_weave": dict(
        description="interleaves grains from each input file round-robin",
        mode="sequence", segment="uniform", grain_size=0.4, hop=2.0,
        order=dict(axis="source", within="random", interleave="round_robin"),
        speed=[0.7, 1.0, 1.4], interp="sinc", envelope="hann", overlap=0.5,
    ),
    "grainfall": dict(
        description="tiny high-rate grains raining across the stereo field",
        mode="cloud", duration=20.0, segment="random", count=400,
        min_grain=0.03, max_grain=0.15,
        grain_rate=[6, 16, 6], order=dict(within="random"),
        speed=[1.0, 0.6, 1.8], interp="hermite", envelope="perc",
        pan_spread=1.0, gain_jitter_db=6, timing_jitter=0.5,
    ),
    # ---- interleave generator showcases -------------------------------------
    "fibo_exchange": dict(
        description="loudness classes traded in runs of Fibonacci length 1,1,2,3,5...",
        mode="sequence", segment="uniform", grain_size=0.25, hop=0.15,
        order=dict(axis="rms", buckets=3, within="random", interleave="fibo"),
        speed=[0.9, 1.2], interp="hermite", envelope="hann", overlap=0.5,
        target_s=75,
    ),
    "fibo_mod_texture": dict(
        description="bucket index = Fibonacci mod 4 over noisiness - quasirandom weave",
        mode="cloud", duration=30.0, segment="uniform", grain_size=0.08, hop=0.05,
        grain_rate=[5, 14, 6],
        order=dict(axis="flatness", buckets=4, within="random", interleave="fibo:mod"),
        speed=[0.6, 1.4, 0.8], interp=["hold", "hermite", "sinc"], envelope="hann",
        pan_spread=0.9, gain_jitter_db=4, timing_jitter=0.35,
    ),
    "expo_accel": dict(
        description="two brightness classes exchanged in doubling runs, accelerating to match",
        mode="sequence", segment="uniform", grain_size=0.2, hop=0.12,
        order=dict(axis="centroid", buckets=2, within="random", interleave="expo:2"),
        speed=[1.0, 2.4], interp=["linear", "hermite", "sinc"], envelope="tukey",
        overlap=0.5, target_s=60,
    ),
    "cycling_weave": dict(
        description="5 pitch classes visited in a rotating stride, drifting stereo cloud",
        mode="cloud", duration=40.0, segment="uniform", grain_size=0.16, hop=0.09,
        grain_rate=[4, 10, 4],
        order=dict(axis="pitch", buckets=5, within="random", interleave="cycling:2"),
        speed=[0.7, 1.3, 0.9], interp="hermite", envelope="hann",
        pan_spread=0.6, timing_jitter=0.2,
    ),
    "euclid_pulse": dict(
        description="Euclidean rhythm of loudness classes on a 0.25s grid, percussive",
        mode="cloud", duration=36.0, grid=0.25, segment="uniform", grain_size=0.2, hop=0.12,
        grain_rate=6,
        order=dict(axis="rms", buckets=3, within="random", interleave="euclid:5:8"),
        speed=[0.9, 1.1], interp="hermite", envelope="perc", pan_spread=0.3,
    ),
}


# --------------------------------------------------------------------------- CLI

def apply_set(d, expr):
    """Apply a KEY=VALUE override, dotted keys reach into nested dicts."""
    key, _, raw = expr.partition("=")
    try:
        val = json.loads(raw)
    except Exception:
        if raw.lstrip().startswith(("{", "[")):
            raise ValueError(
                f"--set {key!r}: value looks like JSON but failed to parse: {raw!r} "
                f"(tip: in bash, use single quotes around it: --set '{key}={{...}}')")
        val = raw.strip()
    cur = d
    keys = key.strip().split(".")
    for kk in keys[:-1]:
        nxt = cur.get(kk)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[kk] = nxt
        cur = nxt
    cur[keys[-1]] = val


def gather_inputs(args_inputs):
    files = []
    for a in args_inputs:
        p = Path(a)
        if p.is_dir():
            files += sorted(q for q in p.iterdir() if q.suffix.lower() in (".wav", ".wave"))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: {a} not found")
    return files


PARAMS_HELP = """\
parameters (all overridable with --set key=value; lists/curves like --set speed=[0.5,3]):
  mode        sequence | cloud                       grain_size    seconds
  segment     uniform | onset | random                hop           seconds (uniform)
  duration    output seconds (cloud)                  grain_rate    grains/sec curve (cloud)
  grid        seconds, quantize events (cloud)         timing_jitter 0..1 (cloud)
  cycles      passes through the order, fractions allowed (sequence)
  overlap     0..0.9 grain overlap (sequence)          target_s      desired seconds (sequence)
  order       dict: axis, buckets, within, interleave  order_speed   advance/grain (curve)
  speed       playback rate (curve)                    speed_jitter  random variation
  interp      hold | linear | hermite | sinc | list    envelope      hann|tri|tukey|perc|const
  reverse_prob, gain, gain_jitter_db, pan_spread, rest_s, count, min_grain, max_grain
  min_grain_db  drop grains below this level (default -60 dB)
order spec:
  axis        feature (rms, peak, centroid, rolloff, flatness, zcr, pitch, crest,
              transient, dur) | "source" | none
  buckets     k quantile classes along axis
  within      as_is | reverse | random | sort:FEATURE:asc|desc
  interleave  sequential | round_robin | zip | random | pattern:0,1,r,-  (r=rand, -=rest)
              | fibo[:mod|rest:P|long]   runs of length 1,1,2,3,5,8... traded between
                buckets; fibo:mod = bucket fib(i)%k (quasirandom texture)
              | expo[:ratio]   run lengths grow geometrically (1,2,4,8,...)
              | cycling[:step] visit buckets as 0,step,2*step,... mod k
              | euclid[:p:s]   Euclidean rhythm, p pulses over s steps, rests between
random experiments:
  --random N  generate N randomized experiments (config sampled from --seed)
"""


def main():
    ap = argparse.ArgumentParser(
        description="Algorithmic-audio experiment bench (see RECIPES in the script).",
        epilog=PARAMS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="*", help="wav files or directories")
    ap.add_argument("-o", "--out", default="granular_out", help="output directory")
    ap.add_argument("--recipe", default="cloud_drift",
                    help="recipe name, comma list, 'all', or 'none'")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="also generate N randomized experiments (seeded by --seed)")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="KEY=VALUE", help="override a parameter (repeatable)")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default 0)")
    ap.add_argument("--each-source", action="store_true",
                    help="also render each source file on its own, besides pooled")
    ap.add_argument("--list", action="store_true", help="list recipes and exit")
    args = ap.parse_args()

    if args.list:
        for name, r in RECIPES.items():
            print(f"{name:20s} {r.get('description', '')}")
        return
    if not args.inputs:
        ap.error("no input files given")

    inputs = gather_inputs(args.inputs)
    if not inputs:
        ap.error("no .wav files found in the given inputs")
    sources = load_sources(inputs)
    print(f"loaded {len(sources)} source(s): "
          + ", ".join(f"{s.name} ({s.audio.shape[0] / s.sr:.0f}s)" for s in sources))

    seed_base = args.seed if args.seed is not None else 0
    if seed_base < 0:
        ap.error("--seed must be >= 0")
    if args.recipe == "all":
        names = list(RECIPES)
    elif args.recipe == "none":
        names = []
    else:
        names = [n.strip() for n in args.recipe.split(",") if n.strip()]
        bad = [n for n in names if n not in RECIPES]
        if bad:
            ap.error(f"unknown recipe(s): {', '.join(bad)} (--list to see them)")
    if not names and args.random <= 0:
        ap.error("nothing to do: give --recipe and/or --random")

    # experiment list: (rand_index, piece_name, overlay_params, is_random);
    # random piece names are built after --set overrides are applied, so the
    # label always reflects what was actually rendered
    experiments = [(-1, n, RECIPES[n], False) for n in names]
    if args.random > 0:
        cfg_rng = np.random.default_rng(seed_base + 911)
        for i in range(args.random):
            experiments.append((i, None, random_params(cfg_rng), True))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    mpath = outdir / "manifest.json"
    manifest = {}                      # merge with previous runs, don't clobber
    if mpath.exists():
        try:
            with open(mpath) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    combos = [sources]
    if args.each_source:
        combos += [[s] for s in sources]

    for exp_i, (rand_i, name, overlay, is_random) in enumerate(experiments):
        p = {k: (dict(v) if isinstance(v, dict) else v)
             for k, v in DEFAULTS.items() if k != "seed"}
        order_overlay = overlay.get("order") if isinstance(overlay.get("order"), dict) else {}
        p.update({k: (dict(v) if isinstance(v, dict) else v)
                  for k, v in overlay.items()
                  if k not in ("description", "order")})
        p["order"] = {**DEFAULTS["order"], **order_overlay}
        for expr in args.sets:
            apply_set(p, expr)
        if p["hop"] is None:
            p["hop"] = p["grain_size"] * 0.5
        if is_random:
            name = f"rand{rand_i:02d}_{_rand_tag(p)}"
        for i, subset in enumerate(combos):
            seed = seed_base + (7919 * exp_i if is_random else 0)
            rng = np.random.default_rng(seed)
            grains = extract_grains(subset, p, rng)
            if not grains:
                print(f"{name}: no grains extracted, skipped")
                continue
            order_seq = build_order(grains, p["order"], rng)
            auto_cycles(p, grains, order_seq)
            audio = render(grains, order_seq, p, subset[0].sr, rng)
            if audio.shape[0] and float(np.abs(audio).max()) < 1e-3:
                print(f"  {name}: warning - output is (near-)silent; try another "
                      f"seed or lower min_grain_db")
            stem = name if len(combos) == 1 else (
                f"{name}_pooled" if i == 0 else f"{name}__{subset[0].name}")
            path = outdir / f"{stem}_{seed}.wav"
            write_wav(path, subset[0].sr, audio)
            dur = audio.shape[0] / subset[0].sr
            print(f"  {name:28s} {len(grains):6d} grains, order {len(order_seq):6d} "
                  f"-> {path.name} ({dur:.1f}s)")
            manifest[str(path)] = dict(
                experiment=name, recipe=None if is_random else name,
                random=is_random, seed=seed, config_seed=seed_base,
                sources=[s.name for s in subset],
                params={k: v for k, v in p.items() if k != "order"},
                order=p["order"],
            )

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"done -> {outdir}/ (manifest.json records every parameter)")


if __name__ == "__main__":
    sys.exit(main())