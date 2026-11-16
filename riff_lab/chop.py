"""chop: the event-slice library — THE CHOP (from dnb_chop_lab / rhythm_lab).

Onset-slice your ./sounds corpus AT ITS TRANSIENTS so every slice begins on
an attack: short, crisp, intricate hits — the raw material minimal dark techy
DnB is actually made of. Slices are feature-measured and role-classified
into riff_lab's voices (bass / noise / fill) by spectral character, then
selected per riff by FARTHEST-POINT with a global usage penalty, so each
riff mines fresh material and the library gets covered, not repeated.

Metadata is cached to riffs/_chop.json; slice audio is read on demand from
the source files (soundfile seek — fast, low memory). The corpus location is
./sounds, overridable with the AUDIO_RESEARCH_SOUNDS environment variable.
"""
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from . import core

CACHE = Path("riffs/_chop.json")

# the source corpus (scanned recursively) + a cap keeping the build quick;
# raise it (or --rebuild) to mine deeper.
SOUNDS = Path(os.environ.get("AUDIO_RESEARCH_SOUNDS", "sounds"))
SCAN_CAP = int(os.environ.get("AUDIO_RESEARCH_SOUNDS_CAP", "1500"))

MIN_GRAIN = 0.045
MAX_GRAIN = 0.55
MIN_RMS = 0.01
MAX_SLICES = 4000
_FEAT_KEYS = ("cent", "flat", "trans", "rms", "dur", "f0")


# ----------------------------------------------------------------- onset chop

def _onset_segments(mono, sr, min_grain=MIN_GRAIN, max_grain=MAX_GRAIN):
    """Slice at energy onsets; [(start, size), ...]. (granular_lab logic.)"""
    n = len(mono)
    hop = max(64, int(sr * 0.01))
    nf = n // hop
    if nf < 8:
        return []
    e = np.abs(mono[: nf * hop]).reshape(nf, hop).mean(axis=1)
    base = float(np.median(e[e > 0])) if np.any(e > 0) else 0.0
    thr = max(base * 2.5, 1e-4)
    peaks, min_gap = [], max(1, int(min_grain * sr / hop))
    for i in range(1, nf - 1):
        if e[i] > thr and e[i] >= e[i - 1] and e[i] > e[i + 1]:
            if peaks and i - peaks[-1] < min_gap:
                continue
            peaks.append(i)
    if not peaks:
        return []
    bounds = [0] + [i * hop for i in peaks] + [n]
    minsz = max(64, int(min_grain * sr))
    maxsz = max(minsz + 1, int(max_grain * sr))
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sz = b - a
        if sz < minsz:
            continue
        segs.append((a, min(sz, maxsz)))
    return segs


def _features(mono, sr):
    n = len(mono)
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    nfft = min(1 << 16, max(256, 1 << int(np.ceil(np.log2(max(2, n))))))
    w = np.hanning(n) if n > 1 else np.ones(1)
    spec = np.abs(np.fft.rfft(mono * w, nfft)) + 1e-9
    fr = np.fft.rfftfreq(nfft, 1.0 / sr)
    cent = float((fr * spec).sum() / spec.sum())
    flat = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-9))
    q = max(1, n // 4)
    trans = float(np.sqrt(np.mean(mono[:q] ** 2)) / (rms + 1e-9))
    f0, conf = core.detect_pitch(mono, sr)
    return dict(cent=cent, flat=flat, trans=trans, rms=rms,
                dur=n / sr, f0=f0 if conf >= 0.3 else 0.0)


# ----------------------------------------------------------------- build

def _iter_files(seed=0):
    rng = np.random.default_rng(seed)
    if not SOUNDS.is_dir():
        return
    files = sorted(q for q in SOUNDS.rglob("*.[wW][aA][vV]"))
    if SCAN_CAP and len(files) > SCAN_CAP:
        idx = np.sort(rng.permutation(len(files))[:SCAN_CAP])
        files = [files[i] for i in idx]
    for f in files:
        yield f


def build(rebuild=False, seed=0, max_slices=MAX_SLICES, verbose=True):
    """Chop the corpus into a role-classified slice library (metadata)."""
    if CACHE.exists() and not rebuild:
        try:
            return json.load(open(CACHE))
        except Exception:
            pass
    recs = []
    nfiles = 0
    for f in _iter_files(seed):
        nfiles += 1
        try:
            x, sr = sf.read(str(f), always_2d=True, dtype="float32")
        except Exception:
            continue
        m = x.mean(axis=1)
        if sr != 44100:
            m = core.resample(m, sr / 44100.0); sr = 44100
        segs = _onset_segments(m, sr)
        if len(segs) < 4:            # not event-dense: a few fixed windows
            sz = int(0.3 * sr); hop = int(0.25 * sr)
            segs = [(st, sz) for st in range(0, max(1, len(m) - sz), hop)][:8]
        for st, sz in segs:
            d = m[st: st + sz]
            if len(d) < 256:
                continue
            ft = _features(d, sr)
            if ft["rms"] < MIN_RMS:
                continue
            recs.append(dict(path=str(f), start=int(st), n=int(sz), **ft))
    if not recs:
        raise SystemExit("chop: no slices extracted — is ./old present?")
    if len(recs) > max_slices:
        rng = np.random.default_rng(seed + 1)
        idx = np.sort(rng.permutation(len(recs))[:max_slices])
        recs = [recs[i] for i in idx]
    _classify(recs)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(recs, fh)
    if verbose:
        from collections import Counter
        rc = Counter(r["role"] for r in recs)
        print(f"chop library: {len(recs)} event-slices from {nfiles} files "
              f"-> {dict(rc)}  ({CACHE})")
    return recs


def _classify(recs):
    """Percentile-rank spectral character -> voice role (rhythm_lab logic,
    mapped to bass/noise/fill)."""
    for key, attr in (("cent", "bright"), ("flat", "noisy"), ("trans", "tr")):
        vals = np.array([r[key] for r in recs], float)
        order = np.argsort(vals, kind="stable")
        rank = np.empty(len(vals)); rank[order] = np.arange(len(vals)) / max(1, len(vals) - 1)
        for r, v in zip(recs, rank):
            r[attr] = float(v)
    for r in recs:
        if r["bright"] < 0.30:
            r["role"] = "bass"
        elif r["bright"] > 0.60:
            r["role"] = "fill"
        else:
            r["role"] = "noise"


# ----------------------------------------------------------------- selection

def _whiten(recs):
    X = np.array([[np.log10(r["cent"] + 1), np.log10(r["rms"] + 1e-9),
                   r["flat"], r["trans"], r["dur"]] for r in recs], np.float32)
    med = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    return (X - med) / iqr


def by_role(lib):
    out = {"bass": [], "noise": [], "fill": []}
    for i, r in enumerate(lib):
        out.setdefault(r["role"], []).append(i)
    return out


def pick_fresh(lib, role_idx, n, rng, usage=None, Xw=None):
    """Farthest-point selection in whitened feature space with a global
    usage penalty — each draw mines material far from what's been used."""
    idx = np.asarray(role_idx)
    if len(idx) == 0:
        return []
    usage = usage if usage is not None else {}
    if Xw is None:
        Xw = _whiten([lib[i] for i in idx])
    else:
        Xw = Xw[idx]
    use = np.array([usage.get(int(i), 0) for i in idx], np.float32)
    seed = int(np.argmin(use + rng.random(len(idx)) * 0.5))
    chosen = [seed]
    usage[int(idx[seed])] = usage.get(int(idx[seed]), 0) + 1
    mind = np.linalg.norm(Xw - Xw[seed], axis=1)
    while len(chosen) < min(n, len(idx)):
        adj = mind / (1.0 + 2.0 * use)
        adj[chosen] = -np.inf
        j = int(np.argmax(adj))
        chosen.append(j)
        usage[int(idx[j])] = usage.get(int(idx[j]), 0) + 1
        use[j] += 1.0
        mind = np.minimum(mind, np.linalg.norm(Xw - Xw[j], axis=1))
    return [int(idx[j]) for j in chosen]


def read_slice(rec, onset_trim=True):
    """Read one slice as (n,2) float32 @44.1k, onset-trimmed to the attack."""
    try:
        x, sr = sf.read(rec["path"], start=rec["start"],
                        stop=rec["start"] + rec["n"], always_2d=True,
                        dtype="float32")
    except Exception:
        return None, 44100
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    if sr != 44100:
        x = core.resample(x, sr / 44100.0); sr = 44100
    if onset_trim and len(x) > 64:
        m = np.abs(x).mean(axis=1)
        pk = float(m.max())
        if pk > 1e-6:
            above = np.flatnonzero(m > 0.04 * pk)
            if len(above) and above[0] > 0:
                s = max(0, int(above[0]) - int(0.001 * sr))
                x = np.ascontiguousarray(x[s:])
    return x, sr
