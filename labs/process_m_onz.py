"""process_m_onz — turn the m-onz-sounds train field-recordings into a
library of processed one-shot sounds.

The three recordings are long, sustained, dense train textures with no sharp
transients (transient onset-chopping returns nothing), so Stage A slices them
with *multi-scale fixed windows* and picks a diverse subset by farthest-point
selection in a whitened feature space. Stage B then runs every raw slice
through the whole oneshot_lab transform registry (45 transforms / 7 families)
plus random mutation chains.

    python3 process_m_onz.py                 # both stages
    python3 process_m_onz.py --stage slice   # Stage A only
    python3 process_m_onz.py --stage process # Stage B only (raw/ must exist)
    python3 process_m_onz.py --dry-run       # plan slice counts only

Output:
    m-onz_oneshots/raw/                      clean, trimmed source one-shots
    m-onz_oneshots/processed/<transform>/    processed one-shots + manifests
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # oneshot_lab + echo live at the repo root

from oneshot_lab import core                  # noqa: E402
from oneshot_lab.engine import run_folder     # noqa: E402

SR = 44100
# raw source recordings to slice & process: your own long textures / field
# recordings. Defaults to the shared ./sounds corpus at the repo root;
# override with AUDIO_RESEARCH_SOUNDS.
IN_DIR = Path(os.environ.get("AUDIO_RESEARCH_SOUNDS", str(HERE.parent / "sounds")))
OUT_DIR = HERE / "m-onz_oneshots"
RAW_DIR = OUT_DIR / "raw"
PROC_DIR = OUT_DIR / "processed"

SCALES_S = (0.3, 0.5, 0.8, 1.2, 1.8)          # multi-scale window lengths
OVERLAP = 0.5                                  # window hop = scale * (1-OVERLAP)
MIN_RMS = 0.01                                 # drop near-silent windows
PER_FILE = 100                                 # diverse slices kept per train


# --------------------------------------------------- feature measurement
# (lifted from riff_lab/chop.py::_features, using oneshot_lab.core)

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


def _whiten(recs):
    """Robust median/IQR whitening (riff_lab/chop.py::_whiten)."""
    X = np.array([[np.log10(r["cent"] + 1), np.log10(r["rms"] + 1e-9),
                   r["flat"], r["trans"], r["dur"]] for r in recs], np.float32)
    med = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    return (X - med) / iqr


def _farthest_point(Xw, n, rng):
    """Greedy farthest-point sampling: pick n spread-out indices."""
    m = len(Xw)
    if m <= n:
        return list(range(m))
    seed = int(rng.integers(m))
    chosen = [seed]
    mind = np.linalg.norm(Xw - Xw[seed], axis=1)
    while len(chosen) < n:
        j = int(np.argmax(mind))
        chosen.append(j)
        mind = np.minimum(mind, np.linalg.norm(Xw - Xw[j], axis=1))
    return chosen


def _attack_trim(m, sr):
    """Trim leading silence to just before the first real sample
    (chop.py::read_slice logic)."""
    if len(m) <= 64:
        return 0
    a = np.abs(m)
    pk = float(a.max())
    if pk <= 1e-6:
        return 0
    above = np.flatnonzero(a > 0.04 * pk)
    if len(above) and above[0] > 0:
        return max(0, int(above[0]) - int(0.001 * sr))
    return 0


# --------------------------------------------------- Stage A: slice

def candidate_windows(n, sr):
    """Multi-scale fixed windows (start, size) spanning the whole file."""
    for scale in SCALES_S:
        w = int(scale * sr)
        hop = max(1, int(w * (1.0 - OVERLAP)))
        for st in range(0, max(1, n - w), hop):
            yield st, w, scale


def slice_file(path, rng, dry_run=False):
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    m = x.mean(axis=1)
    if sr != SR:
        m = core.resample(m, sr / float(SR))
        sr = SR
    n = len(m)
    cands = []
    for st, w, scale in candidate_windows(n, sr):
        d = m[st:st + w]
        rms = float(np.sqrt(np.mean(d * d) + 1e-12))
        if rms < MIN_RMS:
            continue
        ft = _features(d, sr)
        cands.append((st, w, scale, ft))
    if not cands:
        return 0
    Xw = _whiten([c[3] for c in cands])
    keep = _farthest_point(Xw, PER_FILE, rng)
    keep.sort(key=lambda i: cands[i][0])       # chronological output order
    if dry_run:
        return len(keep)
    stem = path.stem
    written = 0
    for i in keep:
        st, w, scale, _ = cands[i]
        d = m[st:st + w]
        t = _attack_trim(d, sr)
        d = d[t:]
        if len(d) < 256:
            continue
        out = RAW_DIR / f"{stem}__s{st:08d}__{scale:g}s.wav"
        core.write_audio(out, d, sr, subtype="PCM_16")
        written += 1
    return written


def stage_slice(dry_run=False):
    files = sorted(IN_DIR.glob("*.wav"))
    if not files:
        raise SystemExit(f"no wavs in {IN_DIR}")
    if not dry_run:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    t0 = time.time()
    for f in files:
        rng = np.random.default_rng(abs(hash(f.stem)) % (2 ** 31))
        k = slice_file(f, rng, dry_run=dry_run)
        total += k
        print(f"  {f.name:30s} -> {k} slices", flush=True)
    verb = "planned" if dry_run else "wrote"
    print(f"Stage A: {verb} {total} raw one-shots "
          f"in {time.time() - t0:.0f}s -> {RAW_DIR}")
    return total


# --------------------------------------------------- Stage B: process

def stage_process():
    if not RAW_DIR.is_dir() or not any(RAW_DIR.glob("*.wav")):
        raise SystemExit(f"no raw slices in {RAW_DIR}; run Stage A first")
    print("Stage B: processing raw one-shots through all 45 transforms "
          "+ 4 mutation chains ...", flush=True)
    return run_folder(str(RAW_DIR), str(PROC_DIR), transforms=("all",),
                      random_chains=4, per_transform_variants=1,
                      max_in_s=3.0, max_out_s=8.0, seed=7, subtype="PCM_16")


# --------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["slice", "process", "all"],
                    default="all")
    ap.add_argument("--dry-run", action="store_true",
                    help="Stage A only: count slices without writing")
    args = ap.parse_args(argv)

    if args.dry_run:
        stage_slice(dry_run=True)
        return 0
    if args.stage in ("slice", "all"):
        stage_slice()
    if args.stage in ("process", "all"):
        stage_process()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
