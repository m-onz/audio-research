"""normalize — loudness-normalise folders of one-shots to the SAME volume.

Measures each file's integrated loudness with an ITU-R BS.1770 K-weighting
filter (proper perceptual loudness, not naive peak/RMS — so a sparse pluck and
a dense wobble end up equally loud), applies the gain to hit a common target,
then an oversampled soft-limiter holds the true peak under a ceiling. Writes
24-bit in place.

    python3 normalize.py FOLDER [FOLDER ...]        # -16 LUFS, -1 dBFS ceiling
    python3 normalize.py --all                      # every m-onz_* bass batch
    python3 normalize.py FOLDER --target -14 --ceiling -1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hifi                                                 # noqa: E402

ALL_FOLDERS = ["m-onz_bass", "m-onz_bass2", "m-onz_bass3", "m-onz_dub",
               "m-onz_squelch", "m-onz_neuro", "m-onz_neuro2", "m-onz_neuro3"]


# ----------------------------------------------------------------- BS.1770 K-weighting

def _k_coeffs(sr):
    """Pre-filter (high-shelf) + RLB high-pass biquads, computed for `sr`."""
    # stage 1: high-frequency shelving boost
    f0, G, Q = 1681.9744509555319, 3.99984385397, 0.7071752369554193
    K = np.tan(np.pi * f0 / sr)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b1 = np.array([(Vh + Vb * K / Q + K * K) / a0,
                   2.0 * (K * K - Vh) / a0,
                   (Vh - Vb * K / Q + K * K) / a0])
    a1 = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
    # stage 2: RLB high-pass (~38 Hz)
    f0, Q = 38.13547087613982, 0.5003270373253953
    K = np.tan(np.pi * f0 / sr)
    a0 = 1.0 + K / Q + K * K
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
    return (b1, a1), (b2, a2)


def _kweight(x, sr):
    (b1, a1), (b2, a2) = _k_coeffs(sr)
    return lfilter(b2, a2, lfilter(b1, a1, x))


def integrated_lufs(y, sr):
    """BS.1770 integrated loudness (LUFS). Channels summed with unit weights."""
    y = np.asarray(y, np.float32)
    if y.ndim == 1:
        y = y[:, None]
    ms = 0.0
    for c in range(y.shape[1]):
        z = _kweight(y[:, c].astype(np.float64), sr)
        ms += float(np.mean(z * z))
    return -0.691 + 10.0 * np.log10(ms + 1e-12)


# ----------------------------------------------------------------- normalise

def normalize_file(path, target=-16.0, ceiling_db=-1.0):
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    L = integrated_lufs(x, sr)
    if not np.isfinite(L):
        return None
    y = x * (10.0 ** ((target - L) / 20.0))                 # loudness gain
    ceil = 10.0 ** (ceiling_db / 20.0)
    if float(np.max(np.abs(y))) > ceil:                     # oversampled soft limit
        y = hifi.os_apply(y, lambda u: ceil * np.tanh(u / ceil))
    hifi.write(path, y.astype(np.float32), sr)
    return integrated_lufs(y, sr)


def normalize_folder(folder, target, ceiling, verbose=True):
    files = sorted(p for p in Path(folder).glob("*.wav") if not p.name.startswith("_"))
    outs = []
    for f in files:
        try:
            after = normalize_file(f, target, ceiling)
            if after is not None:
                outs.append(after)
        except Exception as e:
            print(f"  ! {f.name}: {e}", file=sys.stderr)
    if verbose and outs:
        a = np.array(outs)
        print(f"  {folder}: {len(outs)} files -> "
              f"LUFS mean={a.mean():.1f} spread±{a.std():.2f} (target {target})")
    return outs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Loudness-normalise folders to one volume.")
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--all", action="store_true", help="all m-onz_* bass batches")
    ap.add_argument("--target", type=float, default=-16.0, help="LUFS target")
    ap.add_argument("--ceiling", type=float, default=-1.0, help="peak ceiling dBFS")
    args = ap.parse_args(argv)
    folders = ALL_FOLDERS if args.all else args.folders
    if not folders:
        ap.error("give FOLDER(s) or --all")
    for d in folders:
        if Path(d).is_dir():
            normalize_folder(d, args.target, args.ceiling)
        else:
            print(f"  (skip missing {d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
