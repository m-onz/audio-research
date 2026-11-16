"""uniq — sort a folder of sound files by UNIQUENESS, and remove duplicates.

One job: given a folder of audio, embed every file in a perceptual space, then
rank / dedup / curate by novelty. Built for open-ended algorithms and novelty
search — the thing all the earlier experiments re-invented ad-hoc (bass_lab
donor scoring, squelch_lab.curate, chop.pick_fresh, process_m_onz farthest-point)
unified into one reusable tool with a real embedding.

Embedding: the project's `echo` organ — `echo.encode` → 556-d perceptual vector
(MFCC trajectory + spectral stats + silhouette + dynamics + chroma + spatial),
compared in `echo.EchoMetric`'s fitted, per-block-weighted z-normalized space
(the designed distance). Falls back to a librosa/numpy embedding if echo is
unavailable.

Library:
    import uniq
    paths, X = uniq.embed_folder("m-onz_neuro2")
    Z, _ = uniq.whiten(X)
    order = uniq.farthest_order(Z)           # most-unique-first indices
    nov = uniq.novelty_scores(Z, k=15)       # per-file novelty (kNN mean dist)
    keep = uniq.dedup(Z, quantile=0.1)       # indices with near-dupes removed

CLI:
    python3 uniq.py FOLDER                    # report + write _uniqueness.json
    python3 uniq.py FOLDER --dedup            # move near-duplicates to _dupes/
    python3 uniq.py FOLDER --dedup --delete   # ...delete them instead
    python3 uniq.py FOLDER --keep 100         # keep 100 most-unique (rest -> _dupes)
    python3 uniq.py FOLDER --copy sorted/     # copy all, unique-first, rank-named
    python3 uniq.py FOLDER --novelty          # print per-file novelty ranking
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                        # oneshot_lab
sys.path.insert(0, str(HERE.parent / "echo" / "src"))       # echo

try:
    import echo
    _HAVE_ECHO = True
    EMBED_DIM = echo.DIM
except Exception:                                          # pragma: no cover
    _HAVE_ECHO = False
    EMBED_DIM = None

AUDIO = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}


# ----------------------------------------------------------------- embedding

def _encode_file(path):
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if _HAVE_ECHO:
        return np.asarray(echo.encode(x, sr), np.float32)
    return _encode_fallback(x, sr)


def _encode_fallback(x, sr):
    """librosa/numpy perceptual vector if echo is unavailable."""
    import librosa
    m = x.mean(axis=1)
    mf = librosa.feature.mfcc(y=m, sr=sr, n_mfcc=20)
    sc = librosa.feature.spectral_contrast(y=m, sr=sr)
    ch = librosa.feature.chroma_stft(y=m, sr=sr)
    feats = [mf.mean(1), mf.std(1), sc.mean(1), ch.mean(1),
             [float(np.sqrt(np.mean(m * m) + 1e-9)),
              float(librosa.feature.spectral_flatness(y=m).mean())]]
    return np.concatenate([np.asarray(f, np.float32).ravel() for f in feats])


def embed_folder(folder, recursive=False, cache=True, verbose=True):
    """Embed every audio file. Returns (paths, X[n, dim]). Cached per folder
    keyed by path+mtime so re-runs are instant."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    files = sorted(p for p in it
                   if p.suffix.lower() in AUDIO and p.is_file()
                   and not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"no audio in {folder}")
    cache_path = folder / "_uniq_cache.npz"
    cached = {}
    if cache and cache_path.exists():
        try:
            z = np.load(cache_path, allow_pickle=True)
            for p, mt, v in zip(z["paths"], z["mtimes"], z["X"]):
                cached[str(p)] = (float(mt), np.asarray(v, np.float32))
        except Exception:
            cached = {}
    X, keep_paths, t0, n_new = [], [], time.time(), 0
    for i, f in enumerate(files):
        key, mt = str(f), f.stat().st_mtime
        if key in cached and abs(cached[key][0] - mt) < 1e-6:
            vec = cached[key][1]
        else:
            try:
                vec = _encode_file(f)
                n_new += 1
            except Exception as e:
                if verbose:
                    print(f"  ! skip {f.name}: {e}", file=sys.stderr)
                continue
        X.append(vec)
        keep_paths.append(f)
        if verbose and (i + 1) % 1000 == 0:
            print(f"  embedded {i + 1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(X, np.float32)
    if cache and n_new:
        np.savez(cache_path, paths=[str(p) for p in keep_paths],
                 mtimes=[p.stat().st_mtime for p in keep_paths], X=X)
    if verbose:
        print(f"embedded {len(keep_paths)} files "
              f"({'echo 556-d' if _HAVE_ECHO else 'librosa'}, "
              f"{n_new} new) in {time.time()-t0:.0f}s")
    return keep_paths, X


# ----------------------------------------------------------------- metric space

def whiten(X):
    """Map raw embeddings into the designed comparison space: per-dim
    z-normalization × sqrt(per-block weights) (echo.EchoMetric), else robust
    median/IQR whitening."""
    X = np.asarray(X, np.float64)
    if _HAVE_ECHO and X.shape[1] == echo.DIM:
        m = echo.EchoMetric().fit(X)
        scale = getattr(m, "_scale", np.ones(X.shape[1]))
        Z = (X - m.mu) / m.sigma * scale
        return Z.astype(np.float32), m
    med = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, 0) - np.percentile(X, 25, 0)) / 1.349
    iqr[iqr < 1e-6] = 1.0
    return ((X - med) / iqr).astype(np.float32), None


def novelty_scores(Z, k=15):
    """Per-item novelty = mean distance to its k nearest neighbours (the
    standard novelty-search sparseness measure)."""
    Z = np.asarray(Z, np.float32)
    n = len(Z)
    k = min(k, n - 1)
    if k < 1:
        return np.zeros(n, np.float32)
    tree = cKDTree(Z)
    d, _ = tree.query(Z, k=k + 1)                          # +1 includes self
    return d[:, 1:].mean(axis=1).astype(np.float32)


def farthest_order(Z, scores=None):
    """Greedy farthest-point ordering: most-unique-first. Seeds from the
    highest-novelty item, then repeatedly appends the item farthest (in the
    metric space) from everything chosen so far."""
    Z = np.asarray(Z, np.float32)
    n = len(Z)
    if n <= 1:
        return list(range(n))
    if scores is None:
        scores = novelty_scores(Z)
    seed = int(np.argmax(scores))
    order = [seed]
    mind = np.linalg.norm(Z - Z[seed], axis=1)
    mind[seed] = -1.0
    while len(order) < n:
        j = int(np.argmax(mind))
        order.append(j)
        mind = np.minimum(mind, np.linalg.norm(Z - Z[j], axis=1))
        mind[order] = -1.0
    return order


def _auto_thresh(Z, quantile):
    tree = cKDTree(Z)
    d, _ = tree.query(Z, k=2)
    nn = d[:, 1]
    return float(np.quantile(nn, quantile))


def dedup(Z, thresh=None, quantile=0.10, scores=None):
    """Remove near-duplicates. Walk items most-unique-first; keep an item only
    if no already-kept item lies within `thresh` of it. `thresh` defaults to
    the `quantile` of the nearest-neighbour distance distribution."""
    Z = np.asarray(Z, np.float32)
    n = len(Z)
    if n <= 1:
        return list(range(n))
    if thresh is None:
        thresh = _auto_thresh(Z, quantile)
    order = farthest_order(Z, scores)
    tree = cKDTree(Z)
    kept, kept_set = [], set()
    for i in order:
        near = tree.query_ball_point(Z[i], thresh)
        if any((j in kept_set) for j in near if j != i):
            continue
        kept.append(i)
        kept_set.add(i)
    return kept, thresh


# ----------------------------------------------------------------- CLI

def _relocate(paths, idxs, dst, order=None, move=False, rank=False):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    seq = order if order is not None else idxs
    w = len(str(len(seq)))
    for rank_i, i in enumerate(seq):
        src = Path(paths[i])
        name = (f"{rank_i:0{w}d}_{src.name}" if rank else src.name)
        (shutil.move if move else shutil.copy2)(str(src), str(dst / name))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sort/dedup a folder by uniqueness.")
    ap.add_argument("folder")
    ap.add_argument("-R", "--recursive", action="store_true")
    ap.add_argument("--dedup", action="store_true", help="remove near-duplicates")
    ap.add_argument("--keep", type=int, default=None,
                    help="keep only N most-unique (farthest-point)")
    ap.add_argument("--thresh", type=float, default=None,
                    help="dedup distance threshold (default: auto from --quantile)")
    ap.add_argument("--quantile", type=float, default=0.10,
                    help="auto-thresh = this quantile of nearest-neighbour dists")
    ap.add_argument("--delete", action="store_true",
                    help="delete rejects instead of moving them to _dupes/")
    ap.add_argument("--copy", metavar="DST", default=None,
                    help="copy all files unique-first, rank-renamed, into DST")
    ap.add_argument("--move", metavar="DST", default=None,
                    help="like --copy but move")
    ap.add_argument("--novelty", action="store_true", help="print novelty ranking")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("-k", type=int, default=15, help="kNN for novelty")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    paths, X = embed_folder(folder, args.recursive, cache=not args.no_cache)
    Z, _ = whiten(X)
    scores = novelty_scores(Z, k=args.k)
    order = farthest_order(Z, scores)

    # always write a uniqueness manifest (most-unique-first)
    rank = {int(i): r for r, i in enumerate(order)}
    out = [{"file": Path(paths[i]).name, "novelty": round(float(scores[i]), 4),
            "uniqueness_rank": rank[i]} for i in range(len(paths))]
    out.sort(key=lambda d: d["uniqueness_rank"])
    (folder / "_uniqueness.json").write_text(json.dumps(out, indent=1))
    print(f"\nuniqueness: {len(paths)} files, novelty "
          f"min={scores.min():.2f} med={np.median(scores):.2f} max={scores.max():.2f}")

    if args.novelty:
        for d in out[:30]:
            print(f"  #{d['uniqueness_rank']:3d}  nov={d['novelty']:.3f}  {d['file']}")
        if len(out) > 30:
            print(f"  ... (+{len(out)-30})")

    if args.copy or args.move:
        dst = args.copy or args.move
        _relocate(paths, list(range(len(paths))), dst, order=order,
                  move=bool(args.move), rank=True)
        print(f"wrote {len(order)} files unique-first -> {dst}")
        return 0

    if args.dedup or args.keep:
        if args.keep:
            keep = order[: args.keep]
            thresh = None
        else:
            keep, thresh = dedup(Z, thresh=args.thresh, quantile=args.quantile,
                                 scores=scores)
        keepset = set(keep)
        rejects = [i for i in range(len(paths)) if i not in keepset]
        msg = f"keep {len(keep)}, remove {len(rejects)}"
        if thresh is not None:
            msg += f" (thresh={thresh:.2f})"
        print(msg)
        if args.delete:
            for i in rejects:
                Path(paths[i]).unlink(missing_ok=True)
        else:
            _relocate(paths, rejects, folder / "_dupes", move=True)
            print(f"  moved rejects -> {folder / '_dupes'}")
        # refresh cache so stale paths don't linger
        (folder / "_uniq_cache.npz").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
