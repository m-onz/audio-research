"""index: the cutting room is gone — the pile IS the corpus.

`quilt split` already cut the corpus in place; every file under the
corpus folder is now a ready-made piece. Indexing means: read each
piece (fast path: libsndfile on the pile's native 44.1 kHz wav), run
echo's ear over it once, and pin it into an EchoBank with provenance
parsed from the filename — which original it came from, which scissor
cut it, which number it was. The bank then fits its metric on the
whole population (z-norm per dim — the axes that never move go
inert), so distances between pieces mean what they mean on this
corpus.

The bank stores no audio — only (file, 0, dur) + vector. Rendering
re-reads the pile at walk time. A provenance sidecar
(bank.sqlite.provenance.json) carries the census.

Failures are logged-not-fatal (echo's law). Deterministic: sorted
files, seeded stride subsampling — the same pile yields the
bit-identical bank.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from echo import encode, EchoBank
from echo.probes import SR

from .chunking import MIN_DUR
from .io import read_audio
from .naming import parse_chunk_name
from .forkmap import fork_map

__all__ = ["build_bank", "iter_audio", "provenance_path", "AUDIO_EXT"]

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".aiff", ".aif",
             ".m4a", ".mp4", ".opus", ".au", ".wma"}

_COMMIT_EVERY = 250                # sqlite commits: batched, not per piece


def _auto_workers(workers, n_tasks) -> int:
    """Parallel only pays off on real piles; tiny corpora stay
    single-process (tests, smoke runs — no worker spin-up cost)."""
    if workers is not None and int(workers) >= 1:
        return int(workers)
    if n_tasks < 200:
        return 1
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _index_one(task):
    """One pile file -> (vector, meta) in a worker (or the main
    process). Never raises: failures are decisions, not exceptions."""
    path, root = task
    rel = path.relative_to(root).as_posix()
    try:
        audio = read_audio(path, SR)
        if len(audio) < int(MIN_DUR * SR):
            return {"kind": "dust", "rel": rel}
        dur = len(audio) / SR
        strategy, orig, idx = parse_chunk_name(rel)
        return {"kind": "piece", "rel": rel,
                "vec": encode(audio, SR),
                "meta": {"file": rel, "t0": 0.0, "t1": round(dur, 6),
                         "dur": round(dur, 6), "strategy": strategy,
                         "orig": orig, "idx": idx}}
    except Exception as e:
        return {"kind": "fail", "rel": rel, "error": str(e)}


def provenance_path(bank_path) -> Path:
    """The sidecar that makes a bank re-renderable: corpus root + census."""
    return Path(str(bank_path) + ".provenance.json")


def iter_audio(corpus_dir) -> list:
    """Every decodable audio file under the corpus, sorted (order is
    part of determinism)."""
    root = Path(corpus_dir)
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_EXT)


def build_bank(corpus_dir, out_path, max_files: int | None = None,
               force: bool = False, workers: int | None = None,
               log=print) -> EchoBank:
    """The pile -> fitted bank. One file = one piece.

    max_files  deterministic stride subsample (smoke runs)
    force      replace an existing bank (default: refuse)
    workers    parallel encoding across files — deterministic:
              ordered imap, identical bank for any worker count
    """
    root = Path(corpus_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"corpus {root} does not exist")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prov_p = provenance_path(out)
    for p in (out, prov_p):
        if p.exists():
            if not force:
                raise FileExistsError(f"{p} exists — pass force=True "
                                      f"(or --force) to replace it")
            p.unlink()

    files = iter_audio(root)
    if not files:
        raise FileNotFoundError(f"no audio under {root}")
    if max_files is not None and len(files) > int(max_files):
        sel = np.unique(np.round(np.linspace(
            0, len(files) - 1, int(max_files))).astype(int))
        files = [files[i] for i in sel]
        log(f"subsampled {len(files)} of the pile (deterministic stride)")

    bank = EchoBank(str(out))
    t_start = time.time()
    n_fail = n_dust = 0
    min_samples = int(MIN_DUR * SR)
    tasks = [(f, root) for f in files]
    n_proc = _auto_workers(workers, len(files))
    log(f"indexing {len(files)} pieces from {root} ({n_proc} worker"
        f"{'s' if n_proc > 1 else ''})")
    results = fork_map(_index_one, tasks, n_proc, batch=2000, log=log)
    for res in results:
        if res is None:                               # lost worker slice
            n_fail += 1
            continue
        if res["kind"] == "piece":
            bank.add(res["vec"], res["meta"], role="corpus",
                     kind="chunk", commit=False)
        elif res["kind"] == "dust":
            n_dust += 1
        else:                                          # fail
            n_fail += 1
            log(f"  ! read failed {res['rel']}: {res.get('error')}")
    bank._con.commit()

    X = bank.matrix()
    if len(X) >= 2:
        bank.fit(X)                                   # z-norm on the pieces
    strategies = sorted({m["strategy"] for m in bank.metas})
    prov = {
        "quilt": "index", "mode": "pile", "corpus_root": str(root),
        "strategies": strategies,
        "files_seen": len(files), "files_failed": n_fail, "dust": n_dust,
        "chunks": int(len(bank)), "dim": int(bank.metric.dim),
        "seconds_total": float(sum(m.get("dur", 0.0)
                                   for m in bank.metas)) if len(bank) else 0.0,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    prov_p.write_text(json.dumps(prov, indent=1))
    log(f"bank written: {out} ({len(bank)} pieces, "
        f"{prov['seconds_total'] / 3600:.2f}h of material, "
        f"scissors {','.join(strategies)})")
    return bank