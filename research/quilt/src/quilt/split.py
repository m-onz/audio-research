"""split: the in-place processor — the corpus folder becomes the pile.

Every original is cut *in place*: its chunks are written next to it,
verified, and only then is the original deleted — the folder grows in
file count, not in bytes (splitting, never duplicating). The scissor
is chosen per file, by what the material is:

    onset    event-dense material (many event-sized pieces) -> cut
             at the hits — but only if the onsets carve *events*
             (median piece < ONSET_EVENT_MAX): librosa hears phantom
             onsets in steady drones, and a meaningless cut is not a
             scissor's job
    silence  no events, but real loud/quiet structure — the plan must
             have holes (quiet material actually dropped); a long loud
             run merely subdivided by MAX_SEG is not silence structure,
             it is continuous material (see fixed)
    fixed    continuous material longer than --min-keep -> 2 s windows
    (keep)   short files stay whole — a one-shot is already a piece

Naming carries provenance: `orig.wav` becomes
`orig__onset_0000.wav, orig__onset_0001.wav, ...` — the scissor's
name and the chunk's order survive in the filename, which is how the
`follow` walker walks the corpus's own sequences later, and how walks
can be filtered by scissor (`--strategy onset`).

Idempotent: chunk files carry the marker suffix and are skipped on a
re-run; files that were kept whole simply keep evaluating to "keep".
Failures are logged-not-fatal and never delete the original. An audit
JSONL records every decision, so a split is always accountable:

    {"orig": "x.wav", "action": "split", "strategy": "onset",
     "n": 6, "dur": 3.0, "chunk_durs": [...]}

The law still holds: no processing, no fades, no resampling — a chunk
is a sample-exact span of its original, re-wrapped in a wav header.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .chunking import (chunk_onset, chunk_silence, chunk_fixed, MIN_DUR)
from .naming import CHUNK_RE, is_chunk_file
from .index import iter_audio
from .io import read_audio, write_audio
from .forkmap import fork_map
from echo.probes import SR

__all__ = ["split_corpus", "CHUNK_RE", "is_chunk_file"]

# an onset plan only wins when it is event-cutting: pieces this small
# on the median. Steady drones trigger librosa's flux detector a few
# times per minute; those phantom onsets must not masquerade as events.
ONSET_EVENT_MAX = 3.0

# a silence plan only wins when it has holes: quiet material actually
# dropped (a gate that fired). A single loud run subdivided by MAX_SEG
# tiles contiguously with no holes — that is continuous material, and
# the fixed scissor's business, not the gate's.
SILENCE_HOLE_MIN = 0.08    # s; real gates drop >= SIL_MIN_SIL (0.12 s)

# (parse_chunk_name lives in naming.py — split only writes the
# convention; index and the walkers read it)


def split_corpus(corpus_dir, min_keep: float = 8.0, fixed_dur: float = 2.0,
                 max_files: int | None = None, dry_run: bool = False,
                 audit_path="quilt_out/split_audit.jsonl",
                 workers: int | None = None, log=print) -> dict:
    """Cut the corpus in place. DESTRUCTIVE by design: originals are
    replaced by their chunks (after verification). dry_run reports
    without touching anything. Idempotent — rerun anytime. Large
    corpora parallelize across files (deterministic: ordered imap,
    identical results for any worker count)."""
    root = Path(corpus_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"corpus {root} does not exist")
    files = iter_audio(root)
    todo = [f for f in files if not is_chunk_file(f.name)]
    if max_files is not None and len(todo) > int(max_files):
        todo = todo[:int(max_files)]
    log(f"splitting {len(todo)} original file(s) in place under {root} "
        f"({'DRY RUN' if dry_run else 'originals are replaced by chunks'})")

    audit = None
    if not dry_run:
        ap = Path(audit_path)
        ap.parent.mkdir(parents=True, exist_ok=True)
        audit = open(ap, "a")

    tasks = [(f, min_keep, fixed_dur, dry_run) for f in todo]
    n_proc = _auto_workers(workers, len(todo))
    t0 = time.time()
    counts = {"processed": 0, "split": 0, "kept": 0, "chunks": 0,
              "dust_kept": 0, "failed": 0,
              "already_chunks": len(files) - len(todo), "workers": n_proc}
    try:
        results = fork_map(_split_one, tasks, n_proc, batch=200, log=log)
        for res in results:
            counts["processed"] += 1
            if res is None:                             # lost slice
                counts["failed"] += 1
                continue
            action = res["action"]
            if action == "split":
                counts["split"] += 1
                counts["chunks"] += len(res["chunk_durs"])
                _audit(audit, res["rel"], "split", res["strategy"],
                       res["dur"], res["chunk_durs"])
            elif action == "would-split":
                counts["split"] += 1
                counts["chunks"] += len(res["chunk_durs"])
                _audit(audit, res["rel"], "would-split", res["strategy"],
                       res["dur"], res["chunk_durs"])
            elif action == "keep":
                counts["kept"] += 1
                _audit(audit, res["rel"], "keep", None, res["dur"])
            elif action == "dust":
                counts["dust_kept"] += 1
                _audit(audit, res["rel"], "keep", None, res["dur"])
            else:                                          # fail
                counts["failed"] += 1
                log(f"  ! kept original {res['rel']}: {res.get('error')}")
                _audit(audit, res["rel"], "fail", res.get("strategy"),
                       res.get("dur"))
    finally:
        if audit is not None:
            audit.close()
    counts["elapsed_s"] = round(time.time() - t0, 1)
    log(f"split done: {counts}")
    return counts


def _auto_workers(workers, n_tasks) -> int:
    """Parallel only pays off on real piles; tiny corpora stay
    single-process (tests, smoke runs — no worker spin-up cost)."""
    if workers is not None and int(workers) >= 1:
        return int(workers)
    if n_tasks < 200:
        return 1
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _split_one(task):
    """One original -> one decision, in a worker (or the main process).
    Writes chunks + deletes the original only after every chunk
    verified. Never raises: failures are decisions, not exceptions."""
    path, min_keep, fixed_dur, dry = task
    try:
        audio = read_audio(path, SR)
        total = len(audio) / SR
        action, strategy, segs = _plan(audio, total, min_keep, fixed_dur)
        if action != "split":
            kind = "keep" if total >= MIN_DUR else "dust"
            return {"rel": path.name, "action": kind, "dur": total,
                    "strategy": None, "chunk_durs": None}
        durs = [round(b - a, 3) for a, b in segs]
        if dry:
            return {"rel": path.name, "action": "would-split",
                    "strategy": strategy, "dur": total, "chunk_durs": durs}
        ok, err = _write_chunks(path.parent, path, strategy, segs, audio)
        if not ok:
            return {"rel": path.name, "action": "fail", "error": err,
                    "strategy": strategy, "dur": total, "chunk_durs": None}
        path.unlink()                              # verified: replaced
        return {"rel": path.name, "action": "split", "strategy": strategy,
                "dur": total, "chunk_durs": durs}
    except Exception as e:
        return {"rel": path.name, "action": "fail", "error": str(e),
                "strategy": None, "dur": None, "chunk_durs": None}


def _plan(audio, total: float, min_keep: float, fixed_dur: float):
    """Choose the scissor by what the material is. Returns
    ("split", strategy, segs) or ("keep", None, None)."""
    if total < MIN_DUR:
        return "keep", None, None                     # dust-sized: a piece
    segs = chunk_onset(audio, SR)
    if len(segs) >= 2 and float(np.median(
            [t1 - t0 for t0, t1 in segs])) < ONSET_EVENT_MAX:
        return "split", "onset", segs
    segs = chunk_silence(audio, SR)
    holes = total - sum(t1 - t0 for t0, t1 in segs)
    if len(segs) >= 2 and holes > SILENCE_HOLE_MIN:
        return "split", "silence", segs                  # real gates fired
    if total <= min_keep:
        return "keep", None, None                     # already piece-sized
    return "split", "fixed", chunk_fixed(audio, SR, dur=fixed_dur)


def _write_chunks(root, orig: Path, strategy: str, segs, audio):
    """Write one wav per segment (sample-exact spans), verify each,
    and remove partial writes on any failure. The original is only
    the caller's business once every chunk verified."""
    stem = orig.stem
    targets = [orig.with_name(f"{stem}__{strategy}_{i:04d}.wav")
               for i in range(len(segs))]
    for t in targets:
        if t.exists():
            return False, f"collision {t.name}"
    written = []
    try:
        for t, (a, b) in zip(targets, segs):
            A, B = int(round(a * SR)), int(round(b * SR))
            write_audio(t, audio[A:B], SR)
            written.append(t)
        for t, (a, b) in zip(targets, segs):
            frames = _frames(t)
            expect = int(round((b - a) * SR))
            if abs(frames - expect) > 2:
                raise ValueError(f"{t.name}: {frames} frames, "
                                 f"expected {expect}")
        return True, None
    except Exception as e:
        for t in written:
            t.unlink(missing_ok=True)                 # no orphans, ever
        return False, str(e)


def _frames(path) -> int:
    import soundfile as sf
    return sf.info(str(path)).frames


def _audit(audit, orig, action, strategy, dur, chunk_durs=None):
    if audit is None:
        return
    audit.write(json.dumps({"orig": orig, "action": action,
                            "strategy": strategy, "dur": dur,
                            "chunk_durs": chunk_durs}) + "\n")