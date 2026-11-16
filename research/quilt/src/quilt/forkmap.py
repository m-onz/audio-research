"""forkmap: manual fork parallelism — no multiprocessing.Pool machinery.

Measured on this machine (homebrew CPython 3.11.14, macOS 26): the
`multiprocessing` Pool machinery is not reliable here — spawn's
worker bootstrap deadlocks reading its preparation pipe under any
quilt entry point (console script, `python -c`, nohup), and even the
fork context comes up with its internal handler threads and worker
children silently missing, leaving the parent waiting on results
that can never arrive. Plain `os.fork()` in a script works perfectly
(the /tmp/forktest proof), so the machinery is the problem, not the
metal.

The replacement is the oldest trick in the book:

    fork n children, give each an interleaved slice of the batch,
    each child streams its results to a temp pickle file, the parent
    reaps, reads, reorders.

No queues, no handler threads, no __main__ re-import, no pickled
Process objects. Deterministic: results come back in task order for
ANY worker count. Batches bound the blast radius of a crashed child
(its slice is lost and reported — callers treat failures as
decisions, never exceptions). fork is safe here: the split/index
parents are pure Python (numpy/soundfile, no GUI/CoreAudio/torch).
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

__all__ = ["fork_map"]


def fork_map(worker, tasks, n_proc: int, batch: int = 200,
             log=None, label: str = ""):
    """tasks -> [results] in task order, computed by n_proc forked
    children in batches. worker(task) must return something picklable
    and should never raise (decisions, not exceptions) — a child that
    dies loses its slice: those results come back None and the round
    is reported. log, when given, receives one progress line per
    batch with the running rate/eta."""
    tasks = list(tasks)
    results: list = [None] * len(tasks)
    if not tasks:
        return results
    if n_proc <= 1 or len(tasks) <= 1:
        return [worker(t) for t in tasks]

    done = 0
    import time
    t0 = time.time()
    tmpdir = Path(tempfile.mkdtemp(prefix="quilt-forkmap-"))
    try:
        for base in range(0, len(tasks), batch):
            chunk = list(enumerate(tasks[base: base + batch], start=base))
            _run_batch(worker, chunk, n_proc, tmpdir, results)
            done = min(base + batch, len(tasks))
            if log is not None:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(tasks) - done) / max(rate, 1e-9) / 60.0
                log(f"[{done}/{len(tasks)}]{label} {rate:.1f}/s, "
                    f"eta {eta:.1f}m")
    finally:
        try:
            tmpdir.rmdir()
        except OSError:
            pass
    return results


def _run_batch(worker, indexed_tasks, n_proc: int, tmpdir: Path,
               results: list):
    """One fork round: n children, interleaved slices (i % n), results
    streamed to per-child pickle files, reaped and reordered."""
    pids = []
    for w in range(n_proc):
        mine = [(i, t) for k, (i, t) in enumerate(indexed_tasks)
                if k % n_proc == w]
        if not mine:
            continue
        path = tmpdir / f"r{w}.pkl"
        pid = os.fork()
        if pid == 0:                                   # ---- child
            try:
                with open(path, "wb") as f:
                    for i, t in mine:
                        pickle.dump((i, worker(t)), f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                os._exit(0)
            except BaseException:
                os._exit(1)                            # slice lost
        pids.append((pid, path))
    lost = 0
    for w, (pid, path) in enumerate(pids):             # ---- parent
        _, status = os.waitpid(pid, 0)
        if status == 0 and path.exists():
            try:
                with open(path, "rb") as f:
                    while True:
                        try:
                            i, r = pickle.load(f)
                        except EOFError:
                            break
                        results[i] = r
            except Exception:
                lost += 1
        else:
            lost += 1
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    if lost:
        import sys
        print(f"forkmap: {lost} worker slice(s) lost this round — "
              f"their tasks report as failures (rerun is idempotent)",
              file=sys.stderr, flush=True)