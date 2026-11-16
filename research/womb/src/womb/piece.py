"""piece: original audiovisual composition from organs.

A piece is a timeline of organ segments — like a doom plan, but every
segment GENERATES both its sound and its picture from one evolving system:

    womb.generate(plan=[{"organ": "waveplate"}, {"organ": "swarm"}],
                  length=90, out="piece.mp4", corpus="../loom/loom_out")

Segments without explicit times split the piece evenly. Organs may share
the corpus (grain material from your loom output, the EP, anything), or
run purely on their own synthesis. Every organ gets its own seeded state;
the piece is deterministic and fully offline.
"""

from __future__ import annotations

import time

import numpy as np

from .organs import make_organ
from .writer import AVWriter

__all__ = ["generate", "normalize_plan"]


def normalize_plan(plan, length: float) -> list[dict]:
    """Even-split missing times (segments may also pin start/end)."""
    plan = [dict(s) for s in plan]
    n = len(plan)
    out = []
    for k, seg in enumerate(plan):
        if "start" not in seg and "end" not in seg:
            seg["start"], seg["end"] = k * length / n, (k + 1) * length / n
        else:
            seg.setdefault("start", 0.0 if k == 0 else out[-1]["end"])
            seg.setdefault("end", length)
            seg["end"] = min(float(seg["end"]), length)
        seg.setdefault("organ", seg.get("scene", "waveplate"))
        seg.setdefault("params", {})
        out.append(seg)
    return sorted(out, key=lambda s: s["start"])


def generate(plan, length: float, out="womb_out/piece.mp4", fps: float = 24.0,
             size: tuple = (960, 540), seed: int = 0, corpus=None,
             normalize_audio: bool = True, verbose: bool = False) -> dict:
    """Render an original A/V piece. Returns render stats."""
    plan = normalize_plan(plan, float(length))
    n_frames = int(round(length * fps))
    writer = AVWriter(out, size[0], size[1], fps, sr=44100)

    seg_starts = np.array([s["start"] for s in plan])
    current, seg_i = None, -1
    t0 = time.time()
    for i in range(n_frames):
        t = i / fps
        k = int(np.searchsorted(seg_starts, t, side="right") - 1)
        k = max(0, min(k, len(plan) - 1))
        if k != seg_i:
            seg = plan[k]
            seg_i = k
            current = make_organ(
                seg["organ"], size[0], size[1], fps,
                seed=seed + 977 * k, corpus=corpus, **seg.get("params", {}),
            )
            if verbose:
                print(f"[womb] {t:6.1f}s  organ: {seg['organ']}", flush=True)
        current.step(i)
        hop_audio = current.emit()
        frame = current.render()
        writer.write(frame, hop_audio)
        if verbose and i % max(1, n_frames // 10) == 0:
            print(f"[womb]   frame {i}/{n_frames}", flush=True)

    if normalize_audio:
        peak = float(np.abs(writer.audio).max())
        if peak > 1.0:
            writer.audio = writer.audio / peak * 0.95
    stats = writer.close()
    stats["elapsed"] = round(time.time() - t0, 1)
    stats["plan"] = [{kk: v for kk, v in s.items()} for s in plan]
    if verbose:
        print(f"[womb] wrote {stats['path']} ({stats['seconds']:.1f}s, "
              f"{stats['frames']} frames) in {stats['elapsed']}s")
    return stats