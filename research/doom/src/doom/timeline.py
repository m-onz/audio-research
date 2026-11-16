"""timeline: segment plans + transitions.

A Plan is a list of segments, each:
    {"scene": "plasma", "params": {...}, "fx": [...],
     "start": 0.0, "end": 30.0, "transition": "cut"|"dissolve"|"glitch"}

`plan_from_audio` is the auto-director: it listens to the piece and
schedules scenes on its energy — quiet passages get ambient scenes,
loud stretches get dense ones, boundaries land on structural changes.
Segments may omit start/end; then they split the length evenly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["normalize_plan", "plan_from_audio"]

AMBIENT = ["scope", "flow", "worms", "metaballs", "lenia"]
DRIVING = ["plasma", "tunnel", "starfield", "fire", "lowpoly", "boids"]
TEXTURE = ["spectro", "rd", "feedback", "mandel"]


def normalize_plan(plan, length: float, default_scene="plasma") -> list[dict]:
    """Normalize a plan to explicit [start, end] segments.

    Semantics: a segment with `start`/`end` is fixed in time; a segment with
    neither is *floating* and evenly splits the gap between the previous
    fixed end and the next fixed start (or the end of the piece). Defaults
    for scene/transition/fx/params are filled in, segments sorted by start.
    """
    plan = [dict(s) for s in plan] or [dict(scene=default_scene)]
    n = len(plan)
    kinds = []
    for seg in plan:
        seg.setdefault("scene", default_scene)
        seg.setdefault("transition", "cut")
        seg.setdefault("fx", None)
        seg.setdefault("params", {})
        kinds.append("fixed" if ("start" in seg or "end" in seg) else "float")

    # walk: fix explicit segments, then distribute floats in each gap
    cursor = 0.0
    ordered = []
    k = 0
    while k < n:
        if kinds[k] == "fixed":
            seg = plan[k]
            seg.setdefault("start", cursor)
            end = seg.get("end")
            seg["end"] = min(float(length if end is None else end), length)
            cursor = seg["end"]
            ordered.append(seg)
            k += 1
        else:
            # collect the float run, find next fixed start
            j = k
            while j < n and kinds[j] == "float":
                j += 1
            gap_end = plan[j]["start"] if j < n and "start" in plan[j] else length
            gap_end = max(cursor, min(gap_end, length))
            count = j - k
            for m, seg in enumerate(plan[k:j]):
                seg["start"] = cursor + (gap_end - cursor) * m / count
                seg["end"] = cursor + (gap_end - cursor) * (m + 1) / count
                ordered.append(seg)
            cursor = gap_end
            k = j
    ordered.sort(key=lambda s: s["start"])
    return ordered


def plan_from_audio(feats, seed=0, n_segments=None, scenes=None,
                    fx_pool=None) -> list[dict]:
    """The auto-director: schedule scenes by the audio's own energy arc."""
    rng = np.random.default_rng(seed)
    n_frames = feats.n_frames
    length = n_frames / feats.fps

    # energy per second -> structural boundaries (largest changes in rms)
    per_sec = max(1, n_frames // int(feats.fps))
    energy = feats.rms[: n_frames // per_sec * per_sec].reshape(-1, per_sec).mean(axis=1)
    # smooth + take change points
    k = max(3, len(energy) // 12)
    smooth = np.convolve(energy, np.ones(k) / k, mode="same")
    d = np.abs(np.diff(smooth))
    n_seg = int(n_segments or max(4, min(10, len(energy) // 25)))
    bounds = [0.0]
    min_gap_s = min(max(4.0, length / 10), length / (2 * n_seg))  # no cuts closer than this
    picks: list[float] = []
    if len(d) > 2:
        order = np.argsort(d)[::-1]
        for p in order[: n_seg * 3]:  # take the strongest changes that respect the gap
            t = float(p) / len(energy) * length
            if all(abs(t - q) >= min_gap_s for q in picks) and t >= min_gap_s and t <= length - min_gap_s:
                picks.append(t)
            if len(picks) >= n_seg - 1:
                break
    # too few structural changes -> fill with even spacing
    while len(picks) < n_seg - 1:
        t = (len(picks) + 1) * length / n_seg
        if all(abs(t - q) >= min_gap_s / 2 for q in picks) and t >= min_gap_s and t <= length - min_gap_s:
            picks.append(t)
        else:
            break
    bounds += sorted(picks)
    bounds.append(length)
    bounds = sorted(set(round(b, 2) for b in bounds))

    ambient = list(AMBIENT)
    driving = list(DRIVING)
    texture = list(TEXTURE)
    rng.shuffle(ambient)
    rng.shuffle(driving)
    rng.shuffle(texture)
    pool = scenes or (ambient + driving + texture)

    plan = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        mid = energy[min(len(energy) - 1, int((a + b) / 2 / (length / max(1, len(energy)))))]
        hot = mid > float(np.median(energy))
        if hot:
            scene = driving.pop(0) if driving else pool[len(plan) % len(pool)]
            fx = ["rgb_split", "glow", "grain"]
        else:
            scene = ambient.pop(0) if ambient else texture.pop(0) if texture else pool[len(plan) % len(pool)]
            fx = ["scanlines", "grain"]
        if not hot and rng.random() < 0.35 and texture:
            scene = texture.pop(0)
        plan.append({
            "scene": scene,
            "start": a, "end": b,
            "transition": rng.choice(["cut", "glitch", "dissolve"])
                         if rng.random() < 0.8 else "cut",
            "fx": fx if fx_pool is None else rng.choice(fx_pool),
        })
    return plan