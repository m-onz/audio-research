"""compose: the render pipeline.

    doom.render(audio="piece.wav", out="video.mp4", preset="glitchcore")

Steps: analyze audio (per-frame features = the sync clock) -> build the
segment plan (explicit, auto-directed, or preset) -> for each frame i:
render segment scene -> run the segment FX chain (both receive frame i's
features) -> apply transitions -> pipe RGB24 into ffmpeg alongside the
audio. One pass, frame-exact.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .audio import analyze_audio, probe_duration
from .fx import build_fx_chain, run_chain
from .presets import PRESETS
from .scenes import make_scene
from .timeline import normalize_plan, plan_from_audio
from .writer import FFmpegWriter, verify_mp4

__all__ = ["render"]

SEED_DEFAULT = 0


def render(
    audio,
    out="doom_out/video.mp4",
    length: float | None = None,
    start: float = 0.0,
    fps: float = 24.0,
    size: tuple = (960, 540),
    plan: list | None = None,
    preset: str | None = None,
    auto_plan: bool = False,
    seed: int = 0,
    crf: int = 18,
    x264_preset: str = "medium",
    verbose: bool = False,
    **scene_overrides,
) -> dict:
    """Render a perfectly synced video for `audio`.

    length: seconds; default = the audio's full duration.
    start: render a WINDOW of the track starting here (for chunk videos:
    the frame clock, features, and muxed audio all shift together).
    plan: segment list (see timeline.py) — or auto_plan=True to let the
    audio's energy structure direct the scene schedule, or preset= for a
    named look.
    """
    audio_path = Path(audio)
    duration = probe_duration(audio_path)
    if length is None:
        length = duration - start
    if start < 0 or start + length > duration + 1e-3:
        raise ValueError(f"window [{start}, {start + length}) exceeds audio (0, {duration:.1f})")
    if preset and not plan and not auto_plan:
        if preset not in PRESETS:
            from .presets import list_presets

            raise ValueError(f"unknown preset {preset!r}; have {list_presets()}")
        plan = PRESETS[preset].get("plan")
        fx_default = PRESETS[preset].get("fx")
    else:
        fx_default = None

    t_start = time.time()
    feats_full = analyze_audio(audio_path, fps=fps)
    feats = feats_full.window(start, length) if start > 0 else feats_full
    n_frames = int(min(feats.n_frames, np.ceil(length * fps)))
    if verbose:
        print(f"[doom] audio: {duration:.1f}s, window [{start}, {start + length}) — "
              f"{feats.n_frames} frames of features, {int(feats.onset.sum())} onsets)")

    if plan is None or auto_plan:
        plan = plan_from_audio(feats, seed=seed)
    plan = normalize_plan(plan, length)
    if verbose:
        for seg in plan:
            fx_show = seg.get("fx") or fx_default or []
            print(f"[doom]   {seg['start']:7.2f}-{seg['end']:7.2f}  {seg['scene']:<10} "
                  f"fx={[f if isinstance(f, str) else f[0] for f in fx_show]}")

    # build scene instances + fx chains per segment
    for k, seg in enumerate(plan):
        params = dict(seg.get("params") or {})
        params.update(scene_overrides)
        seg["_scene"] = make_scene(seg["scene"], size[0], size[1], seed + 100 * k, **params)
        fx_spec = seg.get("fx")
        if fx_spec is None and fx_default is not None:
            fx_spec = fx_default
        seg["_fx"] = build_fx_chain(fx_spec)

    rng = np.random.default_rng(seed)
    writer = FFmpegWriter(out, size[0], size[1], fps, audio_path,
                         crf=crf, preset=x264_preset, audio_start=start)
    last_frame = None
    blend_frames = int(0.5 * fps)  # dissolve length
    seg_at = np.array([seg["start"] for seg in plan])
    for i in range(n_frames):
        t = i / fps
        k = int(np.searchsorted(seg_at, t, side="right") - 1)
        k = max(0, min(k, len(plan) - 1))
        seg = plan[k]
        frame = seg["_scene"].render(feats, i)
        frame = run_chain(seg["_fx"], frame, feats, i, rng)

        # transitions
        tr = seg.get("transition", "cut")
        seg_start_f = int(seg["start"] * fps)
        if tr == "dissolve" and last_frame is not None and i - seg_start_f < blend_frames:
            a = (i - seg_start_f) / blend_frames
            frame = (frame.astype(np.float32) * a + last_frame.astype(np.float32) * (1 - a)).astype(np.uint8)
        elif tr == "glitch" and last_frame is not None and i - seg_start_f < int(0.25 * fps):
            frame = run_chain(build_fx_chain(["block_glitch", "rgb_split"]),
                              frame, feats, i, rng)
        last_frame = frame
        writer.write(frame)
        if verbose and i % max(1, n_frames // 20) == 0:
            print(f"[doom]   frame {i}/{n_frames} ({t:.1f}s)", flush=True)

    stats = writer.close()
    info = verify_mp4(stats["path"])
    result = {
        "path": stats["path"],
        "frames": stats["frames"],
        "video_seconds": info["duration"],
        "audio_seconds": duration,
        "window": [start, start + length],
        "streams": info["streams"],
        "size_bytes": info["size_bytes"],
        "elapsed": round(time.time() - t_start, 1),
        "plan": [{k: v for k, v in seg.items() if not k.startswith("_")} for seg in plan],
    }
    if verbose:
        print(f"[doom] wrote {stats['path']} ({info['duration']:.2f}s video / "
              f"{duration:.2f}s audio) in {result['elapsed']}s")
    return result