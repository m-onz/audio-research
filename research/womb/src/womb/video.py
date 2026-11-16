"""video: decode frames + measure them (womb's eyes).

Everything here is ffmpeg piped to numpy — no OpenCV, no imageio. Frames
are decoded in blocks and reduced immediately, so an hour of video never
sits in memory.

`FrameStats` is the video twin of doom's AudioFeatures: per-frame luma,
motion (frame difference), block-displacement energy, and entropy — the
quantities sonification and drift telemetry feed on.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

__all__ = ["probe_video", "decode_frames", "decode_luma", "FrameStats", "video_stats"]

EPS = 1e-9


def probe_video(path) -> dict:
    """Codec/geometry summary via ffprobe."""
    import json

    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
         "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {
        "path": str(path),
        "duration": float(info["format"]["duration"]),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": _parse_fps(v.get("avg_frame_rate", "24/1")),
        "codec": v["codec_name"],
    }


def _parse_fps(rate: str) -> float:
    try:
        num, den = rate.split("/")
        return float(num) / max(1e-9, float(den))
    except Exception:
        return 24.0


def decode_luma(path, scale_w: int = 160, scale_h: int = 90, block: int = 48):
    """Stream-decode a video to small grayscale frames.

    Returns (frames, fps): frames float32 (N, scale_h, scale_w) in 0..1.
    Decodes in blocks of `block` frames so memory stays tiny.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"scale={scale_w}:{scale_h},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frames = []
    n_bytes = scale_w * scale_h
    while True:
        raw = proc.stdout.read(n_bytes * block)
        if not raw:
            break
        n = len(raw) // n_bytes
        frames.append(np.frombuffer(raw, np.uint8)[: n * n_bytes].reshape(n, scale_h, scale_w))
    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}")
    all_frames = np.concatenate(frames) if frames else np.zeros((0, scale_h, scale_w), np.uint8)
    info = probe_video(path)
    return all_frames.astype(np.float32) / 255.0, info["fps"]


def decode_frames(path, fps: float | None = None, size: tuple = (320, 180), block: int = 36):
    """Stream-decode RGB frames at (optionally) a target fps."""
    vf = f"scale={size[0]}:{size[1]}"
    if fps:
        vf = f"fps={fps},{vf}"
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path), "-vf", vf,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    n_bytes = size[0] * size[1] * 3
    frames = []
    while True:
        raw = proc.stdout.read(n_bytes * block)
        if not raw:
            break
        n = len(raw) // n_bytes
        frames.append(np.frombuffer(raw, np.uint8)[: n * n_bytes].reshape(n, size[1], size[0], 3))
    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}")
    return np.concatenate(frames) if frames else np.zeros((0, size[1], size[0], 3), np.uint8)


@dataclass
class FrameStats:
    fps: float
    n_frames: int
    luma: np.ndarray       # (N,) mean brightness 0..1
    motion: np.ndarray     # (N,) frame-difference energy 0..1-ish
    entropy: np.ndarray    # (N,) histogram entropy of luma (bits)
    flow: np.ndarray       # (N,) mean |horizontal displacement| (px, coarse)


def video_stats(path) -> FrameStats:
    """Per-frame video telemetry: the drift loop's measuring stick."""
    frames, fps = decode_luma(path)
    n = len(frames)
    if n == 0:
        return FrameStats(fps, 0, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    luma = frames.mean(axis=(1, 2))
    if n > 1:
        motion = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
        motion = np.concatenate([[motion[0]], motion])
    else:
        motion = np.zeros(n)
    # luma histogram entropy
    ent = np.zeros(n)
    for i in range(n):
        h = np.bincount((frames[i] * 31).astype(np.uint8).ravel(), minlength=32).astype(np.float64)
        p = h / max(1.0, h.sum())
        ent[i] = -(p[p > 0] * np.log2(p[p > 0])).sum()
    # coarse horizontal displacement (global pan/tear proxy), vectorized:
    # every 4th frame, find the roll that best matches the previous sample
    flow = np.zeros(n)
    shifts = np.arange(-6, 7)
    for i in range(4, n, 4):
        a, b = frames[i - 4], frames[i]
        best, bestd = 0, 1e9
        for s in shifts:
            d = np.abs(b - np.roll(a, s, axis=1)).mean()
            if d < bestd:
                bestd, best = d, s
        flow[i - 3: i + 1] = abs(best)
    return FrameStats(fps=fps, n_frames=n, luma=luma, motion=motion, entropy=ent, flow=flow)