"""degrade: generational loss operators.

The metabolism thesis: drift is material, so loss must be *real* — codec
rounds through actual mp3/mp4 encoders, actual resampling wars — not
simulated approximations. Every op is a measured generational step the
drift report can attribute.

Audio ops take (float32 mono, sr) -> (float32 mono, sr) in memory.
Video ops take path -> path (ffmpeg re-encodes).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .video import decode_frames, probe_video

__all__ = ["AUDIO_DEGRADES", "VIDEO_DEGRADES", "degrade_audio", "degrade_video"]


# ------------------------------------------------------------------ audio

def _mp3_round(x, sr, kbps=96, rng=None):
    """Encode to mp3 and back — real codec generational loss."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(sr),
             "-ac", "1", "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", f"{kbps}k", tmp],
            input=x.astype(np.float32).tobytes(), check=True,
        )
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp, "-f", "f32le", "-ac", "1",
             "-ar", str(sr), "pipe:1"],
            capture_output=True, check=True,
        ).stdout
        return np.frombuffer(out, np.float32).copy()
    finally:
        Path(tmp).unlink(missing_ok=True)


def _resample_war(x, sr, rate=8000, rng=None):
    """Downsample to a low rate and back — aliasing as texture."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(sr),
             "-ac", "1", "-i", "pipe:0", "-ar", str(int(rate)), tmp],
            input=pcm.tobytes(), check=True,
        )
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp, "-f", "f32le", "-ac", "1",
             "-ar", str(sr), "pipe:1"],
            capture_output=True, check=True,
        ).stdout
        return np.frombuffer(out, np.float32).copy()
    finally:
        Path(tmp).unlink(missing_ok=True)


def _crush(x, sr, bits=10, rng=None):
    q = 2.0 ** (bits - 1)
    return np.round(x * q) / q


def _bitrot(x, sr, amount=0.0005, rng=None):
    """Zero-runs and sample swaps on the raw PCM — loom's corrupt, on stems."""
    if rng is None:
        rng = np.random.default_rng(0)
    y = x.astype(np.float32).copy()
    hits = int(len(y) * amount)
    for _ in range(hits):
        i = int(rng.integers(len(y)))
        op = rng.integers(3)
        if op == 0:
            run = int(rng.integers(1, 9))
            y[i: i + run] = 0.0
        elif op == 1 and i + 1 < len(y):
            y[i], y[i + 1] = y[i + 1], y[i]
        else:
            y[i] = -y[i]
    return y


AUDIO_DEGRADES = {"mp3_round": _mp3_round, "resample_war": _resample_war,
                  "crush": _crush, "bitrot": _bitrot}


def degrade_audio(x: np.ndarray, sr: int, ops: list, seed: int = 0) -> np.ndarray:
    """Apply a chain of named audio loss ops: [("mp3_round", {"kbps": 96}), ...]"""
    rng = np.random.default_rng(seed)
    for name, params in (ops or []):
        fn = AUDIO_DEGRADES.get(name)
        if fn is None:
            raise ValueError(f"unknown audio degrade {name!r}; have {sorted(AUDIO_DEGRADES)}")
        x = fn(x, sr, rng=rng, **(params or {}))
    return x


# ------------------------------------------------------------------ video

def _v_mp4_round(src, crf=30, rng=None):
    out = str(Path(src).with_name(Path(src).stem + "_round.mp4"))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
         "-c:a", "aac", out],
        check=True,
    )
    return out


def _v_resample_war(src, scale=0.5, rng=None):
    out = str(Path(src).with_name(Path(src).stem + "_war.mp4"))
    info = probe_video(src)
    w, h = max(64, int(info["width"] * scale)), max(36, int(info["height"] * scale))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", f"scale={w}:{h},scale={info['width']}:{info['height']}:flags=neighbor",
         "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", out],
        check=True,
    )
    return out


def _v_frame_drop(src, rate=2, rng=None):
    """Hold every Nth frame — digital stop-motion loss."""
    out = str(Path(src).with_name(Path(src).stem + "_drop.mp4"))
    fps = probe_video(src)["fps"] / max(1, int(rate))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", f"fps={fps}", "-c:v", "libx264", "-crf", "22",
         "-pix_fmt", "yuv420p", "-c:a", "aac", out],
        check=True,
    )
    return out


def _v_palette_crush(src, levels=8, rng=None):
    """Decode -> ordered-dither to N levels (doom's psx) -> re-encode."""
    from doom.palettes import psx_quantize

    info = probe_video(src)
    size = (min(640, info["width"]), min(640, info["height"]))
    frames = decode_frames(src, size=size, block=24)
    out = str(Path(src).with_name(Path(src).stem + "_crush.mp4"))
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", "24",
         "-i", "pipe:0", "-i", str(src), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", out],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        proc.stdin.write(psx_quantize(f, levels=int(levels)).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


VIDEO_DEGRADES = {"mp4_round": _v_mp4_round, "resample_war": _v_resample_war,
                  "frame_drop": _v_frame_drop, "palette_crush": _v_palette_crush}


def degrade_video(src, ops: list, seed: int = 0) -> str:
    """Apply a chain of named video loss ops (each re-encodes — real loss)."""
    rng = np.random.default_rng(seed)
    cur = str(src)
    for name, params in (ops or []):
        fn = VIDEO_DEGRADES.get(name)
        if fn is None:
            raise ValueError(f"unknown video degrade {name!r}; have {sorted(VIDEO_DEGRADES)}")
        cur = fn(cur, rng=rng, **(params or {}))
    return cur