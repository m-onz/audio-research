"""Shared fixtures: a tiny synthetic video + audio with known structure."""

import subprocess
from pathlib import Path

import numpy as np
import pytest


def make_wav(path, seconds=4.0, sr=44100):
    """4 s: sine -> silence -> clicks -> noise (the doom test signal)."""
    n = int(sr * seconds)
    x = np.zeros(n, np.float32)
    t = np.arange(int(sr)) / sr
    x[:sr] = 0.6 * np.sin(2 * np.pi * 220 * t)
    rng = np.random.default_rng(0)
    x[int(3 * sr):] = 0.25 * rng.standard_normal(len(x) - int(3 * sr))
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(sr),
         "-ac", "1", "-i", "pipe:0", str(path)],
        input=pcm.tobytes(), check=True,
    )
    return str(path)


def make_video(path, seconds=3.0, fps=12, w=160, h=120):
    """3 s of test frames: horizontal moving stripes over a gradient."""
    frames = int(seconds * fps)
    rgb = np.zeros((frames, h, w, 3), np.uint8)
    xs = np.arange(w)
    for i in range(frames):
        grad = (xs / w * 255).astype(np.uint8)
        rgb[i, :, :, 0] = grad[None, :]
        stripe = ((xs + i * 6) % 40 < 20).astype(np.uint8) * 200
        rgb[i, :, :, 1] = stripe[None, :]
        rgb[i, :, :, 2] = 120
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(path)],
        input=rgb.tobytes(), check=True,
    )
    return str(path)


@pytest.fixture(scope="session")
def test_wav(tmp_path_factory):
    return make_wav(tmp_path_factory.mktemp("media") / "test.wav")


@pytest.fixture(scope="session")
def test_video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("media") / "test.mp4")