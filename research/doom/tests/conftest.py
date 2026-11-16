"""Shared fixtures: a tiny synthetic audio file with known structure."""

import subprocess
from pathlib import Path

import numpy as np
import pytest


def make_wav(path, seconds=4.0, sr=44100):
    """4s test signal: 1s sine, 1s silence, 1s clicks at 2Hz, 1s noise."""
    n = int(sr * seconds)
    x = np.zeros(n, np.float32)
    t = np.arange(int(sr)) / sr
    x[:sr] = 0.6 * np.sin(2 * np.pi * 220 * t)
    # clicks
    for k in range(4):
        i = int(sr * (2.0 + 0.25 * k))
        wlen = int(0.02 * sr)
        x[i : i + wlen] += 0.8 * np.exp(-np.linspace(0, 6, wlen)) * np.sin(
            2 * np.pi * 440 * t[:wlen]
        )
    rng = np.random.default_rng(0)
    x[int(3 * sr) :] = 0.25 * rng.standard_normal(len(x) - int(3 * sr))
    stereo = np.stack([x, x], axis=1)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(sr),
         "-ac", "2", "-i", "pipe:0", str(path)],
        input=pcm.tobytes(), check=True,
    )
    return str(path)


@pytest.fixture(scope="session")
def test_wav(tmp_path_factory):
    return make_wav(tmp_path_factory.mktemp("audio") / "test.wav")