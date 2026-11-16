"""Shared fixtures: a small synthetic corpus of .wavs in tmp dirs."""

from __future__ import annotations

import numpy as np
import pytest

from loom.io import write_wav


def _click_train(sr: int, seconds: float, interval: float) -> np.ndarray:
    """Percussive clicks every `interval` s over filtered noise."""
    n = int(sr * seconds)
    x = np.zeros((n, 2), dtype=np.float32)
    t = np.arange(n) / sr
    noise = 0.02 * np.sin(2 * np.pi * 90 * t)  # low hum floor
    x[:] = noise[:, None]
    k = 0
    while k * interval < seconds:
        i = int(k * interval * sr)
        wlen = int(0.02 * sr)
        env = np.exp(-np.linspace(0, 8, wlen))
        tone = 0.6 * env * np.sin(2 * np.pi * 440 * np.arange(wlen) / sr)
        if i + wlen <= n:
            x[i : i + wlen, 0] += tone
            x[i : i + wlen, 1] += tone
        k += 1
    return x


def _noise_burst(sr: int, seconds: float, seed: int, brightness: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    white = rng.standard_normal((n, 2)).astype(np.float32)
    # crude brightness: one-pole highpass-ish mix
    hp = white - 0.9 * np.roll(white, 1, axis=0)
    x = brightness * hp + (1 - brightness) * white
    env = np.hanning(n)[:, None] ** 2
    return (0.4 * x * env).astype(np.float32)


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory):
    """Folder with 4 tiny wavs: a click train, two noise bursts, a sine."""
    d = tmp_path_factory.mktemp("corpus")
    sr = 22050
    write_wav(d / "clicks.wav", _click_train(sr, 6.0, 0.5), sr, normalize=True)
    write_wav(d / "dark_noise.wav", _noise_burst(sr, 2.0, seed=1, brightness=0.1), sr)
    write_wav(d / "bright_noise.wav", _noise_burst(sr, 1.5, seed=2, brightness=0.9), sr)
    t = np.arange(int(sr * 1.0)) / sr
    sine = np.stack([0.5 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    write_wav(d / "sine.wav", sine, sr)
    return d


@pytest.fixture(scope="session")
def corpus_wav(tmp_path_factory):
    """A single longer wav for single-file pipelines (~40 s)."""
    d = tmp_path_factory.mktemp("single")
    sr = 22050
    write_wav(d / "long.wav", _click_train(sr, 40.0, 0.7), sr)
    return d / "long.wav"


@pytest.fixture(scope="session")
def big_wav(tmp_path_factory):
    """A 'big' file (few minutes at 48 kHz) proving lazy reads."""
    d = tmp_path_factory.mktemp("big")
    sr = 48000
    write_wav(d / "big.wav", _click_train(sr, 180.0, 1.0), sr)
    return d / "big.wav"