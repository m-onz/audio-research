"""Shared fixtures: a tiny synthetic corpus, split in place like the
real one, then indexed into a bank.

The corpus is designed so the pile has known structure:

    sine_1 / sine_twin   near-duplicate sines (kept whole; the
                         chain's prey)
    noise                a distant archetype (the contrast's destination)
    clicks               a click train (the onset scissor's proof)
    bursts               loud-silence-loud (a split into 2 events)
    drone                9 s continuous (the fixed scissor's proof)
"""

import numpy as np
import pytest

from echo.probes import write_wav, SR

from quilt.split import split_corpus
from quilt.index import build_bank

SR_TEST = SR


def _sine(freq, secs, amp=0.5):
    t = np.arange(int(secs * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    st = np.stack([x, x * 0.95], axis=1)
    return st


def _noise(secs, amp=0.4, seed=7):
    rng = np.random.default_rng(seed)
    x = amp * rng.standard_normal(int(secs * SR))
    return np.stack([x, x], axis=1)


def _click_train(secs, period=0.5):
    x = np.zeros(int(secs * SR))
    for k in range(int(secs / period)):
        i = int(k * period * SR)
        x[i:i + 400] += np.linspace(1, -1, 400)
    return np.stack([x, x], axis=1)


def _bursts(secs=3.0):
    n = int(secs * SR)
    x = np.zeros(n)
    x[:SR] = 0.5 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    x[2 * SR:] = 0.4 * np.sin(2 * np.pi * 800 * np.arange(n - 2 * SR) / SR)
    return np.stack([x, x], axis=1)


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    files = {
        "sine_1.wav": _sine(440.0, 2.0),
        "sine_twin.wav": _sine(441.0, 2.0),
        "noise.wav": _noise(2.0),
        "clicks.wav": _click_train(3.0),
        "bursts.wav": _bursts(3.0),
        "drone.wav": _sine(220.0, 9.0),
    }
    for name, audio in files.items():
        write_wav(str(root / name), audio, SR)
    return root


@pytest.fixture(scope="session")
def pile_dir(corpus_dir):
    """The corpus, cut in place — exactly what the real pipeline does."""
    split_corpus(corpus_dir, audit_path=str(corpus_dir / "audit.jsonl"),
                 log=lambda s: None)
    return corpus_dir


@pytest.fixture(scope="session")
def bank_path(pile_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("banks") / "bank.sqlite"
    build_bank(pile_dir, out, log=lambda s: None)
    return out


@pytest.fixture()
def atlas(bank_path):
    from quilt import ChunkAtlas
    return ChunkAtlas(bank_path)