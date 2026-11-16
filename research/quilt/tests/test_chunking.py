"""the scissors: known-answer cuts."""

import numpy as np
import pytest

from quilt.chunking import (chunk, chunk_fixed, chunk_onset, chunk_silence,
                            chunk_whole, MIN_DUR)
from echo.probes import SR


def _sine(secs, freq=440.0, amp=0.5):
    t = np.arange(int(secs * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x], axis=1)


class TestFixed:
    def test_uniform_windows(self):
        segs = chunk_fixed(_sine(5.0), SR, dur=2.0)
        assert segs == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]

    def test_dust_tail_dropped(self):
        segs = chunk_fixed(_sine(4.05), SR, dur=2.0)
        assert segs == [(0.0, 2.0), (2.0, 4.0)]     # 0.05s tail is dust


class TestWhole:
    def test_short_file_is_one_piece(self):
        assert chunk_whole(_sine(2.0), SR) == [(0.0, 2.0)]

    def test_long_file_refused(self):
        assert chunk_whole(_sine(21.0), SR, max_dur=20.0) == []

    def test_dust_refused(self):
        assert chunk_whole(_sine(0.05), SR) == []


class TestSilence:
    def test_bursts_split_at_the_gap(self):
        n = 3 * SR
        x = np.zeros(n)
        x[:SR] = 0.5 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
        x[2 * SR:] = 0.4 * np.sin(2 * np.pi * 800 * np.arange(SR) / SR)
        segs = chunk_silence(np.stack([x, x], 1), SR)
        assert len(segs) == 2
        assert segs[0][0] < 0.05 and 0.9 < segs[0][1] < 1.1   # the sine
        assert 1.9 < segs[1][0] < 2.1                          # the second

    def test_silent_file_yields_nothing(self):
        assert chunk_silence(np.zeros((3 * SR, 2)), SR) == []

    def test_continuous_tone_is_one_piece(self):
        segs = chunk_silence(_sine(1.0), SR)
        assert len(segs) == 1


class TestOnset:
    def test_click_train_cuts_at_hits(self):
        secs, period = 3.0, 0.5
        x = np.zeros(int(secs * SR))
        for k in range(int(secs / period)):
            i = int(k * period * SR)
            x[i:i + 400] += np.linspace(1, -1, 400)
        segs = chunk_onset(np.stack([x, x], 1), SR)
        # 6 hits; librosa may fuse neighbors — the count is bounded
        assert 4 <= len(segs) <= 8
        for t0, t1 in segs:
            assert t1 - t0 >= 0.15                 # min_seg respected

    def test_deterministic(self):
        x = _click_train_like()
        assert chunk_onset(x, SR) == chunk_onset(x, SR)


def _click_train_like():
    x = np.zeros(int(2.5 * SR))
    for k in range(5):
        i = int(k * 0.5 * SR)
        x[i:i + 400] += np.linspace(1, -1, 400)
    return np.stack([x, x], 1)


class TestRegistry:
    def test_every_strategy_runs(self):
        from quilt.chunking import STRATEGIES
        x = _sine(2.0)
        for name in STRATEGIES:
            segs = chunk(x, SR, name)
            assert isinstance(segs, list)
            for t0, t1 in segs:
                assert t1 - t0 >= MIN_DUR

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            chunk(_sine(1.0), SR, "laser")