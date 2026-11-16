import numpy as np
import pytest

from loom.curves import Curve
from loom.io import peak_normalize, read_wav, resample, to_stereo, write_wav


def test_curve_constant():
    c = Curve(2.5)
    assert c.at(0.0) == 2.5 and c.at(1.0) == 2.5


def test_curve_string():
    assert Curve("3").at(0.3) == 3.0


def test_curve_piecewise():
    c = Curve([0.0, 1.0, 0.0])
    assert c.at(0.0) == pytest.approx(0.0)
    assert c.at(0.25) == pytest.approx(0.5)
    assert c.at(0.5) == pytest.approx(1.0)
    assert c.at(1.0) == pytest.approx(0.0)


def test_curve_clamps_out_of_range():
    c = Curve([1, 2])
    assert c.at(-1) == 1.0
    assert c.at(2) == 2.0


def test_curve_callable():
    c = Curve(lambda t: t * t)
    assert c.at(0.5) == pytest.approx(0.25)


def test_curve_rejects_garbage():
    with pytest.raises((TypeError, ValueError)):
        Curve(object())


def test_wav_roundtrip(tmp_path):
    sr = 44100
    t = np.arange(sr) / sr
    x = np.stack([0.5 * np.sin(2 * np.pi * 300 * t), np.zeros_like(t)], axis=1).astype(np.float32)
    p = tmp_path / "x.wav"
    write_wav(p, x, sr, normalize=False)
    sr2, y = read_wav(p)
    assert sr2 == sr
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-3


def test_peak_normalize():
    x = np.array([[0.1], [-0.4]], dtype=np.float32)
    y = peak_normalize(x, headroom_db=0.0)
    assert np.max(np.abs(y)) == pytest.approx(1.0)


def test_to_stereo():
    mono = np.zeros((10, 1), np.float32)
    assert to_stereo(mono).shape == (10, 2)
    quad = np.zeros((10, 4), np.float32)
    assert to_stereo(quad).shape == (10, 2)


def test_resample_identity_and_ratio():
    x = np.zeros((100, 2), np.float32)
    assert resample(x, 44100, 44100).shape[0] == 100
    y = resample(x, 44100, 22050)
    assert y.shape[0] == 50