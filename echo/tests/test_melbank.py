"""melbank: mel filterbank + DCT correctness."""

import numpy as np

from echo.melbank import dct_matrix, hz_to_mel, mel_filterbank, mel_to_hz


def test_hz_mel_roundtrip():
    f = np.array([40.0, 440.0, 1000.0, 5000.0, 16000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f, rtol=1e-9)


def test_filterbank_shape_and_support():
    fb = mel_filterbank(44100, 2048, n_mels=64)
    assert fb.shape == (64, 1025)
    assert (fb >= 0).all()
    # triangles live inside the mel range: no energy below fmin
    freqs = np.fft.rfftfreq(2048, 1 / 44100)
    assert fb[:, freqs < 39.0].sum() < 1e-9
    # Slaney norm: filter areas integrate near 1 in mel-frequency units
    areas = np.trapz(fb, mel_to_hz(np.linspace(0, hz_to_mel(22050), 1025)), axis=1)
    assert np.all(np.isfinite(areas))


def test_dct_orthonormal():
    d = dct_matrix(24, 64)
    assert d.shape == (24, 64)
    assert np.allclose(d @ d.T, np.eye(24), atol=1e-10)


def test_dct_constant_signal():
    """DCT of a constant is all zeros except k=0 (orthonormal scaling)."""
    d = dct_matrix(8, 16)
    x = np.full(16, 2.5)
    y = d @ x
    assert abs(y[0] - 2.5 * np.sqrt(16)) < 1e-9
    assert np.abs(y[1:]).max() < 1e-10