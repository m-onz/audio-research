"""melbank: pure-numpy mel filterbank + DCT-II — the MFCC machinery.

No librosa, no dependencies beyond numpy (the ecosystem rule: offline,
deterministic, importable anywhere). HTK mel scale, Slaney-normalized
triangles, orthonormal DCT-II. Matrices are built once per (sr, n_fft)
configuration and cached — encoding must fit inside a breeding loop.
"""

from __future__ import annotations

import numpy as np

__all__ = ["hz_to_mel", "mel_to_hz", "mel_filterbank", "dct_matrix",
           "get_mel_dct"]

_MEL_BREAK = 2595.0            # HTK constant
_CACHE: dict[tuple, tuple] = {}


def hz_to_mel(f):
    """HTK mel scale: 2595 * log10(1 + f/700)."""
    return _MEL_BREAK * np.log10(1.0 + np.asarray(f, np.float64) / 700.0)


def mel_to_hz(m):
    m = np.asarray(m, np.float64)
    return 700.0 * (10.0 ** (m / _MEL_BREAK) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 64,
                   fmin: float = 40.0, fmax: float | None = None) -> np.ndarray:
    """(n_mels, 1 + n_fft//2) triangle filter matrix on the mel scale,
    Slaney-normalized (unit area) so log band energies are comparable
    across the filterbank's varying bandwidths."""
    if fmax is None:
        fmax = min(16000.0, sr / 2.0)
    fmax = min(fmax, sr / 2.0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)          # (F,)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, len(freqs)), np.float64)
    # every freq bin belongs to at most two triangles; vectorized assignment
    for i in range(n_mels):
        lo, ce, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - freqs) / max(hi - ce, 1e-9)
        tri = np.maximum(0.0, np.minimum(up, down))
        area = np.trapz(tri, freqs)
        if area > 1e-12:
            tri = tri / area                            # Slaney norm
        fb[i] = tri
    return fb


def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (n_out, n_in): rows are cepstral basis
    vectors; X = dct @ x for x of dim n_in."""
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    basis = np.cos(np.pi * (2 * n + 1) * k / (2 * n_in))
    basis *= np.sqrt(2.0 / n_in)
    basis[0] /= np.sqrt(2.0)                             # k=0 half weight
    return basis


def get_mel_dct(sr: int, n_fft: int, n_mels: int = 64, n_mfcc: int = 24,
                fmin: float = 40.0, fmax: float | None = None):
    """Cached (filterbank, dct) pair — the only state the encoder needs."""
    fmax = (min(16000.0, sr / 2.0) if fmax is None else fmax)
    key = (sr, n_fft, n_mels, n_mfcc, fmin, fmax)
    if key not in _CACHE:
        fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
        dct = dct_matrix(n_mfcc, n_mels)
        _CACHE[key] = (fb, dct)
    return _CACHE[key]