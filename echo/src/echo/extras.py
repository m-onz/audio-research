"""extras: the librosa tier — optional, analysis-grade blocks.

The core encoder (echo.encode) is pure numpy and stays the inner-loop
voice: ~18 ms per 3 s clip, zero imports beyond numpy, deterministic
everywhere. When librosa is importable, `encode_plus` extends the same
anatomy with four blocks the DSP core deliberately doesn't fake:

    chroma_cqt    12   pitch classes via constant-Q transform (the
                       spectral fold in the core is a crude chroma; CQT
                       is the real thing — handles inharmonic partials)
    contrast       7   spectral peak/valley structure per band — the
                       micro-terrain of the spectrum the MFCCs smooth over
    hpss_chunk     8   harmonic-vs-percussive energy ratio per chunk —
                       THE texture axis the character grids approximate
                       with flatness, measured properly (median filtering)
    rhythm         9   onset-strength trajectory per chunk + pulse
                       (autocorrelation periodicity, no tempo commitment)

Cost: ~10x the core encode (still ~a third of a second per clip) —
right for indexing a corpus, atlasing a run, curating by hand; wrong
for the inside of a breeding loop. The bank treats both anatomies
identically (it just refuses to mix them).
"""

from __future__ import annotations

import numpy as np

from .encode import LAYOUT, encode_blocks, _frame_win_hop, _sanitize

__all__ = ["LAYOUT_PLUS", "DIM_PLUS", "encode_plus", "encode_plus_blocks"]

_EPS = 1e-9

LAYOUT_PLUS = LAYOUT + [
    ("chroma_cqt", 12, 0.40),
    ("contrast",   7, 0.50),
    ("hpss_chunk", 8, 0.60),
    ("rhythm",     9, 0.50),   # 8 onset-strength chunk means + pulse
]
DIM_PLUS = sum(dim for _, dim, _ in LAYOUT_PLUS)


def _extras_blocks(x: np.ndarray, sr: int, n_chunks: int, hop: int) -> dict:
    """The four librosa blocks. Pitch-class (CQT chroma), texture split
    (hpss), and onset weather are computed on a 22.05 kHz downsample —
    these are coarse analyses, not fidelity; spectral contrast keeps the
    full rate (it is nearly free)."""
    import librosa
    from scipy.signal import resample_poly

    mono = x.mean(axis=1)
    contrast = librosa.feature.spectral_contrast(y=mono, sr=sr).mean(axis=1)

    y22 = resample_poly(mono, 22050, sr) if sr != 22050 else mono
    sr22, hop22 = 22050, 1024

    # --- chroma (constant-Q pitch classes — the real thing)
    chroma = librosa.feature.chroma_cqt(y=y22, sr=sr22,
                                        hop_length=hop22).mean(axis=1)

    # --- hpss: harmonic / (harmonic + percussive) energy per chunk
    S = librosa.stft(y22, hop_length=hop22)
    H, P = librosa.decompose.hpss(S)
    hE = (np.abs(H) ** 2).sum(axis=0)                 # (T,) frame energies
    pE = (np.abs(P) ** 2).sum(axis=0)
    hpss_chunk = _chunk_mean_1d(hE / (hE + pE + _EPS), n_chunks)

    # --- rhythm: onset strength per chunk + pulse periodicity
    onset = librosa.onset.onset_strength(y=y22, sr=sr22, hop_length=hop22)
    onset_chunk = _chunk_mean_1d(onset, n_chunks)
    pulse = 0.0
    if len(onset) >= 16:
        e = onset - onset.mean()
        ac = np.correlate(e, e, mode="full")[len(e) - 1:]
        ac /= (ac[0] + _EPS)
        lags = np.arange(1, len(ac))
        fps = sr22 / hop22                                # onset frame rate
        sel = (lags > fps * 0.25) & (lags < fps * 2.0)     # 0.25–2 s periods
        if sel.any():
            pulse = float(np.max(ac[1:][sel]) - 1.0)      # 0 = aperiodic
    return {"chroma_cqt": np.asarray(chroma, np.float64),
            "contrast": np.asarray(contrast, np.float64),
            "hpss_chunk": hpss_chunk,
            "rhythm": np.concatenate([onset_chunk, [pulse]])}


def _chunk_mean_1d(v: np.ndarray, n_chunks: int) -> np.ndarray:
    T = len(v)
    if T == 0:
        return np.zeros(n_chunks)
    if T < n_chunks:
        reps = int(np.ceil(n_chunks / T))
        v = np.concatenate([v] * reps)
        T = len(v)
    bounds = np.linspace(0, T, n_chunks + 1).astype(int)
    return np.array([v[bounds[i]: bounds[i + 1]].mean()
                     if bounds[i + 1] > bounds[i] else 0.0
                     for i in range(n_chunks)])


def encode_plus_blocks(audio, sr: int = 44100, n_chunks: int = 8) -> dict:
    """The full anatomy: core blocks + the four librosa blocks."""
    x = _sanitize(audio)
    win, hop = _frame_win_hop(sr)
    blocks = dict(encode_blocks(x, sr, n_chunks))
    blocks.update(_extras_blocks(x, sr, n_chunks, hop))
    return blocks


def encode_plus(audio, sr: int = 44100, n_chunks: int = 8) -> np.ndarray:
    """Audio -> (592,) vector: the analysis-grade tier. Raises
    ImportError with install instructions if librosa is missing."""
    blocks = encode_plus_blocks(audio, sr, n_chunks)
    vec = np.concatenate([blocks[name] for name, _, _ in LAYOUT_PLUS])
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)