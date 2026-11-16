"""ml: the extension point for learned / ML processors.

loom's registries make machine-learning integration trivial: a model is
just a Processor instance. Fit it on your corpus, then pass the *instance*
into any chain spec (build_chain accepts instances), or register it under a
name. No framework lock-in, no GPU requirements, no inference server — the
model runs inline during offline rendering.

Included as a worked reference: `CodebookProcessor` — a miniature
"corpus resynthesis" model. It fits a k-means codebook over STFT magnitude
frames of the corpus, then rewrites each chunk's spectra to the nearest
codebook entry, keeping the chunk's own phase (Griffin-Lim-free
resynthesis). The result: input material *speaks in the corpus's timbres*.
scikit-learn is optional; without it, the codebook falls back to a k-means
implemented in ~30 lines of numpy (in `codebook.py`-style simplicity — see
`NumpyKMeans` below).

Writing your own learned processor
----------------------------------
1. subclass Processor, set `name`,
2. fit whatever model in __init__ or a fit() method,
3. implement process(audio, sr, t, rng, **params) -> (n, 2) float32,
4. use it: `build_chain([my_model, "fades"])` or
   `loom.register_processor_instance("mymodel", my_model)`.
"""

from __future__ import annotations

import numpy as np

from .processors import Processor

__all__ = ["CodebookProcessor", "register_processor_instance", "fit_codebook"]


def register_processor_instance(name: str, instance: Processor) -> None:
    """Register a pre-built (e.g. fitted) processor instance under a name."""
    from .processors import PROCESSORS

    PROCESSORS[name] = instance


# --------------------------------------------------------------------------- tiny k-means

class NumpyKMeans:
    """Minimal k-means (numpy only) for when scikit-learn is unavailable."""

    def __init__(self, k: int, iters: int = 12, seed: int = 0) -> None:
        self.k, self.iters, self.seed = k, iters, seed

    def fit(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        centers = X[rng.choice(n, size=min(self.k, n), replace=False)].astype(np.float64)
        for _ in range(self.iters):
            # squared distances to centers (n, k) via the ||a-b||^2 trick
            d = (
                (X * X).sum(1)[:, None]
                - 2.0 * X @ centers.T
                + (centers * centers).sum(1)[None, :]
            )
            labels = np.argmin(d, axis=1)
            for j in range(len(centers)):
                mask = labels == j
                if mask.any():
                    centers[j] = X[mask].mean(0)
        return centers


# --------------------------------------------------------------------------- codebook processor

def fit_codebook(
    audio_list,
    sr: int,
    k: int = 64,
    n_fft: int = 2048,
    max_frames: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    """Fit a spectral-magnitude codebook on a list of (n, ch) float arrays."""
    hop = n_fft // 4
    frames: list[np.ndarray] = []
    budget = max_frames
    win = np.hanning(n_fft)
    for x in audio_list:
        m = x.mean(axis=1) if x.ndim == 2 else x
        if len(m) < n_fft or budget <= 0:
            continue
        starts = np.arange(0, len(m) - n_fft, hop)
        if len(starts) > budget:
            starts = starts[:: max(1, len(starts) // budget)]
        idx = starts[:, None] + np.arange(n_fft)[None, :]
        S = np.abs(np.fft.rfft(m[idx] * win, axis=1))
        frames.append(np.log1p(S))
        budget -= len(starts)
        if budget <= 0:
            break
    if not frames:
        raise ValueError("no frames collected for codebook fitting")
    X = np.concatenate(frames, axis=0)
    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=seed)
        km.fit(X)
        return np.expm1(km.cluster_centers_)
    except ImportError:
        return np.expm1(NumpyKMeans(k, seed=seed).fit(X))


class CodebookProcessor(Processor):
    """Corpus-timbre resynthesis: rewrite chunk spectra to codebook entries.

    fit() on corpus audio, then process() chunks: each STFT frame's
    magnitude is replaced by (blended with) its nearest codebook magnitude,
    phase kept from the input. Overlap-add ISTFT with hann analysis and
    synthesis windows (COLA-normalized).
    """

    name = "codebook"

    def __init__(self, codebook: np.ndarray | None = None, n_fft: int = 2048, blend: float = 1.0) -> None:
        self.codebook = None if codebook is None else np.asarray(codebook, dtype=np.float64)
        self.n_fft = int(n_fft)
        self.blend = float(blend)

    def fit(self, audio_list, sr: int, k: int = 64, max_frames: int = 20000, seed: int = 0) -> "CodebookProcessor":
        self.codebook = fit_codebook(audio_list, sr, k=k, n_fft=self.n_fft, max_frames=max_frames, seed=seed)
        return self

    def fit_folder(self, path, sr: int | None = None, k: int = 64, max_frames: int = 20000, seed: int = 0) -> "CodebookProcessor":
        """Convenience: fit on a folder of .wavs (or a single file)."""
        from .sources import open_sources

        srcs = open_sources(path, sr=sr)
        audio_list = [s.read_all() for s in srcs.items]
        return self.fit(audio_list, srcs.sr, k=k, max_frames=max_frames, seed=seed)

    def process(self, audio, sr, t, rng, blend=None, dream=0.0, **_):
        if self.codebook is None:
            return audio
        blend = self.blend if blend is None else float(blend)
        n_fft = self.n_fft
        hop = n_fft // 4
        n = audio.shape[0]
        if n < n_fft:
            return audio
        win = np.hanning(n_fft)
        starts = np.arange(0, n - n_fft + 1, hop)
        # pad the tail so the last frames reconstruct fully
        pad = hop if (n - (starts[-1] + n_fft)) > 0 else 0
        m = audio.mean(axis=1)
        m = np.concatenate([m, np.zeros(pad)]) if pad else m
        idx = starts[:, None] + np.arange(n_fft)[None, :]
        F = np.fft.rfft(m[idx] * win, axis=1)
        mag = np.abs(F)
        log_mag = np.log1p(mag)
        # nearest codebook entry per frame (squared-distance shortcut)
        cb = self.codebook
        d = -2.0 * log_mag @ cb.T + (cb * cb).sum(1)[None, :]
        nearest = np.argmin(d, axis=1)
        if float(dream) > 0:
            # dream mode: some frames jump to random codebook entries
            swappers = rng.rng.random(len(nearest)) < float(dream)
            nearest[swappers] = rng.rng.integers(len(cb), size=int(swappers.sum()))
        new_mag = np.expm1(cb[nearest])
        blended = (1.0 - blend) * mag + blend * new_mag
        F2 = blended * np.exp(1j * np.angle(F))
        # overlap-add with hann synthesis window; COLA-normalized
        out = np.zeros(n + pad, dtype=np.float64)
        norm = np.zeros_like(out)
        for i, s in enumerate(starts):
            out[s : s + n_fft] += np.fft.irfft(F2[i], n_fft) * win
            norm[s : s + n_fft] += win * win
        good = norm > 1e-8
        out[good] /= norm[good]
        out = out[:n]
        stereo = np.stack([out, out], axis=1).astype(np.float32)
        return stereo