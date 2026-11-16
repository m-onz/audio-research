"""ears: the perceiver population — feature projections that discriminate.

An ear is m random projections over loom's 14 proven audio features
(rms, peak, crest, zcr, centroid, rolloff, flatness, bass, mid, air,
pitch, pitch_conf, transient, dur) plus a few temporal statistics.
Each ear maps a sound to a point in R^m; discrimination is Euclidean
distance in that space.

The v1.2 refinements:
- feature-space (not raw bands) — meaningfully different for noise vs
  structured sound
- ears are born in two ways: random probes, and probes positioned at
  spectral peaks of living sounds (POET's environments-from-solutions)
- ear MC: must discriminate a pair that NO other ear discriminates
  (unique contribution — Pugh multi-BC pressure)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Ear", "EarQueue", "features_vector", "FEATURE_NAMES"]

SR = 44100

FEATURE_NAMES = (
    "rms", "peak", "crest", "zcr", "centroid", "rolloff", "flatness",
    "bass", "mid", "air", "pitch", "pitch_conf", "transient", "dur",
    "temporal_cv", "onset_rate",
)


def features_vector(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """16-D feature vector (loom's 14 + 2 temporal). All normalized to
    roughly [0, 1] for comparable projections."""
    x = audio.astype(np.float64)
    n = len(x)
    if n < 256:
        return np.zeros(len(FEATURE_NAMES))

    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.abs(x).max())
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))) if n > 1 else 0)

    nfft = min(4096, max(256, 1 << int(np.ceil(np.log2(max(2, n))))))
    w = np.hanning(min(n, nfft))
    seg = x[: len(w)]
    spec = np.abs(np.fft.rfft(seg * w, nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    s = spec.sum() + 1e-9
    centroid = float((freqs * spec).sum() / s)
    i85 = int(np.searchsorted(np.cumsum(spec), 0.85 * spec.sum()))
    rolloff = float(freqs[min(i85, len(freqs) - 1)])
    flatness = float(np.exp(np.mean(np.log(spec + 1e-9))) / (np.mean(spec) + 1e-9))
    tot = float(spec.sum()) + 1e-9
    bass = float(spec[freqs < 250].sum() / tot)
    mid = float(spec[(freqs >= 250) & (freqs < 2000)].sum() / tot)
    air = float(spec[freqs >= 2000].sum() / tot)

    # crude pitch
    dec = max(1, sr // 12000)
    xs = x[::dec]
    srs = sr / dec
    nn = len(xs)
    if nn > 512:
        nfft2 = 1 << int(np.ceil(np.log2(2 * nn)))
        X = np.fft.rfft(xs, nfft2)
        ac = np.fft.irfft(np.abs(X) ** 2, nfft2)[: nn // 2]
        if ac[0] > 1e-9:
            ac = ac / ac[0]
            lagmin = max(2, int(srs / 800))
            lagmax = min(int(srs / 50), nn - 1)
            if lagmax > lagmin + 2:
                k = lagmin + int(np.argmax(ac[lagmin:lagmax]))
                pitch = float(srs / k) if ac[k] > 0.35 else 0.0
                pitch_conf = float(min(1.0, ac[k])) if pitch > 0 else 0.0
            else:
                pitch, pitch_conf = 0.0, 0.0
        else:
            pitch, pitch_conf = 0.0, 0.0
    else:
        pitch, pitch_conf = 0.0, 0.0

    q = max(1, n // 4)
    transient = float(np.sqrt(np.mean(x[:q] ** 2)) / (rms + 1e-9))
    dur = n / sr

    # temporal coefficient of variation (envelope shape)
    win = max(8, n // 100)
    env = np.array([np.sqrt(np.mean(x[i:i + win] ** 2))
                    for i in range(0, n - win, win)])
    temporal_cv = float(np.std(env) / (np.mean(env) + 1e-9)) if len(env) > 2 else 0.0

    # onset rate (energy jumps)
    if len(env) > 4:
        d = np.diff(env)
        onset_rate = float(np.sum(np.abs(d) > 2.0 * np.std(d)) / dur) if dur > 0 else 0.0
    else:
        onset_rate = 0.0

    return np.array([
        min(1.0, rms * 4),
        min(1.0, peak),
        min(1.0, peak / (rms + 1e-9) / 20),
        min(1.0, zcr * 2),
        min(1.0, centroid / 8000),
        min(1.0, rolloff / 12000),
        min(1.0, flatness),
        bass, mid, air,
        min(1.0, pitch / 2000),
        pitch_conf,
        min(1.0, transient / 4),
        min(1.0, dur / 8),
        min(1.0, temporal_cv / 2),
        min(1.0, onset_rate / 20),
    ])


@dataclass
class Ear:
    ear_id: int
    projection: np.ndarray       # (m, 16) random projection matrix
    birth_batch: int = 0
    parent_id: int | None = None

    def hear(self, audio: np.ndarray) -> np.ndarray:
        """Audio -> descriptor point in R^m."""
        f = features_vector(audio)
        return self.projection @ f


class EarQueue:
    """FIFO queue for ears (species = projection-space clustering is
    overkill for v0 — a single FIFO with unique-contribution MC)."""

    def __init__(self, capacity: int = 24):
        self.capacity = int(capacity)
        self._queue: list[Ear] = []
        self._next_id = 0

    @property
    def size(self) -> int:
        return len(self._queue)

    def admit(self, ear: Ear) -> None:
        self._queue.append(ear)
        if len(self._queue) > self.capacity:
            self._queue.pop(0)  # remove oldest

    def dequeue_batch(self, batch_size: int) -> list[Ear]:
        batch = self._queue[:batch_size]
        self._queue = self._queue[batch_size:]
        return batch

    def requeue(self, ears: list[Ear]) -> None:
        self._queue.extend(ears)

    def all_ears(self) -> list[Ear]:
        return list(self._queue)

    def new_ear_id(self) -> int:
        self._next_id += 1
        return self._next_id


def random_ear(rng: np.random.Generator, m: int = 4) -> Ear:
    """A random projection ear: m random directions in 16-D feature space."""
    proj = rng.normal(0, 1, (m, len(FEATURE_NAMES)))
    # normalize rows so distances are comparable across ears
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    return Ear(ear_id=0, projection=proj / (norms + 1e-9))


def ear_from_sounds(rng: np.random.Generator, feature_matrix: np.ndarray,
                    m: int = 4) -> Ear:
    """POET-style: position probes at spectral peaks of living sounds.
    Feature matrix: (n_sounds, 16). Directions = principal axes of the
    population's current feature spread."""
    centered = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
    try:
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return random_ear(rng, m)
    k = min(m, len(s))
    # project the SVD axes back into feature space
    axes = (u[:, :k].T @ centered)[:k]  # (k, 16)
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    proj16 = axes / (norms + 1e-9)
    while len(proj16) < m:
        extra = random_ear(rng, 1).projection
        proj16 = np.vstack([proj16, extra])
    return Ear(ear_id=0, projection=proj16[:m])