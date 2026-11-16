"""Machine-listening features for chunks.

A small, dependency-free feature set computed per chunk: loudness, timbre
(zero-crossing rate, spectral centroid / rolloff / flatness, 3-band energy
ratios), pitch (autocorrelation f0 with confidence), and envelope shape
(crest factor, front-loaded transient-ness). Selection and arrangement work
on *quantile ranks* of these features (see catalog), so absolute scales
never matter — "rms>0.8" always means "the loudest 20%".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

__all__ = ["Features", "FEATURE_NAMES", "analyze", "features_matrix", "feature_help"]

EPS = 1e-10

FEATURE_NAMES = (
    "rms",
    "peak",
    "crest",
    "zcr",
    "centroid",
    "rolloff",
    "flatness",
    "bass",
    "mid",
    "air",
    "pitch",
    "pitch_conf",
    "transient",
    "dur",
)


@dataclass(frozen=True)
class Features:
    rms: float
    peak: float
    crest: float
    zcr: float
    centroid: float
    rolloff: float
    flatness: float
    bass: float
    mid: float
    air: float
    pitch: float
    pitch_conf: float
    transient: float
    dur: float

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in FEATURE_NAMES], dtype=np.float64)

    def as_dict(self) -> dict:
        return asdict(self)


def feature_help() -> str:
    lines = [
        "loom features (queried by quantile rank, 0=lowest .. 1=highest):",
        "  rms        loudness (root-mean-square)",
        "  peak       absolute maximum",
        "  crest      peak/rms (attackiness, transients score high)",
        "  zcr        zero-crossing rate (brightness proxy)",
        "  centroid   spectral centroid in Hz",
        "  rolloff    85% spectral rolloff in Hz",
        "  flatness   spectral flatness (noisiness, 0=tone, 1=noise)",
        "  bass       energy share below 250 Hz",
        "  mid        energy share 250 Hz-2 kHz",
        "  air        energy share above 2 kHz",
        "  pitch      autocorrelation f0 estimate in Hz (0 if unpitched)",
        "  pitch_conf confidence of the f0 estimate (0-1)",
        "  transient  front-quarter rms relative to overall rms",
        "  dur        chunk duration in seconds",
    ]
    return "\n".join(lines)


def analyze(audio: np.ndarray, sr: int) -> Features:
    """Compute the feature set for (n, ch) float audio."""
    x = audio if audio.ndim == 2 else audio[:, None]
    m = x.mean(axis=1).astype(np.float64) if x.shape[1] > 1 else x[:, 0].astype(np.float64)
    n = len(m)
    if n == 0:
        return Features(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)
    rms = float(np.sqrt(np.mean(m * m)))
    peak = float(np.max(np.abs(m)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(m))))) if n > 1 else 0.0

    nfft = min(1 << 16, max(256, 1 << int(np.ceil(np.log2(max(2, n))))))
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(m * w, nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    s_sum = float(spec.sum()) + EPS
    centroid = float((freqs * spec).sum() / s_sum)
    i85 = int(np.searchsorted(np.cumsum(spec), 0.85 * spec.sum()))
    rolloff = float(freqs[min(i85, len(freqs) - 1)])
    flatness = float(np.exp(np.mean(np.log(spec + EPS))) / (np.mean(spec) + EPS))

    # 3-band energy shares
    edges = (0.0, 250.0, 2000.0, sr / 2.0 + 1.0)
    bass = mid = air = 0.0
    for lo, hi, tgt in zip(edges[:-1], edges[1:], ("b", "m", "a")):
        band = float(spec[(freqs >= lo) & (freqs < hi)].sum())
        if tgt == "b":
            bass = band
        elif tgt == "m":
            mid = band
        else:
            air = band
    tot = bass + mid + air + EPS
    bass, mid, air = bass / tot, mid / tot, air / tot

    pitch, conf = _pitch(m, sr)
    q = max(1, n // 4)
    transient = float(np.sqrt(np.mean(m[:q] ** 2)) / (rms + EPS))
    return Features(
        rms=rms,
        peak=peak,
        crest=peak / (rms + EPS),
        zcr=zcr,
        centroid=centroid,
        rolloff=rolloff,
        flatness=flatness,
        bass=bass,
        mid=mid,
        air=air,
        pitch=pitch,
        pitch_conf=conf,
        transient=transient,
        dur=n / sr,
    )


def _pitch(m: np.ndarray, sr: int) -> tuple[float, float]:
    """Crude autocorrelation f0 -> (Hz, confidence). (0, 0) if unpitched."""
    if len(m) < 512:
        return 0.0, 0.0
    dec = max(1, sr // 12000)
    xs = m[::dec]
    srs = sr / dec
    n = len(xs)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(xs, nfft)
    ac = np.fft.irfft(np.abs(X) ** 2, nfft)[: n // 2]
    if ac[0] <= EPS:
        return 0.0, 0.0
    ac = ac / ac[0]
    lagmin = max(2, int(srs / 800))
    lagmax = min(int(srs / 50), n - 1)
    if lagmax <= lagmin + 2:
        return 0.0, 0.0
    k = lagmin + int(np.argmax(ac[lagmin:lagmax]))
    if ac[k] < 0.35:
        return 0.0, 0.0
    # parabolic interpolation around the peak
    d = 0.0
    if 0 < k < n - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if abs(den) > EPS else 0.0
    return float(srs / (k + d)), float(min(1.0, ac[k]))


def features_matrix(feats: list[Features]) -> np.ndarray:
    """Stack Features into an (n, len(FEATURE_NAMES)) matrix."""
    return np.stack([f.vector() for f in feats]) if feats else np.zeros((0, len(FEATURE_NAMES)))