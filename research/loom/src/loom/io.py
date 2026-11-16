"""Wav I/O and resampling.

Design goals for offline rendering:
- Write path never silently truncates: `write_wav` normalizes by default.
- Read path prefers `soundfile` (any format), falls back to a pure-stdlib
  `wave`-based decoder that handles 8/16/24/32-bit PCM — enough to read the
  24-bit corpus files without any third-party dependency.
- `resample` uses scipy's polyphase filter when available and falls back to
  high-quality linear interpolation otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_wav", "write_wav", "resample", "peak_normalize", "wav_info"]

EPS = 1e-10

try:
    import soundfile as _sf
except ImportError:  # pragma: no cover - depends on environment
    _sf = None

try:
    from scipy import signal as _signal
except ImportError:  # pragma: no cover
    _signal = None


# --------------------------------------------------------------------------- read

def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a wav -> (sr, float32 array shaped (n, channels))."""
    path = str(path)
    if _sf is not None:
        data, sr = _sf.read(path, always_2d=True, dtype="float32")
        return int(sr), np.ascontiguousarray(data, dtype=np.float32)
    return _read_wav_stdlib(path)


def _read_wav_stdlib(path: str) -> tuple[int, np.ndarray]:
    import wave

    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if w.getcomptype() != "NONE":
            raise ValueError(f"{path}: compressed wav needs soundfile (`pip install soundfile`)")
        raw = w.readframes(w.getnframes())
    if sw == 1:  # 8-bit wav is unsigned
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 3:  # 24-bit little-endian, hand-packed
        u = np.frombuffer(raw, np.uint8).reshape(-1, ch, 3).astype(np.uint32)
        v = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16)
        v = np.where(v & 0x800000, v - (1 << 24), v).astype(np.int32)
        x = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {sw * 8} bits")
    return int(sr), x.reshape(-1, ch)


def wav_info(path: str | Path) -> dict:
    """Cheap header inspection without decoding the whole file."""
    path = str(path)
    if _sf is not None:
        with _sf.SoundFile(path) as f:
            return {
                "path": path,
                "sr": int(f.samplerate),
                "channels": int(f.channels),
                "frames": int(f.frames),
                "seconds": f.frames / f.samplerate,
                "subtype": f.subtype,
            }
    import wave

    with wave.open(path, "rb") as w:
        return {
            "path": path,
            "sr": int(w.getframerate()),
            "channels": int(w.getnchannels()),
            "frames": int(w.getnframes()),
            "seconds": w.getnframes() / w.getframerate(),
            "subtype": f"PCM_{w.getsampwidth() * 8}",
        }


# --------------------------------------------------------------------------- write

def write_wav(
    path: str | Path,
    data: np.ndarray,
    sr: int,
    bits: int = 16,
    normalize: bool = True,
    headroom_db: float = 0.3,
) -> dict:
    """Write float audio to PCM wav. Returns a small stats dict.

    data: (n,) or (n, ch) float array, values nominally in [-1, 1].
    normalize: peak-normalize to `headroom_db` dBFS before quantizing.
    """
    x = np.asarray(data, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if normalize and peak > EPS:
        x = x * (10.0 ** (-headroom_db / 20.0) / peak)
        peak = float(np.max(np.abs(x)))
    x = np.clip(x, -1.0, 1.0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _sf is not None:
        subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(int(bits), "PCM_16")
        _sf.write(str(path), x, int(sr), subtype=subtype)
    else:
        if int(bits) != 16:
            bits = 16  # stdlib path only writes 16-bit
        y = (x * 32767.0).astype(np.int16)
        import wave

        with wave.open(str(path), "wb") as w:
            w.setnchannels(y.shape[1])
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(y.tobytes())
    return {"path": str(path), "sr": int(sr), "seconds": x.shape[0] / sr, "peak": peak, "bits": int(bits)}


# --------------------------------------------------------------------------- dsp helpers

def resample(x: np.ndarray, sr_from: int, sr_to: int, quality: str = "auto") -> np.ndarray:
    """Resample (n, ch) float32 along axis 0. `quality`: 'poly' | 'linear' | 'auto'."""
    if sr_from == sr_to or x.shape[0] == 0:
        return np.ascontiguousarray(x, dtype=np.float32)
    if quality == "poly" or (quality == "auto" and _signal is not None):
        if _signal is None:
            raise RuntimeError("polyphase resampling requires scipy")
        g = np.gcd(int(sr_from), int(sr_to))
        up, down = int(sr_to) // g, int(sr_from) // g
        y = _signal.resample_poly(x.astype(np.float64), up, down, axis=0)
        return np.ascontiguousarray(y, dtype=np.float32)
    n_new = int(round(x.shape[0] * sr_to / sr_from))
    pos = np.linspace(0.0, x.shape[0] - 1, n_new)
    base = np.arange(x.shape[0], dtype=np.float64)
    out = np.stack([np.interp(pos, base, x[:, c].astype(np.float64)) for c in range(x.shape[1])], axis=1)
    return out.astype(np.float32)


def peak_normalize(x: np.ndarray, headroom_db: float = 0.3) -> np.ndarray:
    """Scale to `headroom_db` dBFS peak. Leaves near-silent input untouched."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= EPS:
        return x
    return x * (10.0 ** (-headroom_db / 20.0) / peak)


def to_stereo(x: np.ndarray) -> np.ndarray:
    """(n,) -> (n, 2) by duplication; (n, ch>2) -> first two channels."""
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] > 2:
        return np.ascontiguousarray(x[:, :2])
    return np.ascontiguousarray(x, dtype=np.float32)