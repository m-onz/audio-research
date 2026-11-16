"""io: fast audio in this ecosystem's format (44.1 kHz stereo wav).

echo decodes through an ffmpeg subprocess — correct for anything, but
~100 ms of process overhead per file. The pile is tens of thousands
of small wavs written by quilt itself, so reading goes through
libsndfile in-process (a few ms per file), falling back to ffmpeg the
moment a file is unusual (not wav, another rate, too many channels).
Writing chunks uses the same fast path (PCM_16, clipped).
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from echo.probes import decode, SR

__all__ = ["read_audio", "write_audio"]


def read_audio(path, sr: int = SR) -> np.ndarray:
    """Any audio file -> (n, 2) float32 at sr. Fast path for plain
    44.1 kHz 1-2 channel wav (the pile's native format); ffmpeg for
    everything else (the ecosystem's universal decoder)."""
    p = str(path)
    try:
        info = sf.info(p)
        if (info.format == "WAV" and info.samplerate == sr
                and 1 <= info.channels <= 2):
            x, _ = sf.read(p, dtype="float32", always_2d=True)
            if x.ndim == 2 and x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            return x
    except Exception:
        pass
    return decode(p, sr, stereo=True)


def write_audio(path, audio, sr: int = SR) -> str:
    """(n,) or (n, 2) float -> 16-bit wav via libsndfile — the pile's
    native format, bytes-equivalent to the ffmpeg writer, ~10x faster
    for the tens of thousands of small files a split produces."""
    x = np.asarray(audio, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.ndim == 2 and x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), x.astype(np.float32), sr, subtype="PCM_16")
    return str(path)