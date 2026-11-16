"""Sources: lazy, seekable handles onto .wav files or folders of .wavs.

This is the module that makes hour-long corpora practical: a `Source` never
loads its whole file. It exposes `read(start_seconds, dur_seconds)` which
seeks into the file and decodes only that window, resampling to the
pipeline's target sample rate on the fly. `Sources` bundles one or more
files into an indexed pool with a common rate, and `iter_blocks` streams a
source sequentially for analysis passes (onset detection, features).

Everything downstream (chunkers, renderer) only ever touches:
    sources[i].read(start, dur)   ->  (n, 2) float32 at `sr`
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .io import read_wav, resample, to_stereo, wav_info

__all__ = ["Source", "Sources", "open_sources"]

try:
    import soundfile as _sf
except ImportError:  # pragma: no cover
    _sf = None


class Source:
    """A single lazy, seekable .wav at a target sample rate (always stereo)."""

    def __init__(self, path: str | Path, target_sr: int | None = None) -> None:
        self.path = Path(path)
        info = wav_info(self.path)
        self.native_sr = int(info["sr"])
        self.channels = int(info["channels"])
        self.sr = int(target_sr) if target_sr else self.native_sr
        self._native_frames = int(info["frames"])
        self._sf_file = None
        self._name = self.path.stem

    # -- basics ---------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def duration(self) -> float:
        """Duration in seconds (rate-independent: frames / native rate)."""
        return self._native_frames / self.native_sr

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Source {self.name}: {self.duration:.1f}s @{self.sr}Hz>"

    # -- lazy reading ----------------------------------------------------------
    def _open(self):
        if self._sf_file is None:
            if _sf is None:
                raise RuntimeError(
                    f"lazy reading of {self.path.name} requires soundfile "
                    "(`pip install soundfile`); it ships with most distributions"
                )
            self._sf_file = _sf.SoundFile(str(self.path))
        return self._sf_file

    def read(self, start: float, dur: float) -> np.ndarray:
        """Read [start, start+dur) seconds -> (n, 2) float32 at self.sr.

        Reads clamp at the end of the file; a read that starts past the end
        returns an empty array. Sample-rate conversion happens per read.
        """
        start = max(0.0, float(start))
        dur = max(0.0, float(dur))
        n0 = min(int(round(start * self.native_sr)), self._native_frames)
        n1 = min(int(round((start + dur) * self.native_sr)), self._native_frames)
        if n1 <= n0:
            return np.zeros((0, 2), dtype=np.float32)
        f = self._open()
        f.seek(n0)
        raw = f.read(n1 - n0, dtype="float32", always_2d=True)
        return self._finalize(raw)

    def read_all(self) -> np.ndarray:
        """Materialize the whole file (only sensible for small sources)."""
        return self._finalize(read_wav(self.path)[1])

    def _finalize(self, raw: np.ndarray) -> np.ndarray:
        x = to_stereo(np.ascontiguousarray(raw, dtype=np.float32))
        if x.shape[0] and self.native_sr != self.sr:
            x = resample(x, self.native_sr, self.sr)
        return x

    def iter_blocks(self, block_seconds: float = 30.0, overlap_seconds: float = 0.0) -> Iterator[np.ndarray]:
        """Stream the source sequentially in (n, 2) blocks at `self.sr`.

        The reference pattern for whole-file analysis without whole-file RAM:
        block boundaries are sample-aligned so concatenated blocks are
        identical to read_all().
        """
        total = self.duration
        t = 0.0
        while t < total:
            dur = min(block_seconds + overlap_seconds, total - t)
            yield self.read(t, dur)
            t += block_seconds

    def close(self) -> None:
        if self._sf_file is not None:
            self._sf_file.close()
            self._sf_file = None


class Sources:
    """An indexed pool of Sources sharing one target sample rate."""

    def __init__(self, items: list[Source], sr: int | None = None) -> None:
        if not items:
            raise ValueError("Sources needs at least one .wav")
        sr = int(sr) if sr else items[0].sr
        for s in items:
            s.sr = sr
        self.items = items
        self.sr = sr

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.items]

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Source:
        return self.items[i]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Sources: {len(self)} files, {self.total_duration:.1f}s @{self.sr}Hz>"


def open_sources(paths, sr: int | None = None) -> Sources:
    """Open one .wav, one folder of .wavs, or an explicit list of either.

    Folders are scanned non-recursively, sorted by name (deterministic order
    is a reproducibility requirement, not a convenience).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            wavs = sorted(
                [f for f in p.iterdir() if f.suffix.lower() == ".wav" and f.is_file()],
                key=lambda f: f.name.lower(),
            )
            if not wavs:
                raise FileNotFoundError(f"no .wav files in folder {p}")
            files.extend(wavs)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(str(p))
    seen: set[Path] = set()
    unique = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(f)
    return Sources([Source(f, sr) for f in unique], sr)