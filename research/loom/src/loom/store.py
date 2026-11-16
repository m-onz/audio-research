"""Stores: audio for *derived* chunks (output of processing passes).

Source chunks need no storage (lazy reads). When a pass transforms a chunk,
the result must live somewhere. Two backends:

- ``memory``: a dict — fast, right for hundreds of chunks.
- ``disk``: one .npy file per chunk in a temp dir — right when passes are
  large or memory is tight (hour-long corpora, many passes).

The renderer is oblivious to the backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["Store", "MemoryStore", "DiskStore", "open_store"]


class Store:
    """Interface: put/get (n, 2) float32 arrays by string key."""

    def put(self, key: str, audio: np.ndarray) -> None:
        raise NotImplementedError

    def get(self, key: str) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        return 0

    def close(self) -> None:
        pass


class MemoryStore(Store):
    def __init__(self) -> None:
        self._d: dict[str, np.ndarray] = {}

    def put(self, key: str, audio: np.ndarray) -> None:
        self._d[key] = np.ascontiguousarray(audio, dtype=np.float32)

    def get(self, key: str) -> np.ndarray:
        return self._d[key]

    def __len__(self) -> int:
        return len(self._d)


class DiskStore(Store):
    """One .npy file per key under a directory (temp dir by default).

    Self-owned temp dirs are removed at interpreter exit so long sessions
    can't leak gigabytes; explicitly-passed roots are the caller's to keep.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        import atexit
        import shutil
        import tempfile

        self._owns_root = root is None
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="loom-store-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._count = 0
        if self._owns_root:
            atexit.register(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.root / f"{safe}.npy"

    def put(self, key: str, audio: np.ndarray) -> None:
        np.save(self._path(key), np.ascontiguousarray(audio, dtype=np.float32))
        self._count += 1

    def get(self, key: str) -> np.ndarray:
        return np.load(self._path(key))

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)


def open_store(kind: str = "memory", **kw) -> Store:
    if kind == "memory":
        return MemoryStore()
    if kind == "disk":
        return DiskStore(**kw)
    raise ValueError(f"unknown store {kind!r} ('memory' | 'disk')")