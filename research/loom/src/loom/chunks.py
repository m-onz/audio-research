"""Chunks: the unit of composition.

A Chunk is deliberately *metadata-first*: it knows where its audio lives
(a (source index, start, duration) window into a lazy Source) or that it was
*derived* by a processing pass (then its audio lives in a Store). Audio is
materialized only when actually needed — analysis, processing, rendering —
which is what keeps hour-long corpora cheap.

Derived chunks keep a full lineage chain and a tag, so re-processing passes
can query them (``"tag=crushed"``) and the manifest can explain exactly how
a piece came to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .features import Features
from .io import read_wav, to_stereo

if TYPE_CHECKING:  # pragma: no cover
    from .sources import Sources
    from .store import Store

__all__ = ["ChunkRef", "Chunk"]


@dataclass(frozen=True)
class ChunkRef:
    """A window into a lazy Source: source index + start/duration in seconds."""

    src: int
    start: float
    dur: float


@dataclass
class Chunk:
    id: int
    """Source window, or None for derived chunks."""
    ref: ChunkRef | None = None
    """Store key, or None for source chunks."""
    stored: str | None = None
    """Pass-applied tag, e.g. 'dark', 'crushed'."""
    tag: str | None = None
    """Lineage: chain of processor names that produced this chunk."""
    lineage: tuple[str, ...] = ()
    parent: int | None = None
    features: Features | None = None

    @property
    def dur(self) -> float:
        return self.features.dur if self.features else (self.ref.dur if self.ref else 0.0)

    def materialize(self, sources: "Sources", store: "Store | None") -> np.ndarray:
        """(n, 2) float32 audio at the pipeline sample rate."""
        if self.stored is not None:
            if store is None:
                raise RuntimeError(f"chunk {self.id} lives in a store, but no store was provided")
            return to_stereo(store.get(self.stored))
        if self.ref is None:
            raise RuntimeError(f"chunk {self.id} has neither ref nor stored audio")
        return sources[self.ref.src].read(self.ref.start, self.ref.dur)

    @classmethod
    def from_file(cls, chunk_id: int, path, **kw) -> "Chunk":
        """Whole-file chunk (convenience for tiny one-shot wavs)."""
        from .sources import Source

        src = Source(path)
        return cls(chunk_id, ref=ChunkRef(-1, 0.0, src.duration), features=None, **kw)

    def describe(self) -> str:
        origin = f"src{self.ref.src}@{self.ref.start:.2f}s" if self.ref else f"store:{self.stored}"
        f = self.features
        desc = f"#{self.id} {origin} dur={self.dur:.2f}s"
        if f is not None:
            desc += f" rms={f.rms:.3f} cent={f.centroid:.0f}Hz flat={f.flatness:.2f}"
        if self.tag:
            desc += f" [{self.tag}]"
        if self.lineage:
            desc += " <- " + ">".join(self.lineage)
        return desc


def materialize_file(path) -> np.ndarray:
    """Read a whole wav as (n, 2) float32 (helper for tests/examples)."""
    sr, x = read_wav(path)
    return to_stereo(x)