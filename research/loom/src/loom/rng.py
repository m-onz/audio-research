"""Deterministic, seedable randomness.

Every stochastic decision in loom flows through a `Seeded` instance, so a
given seed reproduces a piece byte-for-byte. Child generators are spawned
with numpy `SeedSequence`, which gives statistically independent streams
(safe to spawn many children from one parent).
"""

from __future__ import annotations

import numpy as np

__all__ = ["Seeded"]


class Seeded:
    """A thin wrapper around np.random.Generator with deterministic spawning."""

    __slots__ = ("rng", "_seq")

    def __init__(self, seed: int | np.random.SeedSequence) -> None:
        self._seq = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(int(seed))
        self.rng = np.random.default_rng(self._seq)

    def spawn(self, salt: str = "") -> "Seeded":
        """A child generator; `salt` only affects the name, not the stream."""
        return Seeded(self._seq.spawn(1)[0])

    # ---- common conveniences -------------------------------------------------
    def float(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(self.rng.uniform(low, high))

    def ints(self, low: int, high: int | None = None) -> int:
        return int(self.rng.integers(low, high))

    def choice(self, seq):
        if not len(seq):
            raise ValueError("choice from empty sequence")
        return seq[self.ints(len(seq))]

    def shuffle(self, arr: list) -> list:
        """Return a shuffled copy (input is not mutated)."""
        out = list(arr)
        self.rng.shuffle(out)
        return out

    def weighted(self, options: list, weights) -> object:
        if not options:
            raise ValueError("weighted choice from empty options")
        idx = self.rng.choice(len(options), p=np.asarray(weights, dtype=np.float64))
        return options[int(idx)]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Seeded({self._seq.generate_state(1, dtype=np.uint32)[0]})"