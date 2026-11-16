"""sounds: the material population — genomes, queues, speciation.

A SoundGenome is a lineage: a reference to its origin (noise or a founder)
plus an ordered list of ops applied to produce the phenotype. The genome
is tiny; the phenotype materializes on evaluation and is cached.

Speciation is genetics-based (per the MCC paper — no behavior
characterization): the op-type histogram of the lineage. A lineage heavy
in `field` ops is a different species from one heavy in `genesis` ops,
even if their audio happens to sound similar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .ops import mutate, SR

__all__ = ["SoundGenome", "SoundQueue", "SPECIES_KEYS"]


# speciation: by dominant op family in the lineage (genetics, not behavior)
SPECIES_KEYS = ("genesis", "field", "stat", "order", "loss", "structure", "founder")


@dataclass
class SoundGenome:
    genome_id: int
    origin: str                      # "noise" or "founder"
    origin_audio: np.ndarray         # the raw material (never mutated)
    ops: list = field(default_factory=list)   # [(op_name, member_genome_id, params)]
    birth_batch: int = 0
    parent_id: int | None = None
    _phenotype: np.ndarray | None = None

    @property
    def species(self) -> str:
        if not self.ops:
            return "founder"
        fam_of = {n: _family_of(n) for n, _, _ in self.ops}
        counts: dict[str, int] = {}
        for fam in fam_of.values():
            counts[fam] = counts.get(fam, 0) + 1
        return max(counts, key=counts.get) if counts else "founder"

    @property
    def complexity(self) -> int:
        return len(self.ops)

    def materialize(self, population_audio: dict[int, np.ndarray]) -> np.ndarray:
        """Apply the op lineage to the origin audio. `population_audio` maps
        genome_id -> audio for member lookups (stat/structure ops)."""
        if self._phenotype is not None:
            return self._phenotype
        x = self.origin_audio.copy()
        rng = np.random.default_rng(self.genome_id)  # deterministic per genome
        for op_name, member_id, params in self.ops:
            member = population_audio.get(member_id) if member_id is not None else None
            if member is None and op_name in ("morph", "convmate", "histogram", "layer"):
                # member gone from population: self-reference fallback
                member = x
            x, _ = mutate(x, SR, rng, op_name, member=member, **(params or {}))
        self._phenotype = x
        return x


def _family_of(op_name: str) -> str:
    from .ops import FAMILIES

    for fam, members in FAMILIES.items():
        for name, _ in members:
            if name.strip("_") == op_name:
                return fam
    return "structure"


class SoundQueue:
    """Species-wise FIFO queue (per the MCC paper's speciated variant).

    Capacity split evenly across species; each species has its own FIFO;
    overflow removes the oldest *within the overfull species*.
    """

    def __init__(self, capacity: int = 250):
        self.capacity = int(capacity)
        self._species_queues: dict[str, list[SoundGenome]] = {}
        self._all: list[SoundGenome] = []
        self._next_id = 0

    @property
    def size(self) -> int:
        return len(self._all)

    @property
    def species_counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._species_queues.items() if v}

    def per_species_capacity(self) -> int:
        n = max(1, len(self._species_queues))
        return max(1, self.capacity // n)

    def admit(self, genome: SoundGenome) -> None:
        sp = genome.species
        if sp not in self._species_queues:
            self._species_queues[sp] = []
        self._species_queues[sp].append(genome)
        self._all.append(genome)
        self._prune()

    def dequeue_batch(self, batch_size: int) -> list[SoundGenome]:
        """Species-proportionate batch from the front of each FIFO."""
        batch = []
        counts = self.species_counts
        total = sum(counts.values())
        if total == 0:
            return batch
        for sp, n in counts.items():
            take = max(1, int(round(batch_size * n / total)))
            q = self._species_queues[sp]
            for _ in range(min(take, len(q))):
                batch.append(q.pop(0))
        return batch

    def requeue(self, genomes: list[SoundGenome]) -> None:
        """Parents go to the back of their species queue."""
        for g in genomes:
            sp = g.species
            if sp in self._species_queues:
                self._species_queues[sp].append(g)

    def _prune(self) -> None:
        cap = self.per_species_capacity()
        for sp, q in self._species_queues.items():
            if len(q) > cap:
                overflow = len(q) - cap
                removed = q[:overflow]
                self._species_queues[sp] = q[overflow:]
                removed_set = {g.genome_id for g in removed}
                self._all = [g for g in self._all if g.genome_id not in removed_set]

    def get_audio(self, genome_id: int) -> np.ndarray | None:
        for g in self._all:
            if g.genome_id == genome_id:
                return g._phenotype
        return None

    def all_genomes(self) -> list[SoundGenome]:
        return list(self._all)

    def new_genome_id(self) -> int:
        self._next_id += 1
        return self._next_id