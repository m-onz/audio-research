"""Catalog: the chunk pool, analyzed and queryable.

The catalog holds every Chunk (source-derived and pass-derived), computes
features once per chunk, and exposes everything downstream needs:

- ``query(cond)`` — a tiny condition language over feature *quantiles* and
  tags: ``"rms>0.8"``, ``"air<0.3"``, ``"tag=dark"``, ``"all"``. Numbers are
  always quantiles in [0, 1], so queries mean the same thing on any corpus.
- ``cluster(k)`` — labels via k-means when scikit-learn is available,
  quantile-bucket fallback otherwise.
- ``order(ids, kind)`` — orderings that give pieces their motion:
  ``random``, ``by:<feature>``, ``markov`` (corpus grammar), ``farthest``
  (maximal contrast), ``chase`` (similarity pursuit).

Chunk order is deterministic and stable: sources are iterated in sorted
order, chunks in time order, derived chunks appended in pass order.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .chunks import Chunk
from .features import analyze, features_matrix
from .rng import Seeded
from .sources import Sources
from .store import Store

__all__ = ["Catalog", "order_ids"]


class Catalog:
    """The analyzed chunk pool."""

    def __init__(self, sources: Sources, store: Store | None = None) -> None:
        self.sources = sources
        self.store = store
        self.chunks: list[Chunk] = []
        self._ranks: np.ndarray | None = None  # (n, n_features) quantile ranks
        self._labels_cache: dict[int, np.ndarray] = {}  # k -> cluster labels

    # ---------------------------------------------------------------- build
    def add_refs(self, refs: Sequence) -> list[int]:
        """Add source-window chunks (ChunkRef) and analyze them."""
        from .chunks import Chunk

        ids = []
        for ref in refs:
            audio = self.sources[ref.src].read(ref.start, ref.dur)
            if audio.shape[0] < 32:  # degenerate sliver at a file edge
                continue
            cid = len(self.chunks)
            feats = analyze(audio, self.sources.sr)
            self.chunks.append(Chunk(cid, ref=ref, features=feats))
            ids.append(cid)
        self._invalidate()
        return ids

    def add_derived(
        self,
        parent: Chunk,
        audio: np.ndarray,
        tag: str | None,
        lineage: tuple[str, ...],
        replace: bool = False,
    ) -> int:
        """Store processed audio as a new (or replacing) derived chunk."""
        from .chunks import Chunk

        feats = analyze(audio, self.sources.sr)
        if replace:
            key = f"chunk{parent.id}_v{len(self.chunks)}"
            self.store.put(key, audio)
            old = self.chunks[parent.id]
            self.chunks[parent.id] = Chunk(
                id=parent.id,
                ref=None,
                stored=key,
                tag=tag or old.tag,
                lineage=lineage,
                parent=old.parent,
                features=feats,
            )
            self._invalidate()
            return parent.id
        cid = len(self.chunks)
        key = f"chunk{cid}"
        self.store.put(key, audio)
        self.chunks.append(
            Chunk(id=cid, ref=None, stored=key, tag=tag, lineage=lineage, parent=parent.id, features=feats)
        )
        self._invalidate()
        return cid

    def _invalidate(self) -> None:
        self._ranks = None
        self._labels_cache.clear()

    # ---------------------------------------------------------------- features
    @property
    def n(self) -> int:
        return len(self.chunks)

    def matrix(self) -> np.ndarray:
        """(n, d) raw feature matrix."""
        return features_matrix([c.features for c in self.chunks])

    def ranks(self) -> np.ndarray:
        """(n, d) quantile ranks in [0, 1] per feature column.

        The rank transform is the workhorse of every selection query: it is
        scale-free and robust to outliers, and matches how the audio_lab
        experiments always bucketed by quantile, never by absolute value.
        (Stable argsort gives tied values ordered ranks — fine for queries.)
        """
        if self._ranks is None:
            m = self.matrix()
            if m.shape[0] == 0:
                self._ranks = np.zeros_like(m)
            else:
                n, d = m.shape
                order = np.argsort(m, axis=0, kind="stable")  # order[k, b] = row of k-th smallest in column b
                ranks = np.empty((n, d), dtype=np.float64)
                vals = (np.arange(n, dtype=np.float64) / max(1, n - 1))[:, None]  # rank value per position
                cols = np.broadcast_to(np.arange(d)[None, :], (n, d))
                ranks[order, cols] = np.broadcast_to(vals, (n, d))
                self._ranks = ranks
        return self._ranks

    # ---------------------------------------------------------------- queries
    def query(self, cond: str = "all") -> list[int]:
        """Evaluate a condition over chunks -> matching chunk ids.

        Grammar (`and` binds tighter than `or`; both case-insensitive):
            all                       everything
            tag=NAME                  chunk tag equality
            FEATURE op Q             feature quantile, op in > < >= <= ==
                                      (e.g. ``rms>0.8``, ``flatness<=0.5``)
            src>=I                   from source index >= I

        Examples::
            "rms>0.8"                        loudest quintile
            "bass>0.6 and air<0.3"            woody but not airy
            "tag=dark or tag=dark_echo"      either derived family
        """
        cond = (cond or "all").strip()
        if not cond or cond.lower() == "all":
            return list(range(self.n))
        ranks = self.ranks()
        mask = np.zeros(self.n, dtype=bool)
        for group in re.split(r"\s+or\s+|\|\|", cond, flags=re.IGNORECASE):
            gmask = np.ones(self.n, dtype=bool)
            matched = False
            for term in re.split(r"\s+and\s+|&&", group, flags=re.IGNORECASE):
                term = term.strip()
                if not term or term.lower() == "all":
                    continue
                gmask &= self._term_mask(term, ranks)
                matched = True
            if not matched:  # a group of only 'all' terms matches everything
                gmask[:] = True
            mask |= gmask
        return np.nonzero(mask)[0].tolist()

    def _term_mask(self, term: str, ranks: np.ndarray) -> np.ndarray:
        m = re.fullmatch(r"tag\s*=\s*(\S+)", term, flags=re.IGNORECASE)
        if m:
            return np.array([c.tag == m.group(1) for c in self.chunks])
        m = re.fullmatch(r"(src)\s*(>=|<=|==)\s*([0-9]+)", term)
        if m:
            op, val = m.group(2), int(m.group(3))
            s = np.array([c.ref.src if c.ref else -1 for c in self.chunks])
            return {"<=": s <= val, ">=": s >= val, "==": s == val}[op]
        m = re.fullmatch(r"([a-z_]+)\s*(>=|<=|==|>|<)\s*(-?[0-9.]+)", term, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"cannot parse condition term {term!r}")
        name, op, val = m.group(1).lower(), m.group(2), float(m.group(3))
        col = self._feature_col(name)
        v = ranks[:, col]
        return {
            ">": v > val,
            "<": v < val,
            ">=": v >= val,
            "<=": v <= val,
            "==": np.abs(v - val) < 1e-9,
        }[op]

    def _feature_col(self, name: str) -> int:
        from .features import FEATURE_NAMES

        if name not in FEATURE_NAMES:
            raise ValueError(f"unknown feature {name!r}; have {FEATURE_NAMES}")
        return FEATURE_NAMES.index(name)

    def tags(self) -> set[str]:
        return {c.tag for c in self.chunks if c.tag}

    # ---------------------------------------------------------------- clustering
    def cluster(self, k: int = 12) -> np.ndarray:
        """Cluster label per chunk. k-means (sklearn) or quantile fallback."""
        cached = self._labels_cache.get(int(k))
        if cached is not None and len(cached) == self.n:
            return cached
        if self.n == 0:
            return np.zeros(0, dtype=int)
        k = max(1, min(int(k), self.n))
        m = self.ranks()  # cluster in rank space: every feature comparable
        try:
            from sklearn.cluster import MiniBatchKMeans

            km = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(m)
        except ImportError:
            # fallback: hash chunks into k buckets via the top-2 principal
            # directions of the rank matrix (a poor man's clustering)
            x = m - m.mean(axis=0, keepdims=True)
            u, s, _ = np.linalg.svd(x, full_matrices=False)
            proj = x @ (u[:, :2] if s.size >= 2 else u[:, :1])
            labels = np.clip(((proj[:, 0] + proj[:, -1]) * k / 2).astype(int), 0, k - 1)
        labels = np.asarray(labels, dtype=int)
        self._labels_cache[int(k)] = labels
        return labels

    def cluster_of(self, cid: int, k: int = 12) -> int:
        return int(self.cluster(k)[cid])

    # ---------------------------------------------------------------- geometry
    def nearest(self, cid: int, candidates: Sequence[int], exclude_last: int = 0) -> int:
        """Feature-nearest candidate to chunk `cid` (rank-space L2)."""
        r = self.ranks()
        if not candidates:
            raise ValueError("no candidates for nearest")
        d = np.linalg.norm(r[candidates] - r[cid], axis=1)
        order = np.argsort(d, kind="stable")
        if exclude_last and len(order) > exclude_last:
            return int(candidates[order[exclude_last]])
        return int(candidates[order[0]])

    def farthest_set(self, candidates: Sequence[int], n_wanted: int, rng: Seeded) -> list[int]:
        """Greedy farthest-point selection in rank space (maximal spread)."""
        cands = list(candidates)
        if len(cands) <= n_wanted:
            return rng.shuffle(cands)
        r = self.ranks()
        picked = [int(rng.choice(cands))]
        rest = [c for c in cands if c != picked[0]]
        dist = np.linalg.norm(r[rest] - r[picked[0]], axis=1)
        while len(picked) < n_wanted and rest:
            nxt = int(np.argmax(dist))
            picked.append(rest[nxt])
            d2 = np.linalg.norm(r[rest] - r[rest[nxt]], axis=1)
            dist = np.minimum(dist, d2)
            rest.pop(nxt)
            dist = np.delete(dist, nxt)
        return picked

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict:
        from .features import FEATURE_NAMES

        out = {"n_chunks": self.n, "tags": sorted(self.tags()), "sources": self.sources.names}
        if self.n:
            m = self.matrix()
            med = np.median(m, axis=0)
            out["median"] = {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, med)}
            out["median_rms"] = round(float(med[0]), 5)
        return out

    def describe(self, limit: int | None = None) -> str:
        lines = [f"catalog: {self.n} chunks from {len(self.sources)} sources @{self.sources.sr}Hz"]
        tags = sorted(self.tags())
        if tags:
            lines.append("tags: " + ", ".join(tags))
        for c in self.chunks if limit is None else self.chunks[:limit]:
            lines.append("  " + c.describe())
        return "\n".join(lines)


# --------------------------------------------------------------------------- orderings

def order_ids(
    catalog: Catalog,
    ids: Sequence[int],
    kind: str = "random",
    rng: Seeded = None,
    n: int | None = None,
) -> list[int]:
    """Order chunk ids into a playback sequence.

    kinds:
        random      reshuffle
        by:FEATURE  ascending feature quantile (by:-centroid = darkest first)
        harmonic   pitched chunks walked by consonant intervals (fifths,
                   octaves, fourths first) — a harmonic bed, not a shuffle
        markov      order-1 walk over cluster labels fitted on the corpus's
                    own temporal order (the corpus's grammar)
        markov:K    same, with K clusters
        farthest    greedy farthest-point: maximal timbral contrast
        chase       each next chunk is feature-nearest to the previous
                    (similarity pursuit)
    """
    ids = list(dict.fromkeys(ids))  # dedupe, preserve order
    if not ids:
        return []
    n = len(ids) if n is None else n
    head, rest = kind.split(":", 1) if ":" in kind else (kind, None)

    if head == "random":
        out = rng.shuffle(ids)
    elif head == "by":
        col = catalog._feature_col(rest or "rms")
        r = catalog.ranks()
        out = sorted(ids, key=lambda i: float(r[i, col]))
    elif head == "farthest":
        out = catalog.farthest_set(ids, len(ids), rng)
    elif head == "chase":
        cur = int(rng.choice(ids))
        out = [cur]
        remaining = [i for i in ids if i != cur]
        while remaining:
            cur = catalog.nearest(cur, remaining, exclude_last=1)
            remaining.remove(cur)
            out.append(cur)
    elif head == "harmonic":
        out = _harmonic_order(catalog, ids, rng)
    elif head == "markov":
        k = int(rest) if rest else 12
        out = _markov_order(catalog, ids, k, rng)
    else:
        raise ValueError(f"unknown ordering {kind!r} (random | by:f | markov[:k] | farthest | chase)")
    while len(out) < n:  # cycle with reshuffle to reach requested length
        extra = rng.shuffle(out[: min(len(out), n - len(out))]) if out else ids
        out.extend(extra)
    return out[:n]


def _harmonic_order(catalog: Catalog, ids: Sequence[int], rng: Seeded) -> list[int]:
    """Walk pitched chunks by consonant interval from the current one.

    Interval classes scored (semitones from 0): unison/octave best, fifth,
    fourth, thirds/sixths next, seconds/sevenths weak, tritone worst; octave
    distance between the chunks adds a small penalty (stay in register).
    Deterministic given rng; unpitched chunks are appended shuffled.
    """
    ids = list(ids)
    # interval classes 0-11 (semitones): unison/octave best, P5, P4,
    # thirds/sixths next, seconds/sevenths weak, tritone worst
    cons = [4.0, 1.0, 2.0, 2.5, 3.0, 3.5, 0.5, 4.0, 3.0, 2.5, 2.0, 1.0]
    f0: dict[int, float | None] = {}
    for i in ids:
        f = catalog.chunks[i].features.pitch if catalog.chunks[i].features else 0.0
        f0[i] = float(f) if f and f > 40.0 else None
    pitched = [i for i in ids if f0[i]]
    rest = [i for i in ids if not f0[i]]
    out: list[int] = []
    if not pitched:
        return rng.shuffle(ids)
    cur = pitched[int(rng.ints(len(pitched)))]
    pool = [i for i in pitched if i != cur]
    out.append(cur)
    while pool:
        f_cur = f0[cur]

        def score(i: int) -> float:
            f = f0[i]
            if f is None or f_cur is None:
                return -1.0
            d = abs(12.0 * np.log2(f / f_cur))
            s = cons[int(round(d)) % 12] - 0.35 * (d // 12)
            if d < 0.75:
                s -= 1.0  # literal unison: prefer consonant movement, but never fall below seconds
            return s + 0.01 * rng.float()

        nxt = max(pool, key=score)
        out.append(nxt)
        pool.remove(nxt)
        cur = nxt
    return out + rng.shuffle(rest)


def _markov_order(catalog: Catalog, ids: Sequence[int], k: int, rng: Seeded) -> list[int]:
    """Order-1 markov chain over cluster labels, fitted on natural order.

    For source chunks the natural order is where they came from in time —
    this learns the corpus's own transition grammar. Derived chunks inherit
    their parents' position, so pass-transformed material still speaks in
    the corpus's phrasing. Guarantees a permutation: every chunk used once.
    """
    ids = list(ids)
    if len(ids) <= 2:
        return rng.shuffle(ids)
    labels = catalog.cluster(k)
    by_label: dict[int, list[int]] = {}
    for i in ids:
        by_label.setdefault(int(labels[i]), []).append(i)
    for v in by_label.values():
        rng.rng.shuffle(v)

    # natural order: by (parent's source position, derivation depth)
    def sort_key(cid: int) -> tuple:
        c = catalog.chunks[cid]
        p = catalog.chunks[c.parent] if c.parent is not None and c.parent != cid else c
        if p.ref:
            return (p.ref.src, p.ref.start, cid)
        return (0, 0.0, cid)

    natural = sorted(ids, key=sort_key)
    nat_lab = [int(labels[i]) for i in natural]
    counts = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(nat_lab[:-1], nat_lab[1:]):
        counts[a, b] += 1.0
    rows = counts.sum(axis=1, keepdims=True)
    trans = np.divide(counts, rows, out=np.full_like(counts, 1.0 / k), where=rows > 0)

    used: set[int] = set()
    out: list[int] = []
    lab = int(labels[rng.choice(ids)])
    guard = 0
    max_guard = 10 * len(ids) + 100
    while len(out) < len(ids) and guard < max_guard:
        guard += 1
        pool = [i for i in by_label.get(lab, []) if i not in used]
        if not pool:
            # label exhausted: jump, preferring transition-favored labels
            # that still have unused chunks
            row = trans[lab] if lab < len(trans) else np.zeros(len(trans))
            cands = [l for l, v in by_label.items() if any(i not in used for i in v)]
            if not cands:
                break
            weights = np.array([row[l] if l < len(row) else 0.0 for l in cands])
            if weights.sum() > 0:
                lab = cands[int(rng.rng.choice(len(cands), p=weights / weights.sum()))]
            else:
                lab = int(rng.choice(cands))
            continue
        pick = pool[0] if len(pool) == 1 else int(rng.choice(pool))
        used.add(pick)
        out.append(pick)
        row = trans[lab] if lab < len(trans) else np.zeros(len(trans))
        if row.sum() > 0:
            lab = int(rng.rng.choice(len(row), p=row / row.sum()))
        else:
            lab = int(rng.integers(len(row)))
    return out