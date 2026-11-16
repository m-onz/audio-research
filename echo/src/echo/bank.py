"""bank: the vector archive — sqlite persistence, exact k-NN, novelty.

The DB every sound ever heard lands in. Single sqlite file (the
ecosystem's storage medium of choice), in-memory float32 matrix, exact
brute-force search — at 556 dims this stays sub-millisecond into the
tens of thousands of specimens, which is a few hundred generations of
breeding. Distances are weighted L2 over corpus-standardized dims:

    d(a, b) = || (a - b) * sqrt(w) / sigma ||_2          (mu-centered)

novelty(v) = mean distance to the k nearest archive entries — Lehman &
Stanley's score, the number that answers 'have we heard this?' with a
magnitude instead of a cell collision.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .encode import DIM, LAYOUT

__all__ = ["EchoMetric", "EchoBank"]


class EchoMetric:
    """The distance: per-dim z-normalization (mu, sigma — fitted on a
    corpus) with per-block weights (sqrt(w) folded in). Fitted on your
    material, the axes that never move (e.g. a constant) go inert."""

    def __init__(self, dim: int | None = None, weights: dict | np.ndarray | None = None,
                 layout=None):
        self.layout = layout or LAYOUT
        self.dim = int(dim if dim is not None else sum(d for _, d, _ in self.layout))
        if weights is None:                              # layout defaults
            weights = {name: w for name, _, w in self.layout}
        if isinstance(weights, dict):
            w = np.zeros(self.dim)
            pos = 0
            for name, d, default in self.layout:
                if pos + d > self.dim:                   # mismatched layout
                    break
                w[pos: pos + d] = weights.get(name, default)
                pos += d
            if pos < self.dim or w.sum() <= 0.0:
                raise ValueError(
                    f"layout {[(n, d) for n, d, _ in self.layout]} does not "
                    f"cover dim {self.dim} — pass the matching layout "
                    f"(see encode.layout_for)")
        else:
            w = np.asarray(weights, np.float64)
            if len(w) != self.dim:
                raise ValueError(f"weights dim {len(w)} != {self.dim}")
        self.mu = np.zeros(self.dim)
        self.sigma = np.ones(self.dim)
        self.w = w
        self._scale = np.sqrt(w)                         # (mu/sigma applied at distance time)

    # ---------------------------------------------------------- fitting
    def fit(self, X: np.ndarray) -> "EchoMetric":
        """Standardize every dim on a corpus of vectors (n, dim). Dims
        that barely move get sigma floored to 1.0 — inert, not explosive."""
        X = np.atleast_2d(np.asarray(X, np.float64))
        if X.shape[1] != self.dim:
            raise ValueError(f"fit matrix has {X.shape[1]} dims, want {self.dim}")
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-3] = 1.0                              # inert constants
        self.sigma = sd
        return self

    def state(self) -> dict:
        return {"mu": self.mu.tolist(), "sigma": self.sigma.tolist(),
                "w": self.w.tolist(), "dim": self.dim}

    @classmethod
    def from_state(cls, s: dict) -> "EchoMetric":
        m = cls(dim=s["dim"], weights=np.asarray(s["w"], np.float64))
        m.mu = np.asarray(s["mu"], np.float64)
        m.sigma = np.asarray(s["sigma"], np.float64)
        return m

    # ---------------------------------------------------------- distance
    def scale(self) -> np.ndarray:
        return self._scale / self.sigma

    def prepare(self, v) -> np.ndarray:
        """(v - mu) * sqrt(w)/sigma — the space where L2 is the metric."""
        return (np.asarray(v, np.float64) - self.mu) * self.scale()

    def dist(self, a, b) -> float:
        return float(np.linalg.norm(self.prepare(a) - self.prepare(b)))

    def dists(self, a, B: np.ndarray) -> np.ndarray:
        """Distance from a to every row of B (vectorized)."""
        va = self.prepare(a)
        MB = (np.atleast_2d(B) - self.mu) * self.scale()
        d = MB - va
        return np.sqrt(np.maximum((d * d).sum(axis=1), 0.0))


class EchoBank:
    """The archive: sqlite on disk, numpy in memory, exact k-NN."""

    def __init__(self, path: str | Path = ":memory:",
                 metric: EchoMetric | None = None, layout=None):
        self.path = str(path)
        self.layout = layout or LAYOUT                    # vector anatomy
        self.metric = metric or EchoMetric(layout=self.layout)
        self._con = sqlite3.connect(self.path)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " gen INTEGER, role TEXT, kind TEXT, meta TEXT, vec BLOB)")
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
        self._con.commit()
        self._load()

    # ---------------------------------------------------------- persistence
    def _load(self):
        mrow = self._con.execute(
            "SELECT value FROM kv WHERE key='metric'").fetchone()
        if mrow is not None:                              # metric BEFORE scaling
            self.metric = EchoMetric.from_state(json.loads(mrow[0]))
        rows = self._con.execute(
            "SELECT id, gen, role, kind, meta, vec FROM entries ORDER BY id").fetchall()
        self.ids = np.array([r[0] for r in rows], np.int64)
        self.gens = np.array([r[1] for r in rows], np.int64)
        self.roles = [r[2] or "pop" for r in rows]
        self.kinds = [r[3] or "wav" for r in rows]
        self.metas = [json.loads(r[4]) if r[4] else {} for r in rows]
        self._n = len(rows)
        cap = max(64, 2 * self._n)
        self._Mbuf = np.zeros((cap, self.metric.dim), np.float32)
        self._Msbuf = np.zeros((cap, self.metric.dim), np.float64)
        self._normbuf = np.zeros(cap, np.float64)
        if rows:
            self._Mbuf[:self._n] = np.stack(
                [np.frombuffer(r[5], np.float32) for r in rows])
            if self._Mbuf.shape[1] != self.metric.dim:
                raise ValueError(
                    f"bank vectors are {self._Mbuf.shape[1]}-D, metric is "
                    f"{self.metric.dim}-D — refit or re-encode")
        self._rebuild()

    @property
    def _M(self) -> np.ndarray:
        """Raw (n, dim) vectors — a view of the preallocated buffer
        (O(1) inserts, no per-add copies)."""
        return self._Mbuf[:self._n]

    @property
    def _Ms(self) -> np.ndarray:
        return self._Msbuf[:self._n]

    @_Ms.setter
    def _Ms(self, v):
        n = len(v)
        self._Msbuf = np.zeros((max(64, 2 * n), v.shape[1]), np.float64)
        self._Msbuf[:n] = v

    @property
    def _norms(self) -> np.ndarray:
        return self._normbuf[:self._n]

    @_norms.setter
    def _norms(self, v):
        self._normbuf = np.zeros(max(64, 2 * len(v)), np.float64)
        self._normbuf[:len(v)] = v

    def _rebuild(self):
        """Cache the scaled matrix + norms so k-NN is one matmul."""
        n = self._n
        if n and self._Mbuf.shape[1] == self.metric.dim:
            self._Msbuf[:n] = (self._Mbuf[:n] - self.metric.mu) * self.metric.scale()
            self._normbuf[:n] = (self._Msbuf[:n] * self._Msbuf[:n]).sum(axis=1)

    def save_metric(self):
        """Persist metric + the vector anatomy this bank speaks — so a
        bank is never silently mixed across encoder versions."""
        self._con.execute(
            "INSERT OR REPLACE INTO kv VALUES ('metric', ?)",
            (json.dumps(self.metric.state()),))
        self._con.execute(
            "INSERT OR REPLACE INTO kv VALUES ('layout', ?)",
            (json.dumps({"dim": self.metric.dim,
                         "layout": [[n, int(d), float(w)]
                                     for n, d, w in self.layout]}),))
        self._con.commit()

    def close(self):
        self._con.close()

    # ---------------------------------------------------------- writing
    def fit(self, X: np.ndarray, save: bool = True) -> "EchoBank":
        """Fit the metric on a corpus of raw vectors, then re-scale."""
        self.metric.fit(X)
        if save:
            self.save_metric()
        self._rebuild()
        return self

    def add(self, vec, meta: dict | None = None, role: str = "pop",
            kind: str = "wav", gen: int = 0, commit: bool = True) -> int:
        v = np.asarray(vec, np.float64)
        if v.shape != (self.metric.dim,):
            raise ValueError(f"vector is {v.shape}, want ({self.metric.dim},)")
        cur = self._con.execute(
            "INSERT INTO entries (gen, role, kind, meta, vec) VALUES (?,?,?,?,?)",
            (int(gen), role, kind, json.dumps(meta or {}),
             np.asarray(v, np.float32).tobytes()))
        if commit:
            self._con.commit()
        eid = int(cur.lastrowid)
        # incremental cache update (amortized O(1) — the breeding loop
        # adds thousands of children one at a time)
        if self._n >= self._Mbuf.shape[0]:               # grow x2
            cap = 2 * self._Mbuf.shape[0]
            M2 = np.zeros((cap, self.metric.dim), np.float32)
            M2[:self._n] = self._Mbuf[:self._n]
            self._Mbuf = M2
            Ms2 = np.zeros((cap, self.metric.dim), np.float64)
            Ms2[:self._n] = self._Msbuf[:self._n]
            self._Msbuf = Ms2
            self._normbuf = np.concatenate(
                [self._normbuf, np.zeros(cap - self._normbuf.shape[0])])
        self._Mbuf[self._n] = v.astype(np.float32)
        self._Msbuf[self._n] = self.metric.prepare(v)
        self._normbuf[self._n] = float(self._Msbuf[self._n] @ self._Msbuf[self._n])
        self._n += 1
        self.ids = np.append(self.ids, eid)
        self.gens = np.append(self.gens, int(gen))
        self.roles.append(role)
        self.kinds.append(kind)
        self.metas.append(meta or {})
        return int(eid)

    def add_many(self, vecs, metas=None, role: str = "pop", kind: str = "wav",
                 gen: int = 0) -> list[int]:
        ids = []
        metas = metas or [None] * len(vecs)
        for v, m in zip(vecs, metas):
            ids.append(self.add(v, m, role, kind, gen, commit=False))
        self._con.commit()
        return ids

    # ---------------------------------------------------------- reading
    def __len__(self):
        return len(self.ids)

    def _mask(self, roles=None, kinds=None) -> np.ndarray | None:
        if roles is None and kinds is None:
            return None
        m = np.ones(len(self.ids), bool)
        if roles is not None:
            m &= np.isin(np.asarray(self.roles), np.atleast_1d(roles))
        if kinds is not None:
            m &= np.isin(np.asarray(self.kinds), np.atleast_1d(kinds))
        return m

    def knn(self, vec, k: int = 15, roles=None, exclude=None) -> list[dict]:
        """k nearest archive entries — exact, one matmul."""
        if len(self.ids) == 0:
            return []
        vs = self.metric.prepare(vec)
        d2 = self._norms - 2.0 * (self._Ms @ vs) + float(vs @ vs)
        d2 = np.maximum(d2, 0.0)
        mask = self._mask(roles)
        if mask is not None:
            d2 = np.where(mask, d2, np.inf)
        if exclude is not None:
            d2 = np.where(np.isin(self.ids, np.atleast_1d(exclude)), np.inf, d2)
        finite = int(np.isfinite(d2).sum())
        k = int(min(k, finite))
        if k == 0:
            return []
        order = np.argpartition(d2, k - 1)[:k]
        order = order[np.argsort(d2[order])]
        return [{"id": int(self.ids[i]), "dist": float(np.sqrt(d2[i])),
                 "gen": int(self.gens[i]), "role": self.roles[i],
                 "kind": self.kinds[i], "meta": self.metas[i]} for i in order]

    def novelty(self, vec, k: int = 15, roles=None) -> float:
        """Lehman & Stanley: mean distance to the k nearest neighbors.
        inf on an empty archive — nothing has been heard, everything is
        new (or the question is not yet meaningful)."""
        if len(self.ids) == 0:
            return float("inf")
        near = self.knn(vec, k, roles)
        return float(np.mean([n["dist"] for n in near]))

    def min_dist(self, vec, roles=None) -> float:
        """Distance to the single nearest entry — tomb's discriminability
        (the tau question), now against everything ever heard."""
        if len(self.ids) == 0:
            return float("inf")
        vs = self.metric.prepare(vec)
        d2 = np.maximum(self._norms - 2.0 * (self._Ms @ vs) + float(vs @ vs), 0.0)
        mask = self._mask(roles)
        if mask is not None:
            d2 = np.where(mask, d2, np.inf)
        return float(np.sqrt(d2.min()))

    def matrix(self) -> np.ndarray:
        """Raw (n, dim) matrix of every vector in the bank."""
        return self._M.copy()

    def stats(self) -> dict:
        by_role: dict[str, int] = {}
        for r in self.roles:
            by_role[r] = by_role.get(r, 0) + 1
        return {"entries": len(self.ids), "dim": self.metric.dim,
                "roles": by_role, "path": self.path}