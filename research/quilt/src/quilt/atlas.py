"""atlas: the territory — the chunk bank as a map for walking.

EchoBank answers "have we heard this?" (k-NN in, novelty out). A walker
needs more: distance to *everything*, the farthest point, a random
step's worth of neighbors, the principal axes of the land. ChunkAtlas
loads a bank, keeps the strategy-filtered rows, and holds the scaled
matrix — the space where L2 *is* echo's fitted metric:

    d(a, b) = || (a - b) * sqrt(w) / sigma ||_2

**The metal under the map** (device='auto', on by default for big
banks): the scaled matrix also lives as a float32 tensor on the GPU
(Apple M4 via torch MPS), and every distance query — one GEMV per
walker step, the median-NN GEMM, the PCA covariance GEMM — runs there
(measured 45x vs one CPU thread on this machine at 500k pieces). The
CPU path (float64 numpy, exact echo arithmetic) is always available;
device='cpu' forces it. Same device + same seed = same route.

All navigation happens on rows of the scaled matrix. Distances are
exact on either device (brute force, no approximate weather).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from echo import EchoBank

__all__ = ["ChunkAtlas"]

_GPU_MIN_ROWS = 50_000        # below this the CPU path is instant anyway


class ChunkAtlas:
    """The bank as a walkable map. Holds no audio — only vectors and
    the (file, 0, dur) pins into the corpus."""

    def __init__(self, bank_path, strategies=None, corpus_root=None,
                 device: str = "auto"):
        self.path = str(bank_path)
        self.bank = EchoBank(self.path)
        prov_p = Path(self.path + ".provenance.json")
        self.provenance = (json.loads(prov_p.read_text())
                           if prov_p.exists() else {})
        self.corpus_root = Path(
            corpus_root or self.provenance.get("corpus_root") or ".")

        want = set(strategies) if strategies else None
        rows = [i for i, m in enumerate(self.bank.metas)
                if m.get("strategy") and (want is None
                                          or m["strategy"] in want)]
        if not rows:
            raise ValueError(
                f"bank {self.path} has no chunks"
                + (f" with strategy in {sorted(want)}" if want else "")
                + " — build one with `quilt index`")
        self.rows = np.asarray(rows, np.int64)
        self.metas = [self.bank.metas[i] for i in rows]
        self.ids = self.bank.ids[self.rows]
        self.dim = int(self.bank.metric.dim)
        self.durs = np.array([m["dur"] for m in self.metas], np.float64)
        self.t0s = np.array([m["t0"] for m in self.metas], np.float64)
        self.files = np.array([m["file"] for m in self.metas])
        self.strategies = np.array([m["strategy"] for m in self.metas])
        self._pca = None
        self._nn_med = None
        self._macro = None
        self._novelty = None
        self._mean_vec = None

        # the device resolves after we know the bank's size
        from .gpu import resolve_device, mps_available
        if device == "auto":
            device = "mps" if (mps_available()
                               and len(self.rows) >= _GPU_MIN_ROWS) else "cpu"
        self.device = resolve_device(device)
        X = self.bank.matrix()[self.rows]
        if self.gpu:
            import torch
            Xs32 = ((X.astype(np.float32)
                     - self.bank.metric.mu.astype(np.float32))
                    * self.bank.metric.scale().astype(np.float32))
            self._X = torch.as_tensor(Xs32, device=torch.device(self.device))
            self._Xn = (self._X * self._X).sum(dim=1)
        else:
            self.Xs = (X.astype(np.float64) - self.bank.metric.mu) \
                * self.bank.metric.scale()

    # ------------------------------------------------------------ device
    @property
    def gpu(self) -> bool:
        return self.device == "mps"

    @property
    def n(self) -> int:
        return len(self.metas)

    # ------------------------------------------------------------ geometry
    def vec(self, j: int) -> np.ndarray:
        """The scaled-space position of piece j (on the CPU, as numpy)."""
        if self.gpu:
            return self._X[int(j)].cpu().numpy().astype(np.float64)
        return self.Xs[int(j)]

    def vecs_mean(self, rows) -> np.ndarray:
        """Mean scaled-space position of several pieces — the contrast
        walker's reference point."""
        rows = np.asarray(list(rows), np.int64)
        if self.gpu:
            return self._mean_rows(rows)
        return self.Xs[rows].mean(axis=0)

    def _mean_rows(self, rows: np.ndarray) -> np.ndarray:
        import torch
        idx = torch.as_tensor(rows, device=self._X.device)
        return self._X[idx].mean(dim=0).cpu().numpy().astype(np.float64)

    def mean_vec(self) -> np.ndarray:
        """The centroid of the territory (cached)."""
        if self._mean_vec is None:
            if self.gpu:
                self._mean_vec = self._X.mean(dim=0).cpu().numpy() \
                    .astype(np.float64)
            else:
                self._mean_vec = self.Xs.mean(axis=0)
        return self._mean_vec

    def dists_from_row(self, j: int) -> np.ndarray:
        """Distance from piece j to every piece — one GEMV."""
        if self.gpu:
            import torch
            x = self._X[int(j)]
            d2 = self._Xn - 2.0 * (self._X @ x) + float(
                (x * x).sum())
            return torch.clamp(d2, min=0.0).sqrt().cpu().numpy() \
                .astype(np.float64)
        d = self.Xs - self.Xs[int(j)]
        return np.sqrt(np.maximum((d * d).sum(axis=1), 0.0))

    def dists_from_vec(self, v: np.ndarray) -> np.ndarray:
        """Distance from an arbitrary point in scaled space to every
        piece — walkers steer through empty space with this."""
        if self.gpu:
            import torch
            x = torch.as_tensor(np.asarray(v, np.float32),
                                device=self._X.device)
            d2 = self._Xn - 2.0 * (self._X @ x) + float((x * x).sum())
            return torch.clamp(d2, min=0.0).sqrt().cpu().numpy() \
                .astype(np.float64)
        d = self.Xs - np.asarray(v, np.float64)
        return np.sqrt(np.maximum((d * d).sum(axis=1), 0.0))

    def dist(self, a: int, b: int) -> float:
        if self.gpu:
            d = self.vec(a) - self.vec(b)
            return float(np.sqrt((d * d).sum()))
        d = self.Xs[a] - self.Xs[b]
        return float(np.sqrt((d * d).sum()))

    def pair_dists(self, a_rows, b_rows) -> np.ndarray:
        """Elementwise distances between two row lists — the stats
        yardstick's random pairs."""
        a = np.asarray(list(a_rows), np.int64)
        b = np.asarray(list(b_rows), np.int64)
        if self.gpu:
            import torch
            dev = self._X.device
            A = self._X[torch.as_tensor(a, device=dev)]
            B = self._X[torch.as_tensor(b, device=dev)]
            d = A - B
            return torch.sqrt(torch.clamp(
                (d * d).sum(dim=1), min=0.0)).cpu().numpy() \
                .astype(np.float64)
        d = self.Xs[a] - self.Xs[b]
        return np.sqrt(np.maximum((d * d).sum(axis=1), 0.0))

    def median_nn(self, probe: int = 256) -> float:
        """Median nearest-neighbor distance — the atlas's natural unit
        of length (roam's step size). A probe sample vs the whole
        territory, one GEMM on the metal, cached."""
        if self._nn_med is None:
            if self.n < 2:
                self._nn_med = 1.0
            else:
                rng = np.random.default_rng(1234)     # the unit of length
                m = int(min(probe, self.n))           # is a property, not
                r = rng.choice(self.n, size=m, replace=False)  # a roll
                if self.gpu:
                    import torch
                    dev = self._X.device
                    Q = self._X[torch.as_tensor(r, device=dev)]
                    qn = (Q * Q).sum(dim=1)
                    D2 = self._Xn[None, :] - 2.0 * (Q @ self._X.T) \
                        + qn[:, None]
                    for i, j in enumerate(r):
                        D2[i, j] = np.inf
                    mins = torch.clamp(D2, min=0.0).sqrt().min(dim=1).values
                    self._nn_med = float(torch.quantile(
                        mins, 0.5).item())   # np.median semantics —
                                          # torch.median is the lower
                                          # middle, a different animal
                else:
                    Q = self.Xs[r]
                    x2 = (self.Xs * self.Xs).sum(axis=1)
                    D2 = (Q * Q).sum(axis=1)[:, None] \
                        - 2.0 * (Q @ self.Xs.T) + x2[None, :]
                    for i, j in enumerate(r):
                        D2[i, j] = np.inf
                    self._nn_med = float(np.median(
                        np.sqrt(np.maximum(D2, 0.0).min(axis=1))))
        return self._nn_med

    def macro_scale(self, probe: int = 512) -> float:
        """Median distance between random pairs — the territory's
        macro size. roam's step is sized against THIS, not the
        median-NN: a pile dense with near-duplicate chunks has a
        microscopic NN median (0.011 on this corpus) while the
        territory itself spans ~23 — steps scaled to the NN never
        leave the start. Cached; the probe is a fixed yardstick."""
        if self._macro is None:
            if self.n < 2:
                self._macro = 1.0
            else:
                rng = np.random.default_rng(1234)
                a = rng.integers(0, self.n, probe)
                b = rng.integers(0, self.n, probe)
                keep = a != b
                self._macro = float(np.median(
                    self.pair_dists(a[keep], b[keep])))
        return self._macro

    # ------------------------------------------------------------ novelty
    def novelty_scores(self, k: int = 15, block: int = 2048) -> np.ndarray:
        """echo's novelty for every piece at once: the mean distance to
        its k nearest neighbors (Lehman & Stanley's score) — the number
        that ranks a piece by how unlike its surroundings it is. One
        blocked GEMM over the whole territory (a few minutes on the
        metal at 190k pieces), cached beside the bank so it is paid
        once per bank."""
        cache = Path(self.path + f".novelty_k{k}.npy")
        if self._novelty is None and cache.exists():
            v = np.load(cache)
            if v.shape == (self.n,):
                self._novelty = v
        if self._novelty is None:
            kk = int(min(k, max(self.n - 1, 1)))
            scores = np.empty(self.n, np.float64)
            if self.gpu:
                import torch
                for base in range(0, self.n, block):
                    stop = min(base + block, self.n)
                    idx = torch.arange(base, stop, device=self._X.device)
                    Q = self._X[idx]
                    qn = (Q * Q).sum(dim=1)
                    D2 = qn[:, None] - 2.0 * (Q @ self._X.T) \
                        + self._Xn[None, :]
                    vals = torch.topk(torch.clamp(D2, min=0.0), kk + 1,
                                       dim=1, largest=False).values
                    d = vals[:, 1:].sqrt()             # drop self (the 0)
                    scores[base:stop] = d.mean(dim=1).cpu().numpy()
            else:
                if self.n <= 20000:
                    for base in range(0, self.n, block):
                        stop = min(base + block, self.n)
                        Q = self.Xs[base:stop]
                        D2 = ((Q * Q).sum(1)[:, None]
                              - 2.0 * (Q @ self.Xs.T)
                              + (self.Xs * self.Xs).sum(1)[None, :])
                        part = np.partition(np.maximum(D2, 0.0), kk,
                                            axis=1)[:, :kk + 1]
                        part.sort(axis=1)
                        scores[base:stop] = np.sqrt(part[:, 1:]).mean(axis=1)
                else:      # CPU on a huge bank: probe approximation
                    rng = np.random.default_rng(1234)
                    probe = rng.choice(self.n, size=min(10000, self.n),
                                       replace=False)
                    P = self.Xs[probe]
                    for base in range(0, self.n, block):
                        Q = self.Xs[base: base + block]
                        D2 = ((Q * Q).sum(1)[:, None] - 2.0 * (Q @ P.T)
                              + (P * P).sum(1)[None, :])
                        part = np.partition(np.maximum(D2, 0.0), kk,
                                            axis=1)[:, :kk]
                        scores[base: base + len(Q)] = \
                            np.sqrt(part).mean(axis=1)
            self._novelty = scores
            try:
                np.save(cache, self._novelty)
            except OSError:
                pass
        return self._novelty

    def restrict(self, rows) -> "ChunkAtlas":
        """A walkable view of this atlas over the given rows: same
        bank (shared, do not outlive the parent), same device, its own
        matrices and caches. This is how novelty and strategy filters
        become territories."""
        rows = np.asarray(list(rows), np.int64)
        if rows.ndim != 1 or len(rows) == 0 \
                or not np.isin(rows, np.arange(self.n)).all():
            raise ValueError("restrict needs valid, non-empty row indices")
        view = object.__new__(ChunkAtlas)
        view.path, view.bank = self.path, self.bank
        view.provenance, view.corpus_root = self.provenance, self.corpus_root
        view.rows = rows
        view.metas = [self.metas[i] for i in rows]
        view.ids = self.ids[rows]
        view.dim = self.dim
        view.durs = self.durs[rows]
        view.t0s = self.t0s[rows]
        view.files = self.files[rows]
        view.strategies = self.strategies[rows]
        view.device = self.device
        if self.gpu:
            import torch
            idx = torch.as_tensor(rows, device=self._X.device)
            view._X = self._X[idx].contiguous()
            view._Xn = (view._X * view._X).sum(dim=1)
        else:
            view.Xs = self.Xs[rows]
        view._pca = view._nn_med = view._macro = None
        view._mean_vec = view._novelty = None
        return view

    def novelty_view(self, keep: float = 0.1, k: int = 15) -> "ChunkAtlas":
        """The strangest keep-fraction of the territory as a walkable
        view — scores measured against the FULL territory, so 'novel'
        means unlike the whole corpus, not unlike the other
        leftovers."""
        s = self.novelty_scores(k)
        keep = min(max(float(keep), 1.0 / self.n), 1.0)
        thresh = np.quantile(s, 1.0 - keep)
        rows = np.where(s >= thresh)[0]
        if len(rows) < 2:
            rows = np.argsort(s)[-2:]
        return self.restrict(rows)

    def pca(self, k: int = 3):
        """Top-k principal axes of the scaled space (exact eigh on the
        covariance — the covariance GEMM runs on the metal when the
        atlas is big). Returns (P, coords, cmean, cstd): P maps
        scaled-space -> coords; coords is every piece's (n, k)
        position, cmean/cstd standardize it. Attractor and drift steer
        through these axes."""
        if self._pca is None or self._pca[0].shape[1] != k:
            mu = self.mean_vec()
            if self.gpu:
                import torch
                Xc = self._X - torch.as_tensor(
                    mu.astype(np.float32), device=self._X.device)
                cov_t = (Xc.T @ Xc) / float(max(self.n - 1, 1))
                cov = cov_t.cpu().numpy().astype(np.float64)
            else:
                Xc = self.Xs - mu
                cov = (Xc.T @ Xc) / max(self.n - 1, 1)
            evals, evecs = np.linalg.eigh(cov)         # symmetric: orthonormal
            order = np.argsort(evals)[::-1][:k]
            P = evecs[:, order]                        # (dim, k)
            if self.gpu:
                import torch
                Pt = torch.as_tensor(P.astype(np.float32),
                                      device=self._X.device)
                coords = (Xc @ Pt).cpu().numpy().astype(np.float64)
            else:
                coords = Xc @ P                        # (n, k)
            cstd = coords.std(axis=0)
            cstd[cstd < 1e-9] = 1.0
            self._pca = (P, coords, coords.mean(axis=0), cstd)
        return self._pca

    # ------------------------------------------------------------ census
    def stats(self) -> dict:
        by_strat: dict[str, int] = {}
        for s in self.strategies:
            by_strat[str(s)] = by_strat.get(str(s), 0) + 1
        return {
            "bank": self.path, "pieces": int(self.n), "dim": int(self.dim),
            "device": self.device,
            "unique_files": int(len(set(self.files.tolist()))),
            "strategies": by_strat,
            "dur_min_s": float(self.durs.min()),
            "dur_med_s": float(np.median(self.durs)),
            "dur_max_s": float(self.durs.max()),
            "material_h": float(self.durs.sum() / 3600.0),
            "median_nn": round(self.median_nn(), 3),
            "corpus_root": str(self.corpus_root),
        }