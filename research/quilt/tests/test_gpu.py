"""the metal under the map: GpuEar equivalence + GPU atlas queries.

These tests pin the two claims gpu.py makes:

1. the GPU ear is echo's ear at float32 precision — vectors agree,
   nearest neighbors are preserved, results are run-to-run
   deterministic;
2. the GPU atlas answers the same distance questions as the CPU
   atlas (within float32 tolerance) and the walkers can walk on it.

Skipped automatically when torch/MPS is not on the machine — the
numpy path is always the reference.
"""

import numpy as np
import pytest

import quilt  # noqa: F401  (env fix before numpy loads)
from echo import encode
from echo.probes import SR

from quilt.gpu import GpuEar, mps_available, resolve_device

pytestmark = pytest.mark.skipif(not mps_available(),
                                reason="torch/MPS not available")


def _real_chunks(n=40):
    """A spread of real pile chunks, bucketed by exact length."""
    from pathlib import Path
    from quilt.io import read_audio
    root = Path(__file__).resolve().parents[2] / "corpus"
    files = sorted(root.glob("*__onset_*.wav"))[:n // 2] \
        + sorted(root.glob("*.wav"))[:n // 2]
    by_len = {}
    for f in files:
        try:
            a = read_audio(f)
        except Exception:
            continue
        if len(a) >= int(0.12 * SR):
            by_len.setdefault(len(a), []).append(a)
    return by_len


class TestGpuEar:
    def test_equivalence_with_echo(self):
        by_len = _real_chunks()
        if not by_len:
            pytest.skip("no real chunks available")
        ear = GpuEar()
        for L, group in list(by_len.items())[:6]:
            gv = ear.encode_batch(group)
            cv = np.stack([encode(a, SR) for a in group])
            cos = (gv * cv).sum(1) / (np.linalg.norm(gv, axis=1)
                                      * np.linalg.norm(cv, axis=1))
            assert cos.min() > 0.9999          # same direction, f32 era

    def test_neighbor_preservation(self):
        by_len = _real_chunks(60)
        if not by_len:
            pytest.skip("no real chunks available")
        ear = GpuEar()
        gs, cs = [], []
        for L, group in list(by_len.items())[:8]:
            gv = ear.encode_batch(group)
            cv = np.stack([encode(a, SR) for a in group])
            gs.append(gv)
            cs.append(cv)
        G, C = np.concatenate(gs), np.concatenate(cs)

        def nn(V):
            D = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(D, np.inf)
            return D.argmin(1)

        assert nn(G) is not None
        same = (nn(G) == nn(C)).mean()
        assert same >= 0.9                     # the space is the same

    def test_deterministic(self):
        by_len = _real_chunks(30)
        if not by_len:
            pytest.skip("no real chunks available")
        ear = GpuEar()
        L, group = next(iter(by_len.items()))
        assert np.array_equal(ear.encode_batch(group),
                              ear.encode_batch(group))

    def test_batch_length_contract(self):
        ear = GpuEar()
        a = np.zeros((SR, 2), np.float32)
        b = np.zeros((SR // 2, 2), np.float32)
        with pytest.raises(AssertionError):
            ear.encode_batch([a, b])


class TestGpuAtlas:
    def test_distances_agree_with_cpu(self, bank_path):
        from quilt import ChunkAtlas
        cpu = ChunkAtlas(bank_path, device="cpu")
        gpu = ChunkAtlas(bank_path, device="mps")
        assert gpu.gpu and cpu.device == "cpu"
        for j in (0, cpu.n // 3, cpu.n - 1):
            dc = cpu.dists_from_row(j)
            dg = gpu.dists_from_row(j)
            assert np.allclose(dc, dg, rtol=1e-3, atol=1e-3)
        v = cpu.vec(0)
        assert np.allclose(cpu.dists_from_vec(v), gpu.dists_from_vec(v),
                           rtol=1e-3, atol=1e-3)
        assert np.isclose(cpu.median_nn(), gpu.median_nn(), rtol=1e-2)

    def test_walkers_walk_on_gpu(self, bank_path):
        from quilt import walk, ChunkAtlas
        gpu = ChunkAtlas(bank_path, device="mps")
        r = walk(gpu, mode="roam", seconds=8.0, seed=2)
        assert r.total_dur >= 8.0
        r = walk(gpu, mode="attractor", seconds=8.0, seed=2)
        assert r.total_dur >= 8.0

    def test_device_resolution(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("auto") in ("cpu", "mps")
        assert resolve_device(None) in ("cpu", "mps")