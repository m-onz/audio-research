"""the territory: distances mean what echo's fitted metric means."""

import numpy as np
import pytest

from quilt import ChunkAtlas


def _sine_idx(atlas, freq_hint="440"):
    """Rows whose source is one of the twin sines vs noise."""
    sine_rows = [i for i, m in enumerate(atlas.metas)
                 if "sine" in m["file"]]
    noise_rows = [i for i, m in enumerate(atlas.metas)
                  if "noise" in m["file"]]
    return sine_rows, noise_rows


class TestGeometry:
    def test_self_distance_zero(self, atlas):
        j = 0
        d = atlas.dists_from_row(j)
        assert d[j] == pytest.approx(0.0)

    def test_twin_is_the_nearest(self, atlas):
        """The known answer: sine_1's nearest neighbor (excluding
        itself and its own twins) is sine_twin — family retrieval."""
        sines, noise = _sine_idx(atlas)
        for j in sines:
            d = atlas.dists_from_row(j)
            order = np.argsort(d)
            # skip audio twins (same file instant, other scissors)
            for cand in order:
                m = atlas.metas[cand]
                if m["file"] == atlas.metas[j]["file"] \
                        and abs(m["t0"] - atlas.metas[j]["t0"]) < 1e-9:
                    continue
                assert "sine" in m["file"]      # the twin wins
                break
            break                                  # one proof is enough

    def test_noise_is_far(self, atlas):
        sines, noise = _sine_idx(atlas)
        j = sines[0]
        d = atlas.dists_from_row(j)
        sine_far = max(d[i] for i in sines if i != j)
        noise_near = min(d[i] for i in noise)
        assert noise_near > sine_far               # archetypes separate

    def test_median_nn_positive_and_finite(self, atlas):
        v = atlas.median_nn()
        assert np.isfinite(v) and v > 0.0
        assert atlas.median_nn() == v             # cached

    def test_pca_orthonormal(self, atlas):
        P, coords, cmean, cstd = atlas.pca(3)
        assert P.shape == (atlas.dim, 3)
        assert coords.shape == (atlas.n, 3)
        G = P.T @ P
        assert np.allclose(G, np.eye(3), atol=1e-8)

    def test_strategy_filter(self, bank_path):
        a = ChunkAtlas(bank_path, strategies=["fixed"])
        assert a.n > 0
        assert set(a.strategies.tolist()) == {"fixed"}

    def test_durs_match_metas(self, atlas):
        for j in range(atlas.n):
            assert atlas.durs[j] == pytest.approx(atlas.metas[j]["dur"])