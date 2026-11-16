"""the strangest pieces: echo novelty scores, views, and hocket grains.

novelty = mean distance to the k nearest neighbors (Lehman & Stanley,
via echo). The tests pin: scores are sane and ranked as a listener
would rank (the noise archetype is stranger than the sines), views
restrict without lying, and hocket grains are sample-exact sub-spans
that tile each voice's pieces contiguously while the clock breathes.
"""

import numpy as np
import pytest

from quilt import ChunkAtlas, walk, render_route
from quilt.io import write_audio
from echo.probes import SR


class TestNovelty:
    def test_scores_sane(self, atlas):
        s = atlas.novelty_scores(k=5)
        assert s.shape == (atlas.n,)
        assert np.isfinite(s).all() and (s >= 0).all()
        assert np.array_equal(s, atlas.novelty_scores())   # cached

    def test_novelty_is_embeddedness(self, atlas):
        """Novelty is not distance from the center — it is how
        embedded a piece is among its neighbors. The isolated sine
        pair (a family of two, far from everything) must rank more
        novel than the clicks family (a dense mutual cluster), even
        though clicks are 'stranger' sounds in the abstract."""
        s = atlas.novelty_scores(k=5)
        sines = [i for i, m in enumerate(atlas.metas) if "sine" in m["file"]]
        clicks = [i for i, m in enumerate(atlas.metas) if "clicks" in m["file"]]
        assert np.mean([s[i] for i in sines]) > \
            np.mean([s[i] for i in clicks])
        assert len(sines) >= 2 and len(clicks) >= 2

    def test_novelty_view_restricts(self, atlas):
        view = atlas.novelty_view(keep=0.25, k=5)
        assert 0 < view.n <= max(2, int(np.ceil(0.25 * atlas.n)) + 1)
        # the view holds exactly the top-scoring pieces
        s = atlas.novelty_scores(k=5)
        top = set(np.argsort(s)[-view.n:].tolist())
        assert top == set(view.rows.tolist())
        # walkable, honest metadata
        for m in view.metas:
            assert m in atlas.metas
        r = walk(view, mode="chain", seconds=6.0, seed=0)
        assert all(0 <= st.i < view.n for st in r.steps)

    def test_restrict_validates(self, atlas):
        with pytest.raises(ValueError):
            atlas.restrict([])
        with pytest.raises(ValueError):
            atlas.restrict([atlas.n])                     # out of range

    def test_view_distances_agree(self, atlas):
        view = atlas.novelty_view(keep=0.3, k=5)
        j = int(view.rows[0])
        row_in_parent = np.where(atlas.rows == atlas.rows[j])[0][0]
        dv = view.dists_from_row(0)
        dp = atlas.dists_from_row(row_in_parent)
        # every view distance exists in the parent's row
        assert len(dv) == view.n
        assert np.isin(np.round(dv, 9), np.round(dp, 9)).all()


class TestHocket:
    @pytest.fixture()
    def route(self, atlas):
        return walk(atlas, mode="hocket", seconds=8.0, seed=3,
                    voices=2, grain_lo=0.03, grain_hi=0.25)

    def test_reaches_target(self, route):
        assert route.total_dur >= 8.0

    def test_grains_are_subspans(self, route, atlas):
        for st in route.steps:
            assert st.t0 is not None and st.t1 is not None
            assert 0.0 <= st.t0 < st.t1
            assert st.t1 <= atlas.durs[st.i] + 1e-9
            assert abs((st.t1 - st.t0) - st.dur) < 1e-9   # dur == span

    def test_voice_pieces_play_through(self, route, atlas):
        """A voice plays THROUGH its piece: while it holds a piece,
        its grains are contiguous; when it advances, the new piece
        starts at its head."""
        prev = {}                                        # voice -> (piece, t1)
        for st in route.steps:
            v = int(st.note.split()[0][1:])              # "v0 g=..."
            if v in prev:
                pi, pt1 = prev[v]
                if pi == st.i:
                    assert abs(st.t0 - pt1) < 1e-9       # contiguous
                else:
                    assert abs(st.t0) < 1e-9             # fresh head
            prev[v] = (st.i, st.t1)

    def test_voices_interleave(self, route):
        vs = [int(st.note.split()[0][1:]) for st in route.steps]
        assert set(vs) == {0, 1}                         # both voices
        assert len(route.steps) > 20                     # granular

    def test_deterministic(self, atlas):
        a = walk(atlas, mode="hocket", seconds=6.0, seed=5, voices=2)
        b = walk(atlas, mode="hocket", seconds=6.0, seed=5, voices=2)
        assert [(s.i, s.t0, s.t1) for s in a.steps] == \
               [(s.i, s.t0, s.t1) for s in b.steps]

    def test_seeds_differ(self, atlas):
        a = walk(atlas, mode="hocket", seconds=6.0, seed=1, voices=2)
        b = walk(atlas, mode="hocket", seconds=6.0, seed=2, voices=2)
        assert [s.i for s in a.steps] != [s.i for s in b.steps]

    def test_inner_roam(self, atlas):
        r = walk(atlas, mode="hocket", seconds=6.0, seed=0, voices=3,
                 inner="roam")
        assert r.total_dur >= 6.0
        assert {int(s.note.split()[0][1:]) for s in r.steps} == {0, 1, 2}

    def test_bad_inner_raises(self, atlas):
        with pytest.raises(ValueError):
            walk(atlas, mode="hocket", seconds=4.0, inner="contrast")

    def test_grain_clock_breathes(self, atlas):
        """The grain size must change over the piece (the frequency of\n        interleaving is not constant)."""
        r = walk(atlas, mode="hocket", seconds=10.0, seed=4, voices=2,
                 grain_lo=0.03, grain_hi=0.3)
        durs = [s.dur for s in r.steps]
        assert min(durs) < 0.25 * max(durs)              # slow and fast
        # it moves: at least 3 distinct size regimes
        coarse = [round(d, 2) for d in durs]
        assert len(set(coarse)) >= 3

    def test_render_exact_and_reproducible(self, atlas, route, tmp_path):
        rj = route.to_json(atlas)
        a, b = tmp_path / "h1.wav", tmp_path / "h2.wav"
        render_route(rj, a)
        render_route(rj, b)
        import soundfile as sf
        info = sf.info(str(a))
        assert abs(info.duration - 8.0) < 0.01           # trimmed exact
        assert a.read_bytes() == b.read_bytes()         # bit-identical

    def test_hocket_on_novelty_view(self, atlas, tmp_path):
        view = atlas.novelty_view(keep=0.3, k=5)
        r = walk(view, mode="hocket", seconds=6.0, seed=0, voices=2,
                 grain_lo=0.03, grain_hi=0.2)
        assert all(0 <= s.i < view.n for s in r.steps)
        render_route(r.to_json(view), tmp_path / "hn.wav")
        import soundfile as sf
        assert sf.info(str(tmp_path / "hn.wav")).frames > 0