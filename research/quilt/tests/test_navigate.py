"""the walkers: each policy leaves a measurable trace in its steps."""

import numpy as np
import pytest

from quilt import walk, MODES


def _dists(atlas, route):
    return np.array([s.dist_prev for s in route.steps[1:]])


class TestAllModes:
    @pytest.mark.parametrize("mode", sorted(MODES))
    def test_reaches_the_target(self, atlas, mode):
        route = walk(atlas, mode=mode, seconds=6.0, seed=0)
        assert route.total_dur >= 6.0
        assert all(0 <= s.i < atlas.n for s in route.steps)
        assert route.stats["n_steps"] == len(route.steps)
        assert route.mode == mode

    @pytest.mark.parametrize("mode", sorted(MODES))
    def test_deterministic_in_seed(self, atlas, mode):
        r1 = walk(atlas, mode=mode, seconds=6.0, seed=5)
        r2 = walk(atlas, mode=mode, seconds=6.0, seed=5)
        assert [s.i for s in r1.steps] == [s.i for s in r2.steps]

    def test_seeds_differ(self, atlas):
        r1 = walk(atlas, mode="roam", seconds=6.0, seed=1)
        r2 = walk(atlas, mode="roam", seconds=6.0, seed=2)
        assert [s.i for s in r1.steps] != [s.i for s in r2.steps]

    def test_unknown_mode_raises(self, atlas):
        with pytest.raises(ValueError):
            walk(atlas, mode="teleport")


class TestChain:
    def test_similar_in_sequence(self, atlas):
        """chain steps are much closer than random pairs — the
        'similar in sequence' guarantee, measured."""
        route = walk(atlas, mode="chain", seconds=10.0, seed=0,
                     temp=0.2, exclude_window=12)
        dp = _dists(atlas, route)
        med_rand = route.stats["random_pair_median"]
        assert np.median(dp) < 0.6 * med_rand

    def test_no_recent_recurrence(self, atlas):
        route = walk(atlas, mode="chain", seconds=20.0, seed=3)
        # the effective window (tiny banks cap it at n // 3)
        w = min(12, max(1, atlas.n // 3))
        seen = []
        for s in route.steps:
            assert s.i not in seen[-w:]             # exclusion window holds
            seen.append(s.i)


class TestContrast:
    def test_most_different(self, atlas):
        """contrast steps are farther than random pairs — the
        'most different next' guarantee, measured."""
        route = walk(atlas, mode="contrast", seconds=10.0, seed=0)
        dp = _dists(atlas, route)
        med_rand = route.stats["random_pair_median"]
        assert np.median(dp) > 0.9 * med_rand


class TestRoam:
    def test_roams(self, atlas):
        """neither chain nor contrast: the trace sits between them."""
        r_chain = walk(atlas, mode="chain", seconds=10.0, seed=0, temp=0.2)
        r_contr = walk(atlas, mode="contrast", seconds=10.0, seed=0)
        r_roam = walk(atlas, mode="roam", seconds=10.0, seed=0, speed=1.0)
        assert (np.median(_dists(atlas, r_roam))
                > np.median(_dists(atlas, r_chain)))
        assert (np.median(_dists(atlas, r_roam))
                < np.median(_dists(atlas, r_contr)))

    def test_speed_scales_steps(self, atlas, tmp_path):
        """The dial must matter. The extended-run survey found the
        original NN-scaled steps degenerate (every speed produced
        bit-identical routes — the step was microscopic against the
        territory). Now: (1) the step is speed x 0.1 x macro scale,
        (2) different speeds actually walk different routes."""
        import numpy as np
        from echo.probes import write_wav, SR
        from quilt.index import build_bank
        from quilt import ChunkAtlas
        from quilt.navigate import _Roam, _Nav
        # 1. the mechanism, directly
        nav = _Nav(atlas, 12)
        rng = np.random.default_rng(0)
        r1 = _Roam(atlas, rng, nav, {"speed": 1.0})
        r4 = _Roam(atlas, rng, nav, {"speed": 4.0})
        assert abs(r1.step_len - 0.1 * atlas.macro_scale()) < 1e-9
        assert abs(r4.step_len - 0.4 * atlas.macro_scale()) < 1e-9
        # 2. the behavior: on a dense territory (64 eighth-tone steps
        #    across 8 octaves — pieces ~1-2 apart at a macro scale of
        #    ~25) the speeds must actually walk different routes.
        root = tmp_path / "line"
        root.mkdir()
        for k in range(64):
            freq = 55.0 * (2 ** (k / 8))              # eighth tones
            t = np.arange(int(1.2 * SR)) / SR
            x = 0.4 * np.sin(2 * np.pi * freq * t)
            write_wav(str(root / f"tone_{k:02d}.wav"),
                      np.stack([x, x], 1), SR)      # short: kept whole
        bank = tmp_path / "line.sqlite"
        build_bank(root, bank, log=lambda s: None)
        line = ChunkAtlas(bank)
        slow = walk(line, mode="roam", seconds=24.0, seed=0, speed=0.25)
        fast = walk(line, mode="roam", seconds=24.0, seed=0, speed=4.0)
        assert [s.i for s in slow.steps] != [s.i for s in fast.steps]
        # and the fast walk leaps farther than the slow one strolls
        ds = np.array([s.dist_prev for s in slow.steps[1:]])
        df = np.array([s.dist_prev for s in fast.steps[1:]])
        assert np.median(df) > np.median(ds)


class TestFollow:
    def test_continues_sequences(self, atlas):
        route = walk(atlas, mode="follow", seconds=12.0, seed=1,
                     branch=0.0)                     # never branch
        notes = [s.note for s in route.steps]
        assert sum(n.startswith("continue") for n in notes) >= 2
        # every 'continue' is the true successor: same original, same
        # scissor, next chunk index (the pile's naming convention)
        for prev, cur in zip(route.steps, route.steps[1:]):
            if cur.note == "continue":
                mp, mc = atlas.metas[prev.i], atlas.metas[cur.i]
                assert mp["orig"] == mc["orig"]
                assert mp["strategy"] == mc["strategy"]
                assert mc["idx"] == mp["idx"] + 1


class TestStats:
    def test_temperature_trace(self, atlas):
        r = walk(atlas, mode="chain", seconds=8.0, seed=0, temp=0.0)
        assert r.stats["temperature"] < 1.0         # chain runs cooler
        r = walk(atlas, mode="contrast", seconds=8.0, seed=0)
        assert r.stats["temperature"] > 1.0          # contrast runs hot

    def test_route_json_round_trips(self, atlas):
        route = walk(atlas, mode="roam", seconds=6.0, seed=4)
        j = route.to_json(atlas)
        assert j["mode"] == "roam" and j["seed"] == 4
        assert len(j["steps"]) == len(route.steps)
        s0 = j["steps"][0]
        assert {"file", "t0", "t1", "dur", "strategy", "note"} <= set(s0)