"""the glue: routes become exact, bit-reproducible audio."""

import numpy as np
import pytest
import soundfile as sf

from quilt import walk, render_route, write_route_json, load_route


@pytest.fixture()
def route_json(atlas):
    route = walk(atlas, mode="chain", seconds=6.0, seed=0)
    return route.to_json(atlas)


class TestRender:
    def test_exact_duration(self, atlas, route_json, tmp_path):
        out = tmp_path / "piece.wav"
        rep = render_route(route_json, out)
        assert rep["pieces"] == len(route_json["steps"])
        info = sf.info(str(out))
        assert abs(info.duration - 6.0) < 0.01      # trimmed to target

    def test_no_exact_ends_on_boundary(self, atlas, route_json, tmp_path):
        out = tmp_path / "piece.wav"
        rep = render_route(route_json, out, exact=False)
        assert rep["duration_s"] >= 6.0

    def test_bit_reproducible(self, atlas, route_json, tmp_path):
        a, b = tmp_path / "a.wav", tmp_path / "b.wav"
        render_route(route_json, a)
        render_route(route_json, b)
        assert a.read_bytes() == b.read_bytes()    # same route, same bytes

    def test_crossfade_still_stitches(self, atlas, route_json, tmp_path):
        out = tmp_path / "xf.wav"
        rep = render_route(route_json, out, crossfade_ms=20.0)
        info = sf.info(str(out))
        assert info.frames > 0
        assert rep["crossfade_ms"] == 20.0

    def test_route_file_round_trip(self, atlas, route_json, tmp_path):
        p = tmp_path / "r.json"
        write_route_json(route_json, p)
        rep = render_route(load_route(p), tmp_path / "again.wav")
        assert rep["pieces"] == len(route_json["steps"])

    def test_missing_corpus_raises(self, route_json, tmp_path):
        route_json["corpus_root"] = str(tmp_path / "nowhere")
        with pytest.raises(FileNotFoundError):
            render_route(route_json, tmp_path / "x.wav")

    def test_stereo_preserved(self, atlas, route_json, tmp_path):
        out = tmp_path / "st.wav"
        render_route(route_json, out)
        info = sf.info(str(out))
        assert info.channels == 2