"""form tests — the tempo map is exact math, then it is music."""

import numpy as np
import pytest

from maestro.form import FORMS, Form, build_form, describe_form


def test_flat_form_is_uniform():
    f = build_form("flat")
    tt = f.tick_times()
    assert np.allclose(np.diff(tt), 60.0 / 120.0 / 4)     # 120 bpm x 4 tpb
    assert len(tt) == 128 * 4 * 4 + 1                      # bars*beats*tpb + 1


def test_section_durations_are_exact():
    f = Form([(90, 2, "a"), (180, 4, "b")])
    tt = f.tick_times()
    bounds = f.section_bounds()
    assert bounds[0]["t1"] == pytest.approx(2 * 4 * 60.0 / 90.0, abs=0.01)
    assert bounds[1]["t1"] == pytest.approx(2 * 4 * 60.0 / 90.0
                                            + 4 * 4 * 60.0 / 180.0, abs=0.01)
    # inside a section, ticks are uniform at that section's rate
    n0 = 2 * 4 * 4
    assert np.allclose(np.diff(tt[:n0 + 1]), 60.0 / 90.0 / 4)
    assert np.allclose(np.diff(tt[n0:]), 60.0 / 180.0 / 4)


def test_length_extension_holds_last_tempo():
    f = Form([(100, 2), (200, 2)])
    tt = f.tick_times(length_s=10.0)
    assert tt[-1] >= 10.0
    assert np.allclose(np.diff(tt[-8:]), 60.0 / 200.0 / 4)


def test_tick_times_strictly_increasing():
    for name in FORMS:
        tt = build_form(name).tick_times(length_s=200.0)
        assert np.all(np.diff(tt) > 0)


def test_named_forms_land_near_three_minutes():
    for name in ("tectonic", "accel", "flip"):
        f = build_form(name)
        total = sum(s.seconds for s in f.sections)
        assert 120.0 <= total <= 240.0, f"{name}: {total}s"


def test_build_form_variants(tmp_path):
    # a json list of sections
    f = build_form([[100, 4, "a"], [300, 4, "b"]])
    assert f.sections[1].bpm == 300
    # a json file
    p = tmp_path / "form.json"
    p.write_text('{"ticks_per_beat": 8, "sections": [[100, 4], [300, 4]]}')
    f = build_form(str(p))
    assert f.ticks_per_beat == 8
    # name@tpb
    f = build_form("flat@8")
    assert f.ticks_per_beat == 8
    with pytest.raises(ValueError):
        build_form("nonexistent")


def test_describe_form_runs():
    assert "bpm" in describe_form(build_form("tectonic"))