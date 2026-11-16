import numpy as np
import pytest

import doom
from doom.audio import analyze_audio, probe_duration
from doom.fx import build_fx_chain, list_fx, run_chain
from doom.palettes import get_palette, psx_quantize
from doom.scenes import list_scenes, make_scene
from doom.timeline import normalize_plan, plan_from_audio


def test_probe_duration(test_wav):
    assert probe_duration(test_wav) == pytest.approx(4.0, abs=0.05)


def test_analyze_features(test_wav):
    f = analyze_audio(test_wav, fps=24.0)
    assert f.n_frames == pytest.approx(4 * 24, abs=2)
    for k in ("rms", "bass", "mid", "treble", "centroid", "flux", "onset"):
        assert len(getattr(f, k)) == f.n_frames
    # the loud parts should register
    assert f.rms[:20].max() > f.rms[24:44].max()  # sine vs silence
    # clicks are transients
    assert f.onset[48:72].sum() >= 1
    # waveform kept for scope scenes
    assert len(f.wave) > 0
    w = f.wave_window(10, seconds=0.05)
    assert len(w) == pytest.approx(0.05 * 8000, abs=8)


def test_every_scene_renders(test_wav):
    feats = analyze_audio(test_wav, fps=12.0)
    for name in list_scenes():
        scene = make_scene(name, 160, 120, seed=0)
        for i in (0, 10, 20):
            frame = scene.render(feats, i)
            assert frame.shape == (120, 160, 3), name
            assert frame.dtype == np.uint8, name
            assert frame.min() >= 0 and frame.max() <= 255, name
            # scene must actually produce something over a few frames
        assert frame.astype(np.float64).mean() > 0.0, name


def test_every_fx_runs(test_wav):
    feats = analyze_audio(test_wav, fps=12.0)
    frame = (np.random.default_rng(0).uniform(0, 255, (120, 160, 3))).astype(np.uint8)
    rng = np.random.default_rng(0)
    for name in list_fx():
        chain = build_fx_chain([name])
        out = run_chain(chain, frame.copy(), feats, 5, rng)
        assert out.shape == frame.shape and out.dtype == np.uint8, name
        assert np.isfinite(out.astype(np.float64)).all(), name


def test_fx_inline_params():
    chain = build_fx_chain(["psx(levels=16)", ["vhs", {"noise": 0.05}], "glow"])
    assert len(chain) == 3


def test_unknown_fx_scene():
    with pytest.raises(ValueError):
        build_fx_chain(["nope"])
    with pytest.raises(ValueError):
        make_scene("nope", 64, 64, 0)


def test_normalize_plan_mixed():
    plan = [
        dict(scene="plasma", start=0, end=2.0),
        dict(scene="tunnel"),           # floating
        dict(scene="fire"),             # floating
        dict(scene="starfield", start=6.0, end=8.0),
        dict(scene="scope"),            # floating until the end
    ]
    out = normalize_plan(plan, 10.0)
    assert [round(s["start"], 1) for s in out] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert [round(s["end"], 1) for s in out] == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_auto_plan(test_wav):
    feats = analyze_audio(test_wav, fps=24.0)
    plan = plan_from_audio(feats, seed=0)
    assert 3 <= len(plan) <= 12
    assert plan[0]["start"] == 0.0
    assert plan[-1]["end"] == pytest.approx(feats.n_frames / feats.fps, abs=0.1)


def test_render_end_to_end(test_wav, tmp_path):
    out = tmp_path / "video.mp4"
    result = doom.render(
        audio=test_wav, out=out, fps=12.0, size=(160, 120),
        plan=[dict(scene="plasma"), dict(scene="starfield", transition="glitch")],
        seed=3, verbose=False,
    )
    assert result["frames"] == pytest.approx(4 * 12, abs=1)
    assert result["video_seconds"] == pytest.approx(4.0, abs=0.2)
    assert "h264" in result["streams"]["video"]["codec"]
    assert "aac" in result["streams"]["audio"]["codec"]
    assert result["size_bytes"] > 0


def test_render_determinism(test_wav, tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    kw = dict(audio=test_wav, fps=12.0, size=(160, 120),
              plan=[dict(scene="fire")], seed=9, crf=22)
    doom.render(out=a, **kw)
    doom.render(out=b, **kw)
    ha = open(a, "rb").read()
    hb = open(b, "rb").read()
    # x264 embeds no timestamps of creation; same input -> same bytes
    assert ha == hb


def test_render_length_shorter(test_wav, tmp_path):
    result = doom.render(audio=test_wav, out=tmp_path / "short.mp4",
                         length=2.0, fps=12.0, size=(160, 120),
                         plan=[dict(scene="plasma")])
    assert result["video_seconds"] == pytest.approx(2.0, abs=0.2)


def test_render_presets(test_wav, tmp_path):
    for preset in ("demo", "ps2", "vhs", "glitchcore", "noc"):
        result = doom.render(audio=test_wav, out=tmp_path / f"{preset}.mp4",
                             preset=preset, fps=12.0, size=(160, 120), crf=24)
        assert result["frames"] >= 4 * 12 - 2, preset


def test_palette_quantize():
    lut = get_palette("inferno")
    assert lut.shape == (256, 3)
    frame = np.full((8, 8, 3), 137, np.uint8)
    q = psx_quantize(frame, levels=32)
    assert q.shape == frame.shape
    # RGB555: every channel value is a multiple of 8
    assert (q % 8 == 0).all()

def test_windowed_render(test_wav, tmp_path):
    """Chunk renders: a 60s window of a longer track, sync via feature views."""
    result = doom.render(audio=test_wav, out=tmp_path / "chunk.mp4",
                         start=1.0, length=1.5, fps=12.0, size=(160, 120),
                         plan=[dict(scene="plasma")])
    assert result["window"] == [1.0, 2.5]
    assert result["video_seconds"] == pytest.approx(1.5, abs=0.2)
    # the window starts in the silent second (1-2s of the fixture):
    # rms features must be the SLICE, not the file head
    feats = doom.analyze_audio(test_wav, fps=12.0).window(1.0, 1.5)
    assert feats.rms[0] < 0.1


def test_new_scenes_render(test_wav):
    feats = doom.analyze_audio(test_wav, fps=12.0)
    for name in ("mask", "newton"):
        s = doom.make_scene(name, 160, 120, seed=0)
        f = s.render(feats, 5)
        assert f.shape == (120, 160, 3) and f.mean() > 0.0, name
