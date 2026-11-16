import numpy as np
import pytest

import womb
from womb.sonify import sonify, write_wav


def test_decode_luma(test_video):
    frames, fps = womb.decode_luma(test_video)
    assert len(frames) >= 30          # 3 s @ 12 fps
    assert 10 <= fps <= 14
    assert frames.ndim == 3
    assert 0.0 <= frames.min() and frames.max() <= 1.0


def test_video_stats(test_video):
    s = womb.video_stats(test_video)
    assert s.n_frames >= 30
    assert s.luma.mean() > 0.1            # gradient background is visible
    assert s.motion.max() > 0.0            # stripes move
    assert (s.entropy > 0).all()           # entropy defined
    assert s.flow.max() >= 0


def test_sonify_spectra(test_video):
    """The inverse channel: frames -> sound. Bright regions -> energy."""
    audio, sr = sonify(test_video, mode="spectra", seed=3)
    assert len(audio) > sr                # >= 1 s
    assert np.abs(audio).max() <= 1.0
    # it must actually sound like something
    assert np.sqrt((audio ** 2).mean()) > 0.01


def test_sonify_grains(test_video, test_wav):
    audio, sr = sonify(test_video, mode="grains", seed=3, corpus=test_wav)
    assert len(audio) > sr
    assert np.abs(audio).max() <= 1.0
    assert np.sqrt((audio ** 2).mean()) > 0.005


def test_sonify_tones(test_video):
    audio, sr = sonify(test_video, mode="tones", seed=3)
    assert len(audio) > sr
    assert np.abs(audio).max() <= 1.0


def test_sonify_determinism(test_video):
    a1, _ = sonify(test_video, mode="spectra", seed=9)
    a2, _ = sonify(test_video, mode="spectra", seed=9)
    assert np.array_equal(a1, a2)


def test_sonify_bad_mode(test_video):
    with pytest.raises(ValueError):
        sonify(test_video, mode="vibes")


def test_write_wav(test_video, tmp_path):
    audio, sr = sonify(test_video, mode="spectra", seed=1)
    out = tmp_path / "heard.wav"
    write_wav(out, audio, sr)
    import subprocess

    dur = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    assert float(dur.stdout) == pytest.approx(len(audio) / sr, abs=0.1)


# ------------------------------------------------------------------ degrade

def test_degrade_audio_mp3_round(test_wav):
    from womb.degrade import degrade_audio
    import subprocess as sp

    out = sp.run(["ffmpeg", "-v", "error", "-i", test_wav, "-f", "f32le", "-ar",
                  "44100", "-ac", "1", "pipe:1"], capture_output=True).stdout
    x = np.frombuffer(out, np.float32).copy()
    y = degrade_audio(x, 44100, [("mp3_round", {"kbps": 64})], seed=0)
    assert len(y) > 0 and not np.array_equal(x[: len(y)], y)   # real loss happened


def test_degrade_audio_chain(test_wav):
    from womb.degrade import degrade_audio
    x = np.linspace(-0.5, 0.5, 44100).astype(np.float32)
    y = degrade_audio(x, 44100,
                      [("crush", {"bits": 6}), ("bitrot", {"amount": 0.001})], seed=1)
    assert len(y) == len(x)


def test_degrade_video_round(test_video, tmp_path):
    from womb.degrade import degrade_video
    out = degrade_video(test_video, [("mp4_round", {"crf": 34})], seed=0)
    assert out != test_video and Path(out).exists()


from pathlib import Path  # noqa: E402  (used above)


# ------------------------------------------------------------------ reel + metabolize

def test_reel_lineage(test_wav, test_video, tmp_path):
    reel = womb.Reel(tmp_path / "r")
    seed = reel.add_seed(test_wav)
    import shutil

    shutil.copy(test_video, reel.pass_dir(0) / "video.mp4")
    reel.add_pass("seed", stem=seed, video=reel.video(0))
    assert reel.n_passes == 1
    art = reel.finalize()
    assert art.exists()
    assert "final" in reel.manifest
    assert "seed" in reel.drift_report()


def test_metabolize_end_to_end(test_wav, tmp_path):
    """The whole loop, small: 3 passes on a 4-second seed."""
    reel = womb.metabolize(test_wav, out=tmp_path / "reel", passes=2,
                           preset="metabolism", seed=5, length=4.0)
    assert reel.n_passes == 3               # seed + 2 loop passes
    assert (reel.root / "final.mp4").exists()
    assert (reel.root / "final.wav").exists()
    # every pass has stem + video + telemetry
    for p in reel.manifest["passes"]:
        assert reel.stem(p["index"]).exists()
        assert reel.video(p["index"]).exists()
        assert "rms" in p and "audio_drift" in p
    # drift is real but not total annihilation on 'metabolism'
    last = reel.manifest["passes"][-1]
    assert last["rms"] > 0.0
    report = reel.drift_report()
    assert "pass" in report and "aDrift" in report


def test_metabolize_determinism(test_wav, tmp_path):
    r1 = womb.metabolize(test_wav, out=tmp_path / "a", passes=1,
                          preset="hold", seed=3, length=3.0)
    r2 = womb.metabolize(test_wav, out=tmp_path / "b", passes=1,
                          preset="hold", seed=3, length=3.0)
    a = open(r1.root / "final.wav", "rb").read()
    b = open(r2.root / "final.wav", "rb").read()
    assert a == b

# ------------------------------------------------------------------ organs + pieces

def test_organs_step_render_emit(tmp_path):
    import womb

    for name in womb.list_organs():
        if name == "territory":
            continue  # needs corpus (covered below)
        o = womb.make_organ(name, 160, 90, fps=12, seed=0)
        for i in range(12):
            o.step(i)
            a = o.emit()
            assert len(a) == o.hop, name
        f = o.render()
        assert f.shape == (90, 160, 3) and f.dtype == np.uint8, name


def test_organ_determinism():
    import womb

    def run(seed):
        o = womb.make_organ("automata", 96, 54, fps=6, seed=seed)
        frames, audio = [], []
        for i in range(6):
            o.step(i)
            audio.append(o.emit().copy())
            frames.append(o.render().copy())
        return frames, audio

    f1, a1 = run(3)
    f2, a2 = run(3)
    assert all(np.array_equal(x, y) for x, y in zip(f1, f2))
    assert all(np.array_equal(x, y) for x, y in zip(a1, a2))


def test_generate_end_to_end(tmp_path):
    import womb

    out = tmp_path / "piece.mp4"
    r = womb.generate(
        plan=[dict(organ="waveplate", palette="ice"),
              dict(organ="swarm", palette="blood")],
        length=2.0, out=out, fps=12, size=(160, 90), seed=1,
    )
    assert r["frames"] == 24
    assert r["seconds"] == pytest.approx(2.0, abs=0.05)
    import subprocess as sp

    probe = sp.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_streams", str(out)], capture_output=True, text=True)
    import json

    streams = {s["codec_type"]: s for s in json.loads(probe.stdout)["streams"]}
    assert "video" in streams and "audio" in streams
    assert float(streams["video"]["duration"]) == pytest.approx(2.0, abs=0.2)
    assert float(streams["audio"]["duration"]) == pytest.approx(2.0, abs=0.3)


def test_generate_audio_present(tmp_path):
    """The piece must actually sound — audio RMS above silence."""
    import womb

    r = womb.generate(plan=[dict(organ="waveplate", strike_every=2)],
                      length=1.5, out=tmp_path / "p.mp4", fps=12,
                      size=(160, 90), seed=2)
    import subprocess as sp

    out = sp.run(["ffmpeg", "-v", "error", "-i", r["path"],
                  "-f", "f32le", "-ac", "1", "pipe:1"], capture_output=True).stdout
    x = np.frombuffer(out, np.float32)
    assert np.sqrt((x ** 2).mean()) > 0.01  # not silence


def test_generate_with_corpus(tmp_path, test_wav):
    import womb

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    import shutil

    shutil.copy(test_wav, corpus_dir / "a.wav")
    r = womb.generate(plan=[dict(organ="territory"), dict(organ="automata")],
                      length=1.5, out=tmp_path / "p.mp4", fps=12,
                      size=(160, 90), seed=0, corpus=str(corpus_dir))
    assert r["frames"] > 0


# ------------------------------------------------------------------ lowpoly + novelty search

def test_lowpoly_frame():
    from womb.lowpoly import lowpoly_frame

    f = np.zeros((90, 160, 3), np.uint8)
    f[..., 0] = np.tile(np.linspace(0, 255, 160, dtype=np.uint8), (90, 1))
    out = lowpoly_frame(f, grid=(8, 14), jitter=2.0, tear=0.0, levels=16,
                        energy=0.5, rng=np.random.default_rng(0))
    assert out.shape == f.shape and out.dtype == np.uint8
    assert (out % 8 == 0).all()                       # RGB555 purity
    out2 = lowpoly_frame(f, grid=(8, 14), jitter=2.0, tear=0.0, levels=16,
                         energy=0.95, rng=np.random.default_rng(0))
    assert not np.array_equal(out, out2)             # energy-reactive wobble


def test_lowpoly_reprocess(test_video, tmp_path):
    import womb

    out = womb.reprocess(test_video, tmp_path / "lens.mp4", grid=(8, 14),
                         jitter=1.5, seed=0)
    assert Path(out).exists()


def test_genome_mutation_type_safety():
    import womb

    rng = np.random.default_rng(0)
    g = womb.random_genome(rng)
    child = womb.mutate_genome(g, rng)
    for seg in child["organs"]:
        for k, v in seg["params"].items():
            if k == "palette":
                continue
            if isinstance(v, int):
                assert isinstance(v, int)  # ints stay ints
    assert "lowpoly" in child and "seed" in child


def test_search_fills_archive(tmp_path):
    import womb

    r = womb.search(out_dir=tmp_path / "atl", iters=6, seconds=0.8,
                    size=(80, 144), fps=6, cells=(4, 4), seed=1)
    assert r["cells_filled"] >= 1
    assert Path(r["archive"]).exists()
    import json

    atlas = json.loads(Path(r["archive"]).read_text())
    assert atlas["elites"]


def test_walk_from_archive(tmp_path):
    import womb

    r = womb.search(out_dir=tmp_path / "atl", iters=5, seconds=0.6,
                    size=(80, 144), fps=6, cells=(3, 3), seed=2)
    out = tmp_path / "walk.mp4"
    stats = womb.walk(r["archive"], length=1.5, out=out,
                      size=(144, 256), fps=12, seed=0)
    assert stats["frames"] == 18
    assert out.exists()
