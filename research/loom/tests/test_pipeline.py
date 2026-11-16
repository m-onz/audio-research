import hashlib

import numpy as np
import pytest

from loom import compose


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_compose_end_to_end(corpus_dir, tmp_path):
    out = tmp_path / "piece.wav"
    r = compose(
        sources=corpus_dir,
        length=8.0,
        seed=42,
        chunker="fixed",
        chunker_params={"seconds": 0.5},
        arranger="sequence",
        out=out,
        verbose=False,
    )
    assert out.exists()
    from loom.io import read_wav, wav_info

    info = wav_info(out)
    assert abs(info["seconds"] - 8.0) < 0.01
    sr, y = read_wav(out)
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 1.0
    manifest = out.with_suffix(".manifest.json")
    assert manifest.exists()
    import json

    m = json.loads(manifest.read_text())
    assert m["seed"] == 42 and m["sha256"] == _sha(out)
    assert r.summary()


def test_reproducibility_same_seed(corpus_dir, tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    kw = dict(sources=corpus_dir, length=6.0, seed=9, chunker="random",
              chunker_params={"n": 8, "min_seconds": 0.2, "max_seconds": 0.8})
    compose(**kw, arranger="cloud", arranger_params={"density": 200}, out=a)
    compose(**kw, arranger="cloud", arranger_params={"density": 200}, out=b)
    assert _sha(a) == _sha(b)


def test_different_seeds_differ(corpus_dir, tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    kw = dict(sources=corpus_dir, length=6.0, chunker="random",
              chunker_params={"n": 8, "min_seconds": 0.2, "max_seconds": 0.8},
              arranger="cloud", arranger_params={"density": 200})
    compose(**kw, seed=1, out=a)
    compose(**kw, seed=2, out=b)
    assert _sha(a) != _sha(b)


def test_single_big_file_lazy(big_wav, tmp_path):
    """The flagship case: compose from one long file, never loading it."""
    out = tmp_path / "big_piece.wav"
    r = compose(
        sources=big_wav,
        length=20.0,
        seed=3,
        chunker="onset",
        chunker_params={"min_seconds": 0.3, "max_seconds": 3.0},
        arranger="sequence",
        arranger_params={"order": "by:rms", "gap": -0.1},
        out=out,
    )
    from loom.io import wav_info

    assert abs(wav_info(out)["seconds"] - 20.0) < 0.01
    assert r.manifest["score"]["events"] > 5


def test_passes_derive_and_tag(corpus_dir, tmp_path):
    out = tmp_path / "ghost.wav"
    r = compose(
        sources=corpus_dir,
        length=6.0,
        seed=5,
        chunker="fixed",
        chunker_params={"seconds": 0.5},
        passes=[
            dict(select="rms>0.5", processors=[["lowpass", {"cut_hz": 800}], "reverse"], tag="dark"),
        ],
        select="tag=dark",
        arranger="sequence",
        out=out,
    )
    assert "dark" in r.manifest["catalog"]["tags"]
    assert r.catalog.query("tag=dark")


def test_preset_overrides(corpus_dir, tmp_path):
    out = tmp_path / "p.wav"
    r = compose(sources=corpus_dir, preset="drift", length=5.0, seed=1,
                arranger_params={"density": 300}, out=out)
    # explicit arranger_params merged with preset's cloud params
    ev = r.score.events
    assert len(ev) > 3
    assert abs(r.manifest["length"] - 5.0) < 1e-9


def test_unknown_preset_and_arg(corpus_dir):
    with pytest.raises(ValueError):
        compose(sources=corpus_dir, preset="nope")
    with pytest.raises(TypeError):
        compose(sources=corpus_dir, wobble=3)


def test_dry_run(corpus_dir):
    r = compose(sources=corpus_dir, length=5.0, dry_run=True, seed=0,
                chunker="fixed", chunker_params={"seconds": 0.5})
    assert r.out is None
    assert len(r.score.events) > 0
    assert r.score.n_samples > 0


def test_layers_arranger(corpus_dir, tmp_path):
    out = tmp_path / "layers.wav"
    r = compose(
        sources=corpus_dir,
        length=6.0,
        seed=2,
        chunker="fixed",
        chunker_params={"seconds": 0.4},
        arranger="layers",
        arranger_params={"voices": [
            {"select": "all", "arranger": "euclid", "params": {"bpm": 120, "steps": 8, "pulses": 3}},
            {"select": "all", "arranger": "cloud", "params": {"density": 100}, "gain": 0.4},
        ]},
        out=out,
    )
    assert any(e.note == "voice0" for e in r.score.events)
    assert any(e.note == "voice1" for e in r.score.events)


def test_score_json_roundtrip(corpus_dir, tmp_path):
    r = compose(sources=corpus_dir, length=4.0, seed=7, chunker="fixed",
                chunker_params={"seconds": 0.5}, dry_run=True)
    p = tmp_path / "score.json"
    r.score.write(p)
    from loom.score import Score

    s2 = Score.read(p)
    assert s2.summary() == r.score.summary()
    assert len(s2.events) == len(r.score.events)
    assert s2.events[0].chain is not None