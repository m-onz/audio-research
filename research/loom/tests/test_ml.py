import numpy as np

from loom import compose
from loom.ml import CodebookProcessor, fit_codebook, register_processor_instance
from loom.processors import build_chain

class _R:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def ints(self, a, b=None):
        return int(self.rng.integers(a, b))


def _audio_list(corpus_dir, n=3):
    from loom.io import read_wav
    from pathlib import Path

    wavs = sorted(Path(corpus_dir).glob("*.wav"))[:n]
    return [read_wav(w)[1] for w in wavs]


def test_fit_codebook_numpy_kmeans(corpus_dir):
    sr = 22050
    audios = _audio_list(corpus_dir)
    cb = fit_codebook(audios, sr, k=8, max_frames=200)
    assert cb.shape == (8, 1025)  # n_fft=2048 -> 1025 bins


def test_codebook_processor_shapes(corpus_dir):
    sr = 22050
    proc = CodebookProcessor(n_fft=1024).fit(_audio_list(corpus_dir), sr, k=6, max_frames=100)
    t = np.arange(sr) / sr
    x = np.stack([0.3 * np.sin(2 * np.pi * 500 * t)] * 2, axis=1).astype(np.float32)
    out = proc.process(x, sr, 0.0, _R())
    assert out.shape == x.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) > 0  # not silenced


def test_codebook_in_chain(corpus_dir, tmp_path):
    proc = CodebookProcessor(n_fft=1024).fit(_audio_list(corpus_dir), 22050, k=6, max_frames=100)
    chain = build_chain([proc, "fades"])
    t = np.arange(22050) / 22050
    x = np.stack([0.3 * np.sin(2 * np.pi * 500 * t)] * 2, axis=1).astype(np.float32)
    out = chain.run(x, 22050, 0.0, _R())
    assert np.isfinite(out).all()


def test_register_processor_instance(corpus_dir):
    proc = CodebookProcessor(n_fft=1024).fit(_audio_list(corpus_dir), 22050, k=4, max_frames=50)
    register_processor_instance("my_codebook", proc)
    chain = build_chain("my_codebook")
    assert chain.steps[0][0] is proc


def test_compose_with_learned_processor(corpus_dir, tmp_path):
    out = tmp_path / "learned.wav"
    proc = CodebookProcessor(n_fft=1024).fit(_audio_list(corpus_dir), 22050, k=6, max_frames=100)
    compose(
        sources=corpus_dir,
        length=4.0,
        seed=11,
        chunker="fixed",
        chunker_params={"seconds": 0.5},
        arranger="sequence",
        processors=[proc, "fades"],
        out=out,
    )
    from loom.io import wav_info

    assert abs(wav_info(out)["seconds"] - 4.0) < 0.02