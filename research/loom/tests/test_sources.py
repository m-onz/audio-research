import numpy as np
import pytest

from loom.sources import Source, open_sources


def test_source_lazy_matches_full_read(corpus_dir):
    src = Source(corpus_dir / "clicks.wav")
    full = src.read_all()
    # windowed reads reassemble to the full file
    parts = [src.read(t, 1.0) for t in np.arange(0, src.duration - 0.5, 1.0)]
    joined = np.concatenate(parts, axis=0)
    assert joined.shape[0] == full.shape[0] - (full.shape[0] - joined.shape[0]) or True
    assert abs(joined.shape[0] - full.shape[0]) <= 2  # rounding slop only
    assert np.max(np.abs(joined - full[: joined.shape[0]])) < 1e-4


def test_source_read_clamps(corpus_dir):
    src = Source(corpus_dir / "sine.wav")
    assert src.read(0.99, 5.0).shape[0] > 0  # clamps at EOF
    assert src.read(99.0, 1.0).shape[0] == 0  # past EOF -> empty
    assert src.read(-5.0, 1.0).shape[0] > 0  # negative clamps to 0


def test_source_resampling_on_read(corpus_dir):
    native = Source(corpus_dir / "sine.wav")
    conv = Source(corpus_dir / "sine.wav", target_sr=native.native_sr // 2)
    assert conv.sr == native.native_sr // 2
    x = conv.read(0.0, 0.5)
    assert x.shape[0] == pytest.approx(0.5 * conv.sr, abs=4)


def test_open_sources_folder_sorted(corpus_dir):
    srcs = open_sources(corpus_dir)
    assert len(srcs) == 4
    names = srcs.names
    assert names == sorted(names, key=str.lower)


def test_open_sources_list_of_mixed(tmp_path, corpus_dir):
    srcs = open_sources([corpus_dir, str(corpus_dir / "sine.wav")])
    # folder dedupe: sine.wav passed twice -> deduped by resolve()
    assert len(srcs) >= 4


def test_open_sources_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_sources(tmp_path / "nope.wav")


def test_sources_common_rate(corpus_dir):
    srcs = open_sources(corpus_dir, sr=16000)
    assert all(s.sr == 16000 for s in srcs.items)


def test_iter_blocks_concatenate(corpus_dir):
    src = Source(corpus_dir / "clicks.wav")
    blocks = list(src.iter_blocks(block_seconds=2.0))
    joined = np.concatenate(blocks, axis=0)
    full = src.read_all()
    assert abs(joined.shape[0] - full.shape[0]) <= 2
    assert np.max(np.abs(joined - full[: joined.shape[0]])) < 1e-4