import numpy as np

from loom.chunks import ChunkRef
from loom.chunkers import fixed, list_chunkers, make_chunker, onset, random
from loom.rng import Seeded
from loom.sources import open_sources


def test_fixed_chunker(corpus_dir):
    srcs = open_sources(corpus_dir)
    refs = fixed(srcs, Seeded(0), seconds=1.0)
    assert len(refs) >= 8
    assert all(isinstance(r, ChunkRef) for r in refs)
    # windows tile each source
    durs = {}
    for r in refs:
        durs.setdefault(r.src, []).append((r.start, r.dur))
    for si, spans in durs.items():
        assert spans[0][0] == 0.0  # starts at zero
        assert max(d for _, d in spans) <= 1.0 + 1e-6


def test_fixed_overlap_doubles(corpus_dir):
    srcs = open_sources(corpus_dir)
    a = len(fixed(srcs, Seeded(0), seconds=1.0))
    b = len(fixed(srcs, Seeded(0), seconds=1.0, overlap=0.5))
    assert b > a


def test_random_chunker(corpus_dir):
    srcs = open_sources(corpus_dir)
    refs = random(srcs, Seeded(1), n=10, min_seconds=0.2, max_seconds=0.5)
    assert len(refs) == 10 * len(srcs)
    assert all(r.dur <= 0.5 + 1e-6 for r in refs)
    assert all(r.start + r.dur <= srcs[r.src].duration + 1e-6 for r in refs)


def test_onset_chunker_finds_clicks(corpus_wav):
    """A click train every 0.7 s should segment into ~57 onset chunks."""
    srcs = open_sources(corpus_wav)
    refs = onset(srcs, Seeded(0), min_seconds=0.1, max_seconds=4.0)
    assert len(refs) > 10
    durs = np.array([r.dur for r in refs])
    assert durs.max() <= 4.0 + 1e-6
    # chunks tile the file without big holes
    total = sum(durs)
    assert total > srcs.total_duration * 0.9


def test_onset_chunker_flat_fallback(corpus_dir):
    """A pure sine has no onsets -> fixed fallback windows."""
    srcs = open_sources([corpus_dir / "sine.wav"])
    refs = onset(srcs, Seeded(0), fallback_seconds=0.3)
    assert len(refs) == 4  # 1.0 s -> 0.3 + 0.3 + 0.3 + 0.1 tail
    durs = [r.dur for r in refs]
    assert all(abs(d - 0.3) < 0.05 for d in durs[:-1])
    assert 0.02 < durs[-1] <= 0.3 + 1e-9


def test_make_chunker_registry():
    assert "fixed" in list_chunkers()
    fn = make_chunker("onset")
    assert callable(fn)
    try:
        make_chunker("nope")
        assert False, "should raise"
    except ValueError:
        pass


def test_beats_chunker(corpus_dir):
    from loom.chunkers import beats

    srcs = open_sources(corpus_dir)
    refs = beats(srcs, Seeded(0), bpm=120, bars=1)
    # 4 beats at 120bpm = 2 s windows; the 6 s clicks source (index 1) gets 3
    full = [r for r in refs if r.src == 1]
    assert len(full) == 3
    assert all(abs(r.dur - 2.0) < 0.02 for r in full)