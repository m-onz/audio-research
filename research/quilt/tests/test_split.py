"""the in-place processor: known-answer splits, idempotency, safety."""

import numpy as np
import pytest
import soundfile as sf

from quilt.split import split_corpus, is_chunk_file
from quilt.naming import parse_chunk_name, CHUNK_RE
from quilt.io import write_audio
from echo.probes import SR, write_wav


def _sine(secs, freq=440.0, amp=0.5):
    t = np.arange(int(secs * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x], axis=1)


def _click_train(secs, period=0.5):
    x = np.zeros(int(secs * SR))
    for k in range(int(secs / period)):
        i = int(k * period * SR)
        x[i:i + 400] += np.linspace(1, -1, 400)
    return np.stack([x, x], axis=1)


def _bursts(secs=3.0):
    n = int(secs * SR)
    x = np.zeros(n)
    x[:SR] = 0.5 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    x[2 * SR:] = 0.4 * np.sin(2 * np.pi * 800 * np.arange(n - 2 * SR) / SR)
    return np.stack([x, x], axis=1)


@pytest.fixture()
def fresh_corpus(tmp_path):
    return tmp_path


class TestNaming:
    def test_marker_detection(self):
        assert is_chunk_file("x__onset_0000.wav")
        assert is_chunk_file("x__silence_0012.wav")
        assert is_chunk_file("y__fixed_0003.wav")
        assert not is_chunk_file("x.wav")
        assert not is_chunk_file("alien_rhythms_py_138.wav")  # corpus-style

    def test_parse_round_trip(self):
        s, orig, idx = parse_chunk_name("a__b__onset_0007.wav")
        assert (s, orig, idx) == ("onset", "a__b", 7)
        s, orig, idx = parse_chunk_name("plain.wav")
        assert (s, orig, idx) == ("whole", "plain", -1)


class TestSplits:
    def test_click_train_becomes_onset_chunks(self, fresh_corpus):
        write_wav(str(fresh_corpus / "hits.wav"), _click_train(3.0), SR)
        split_corpus(fresh_corpus, audit_path=str(fresh_corpus / "a.jsonl"),
                     log=lambda s: None)
        chunks = sorted(fresh_corpus.glob("hits__onset_*.wav"))
        assert not (fresh_corpus / "hits.wav").exists()   # replaced
        assert 4 <= len(chunks) <= 8
        assert sum(sf.info(str(c)).frames for c in chunks) \
            <= 3 * SR + len(chunks) * 2                   # nothing duplicated

    def test_long_drone_gets_fixed_windows(self, fresh_corpus):
        write_wav(str(fresh_corpus / "drone.wav"), _sine(9.0), SR)
        split_corpus(fresh_corpus, audit_path=str(fresh_corpus / "a.jsonl"),
                     log=lambda s: None)
        chunks = sorted(fresh_corpus.glob("drone__fixed_*.wav"))
        assert len(chunks) == 5                            # 2+2+2+2+1
        assert not (fresh_corpus / "drone.wav").exists()

    def test_short_file_kept_whole_unrenamed(self, fresh_corpus):
        write_wav(str(fresh_corpus / "one_shot.wav"), _sine(1.0), SR)
        split_corpus(fresh_corpus, audit_path=str(fresh_corpus / "a.jsonl"),
                     log=lambda s: None)
        assert (fresh_corpus / "one_shot.wav").exists()     # untouched
        assert list(fresh_corpus.glob("one_shot__*")) == []

    def test_dust_kept(self, fresh_corpus):
        write_wav(str(fresh_corpus / "dust.wav"), _sine(0.05), SR)
        split_corpus(fresh_corpus, audit_path=str(fresh_corpus / "a.jsonl"),
                     log=lambda s: None)
        assert (fresh_corpus / "dust.wav").exists()

    def test_bursts_split_at_the_gap(self, fresh_corpus):
        write_wav(str(fresh_corpus / "bursts.wav"), _bursts(), SR)
        split_corpus(fresh_corpus, audit_path=str(fresh_corpus / "a.jsonl"),
                     log=lambda s: None)
        on = list(fresh_corpus.glob("bursts__onset_*.wav"))
        si = list(fresh_corpus.glob("bursts__silence_*.wav"))
        assert 2 <= len(on) + len(si) <= 4          # the events separated
        assert not (fresh_corpus / "bursts.wav").exists()

    def test_size_not_duplicated(self, fresh_corpus):
        """Splitting, never duplicating: the pile holds ~the original's
        frames (within scissor dust and per-file headers)."""
        write_wav(str(fresh_corpus / "hits.wav"), _click_train(2.0), SR)
        before = sum(f.stat().st_size for f in fresh_corpus.glob("*.wav"))
        split_corpus(fresh_corpus, audit_path=str(fresh_corpus / "a.jsonl"),
                     log=lambda s: None)
        after = sum(f.stat().st_size for f in fresh_corpus.glob("*.wav"))
        assert after <= before * 1.05


class TestSafety:
    def test_dry_run_touches_nothing(self, fresh_corpus):
        write_wav(str(fresh_corpus / "hits.wav"), _click_train(2.0), SR)
        counts = split_corpus(fresh_corpus, dry_run=True,
                              audit_path=str(fresh_corpus / "a.jsonl"),
                              log=lambda s: None)
        assert counts["split"] >= 1
        assert (fresh_corpus / "hits.wav").exists()
        assert list(fresh_corpus.glob("hits__*")) == []

    def test_idempotent(self, fresh_corpus):
        write_wav(str(fresh_corpus / "hits.wav"), _click_train(2.0), SR)
        split_corpus(fresh_corpus, audit_path=str(fresh_corpus / "a.jsonl"),
                     log=lambda s: None)
        once = sorted(f.name for f in fresh_corpus.glob("*.wav"))
        split_corpus(fresh_corpus, audit_path=str(fresh_corpus / "a.jsonl"),
                     log=lambda s: None)
        twice = sorted(f.name for f in fresh_corpus.glob("*.wav"))
        assert once == twice                                # stable pile

    def test_audit_written(self, fresh_corpus):
        import json
        write_wav(str(fresh_corpus / "hits.wav"), _click_train(2.0), SR)
        audit = fresh_corpus / "a.jsonl"
        split_corpus(fresh_corpus, audit_path=str(audit), log=lambda s: None)
        lines = [json.loads(x) for x in audit.read_text().splitlines()]
        assert any(l["action"] == "split" for l in lines)
        sp = next(l for l in lines if l["action"] == "split")
        assert sp["strategy"] in ("onset", "silence", "fixed")
        assert len(sp["chunk_durs"]) >= 2

    def test_unreadable_original_kept(self, fresh_corpus):
        (fresh_corpus / "broken.wav").write_bytes(b"not a wav at all")
        split_corpus(fresh_corpus, audit_path=str(fresh_corpus / "a.jsonl"),
                     log=lambda s: None)
        assert (fresh_corpus / "broken.wav").exists()       # kept, logged