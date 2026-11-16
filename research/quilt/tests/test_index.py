"""the cutting room is gone: pile -> bank, with provenance and determinism."""

import numpy as np
import pytest

from quilt.index import build_bank, provenance_path, iter_audio
from quilt.chunking import MIN_DUR
from quilt.naming import parse_chunk_name
from echo import DIM


def test_bank_builds_and_fits(bank_path, pile_dir):
    from echo import EchoBank
    bank = EchoBank(str(bank_path))
    assert len(bank) > 10
    assert bank.metric.dim == DIM
    # the fit happened: sigma is no longer the identity
    assert not np.allclose(bank.metric.sigma, 1.0)
    # every entry is one pile file, pinned with its name's provenance
    for m in bank.metas:
        assert set(m) >= {"file", "t0", "t1", "dur", "strategy",
                          "orig", "idx"}
        assert m["t0"] == 0.0                       # a piece is the whole file
        assert m["dur"] >= MIN_DUR - 1e-6
        strat, orig, idx = parse_chunk_name(m["file"])
        assert m["strategy"] == strat and m["orig"] == orig \
            and m["idx"] == idx                    # names are truth
    assert all(r == "corpus" for r in bank.roles)


def test_provenance_sidecar(bank_path, pile_dir):
    import json
    prov = json.loads(provenance_path(bank_path).read_text())
    assert prov["corpus_root"] == str(pile_dir)
    assert prov["mode"] == "pile"
    assert prov["chunks"] > 10
    assert "onset" in prov["strategies"] and "fixed" in prov["strategies"]


def test_strategies_of_the_pile(bank_path):
    from echo import EchoBank
    bank = EchoBank(str(bank_path))
    strats = {m["strategy"] for m in bank.metas}
    assert {"onset", "fixed", "whole"} <= strats    # hits, drone, sines
    assert strats <= {"fixed", "onset", "silence", "whole"}


def test_refuses_to_clobber(bank_path, pile_dir):
    with pytest.raises(FileExistsError):
        build_bank(pile_dir, bank_path, log=lambda s: None)
    # ...and force replaces it with an identical bank (determinism)
    from echo import EchoBank
    before = EchoBank(str(bank_path)).matrix()
    build_bank(pile_dir, bank_path, force=True, log=lambda s: None)
    after = EchoBank(str(bank_path)).matrix()
    assert before.shape == after.shape
    assert np.allclose(before, after)


def test_iter_audio_sorted_and_chunk_aware(pile_dir):
    files = iter_audio(pile_dir)
    assert files == sorted(files)
    assert any(is_chunk_name(f.name) for f in files)
    assert all(f.suffix == ".wav" for f in files)


def is_chunk_name(name):
    from quilt.split import is_chunk_file
    return is_chunk_file(name)