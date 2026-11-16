"""validate: the at-scale pipeline on a tiny synthetic corpus."""

import json

import numpy as np
import pytest

from echo.probes import SR, contrasts, write_wav
from echo.validate import validate_corpus


def _tiny_corpus(tmp_path):
    """Two families of known character: tonal archetypes vs noise
    archetypes — the validation pipeline's ground truth in miniature."""
    d = tmp_path / "corpus"
    d.mkdir()
    syn = contrasts(SR, seconds=2.0)
    fam = {}
    for name, x in syn.items():
        tonal = name.startswith(("sine", "fm", "rumble"))
        fam[name + ".wav"] = "tonal" if tonal else "noise"
        write_wav(d / f"{name}.wav", np.stack([x, x], axis=1), SR)
    (d / "manifest.json").write_text(json.dumps(
        [{"file": k, "family": v} for k, v in fam.items()]))
    return d


def test_validate_smoke(tmp_path):
    pytest.importorskip("librosa")  # not needed, but keeps env assumptions equal
    d = _tiny_corpus(tmp_path)
    out = tmp_path / "out"
    res = validate_corpus(str(d), out_dir=str(out), n_probes=8, n_far=6,
                          n_mp3=0, verbose=False)
    assert res["index"]["vectors"] >= 9            # 9 files, none failed
    assert res["index"]["failures"] == []
    # family retrieval: tonal archetypes must recognize home
    assert res["family"]["n_queries"] == 9
    assert res["family"]["recall_at_1"] >= 0.5
    # near-dups (perturbations) must separate from cross-family far pairs
    assert res["discrimination"]["auc"] > 0.8
    assert 0.0 <= res["novelty"]["median"] < float("inf")
    assert len(res["spot_checks"]) >= 1
    assert (out / "validate_report.md").exists()
    assert (out / "results.json").exists()
    # the bank persisted
    assert (out / "bank.sqlite").exists()