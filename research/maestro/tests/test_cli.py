"""cli tests — the commands run clean end to end."""

import numpy as np

from echo.probes import SR, decode, write_wav

from maestro.cli import main

TICK_HZ = 8.0


def click_spine(onsets_s, seconds=3.5, sr=SR):
    x = np.zeros(int(seconds * sr), np.float32)
    for t in onsets_s:
        p0, klen = int(t * sr), int(0.02 * sr)
        x[p0: p0 + klen] += np.exp(-np.linspace(0, 6, klen)) * 0.8
    return x


def test_cli_score_preview_info_champions(tmp_path):
    spine = tmp_path / "spine.wav"
    write_wav(spine, click_spine([0.5, 1.5, 2.5]), SR)

    out = tmp_path / "score.json"
    assert main(["score", "--spine", str(spine), "--champion", "follower",
                 "--tick-hz", str(TICK_HZ), "-o", str(out)]) == 0
    assert out.exists()
    assert out.with_name("score_preview.wav").exists()
    x = decode(out.with_name("score_preview.wav"), sr=SR)
    assert x.max() > 0.05

    assert main(["preview", str(out), "--spine-gain", "0.3",
                 "-o", str(tmp_path / "pv.wav")]) == 0
    assert (tmp_path / "pv.wav").exists()

    assert main(["info", str(out)]) == 0
    assert main(["champions"]) == 0


def test_cli_random_founder(tmp_path):
    spine = tmp_path / "spine.wav"
    write_wav(spine, click_spine([0.5, 1.5, 2.5]), SR)
    out = tmp_path / "r.json"
    assert main(["score", "--spine", str(spine), "--random",
                 "--random-seed", "7", "--tick-hz", str(TICK_HZ),
                 "-o", str(out)]) == 0
    import json
    sc = json.loads(out.read_text())
    assert set(sc["genome"]["outputs"]) == {"pulse", "flow"}