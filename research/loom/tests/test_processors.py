import numpy as np
import pytest

from loom.curves import Curve
from loom.processors import (
    Processor,
    build_chain,
    list_processors,
    register_processor,
)
SR = 22050


class _R:
    """Shim matching the Seeded interface processors use."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def ints(self, a, b=None):
        return int(self.rng.integers(a, b))

    def float(self, a=0.0, b=1.0):
        return float(self.rng.uniform(a, b))


def _audio(seconds=1.0, f=220.0):
    t = np.arange(int(SR * seconds)) / SR
    x = 0.5 * np.sin(2 * np.pi * f * t)
    return np.stack([x, x], axis=1).astype(np.float32)


def _run_all(name, params=None):
    """Every registered processor must: accept audio, return finite audio."""
    from loom.processors import PROCESSORS

    cls = PROCESSORS[name]
    proc = cls() if isinstance(cls, type) else cls
    out = proc.process(_audio(), SR, 0.5, _R(), **(params or {}))
    assert out.ndim == 2 and out.shape[1] == 2
    assert np.isfinite(out).all(), f"{name} produced non-finite audio"
    assert out.shape[0] > 0
    return out


def test_all_processors_run():
    for name in list_processors():
        _run_all(name)


def test_speed_changes_length():
    out = _run_all("speed", {"x": 2.0})
    assert out.shape[0] == pytest.approx(SR / 2, abs=2)


def test_reverse():
    x = _audio()
    out = _run_all("reverse")
    assert np.allclose(out[::-1], x, atol=1e-6)


def test_lowpass_kills_highs():
    out = _run_all("lowpass", {"cut_hz": 200.0})
    # 220 Hz sine should be heavily attenuated by a 200 Hz lowpass
    assert np.max(np.abs(out)) < 0.1 * np.max(np.abs(_audio()))


def test_gain_and_normalize():
    out = _run_all("gain", {"amount": 0.1})
    assert np.max(np.abs(out)) < 0.06
    out = _run_all("normalize")
    assert np.max(np.abs(out)) == pytest.approx(0.966, abs=0.01)  # -0.3 dB


def test_repeat_extends():
    out = _run_all("repeat", {"count": 3, "decay": 0.5})
    assert out.shape[0] == 3 * SR


def test_trim_and_gate():
    out = _run_all("trim", {"threshold_db": -30.0})
    assert out.shape[0] <= SR
    out = _run_all("gate", {"threshold_db": -10.0})
    assert np.max(np.abs(out)) <= np.max(np.abs(_audio())) + 1e-6


def test_chain_with_curves():
    chain = build_chain([["gain", {"amount": Curve([1.0, 0.0])}], "fades"])
    loud = chain.run(_audio(0.5), SR, 0.0, _R())
    quiet = chain.run(_audio(0.5), SR, 1.0, _R())
    assert np.max(np.abs(loud)) > np.max(np.abs(quiet))


def test_build_chain_specs():
    c1 = build_chain("lowpass")
    assert len(c1.steps) == 1
    c2 = build_chain(["lowpass", ["echo", {"delay_s": 0.1}]])
    assert len(c2.steps) == 2
    c3 = build_chain({"lowpass": {"cut_hz": 1000}})
    assert c3.steps[0][1]["cut_hz"] == 1000
    c4 = build_chain(None)
    assert len(c4.steps) == 0
    c5 = build_chain("crush(bits=3)")
    assert len(c5.steps) == 1


def test_chain_inline_kwargs():
    c = build_chain("lowpass(cut_hz=500)")
    assert c.steps[0][1] == {"cut_hz": 500}


def test_unknown_processor_raises():
    with pytest.raises(ValueError):
        build_chain("wobble")


def test_custom_processor_registration():
    @register_processor
    class Swap(Processor):
        name = "swap_test_proc"

        def process(self, audio, sr, t, rng, **kw):
            out = audio.copy()
            out[:, 0], out[:, 1] = audio[:, 1], audio[:, 0]
            return out

    c = build_chain("swap_test_proc")
    x = np.zeros((10, 2), np.float32)
    x[0, 0] = 1.0
    out = c.run(x, SR, 0, _R())
    assert out[0, 1] == 1.0


def test_finite_guard():
    bad = np.full((100, 2), np.nan, np.float32)
    chain = build_chain("gain")
    out = chain.run(bad, SR, 0.0, _R())
    assert np.isfinite(out).all()

# ---- new processors: stretch / chop / stutter / morph ------------------------

def test_stretch_lengthens_preserves_pitch():
    from loom.processors import Stretch

    sr = SR
    t = np.arange(sr) / sr
    x = np.stack([0.5 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    out = Stretch().process(x, sr, 0.0, _R(), factor=2.0, grain_s=0.05)
    assert out.shape[0] == pytest.approx(2 * sr, abs=2000)  # 2x longer
    # pitch preserved: dominant frequency still ~220 Hz
    m = out[: sr * 2 // 2, 0]
    S = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    fr = np.fft.rfftfreq(len(m), 1 / sr)
    assert abs(fr[S.argmax()] - 220) < 15


def test_stretch_compresses():
    from loom.processors import Stretch

    out = Stretch().process(_audio(1.0), SR, 0.0, _R(), factor=0.5, grain_s=0.05)
    assert out.shape[0] == pytest.approx(SR / 2, abs=2000)


def test_chop_mutes_steps():
    from loom.processors import Chop

    x = np.ones((SR, 2), np.float32)  # constant DC-ish chunk
    out = Chop().process(x, SR, 0.0, _R(), steps=4, pulses=2, rotate=0, fade_s=0.001)
    quarter = SR // 4
    # euclid 2-of-4 -> pattern [1, 0, 1, 0]: steps 0 and 2 kept, 1 and 3 muted
    assert np.abs(out[: quarter - 1000]).max() > 0.5
    assert np.abs(out[quarter + 1000 : 2 * quarter - 1000]).max() == 0.0
    assert np.abs(out[2 * quarter + 1000 : 3 * quarter - 1000]).max() > 0.5
    assert np.abs(out[3 * quarter + 1000 :]).max() == 0.0


def test_stutter_retriggers_with_pitch():
    from loom.processors import Stutter

    sr = SR
    t = np.arange(sr) / sr
    x = np.stack([0.5 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    out = Stutter().process(x, sr, 0.0, _R(), count=4, step_semi=3.0,
                            direction="down", slice_s=0.1, decay=0.9)
    slice_n = int(0.1 * sr)
    assert out.shape[0] == 4 * slice_n
    # first slice should be ~440 Hz, later slices lower
    def domFreq(seg):
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        fr = np.fft.rfftfreq(len(seg), 1 / sr)
        return fr[S.argmax()]

    f0 = domFreq(out[0:slice_n, 0])
    f1 = domFreq(out[2 * slice_n : 3 * slice_n, 0])
    assert abs(f0 - 440) < 30
    assert f1 < f0  # pitched down progressively


def test_stutter_random_direction_deterministic():
    from loom.processors import Stutter

    x = _audio(0.5, f=330)
    a = Stutter().process(x, SR, 0.0, _R(7), count=3, direction="random")
    b = Stutter().process(x, SR, 0.0, _R(7), count=3, direction="random")
    assert np.array_equal(a, b)


def test_morph_applies_one_effect_per_event():
    from loom.processors import build_chain

    chain = build_chain([["morph", {"processors": [
        ["gain", {"amount": 0.1}],
        ["gain", {"amount": 0.9}],
    ]}], "fades"])
    x = _audio(0.5)
    outs = [np.max(np.abs(chain.run(x, SR, 0.0, _R(s)))) for s in range(8)]
    lo = [o for o in outs if o < 0.3]
    hi = [o for o in outs if o > 0.3]
    assert lo and hi, "morph should pick different effects across events"
    # determinism: same seed -> same effect
    assert np.max(np.abs(chain.run(x, SR, 0.0, _R(3)))) == \
           np.max(np.abs(chain.run(x, SR, 0.0, _R(3))))


def test_morph_weights():
    from loom.processors import build_chain

    chain = build_chain([["morph", {"processors": [
        ["gain", {"amount": 0.1}],
        ["gain", {"amount": 0.9}],
    ], "weights": [0.99, 0.01]}]])
    x = _audio(0.5)
    outs = [np.max(np.abs(chain.run(x, SR, 0.0, _R(s)))) for s in range(6)]
    assert sum(o < 0.3 for o in outs) >= 5  # heavily biased to the first


def test_stretch_no_edge_blowup():
    """Regression: OLA edge regions must not explode (single-grain coverage)."""
    from loom.processors import Stretch

    sr = SR
    rng = np.random.default_rng(3)
    # long, hot, noisy-sine chunk — the shape that found the bug
    t = np.arange(2 * sr) / sr
    x = np.stack([0.9 * np.sin(2 * np.pi * 173 * t) + 0.05 * rng.standard_normal(2 * sr)] * 2,
                 axis=1).astype(np.float32)
    for factor in (3.0, 0.6, 1.7):
        out = Stretch().process(x, sr, 0.0, _R(), factor=factor, grain_s=0.05)
        assert np.isfinite(out).all()
        assert np.abs(out).max() < 2.0 * np.abs(x).max() + 0.1, f"factor {factor} blew up"
