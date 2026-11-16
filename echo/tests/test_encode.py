"""encode: the vector anatomy, determinism, and the temporal argument."""

import numpy as np

from echo.bank import EchoMetric
from echo.encode import DIM, LAYOUT, encode, encode_blocks


def _rng():
    return np.random.default_rng(42)


def test_dims_match_layout():
    x = _rng().normal(0, 0.3, (44100 * 2, 2)).astype(np.float32)
    b = encode_blocks(x)
    pos = 0
    for name, dim, _ in LAYOUT:
        assert len(b[name]) == dim, name
        pos += dim
    assert pos == DIM == 556
    assert encode(x).shape == (DIM,)


def test_deterministic():
    x = _rng().normal(0, 0.2, (44100, 2)).astype(np.float32)
    assert np.array_equal(encode(x), encode(x))


def test_finite_on_silence_and_nan():
    v = encode(np.zeros((44100, 2), np.float32))
    assert np.isfinite(v).all()
    bad = np.full((44100, 2), np.nan, np.float32)
    assert np.isfinite(encode(bad)).all()


def test_mono_equals_duplicated_stereo():
    x = _rng().normal(0, 0.2, 44100).astype(np.float32)
    v1 = encode(x)
    v2 = encode(np.stack([x, x], axis=1))
    assert np.allclose(v1, v2, atol=1e-9)


def test_short_clip_full_anatomy():
    v = encode(_rng().normal(0, 0.2, 2000).astype(np.float32))
    assert v.shape == (DIM,) and np.isfinite(v).all()
    v = encode(np.zeros(100, np.float32))
    assert v.shape == (DIM,) and np.isfinite(v).all()


def _order_swap_pair():
    """click-then-drone vs drone-then-click: same ingredients, opposite
    order — the whole reason the vector is chunked."""
    sr = 44100
    t = np.arange(sr * 3) / sr
    burst = _rng().normal(0, 0.5, int(sr * 0.5))
    drone = 0.3 * np.sin(2 * np.pi * 220 * t[int(sr * 0.5):])
    a = np.concatenate([burst, drone])
    b = np.concatenate([drone, burst])
    return a.astype(np.float32), b.astype(np.float32)


def test_order_matters_chunked():
    """The temporal claim: swapped order must land farther than a near
    duplicate of the same clip (gain jitter)."""
    a, b = _order_swap_pair()
    metric = EchoMetric()
    d_swap = metric.dist(encode(a), encode(b))
    d_dup = metric.dist(encode(a), encode(a * 0.95))
    assert d_swap > 3 * d_dup, (d_swap, d_dup)


def test_order_invisible_without_chunks():
    """Ablation: with n_chunks=1 the order signal must shrink — proving
    the chunked vector, not just any vector, carries the time axis."""
    from echo.encode import LAYOUT_FLAT
    a, b = _order_swap_pair()
    metric = EchoMetric(layout=LAYOUT_FLAT)
    d_flat = metric.dist(encode(a, n_chunks=1), encode(b, n_chunks=1))
    metric_c = EchoMetric()
    d_chunk = metric_c.dist(encode(a), encode(b))
    assert d_chunk > d_flat, (d_chunk, d_flat)


def test_long_clip_blocked_path_matches_single_pass():
    """The blocked (memory-flat) path must compute the same vector as
    the single-pass path — force tiny blocks on a 60 s clip."""
    import importlib
    E = importlib.import_module("echo.encode")   # not the function
    sr = 44100
    t = np.arange(sr * 60) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t)
         + 0.1 * np.random.default_rng(0).normal(0, 1, len(t))).astype(np.float32)
    big = E._MAX_FRAMES_PER_PASS
    try:
        E._MAX_FRAMES_PER_PASS = 512                     # many blocks
        v_blocked = E.encode(x, sr)
        E._MAX_FRAMES_PER_PASS = 10 ** 9                 # single pass
        v_single = E.encode(x, sr)
    finally:
        E._MAX_FRAMES_PER_PASS = big
    assert np.allclose(v_blocked, v_single, atol=1e-8), \
        np.abs(v_blocked - v_single).max()


def test_block_weights_change_distance():
    """Weights are real: zeroing all but MFCCs must change the metric."""
    a, b = _order_swap_pair()
    full = EchoMetric().dist(encode(a), encode(b))
    mfcc = EchoMetric(weights={n: (1.0 if n == "mfcc_mean" else 0.0)
                               for n, _, _ in LAYOUT}).dist(encode(a), encode(b))
    assert full != mfcc and mfcc > 0