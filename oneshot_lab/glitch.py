"""Glitch family: raw-byte databend (sonify the file's own bytes through many
encodings, optionally corrupted), bitcrush/decimate, ring mod / AM,
waveshaping, machine-gun stutter, granular scatter (micro-grain explosion
with per-grain corruption), irrational-period echo replicate, and
cellular-automaton chopping."""
from pathlib import Path

import numpy as np

from .core import (EPS, fade_edges, mono, resample, safe)
from .core import resample_curve

# chaotic (Class 3) Wolfram rules only — arbitrary rule bytes can converge to
# dead fixed points that mute everything (learned the hard way with rule 114)
CHAOS_RULES = (22, 30, 45, 60, 90, 105, 106, 110, 122, 150, 161)


# ----------------------------------------------------------------- databend

def _norm(x):
    x = x - np.mean(x)
    s = np.std(x)
    if s > EPS:
        x = x / s * 0.25
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def sonify_bytes(u8):
    """Every encoding is a different way of listening to the same bytes.

    From the databend labs: int8 / bit planes / delta / mu-law / nibbles /
    16-bit little & big endian / MSB & LSB of 16-bit PCM."""
    out = {}
    out["int8"] = _norm(u8.astype(np.float32) - 128.0)
    for shift in (0, 4):
        out[f"bits{shift}"] = _norm(((u8 >> shift) & 3).astype(np.float32) - 1.5)
    out["delta"] = _norm(np.diff(u8.astype(np.int16), prepend=0))
    u = (~u8.astype(np.uint16)) & 0xFF
    t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
    t = np.where(u & 0x80, 0x84 - t, t - 0x84)
    out["mulaw"] = _norm(t.astype(np.float32))
    out["nib_lo"] = _norm((u8 & 15).astype(np.float32) - 8)
    out["nib_hi"] = _norm((u8 >> 4).astype(np.float32) - 8)
    if len(u8) >= 4:
        n2 = len(u8) // 2
        b = u8[: n2 * 2]
        out["i16le"] = _norm(np.frombuffer(b.tobytes(), "<i2").astype(np.float32))
        out["i16be"] = _norm(np.frombuffer(b.tobytes(), ">i2").astype(np.float32))
        b2 = u8[: n2 * 2].reshape(-1, 2)
        out["shape8"] = _norm(b2[:, 1].astype(np.float32) - 128.0)  # MSB: waveform
        out["noise8"] = _norm(b2[:, 0].astype(np.float32) - 128.0)  # LSB: dither
    return {k: y for k, y in out.items() if len(y) > 64}


def corrupt_bytes(u8, rng):
    """A family of corrupted readings of the same bytes."""
    v = {"raw": u8}
    u16 = u8.astype(np.uint16)
    v["rot"] = (((u16 << 3) | (u16 >> 5)) & 0xFF).astype(np.uint8)
    v["xor"] = np.frombuffer(u8.tobytes(), np.uint8) ^ \
        np.frombuffer(rng.bytes(len(u8)), np.uint8)
    sw = u8.copy()
    for _ in range(min(24, max(1, len(sw) // 4096))):
        m = int(rng.integers(64, min(2048, max(65, len(sw)))))
        a = int(rng.integers(0, max(1, len(sw) - m)))
        b = int(rng.integers(0, max(1, len(sw) - m)))
        sa, sb = sw[a:a + m].copy(), sw[b:b + m].copy()
        sw[a:a + m], sw[b:b + m] = sb, sa
    v["swap"] = sw
    z = u8.copy()
    for _ in range(min(14, max(1, len(z) // 2048))):
        a = int(rng.integers(0, max(1, len(z) - 800)))
        z[a: a + int(rng.integers(100, 800))] = 0
    v["zeroed"] = z
    return v


def databend(x, sr, rng, path=None, corrupt_p=0.6, max_s=4.0, max_bytes=4_000_000):
    """DATABEND: the wav file's own raw bytes (headers included) become the
    sound. Pick an encoding (mu-law snaps, MSB bodies, LSB dither ticks,
    16-bit structures...), optionally through a corruption (swap / zero /
    xor / rotate), and lift a random window."""
    if path is None:
        # no file path: databend the samples themselves
        u8 = np.frombuffer(np.ascontiguousarray(
            (np.clip(mono(x), -1, 1) * 12000).astype(np.int16)).tobytes(),
            np.uint8)
    else:
        # read a window of the file's RAW bytes — headers and all
        fsize = Path(str(path)).stat().st_size
        take = min(fsize, max_bytes)
        start = int(rng.integers(0, max(1, fsize - take)))
        with open(str(path), "rb") as fh:
            fh.seek(start)
            u8 = np.frombuffer(fh.read(take), np.uint8)
    if len(u8) < 4096:
        u8 = np.tile(u8, max(2, 4096 // max(1, len(u8))))
    variants = corrupt_bytes(u8, rng) if rng.random() < corrupt_p else {"raw": u8}
    key = str(rng.choice(sorted(variants.keys())))
    encs = sonify_bytes(variants[key])
    if not encs:
        u8 = np.tile(u8, 4)
        encs = sonify_bytes(u8)
    enc_key = str(rng.choice(sorted(encs.keys())))
    y = encs[enc_key]
    want = int(max_s * sr)
    if len(y) > want:
        st = int(rng.integers(0, len(y) - want))
        y = y[st: st + want]
    elif len(y) < want // 4:
        reps = int(np.ceil(want // 4 / max(1, len(y))))
        y = np.tile(y, reps)
    y = fade_edges(y.astype(np.float32))
    return np.stack([y, np.roll(y, int(rng.integers(1, 300)))], axis=1)


# ----------------------------------------------------------------- crush / mod / shape


def crush(x, sr, rng, bits=None, downsample=None, jitter=0.3):
    """Bitcrush + sample-hold decimation (with optional drunk jitter)."""
    x = np.asarray(x, np.float32)
    bits = int(bits if bits is not None else rng.integers(2, 8))
    lv = 2.0 ** (bits - 1)
    y = np.round(x * lv) / lv
    k = int(downsample if downsample is not None else rng.integers(2, 10))
    if k > 1:
        y = np.repeat(y[::k], k, axis=0)[: len(y)]
        if jitter:
            n = len(y)
            drift = np.cumsum(rng.uniform(-jitter, jitter, n))
            pos = np.clip(np.arange(n) + np.round(drift), 0, n - 1)
            y = y[pos.astype(int)]
    return safe(y)


def ringmod(x, sr, rng, f_hz=None, mix=1.0, drift_p=0.4, side="both"):
    """Ring modulation with a carrier at f_hz (auto: below/near the hit's f0).
    Optional drifting carrier (wobbling metal)."""
    x = np.asarray(x, np.float32)
    if f_hz is None:
        f_hz = float(rng.uniform(40, 1800))
    t = np.arange(len(x)) / sr
    ph = 2 * np.pi * f_hz * t
    if rng.random() < drift_p:
        ph += rng.uniform(0.5, 9.0) * np.sin(2 * np.pi *
                                            rng.uniform(0.3, 6.0) * t)
    car = np.sin(ph).astype(np.float32)
    y = x * car[:, None] if x.ndim > 1 else x * car
    if side == "upper":
        y = y + np.abs(y)
    elif side == "lower":
        y = y - np.abs(y)
    return (1.0 - mix) * x + mix * y


def am_gate(x, sr, rng, f_hz=None, depth=0.9, shape="square"):
    """AM tremolo gate: square or sine gate at f_hz (stutter-y machine gun)."""
    x = np.asarray(x, np.float32)
    if f_hz is None:
        f_hz = float(rng.uniform(2, 40))
    t = np.arange(len(x)) / sr
    if shape == "square":
        g = np.sign(np.sin(2 * np.pi * f_hz * t))
        g = np.where(g < 0, 1.0 - depth, 1.0)
    else:
        g = 1.0 - depth * 0.5 * (1 - np.sin(2 * np.pi * f_hz * t))
    return x * g[:, None].astype(np.float32) if x.ndim > 1 else x * g


def waveshape(x, sr, rng, mode=None, drive=None, mix=1.0):
    """Waveshaping: tanh drive / wavefolder / chebyshev / power asymmetric."""
    x = np.asarray(x, np.float32)
    mode = mode or str(rng.choice(["tanh", "fold", "cheby", "pow"]))
    drive = float(drive if drive is not None else rng.uniform(1.5, 6.0))
    if mode == "tanh":
        y = np.tanh(x * drive) / np.tanh(drive)
    elif mode == "fold":                       # sine wavefolder
        y = np.sin(x * drive * np.pi / 2.0)
    elif mode == "cheby":                     # chebyshev harmonic mangler
        n = int(rng.integers(2, 6))
        y = np.zeros_like(x)
        for k in range(1, n + 1):
            y += np.cos(k * np.arcsin(np.clip(x, -1, 1))) / k
        y /= n * 0.5
    else:                                     # asymmetric power
        a = rng.uniform(1.5, 3.5)
        y = np.sign(x) * np.abs(x) ** a
    return (1.0 - mix) * x + mix * y


# ----------------------------------------------------------------- rhythm glitch


def stutter(x, sr, rng, slice_ms=None, count=None, decay=None,
            pitch_semi=None, out_mult=2.0, max_s=8.0):
    """Machine-gun loop-lock: grab a slice, repeat at a fast period with a
    decay envelope and an optional pitch ladder. From the storm loop-locks.

    All randomness is pre-drawn so stereo channels stay coherent."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    sl = int((slice_ms if slice_ms is not None else rng.uniform(50, 180))
             / 1000.0 * sr)
    sl = max(64, min(sl, n))
    e = np.abs(mono(x))
    win = max(0, int(np.argmax(np.convolve(e, np.ones(sl) / sl, "same")) - sl / 2))
    period = int(rng.uniform(0.030, 0.11) * sr)
    count = int(count if count is not None else rng.integers(8, 24))
    decay = float(decay if decay is not None else rng.uniform(0.55, 0.85))
    pitch_ladder = pitch_semi is not None or rng.random() < 0.5
    out_n = int(min(max(n * out_mult, period * (count + 2)), max_s * sr))
    out = np.zeros((out_n, x.shape[1]), np.float32)
    # pre-draw the per-hit randomness: drops, pitch steps
    drops = rng.random(count + 2) < 0.12
    steps = rng.random(count + 2)
    step_semis = rng.choice([-1.0, 1.0, -2.0, 3.0], count + 2)
    for c in range(x.shape[1]):
        y0 = fade_edges(x[win: win + sl, c].copy(), s=0.003)
        g = 0.9
        yb = y0
        j = 0
        while j * period < out_n and j <= count:
            if pitch_ladder and j % 4 == 3 and steps[j] < 0.5:
                yb = resample(y0, 2.0 ** (float(pitch_semi or step_semis[j]) / 12.0))
                if len(yb) > len(y0):
                    yb = yb[: len(y0)]
                elif len(yb) < len(y0):
                    yb = np.pad(yb, (0, len(y0) - len(yb)))
            if drops[j]:                  # drop a hit for groove
                j += 1
                continue
            tt = j * period
            e2 = min(out_n - tt, len(yb))
            out[tt: tt + e2, c] += yb[:e2] * g
            g *= decay
            j += 1
    return out


def granular_scatter(x, sr, rng, count=90, grain_ms=(12, 160), reverse_p=0.2,
                     crush_p=0.2, decim_p=0.15, ring_p=0.12, spread=0.9,
                     out_mult=1.6, max_s=10.0):
    """Explode the oneshot into a cloud of micro-grains, each independently
    corrupted (reverse / crush / decimate / ringmod) and scattered across the
    stereo field with a density arc. From the granular grain_fx + micro_grains."""
    x = np.asarray(x, np.float32)
    m = mono(x)
    n_out = int(min(max(len(x) * out_mult, int(1.0 * sr)), max_s * sr))
    out = np.zeros((n_out, 2), np.float32)
    lo = int(grain_ms[0] / 1000.0 * sr)
    hi = int(grain_ms[1] / 1000.0 * sr)
    if len(m) <= hi:
        m = np.tile(m, int(np.ceil((hi + 2) / max(1, len(m)))))
    arc = np.sin(np.linspace(0, np.pi, 1000)) ** 0.7  # density arc
    for j in range(int(count)):
        g = int(rng.integers(lo, hi))
        st = int(rng.integers(0, len(m) - g))
        gr = m[st: st + g].astype(np.float32).copy()
        if rng.random() < reverse_p:
            gr = gr[::-1]
        if rng.random() < crush_p:
            bits = int(rng.integers(3, 9))
            lv = 2.0 ** (bits - 1)
            gr = np.round(gr * lv) / lv
        if rng.random() < decim_p:
            k = int(rng.integers(2, 10))
            gr = np.repeat(gr[::k], k)[: len(gr)]
        if rng.random() < ring_p:
            t = np.arange(len(gr)) / sr
            gr = gr * np.sin(2 * np.pi * rng.uniform(80, 3000) * t)
        at = float(rng.uniform(0, 1.0))
        dens = float(np.interp(at, np.linspace(0, 1, len(arc)), arc))
        if rng.random() > 0.25 + 0.75 * dens:
            continue
        pos = int(at * (n_out - g))
        pan = rng.uniform(-spread, spread)
        th = (pan + 1) * np.pi / 4
        gl, gr_ = np.cos(th), np.sin(th)
        w = np.hanning(len(gr)).astype(np.float32)
        amp = float(rng.uniform(0.3, 1.0))
        out[pos: pos + g, 0] += gr * w * amp * gl
        out[pos: pos + g, 1] += gr * w * amp * gr_
    return safe(out)


def echo_replicate(x, sr, rng, period_mult=None, k=None, decay=None,
                   pitch_p=0.5, max_s=10.0):
    """Echo loops at rates UNRELATED to the hit's own length — irrational
    multiples (0.37 / 0.61 / 1.41 / 2.33 of the original duration), drifting
    against it -> glitch. From the storm glitch_replicate."""
    x = np.asarray(x, np.float32)
    dur = len(x) / sr
    pm = float(period_mult or
               rng.uniform(0.03, 0.35) * rng.choice([0.37, 0.61, 1.0, 1.41, 2.33]))
    k = int(k or rng.integers(3, 9))
    decay = float(decay or rng.uniform(0.55, 0.88))
    period = pm * dur
    n_out = int(min((len(x) / sr + k * period + 0.1) * sr, max_s * sr))
    out = np.zeros((n_out, x.shape[1] if x.ndim > 1 else 1), np.float32)
    if x.ndim == 1:
        x = x[:, None]
        out = out[:, :1]
    s0 = x
    if len(out) > len(s0):
        out[: len(s0)] += s0
    else:
        out += s0[: len(out)]
    for j in range(1, k + 1):
        tt = int(j * period * sr)
        if tt >= n_out:
            break
        yb = s0
        if rng.random() < pitch_p and rng.random() < 0.6:
            yb = resample(s0, 2.0 ** (float(rng.choice([-1, 1, 2, -2, 5, 7])) / 12.0))
        g = float(rng.uniform(0.35, 0.75)) * decay ** j
        e = min(n_out, tt + len(yb))
        out[tt: e] += (yb * g)[: e - tt]
    return safe(out)


def ca_chop(x, sr, rng, step_ms=None, rule=None, bars=2, wild=0.5, max_s=12.0):
    """Cellular-automaton chop: the hit is sliced to a step grid; a Wolfram
    CA (chaotic rules only) gates which steps play, each active step hit by a
    random process (pitch / reverse / decimate / drive / one-pole filter).
    From the databend glitchkit + eventgrid hit processing."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    step = int((step_ms or rng.uniform(50, 120)) / 1000.0 * sr)
    n_steps = max(4, int(np.ceil(len(x) / step)))
    step = max(16, int(len(x) / n_steps))
    rule = int(rule if rule is not None else CHAOS_RULES[int(rng.integers(len(CHAOS_RULES)))])
    cells = (rng.random(n_steps) > 0.5).astype(np.uint8)
    out = np.zeros((n_steps * step * bars, x.shape[1]), np.float32)
    for bar in range(bars):
        cells = ca_step(rule, cells)
        if cells.sum() == 0 or cells.sum() == n_steps:
            cells = (rng.random(n_steps) > 0.5).astype(np.uint8)
        for i in range(n_steps):
            if not int(cells[i]):
                continue
            if rng.random() > 0.15 + 0.85 * wild:
                continue
            s0 = x[i * step: (i + 1) * step]
            y = s0
            r = rng.random()
            if r < 0.35:
                y = resample(y, 2.0 ** rng.uniform(-1, 1))
            elif r < 0.45:
                y = y[::-1]
            elif r < 0.55:                       # sample-hold decimation
                k = int(rng.integers(2, 7))
                y = np.repeat(y[::k], k, axis=0)[: len(y)]
            elif r < 0.7:                        # tanh drive
                g = float(rng.uniform(1.5, 4.0))
                y = np.tanh(y * g) / np.tanh(g)
            if len(y) < step:
                y = np.pad(y, ((0, step - len(y)), (0, 0)))
            f = min(256, step // 12)
            if f > 4:
                y = y.copy()
                y[:f] *= np.linspace(0, 1, f)[:, None]
                y[-f:] *= np.linspace(1, 0, f)[:, None]
            p = (bar * n_steps + i) * step
            if p + step > len(out):
                break
            out[p: p + step] += y[:step]
    return safe(out[: min(len(out), int(max_s * sr))])


def ca_step(rule, cells):
    """One step of an elementary 1-D cellular automaton (Wolfram rule)."""
    l = np.roll(cells, 1)
    r = np.roll(cells, -1)
    idx = (l.astype(np.int64) << 2) | (cells.astype(np.int64) << 1) | r.astype(np.int64)
    return ((rule >> idx) & 1).astype(np.uint8)


# ----------------------------------------------------------------- misc glitch


def reverse(x, sr, rng, max_s=None):
    """Plain reverse — the simplest variation."""
    x = np.asarray(x, np.float32)
    return x[::-1].copy()


def samplehold_wobble(x, sr, rng, base_hz=6.0, jitter=0.4, depth=0.5):
    """Drunk sample-and-hold wobble: playback rate skitters around 1.0 with
    a noisy LFO — a broken-tape skitter."""
    x = np.asarray(x, np.float32)
    n = len(x)
    t = np.arange(n) / sr
    lfo = np.sin(2 * np.pi * base_hz * t +
                 np.cumsum(rng.normal(0, jitter, n)) * 0.02)
    rate = np.maximum(1.0 + depth * 0.5 * lfo, 0.05)
    y = resample_curve(x, rate)
    return y if len(y) else x