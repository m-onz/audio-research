"""render: preview the riff as the patch plays it.

The patch clocks the alien sequencers at a fixed 80 ms per step, so the
preview renders on an 80 ms grid (STEP_S) — what you hear is what the
patch plays. Alien expressions are evaluated across the whole preview
(they breathe / phase as polymeter), folder indices wrap modulo as
playdir~ does, DRUMS (from the patch's own kit) lock the groove, and
melodic hits get per-hit stochastic processing so repeats differ.
"""
import numpy as np

from . import core, patterns as P

STEP_S = 0.080                 # the patch's clock interval (80 ms / step)
MAX_PREVIEW_STEPS = 128        # ~10 s cap so long riffs still preview fast

DRUM_GAINS = {"kick": 0.95, "snare": 0.8, "hihat": 0.45,
              "openhh": 0.4, "perc": 0.5}


def _process_hit(y, sr, rng, g, voice):
    """Stochastic per-trigger processing on melodic voices."""
    y = np.asarray(y, np.float32)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    speed = 1.0
    if rng.random() < g.get("hit_pitch_p", 0.0):
        speed = float(rng.choice([0.5, 0.75, 0.87, 1.0, 1.26, 1.5, 2.0]))
    if speed != 1.0:
        y = core.resample(y, speed)
    if rng.random() < g.get("hit_rev_p", 0.0) and len(y) > 400:
        y = y[::-1].copy()
    if rng.random() < g.get("hit_drive_p", 0.0):
        d = float(rng.uniform(1.5, 3.0))
        y = np.tanh(y * d) / np.tanh(d)
    if rng.random() < g.get("hit_crush_p", 0.0):
        hold = int(rng.integers(2, 6))
        y = np.repeat(y[::hold], hold, axis=0)[: len(y)]
    return core.safe(np.asarray(y, np.float32))


def _wrap(v, n):
    return ((int(round(v)) - 1) % n) + 1 if n else 1


def _voice_stream(expr, buf, n_steps, seed, salt):
    """Evaluate one voice expression -> [(step, sound_index)] hits."""
    nsnd = len(buf)
    srng = np.random.default_rng((seed + salt) & 0x7fffffff)
    stream = P.eval_stream(expr, srng, n_steps)
    rr, hits = 0, []
    for i, v in enumerate(stream):
        if not P.is_hit(v):
            continue
        if float(v) >= 1.5:
            k = _wrap(v, nsnd)
        else:
            k = (rr % nsnd) + 1; rr += 1
        hits.append((i, k))
    return hits


def collect_events(g, sounds, drum_sounds, sr=44100):
    """Melodic voices (bass/noise/fill) + drum voices on the 80 ms grid,
    with the sub line following the bass index stream."""
    bars = g["bars"]
    n_steps = min(bars * g["steps_per_bar"], MAX_PREVIEW_STEPS)
    swing = g.get("swing", 0.0)
    seed = int(g["seed"]) & 0x7fffffff
    events = []

    def step_time(i):
        t = i * STEP_S
        if i % 2 == 1:
            t += swing * STEP_S * 0.5
        return t

    # melodic voices (processed per hit)
    bass_idx = {}
    for voice, expr in (("bass", g["bass_expr"]), ("noise", g["noise_expr"]),
                        ("fill", g["fill_expr"])):
        buf = sounds.get(voice) or []
        if not buf:
            continue
        hrng = np.random.default_rng(seed + 7 + hash(voice) % 9973)
        for i, k in _voice_stream(expr, buf, n_steps, seed, hash(voice) % 9973):
            if voice == "bass":
                bass_idx[i] = k
            y = _process_hit(buf[k - 1], sr, hrng, g, voice)
            events.append(dict(t_s=step_time(i), voice=voice, sound=y))

    # drums (clean; the patch's kit) — tiny micro-timing only
    jit = g.get("hit_jit_ms", 0.0) / 1000.0
    for voice, expr in (g.get("drum_exprs") or {}).items():
        buf = drum_sounds.get(voice) or []
        if not buf:
            continue
        jrng = np.random.default_rng(seed + 31 + hash(voice) % 9973)
        for i, k in _voice_stream(expr, buf, n_steps, seed, hash("d" + voice) % 9973):
            t = step_time(i) + (jrng.uniform(-1, 1) * jit if jit else 0.0)
            events.append(dict(t_s=max(0.0, t), voice="drum",
                               dvoice=voice, sound=buf[k - 1]))

    # sub line follows the bass index at strong steps
    notes = g["notes"]
    for i in range(n_steps):
        k = bass_idx.get(i)
        if k:                                      # sub follows every bass hit
            events.append(dict(t_s=step_time(i), voice="sub",
                               note=int(notes[(k - 1) % len(notes)]),
                               dur_s=STEP_S * 2.0))
    events.sort(key=lambda e: e["t_s"])
    return events, bass_idx, n_steps


def _pan(y, pan):
    th = (pan + 1) * np.pi / 4
    return y * np.float32([np.cos(th), np.sin(th)])


def render_preview(g, events, n_steps, sr=44100):
    total = n_steps * STEP_S + 1.0
    n = int(total * sr)
    out = np.zeros((n, 2), np.float32)
    gains = g["gains"]
    pans = g.get("pans", {})
    for e in events:
        i0 = int(e["t_s"] * sr)
        if i0 >= n:
            continue
        if e["voice"] == "sub":
            dur = e.get("dur_s", STEP_S * 2)
            ns = min(int(dur * sr), n - i0)
            tt = np.arange(ns) / sr
            env = np.minimum(tt / 0.008, 1.0) * np.exp(-tt * 5.0)
            buf = (np.sin(2 * np.pi * 440.0 * 2 ** ((e["note"] - 69) / 12) * tt)
                   * env).astype(np.float32)
            out[i0: i0 + ns, 0] += buf * gains["sub"]
            out[i0: i0 + ns, 1] += buf * gains["sub"]
            continue
        y = np.asarray(e["sound"], np.float32)
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)
        if e["voice"] == "drum":
            gain = DRUM_GAINS.get(e["dvoice"], 0.5)
            pan = 0.0 if e["dvoice"] in ("kick", "snare") else 0.25
        else:
            gain = gains[e["voice"]]
            pan = pans.get(e["voice"], 0.0)
        if pan:
            y = _pan(y, pan)
        f = min(64, int(0.002 * sr))
        if f > 2:
            y = y.copy(); y[:f] *= np.linspace(0, 1, f)[:, None]
        e2 = min(n, i0 + y.shape[0])
        out[i0:e2] += y[: e2 - i0] * gain
    # master: HP rumble, LP hiss, glue, normalize
    from scipy.signal import butter, lfilter, lfilter_zi
    for cut, bt in ((26, "high"), (16000, "low")):
        b, a = butter(2, cut / (sr / 2), btype=bt)
        zi = lfilter_zi(b, a)
        out = np.stack([lfilter(b, a, out[:, c], zi=zi * out[0, c])[0]
                        for c in range(2)], axis=1)
    out = np.tanh(out * 1.2) / np.tanh(1.2)
    pk = float(np.abs(out).max())
    if pk > 1e-4:
        out *= 0.95 / pk
    return out
