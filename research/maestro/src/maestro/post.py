"""post: the mastering hand — damage placed where it lives best.

The composition is done; this is the fun part. A post LEDGER of
random-but-seeded events is planned against the form: stutters
belong in the FAST sections (a 320bpm rush is begging for a
ladder-pitch stutter), time-stretches belong in the SLOW bits (a
75bpm sink should hang in the air), tape speed bends the transitions,
reverses and euclid gates are the seasoning. Then the piece is cut
at event boundaries, each segment transformed, and reassembled —
length changes propagate honestly (a stretched section really is
longer; the piece breathes).

Informed by the audio_lab: club_lab's stutter (const/ramp/ladder
retrigger modes, beat-biased placement), storm_lab's tape warp,
loom's chop and stretch. Everything deterministic from the seed; the
ledger ships with the piece (provenance, like everywhere here).
"""

from __future__ import annotations

import numpy as np

__all__ = ["post_process", "plan_post", "apply_post", "duck",
           "seg_stretch", "seg_stutter", "seg_speed", "seg_reverse",
           "seg_gate", "seg_crush", "FAST_BPM", "SLOW_BPM"]

SR = 44100
FAST_BPM = 170.0          # sections at/above this stutter
SLOW_BPM = 105.0          # sections at/below this stretch
_XF = 192                 # join fade, samples (~4 ms)


# ------------------------------------------------------------------ transforms


def seg_speed(seg: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Tape: varispeed — pitch and time move together. speed < 1 is
    the storm_lab warp: slower, longer, darker."""
    if abs(speed - 1.0) < 1e-6 or seg.shape[0] < 4:
        return seg
    n = max(4, int(seg.shape[0] / speed))
    pos = np.linspace(0.0, seg.shape[0] - 1, n)
    base = np.arange(seg.shape[0])
    if seg.ndim == 1:
        return np.interp(pos, base, seg)
    return np.stack([np.interp(pos, base, seg[:, c])
                     for c in range(seg.shape[1])], axis=1)


def seg_stretch(seg: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Granular time-stretch, pitch preserved (loom's stretch, the
    granular classic): grain 8192, hop 2048, hann OLA — big grains
    keep the sidebands tight, so a stretched tone is still its tone.
    factor > 1 is longer — the slow bits hang."""
    x = np.asarray(seg, np.float64)
    if factor <= 1.02 or x.shape[0] < 16384:
        return seg
    if x.ndim == 1:
        x = x[:, None]
    grain, hop = 8192, 2048
    win = np.hanning(grain)[:, None]
    n_out = int(x.shape[0] * factor)
    out = np.zeros((n_out + grain, x.shape[1]))
    j = 0
    while j * hop < n_out:
        r = int(j * hop / factor)
        if r + grain > x.shape[0]:
            break
        out[j * hop: j * hop + grain] += x[r: r + grain] * win
        j += 1
    return out[:n_out]


def seg_stutter(seg: np.ndarray, sr: int, rng, slice_s: float = 0.06,
                want_s: float = 0.8, mode: str = "ladder",
                step: float = 1.0) -> np.ndarray:
    """club_lab's stutter, post-slice flavor: retrigger the segment's
    first `slice_s` over and over until `want_s` is filled.
      const   the same slice, hypnotic
      ramp    each rep longer/shorter than the last (ratio drawn here)
      ladder  each rep pitched up/down `step` semitones — the riser
    """
    klen = int(slice_s * sr)
    if klen < 64 or seg.shape[0] < klen * 2:
        return seg
    head = seg[: klen].astype(np.float64)
    want = int(want_s * sr)
    ramp = float(rng.uniform(0.75, 1.3))
    built = [head]
    total = klen
    rep = head
    while total < want:
        if mode == "ladder":
            rep = seg_speed(rep, sr, 2.0 ** (step / 12.0))
        elif mode == "ramp":
            n_len = int(np.clip(len(rep) * ramp, 256, int(0.4 * sr)))
            pos = np.linspace(0.0, len(rep) - 1, n_len)
            base = np.arange(len(rep))
            rep = np.stack([np.interp(pos, base, rep[:, c])
                            for c in range(rep.shape[1])], axis=1)
        built.append(rep)
        total += len(rep)
    block = np.concatenate(built)[: want]
    f = min(len(block) // 4, 512)
    block[:f] *= np.linspace(0, 1, f)[:, None]
    block[-f:] *= np.linspace(1, 0, f)[:, None]
    return block


def seg_reverse(seg: np.ndarray) -> np.ndarray:
    return np.asarray(seg)[::-1].copy()


def seg_gate(seg: np.ndarray, sr: int, steps: int = 16, pulses: int = 5,
             rot: int = 0) -> np.ndarray:
    """Euclid silence inside the segment (loom's chop): the pattern
    turns a stretch of audio into rhythm."""
    x = np.asarray(seg, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if steps < 2 or n < 512:
        return x
    pat = np.zeros(steps, bool)
    for i in range(steps):                      # the classic euclid fill
        pat[(i * pulses + rot) % steps] = True
    per = n / steps
    out = np.zeros_like(x)
    nf = max(32, int(0.004 * sr))
    for i in range(steps):
        if not pat[i]:
            continue
        a, b = int(i * per), int((i + 1) * per)
        chunk = x[a:b].copy()
        k = min(nf, chunk.shape[0] // 2)
        if k > 0:
            chunk[:k] *= np.linspace(0, 1, k)[:, None]
            chunk[-k:] *= np.linspace(1, 0, k)[:, None]
        out[a:b] = chunk
    return out


def seg_crush(seg: np.ndarray, bits: int = 5) -> np.ndarray:
    levels = 2.0 ** max(2, bits)
    return np.round(np.asarray(seg) * levels) / levels


# ------------------------------------------------------------------ planning


def _weighted_section(rng, bounds, want_fast: bool):
    """Pick a section, weighted toward its character."""
    if not bounds:
        return None
    key = lambda s: s["bpm"]
    if want_fast:
        w = np.array([max(s["bpm"], 1.0) for s in bounds]) ** 2
    else:
        w = np.array([1.0 / max(s["bpm"], 1.0) for s in bounds]) ** 2
    w /= w.sum()
    return bounds[int(rng.choice(len(bounds), p=w))]


def plan_post(rng, bounds, length_s: float, intensity: float = 1.0,
              n_stutters=None, n_stretches=None, n_tapes=None,
              n_reverses=None, n_gates=None) -> list[dict]:
    """The post ledger: section-aware, non-overlapping, deterministic.
    Counts scale with piece length (per 180 s) and the intensity dial."""
    scale = intensity * max(length_s, 1.0) / 180.0
    n_stutters = n_stutters if n_stutters is not None else int(6 * scale)
    n_stretches = n_stretches if n_stretches is not None else int(3 * scale)
    n_tapes = n_tapes if n_tapes is not None else int(2 * scale)
    n_reverses = n_reverses if n_reverses is not None else int(3 * scale)
    n_gates = n_gates if n_gates is not None else int(2 * scale)
    fast = [s for s in bounds if s["bpm"] >= FAST_BPM]
    slow = [s for s in bounds if s["bpm"] <= SLOW_BPM]
    events: list[dict] = []

    def _place(kind, pool_fast, pool_slow, dur_params_fn):
        want_fast = bool(pool_fast) and (not pool_slow or rng.random() < 0.75)
        sec = _weighted_section(rng, pool_fast if want_fast else pool_slow,
                               want_fast)
        if sec is None:
            sec = bounds[int(rng.integers(len(bounds)))]
        margin = 1.0
        t0 = rng.uniform(sec["t0"] + margin, max(sec["t0"] + margin,
                                                 sec["t1"] - margin))
        dur, params = dur_params_fn(t0, sec)
        events.append({"type": kind, "t0": round(float(t0), 3),
                       "dur": round(float(dur), 3), **params})

    for _ in range(n_stutters):
        _place("stutter", fast, [], lambda t0, s: (
            float(rng.uniform(0.35, 1.3)),
            {"slice": float(rng.choice([0.03, 0.05, 0.08, 0.12])),
             "mode": str(rng.choice(["const", "ramp", "ladder"])),
             "step": float(rng.choice([-2, -1, 1, 1, 2, 3]))}))
    for _ in range(n_stretches):
        _place("stretch", [], slow, lambda t0, s: (
            float(rng.uniform(2.0, min(6.0, max(s["t1"] - t0, 2.5)))),
            {"factor": float(rng.uniform(1.4, 2.6))}))
    for _ in range(n_tapes):
        _place("tape", [], slow + bounds, lambda t0, s: (
            float(rng.uniform(1.0, 3.0)),
            {"speed": float(rng.uniform(0.55, 0.9)) if rng.random() < 0.8
             else float(rng.uniform(1.25, 1.6))}))
    for _ in range(n_reverses):
        sec = bounds[int(rng.integers(len(bounds)))]
        t0 = rng.uniform(sec["t0"] + 0.5, max(sec["t0"] + 0.5,
                                             sec["t1"] - 0.5))
        events.append({"type": "reverse", "t0": round(float(t0), 3),
                       "dur": round(float(rng.uniform(0.5, 2.0)), 3)})
    for _ in range(n_gates):
        _place("gate", fast, [], lambda t0, s: (
            float(rng.uniform(1.0, 2.2)),
            {"steps": int(rng.choice([8, 12, 16])),
             "pulses": 0, "rot": int(rng.integers(16))}))
    for e in events:                            # euclid pulses: 2..steps-2
        if e["type"] == "gate":
            e["pulses"] = int(rng.integers(2, max(3, e["steps"] - 1)))

    # non-overlap: greedy accept in planned order, keep 1.5 s apart
    events.sort(key=lambda e: e["t0"])
    kept, last_end = [], -10.0
    for e in events:
        if e["t0"] - last_end >= 1.5 and e["t0"] + e["dur"] < length_s - 1.0:
            kept.append(e)
            last_end = e["t0"] + e["dur"]
    return kept


# ------------------------------------------------------------------ reassembly


def _fade_edges(x: np.ndarray, sr: int) -> np.ndarray:
    nf = min(_XF, x.shape[0] // 2)
    if nf > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
        x[:nf] *= ramp[:, None] if x.ndim == 2 else ramp
        x[-nf:] *= (ramp[::-1])[:, None] if x.ndim == 2 else ramp[::-1]
    return x


def apply_post(audio, sr: int, events: list[dict]):
    """Cut the piece at event boundaries, transform each segment,
    reassemble. Stretched sections really get longer — the piece
    breathes. Returns (audio, ledger with realized out lengths)."""
    src = np.asarray(audio, np.float64)
    if src.ndim == 1:
        src = src[:, None]
    mono = src.shape[1] == 1
    rng = None
    out_chunks, ledger, cur = [], [], 0
    events = sorted(events, key=lambda e: e["t0"])
    for ev in events:
        i0 = int(ev["t0"] * sr)
        dur_i = int(ev["dur"] * sr)
        if i0 <= cur or i0 + dur_i > src.shape[0]:
            continue
        seg = src[cur: i0]
        if seg.shape[0] > 2 * _XF:
            seg = _fade_edges(seg.copy(), sr)
        out_chunks.append(seg)
        piece = src[i0: i0 + dur_i].copy()
        if ev["type"] == "stutter":
            if rng is None:
                import numpy as _np
                rng = _np.random.default_rng(int(ev["t0"] * 1000) & 0x7FFFFFFF)
            piece = seg_stutter(piece, sr, rng, ev.get("slice", 0.06),
                                ev["dur"], ev.get("mode", "ladder"),
                                ev.get("step", 1.0))
        elif ev["type"] == "stretch":
            piece = seg_stretch(piece, sr, ev.get("factor", 1.6))
        elif ev["type"] == "tape":
            piece = seg_speed(piece, sr, ev.get("speed", 0.8))
        elif ev["type"] == "reverse":
            piece = seg_reverse(piece)
        elif ev["type"] == "gate":
            piece = seg_gate(piece, sr, ev.get("steps", 16),
                             ev.get("pulses", 5), ev.get("rot", 0))
        piece = np.asarray(piece)
        if piece.ndim == 1:
            piece = piece[:, None]
        piece = _fade_edges(piece.copy(), sr)
        out_chunks.append(piece)
        ledger.append({**ev, "out_dur": round(float(piece.shape[0] / sr), 3)})
        cur = i0 + dur_i
    tail = src[cur:]
    if tail.shape[0] > 2 * _XF:
        tail = _fade_edges(tail.copy(), sr)
    out_chunks.append(tail)
    out = np.concatenate([c for c in out_chunks if c.shape[0] > 0], axis=0)
    if mono and out.shape[1] == 2:
        out = out
    return out, ledger


# ------------------------------------------------------------------ glue


def duck(audio, sr: int, times, depth: float = 0.28, tau: float = 0.12):
    """Sidechain: every time in `times` (the low voice's events) pulls
    the mix down and lets it recover — club_lab's duck, the groove
    glue that makes placed hits feel like a band."""
    x = np.asarray(audio, np.float64)
    n = x.shape[0]
    env = np.ones(n)
    t = np.arange(n) / sr
    for tk in times:
        i0 = int(tk * sr)
        i1 = min(n, i0 + int(0.35 * sr))
        if i0 >= n:
            continue
        seg = np.exp(-(t[i0:i1] - tk) / tau)
        env[i0:i1] = np.minimum(env[i0:i1], 1.0 - depth * seg)
    return x * env[:, None] if x.ndim == 2 else x * env


def post_process(audio, sr: int, bounds, seed: int = 0,
                 intensity: float = 1.0):
    """The whole hand: plan against the form, apply, normalize.
    Returns (audio, ledger)."""
    x = np.asarray(audio, np.float64)
    length = x.shape[0] / sr
    rng = np.random.default_rng(seed)
    events = plan_post(rng, bounds, length, intensity)
    out, ledger = apply_post(x, sr, events)
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 1e-9:
        out = out / peak * 0.92
    return out, ledger