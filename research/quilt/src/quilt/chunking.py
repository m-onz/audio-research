"""chunking: the scissors — where the corpus is cut.

A strategy is a pure function (audio, sr) -> [(t0, t1), ...] in
seconds: non-overlapping, sorted, every piece at least MIN_DUR long.
The strategies are deliberately different *views* of the same corpus:

    fixed    uniform windows of FIXED_DUR seconds — the metronome cut;
             treats a drone and a breakbeat identically, which is the
             point: one corpus, many scissors, the bank tags which
             scissor cut what.
    onset    cut at detected onsets (librosa, backtracked) — event-
             aligned; a breakbeat becomes its hits, a drone one piece.
    silence  cut where the signal drops SIL_GATE_DB below its own
             peak for SIL_MIN_SIL — the ear's most ancient scissor.
    whole    the file as one piece (only files <= WHOLE_MAX) — the
             long-form view; many corpus files are already pieces.

All deterministic, pure numpy except onset (librosa). Boundaries are
suggestions the bank believes: navigation only ever sees the encoded
piece, never the boundary.
"""

from __future__ import annotations

import numpy as np

__all__ = ["chunk", "chunk_fixed", "chunk_onset", "chunk_silence",
           "chunk_whole", "STRATEGIES", "DEFAULT_STRATEGIES",
           "MIN_DUR", "MAX_SEG", "FIXED_DUR", "WHOLE_MAX",
           "ONSET_MIN", "SIL_GATE_DB", "SIL_MIN_SIL", "SIL_MIN_SEG"]

# ------------------------------------------------------------------ constants
MIN_DUR = 0.12      # s — the smallest piece worth encoding (echo's ear
                    #      needs a handful of frames; shorter is dust)
MAX_SEG = 8.0       # s — any detected segment longer gets subdivided:
                    #      no single piece should dominate a 3-min quilt
FIXED_DUR = 2.0     # s — the fixed scissor's window
WHOLE_MAX = 20.0    # s — whole-file pieces only for files this short
ONSET_MIN = 0.15    # s — min piece under the onset scissor (else merge)
SIL_GATE_DB = -40.0  # dB below the file's own peak = silence
SIL_MIN_SIL = 0.12  # s — quiet must last this long to count as a cut
SIL_MIN_SEG = 0.20   # s — min piece under the silence scissor (else merge)


# ------------------------------------------------------------------ helpers

def _mono(audio) -> np.ndarray:
    x = np.asarray(audio, np.float64)
    if x.ndim == 2:
        return x.mean(axis=1)
    return x


def _merge_short(segs: list, min_seg: float) -> list:
    """Absorb pieces shorter than min_seg into their predecessor (a
    boundary is deleted); a short head merges into its successor."""
    if not segs:
        return []
    out = [[float(segs[0][0]), float(segs[0][1])]]
    for t0, t1 in segs[1:]:
        if t1 - t0 < min_seg and out:
            out[-1][1] = float(t1)
        else:
            out.append([float(t0), float(t1)])
    if len(out) > 1 and out[0][1] - out[0][0] < min_seg:
        out[1][0] = out[0][0]
        out.pop(0)
    return [tuple(s) for s in out]


def _subdivide(segs: list, max_seg: float) -> list:
    """Pieces longer than max_seg become equal-width cuts."""
    out = []
    for t0, t1 in segs:
        d = t1 - t0
        if d <= max_seg:
            out.append((t0, t1))
            continue
        n = int(np.ceil(d / max_seg))
        w = d / n
        out.extend((t0 + k * w, t0 + (k + 1) * w) for k in range(n))
    return out


def _keep(segs: list) -> list:
    return [(float(a), float(b)) for a, b in segs if b - a >= MIN_DUR]


# ------------------------------------------------------------------ strategies

def chunk_fixed(audio, sr: int, dur: float = FIXED_DUR) -> list:
    """Uniform windows; a partial tail is kept if it clears MIN_DUR."""
    total = len(audio) / sr
    out = []
    t = 0.0
    while t < total - 1e-9:
        t1 = min(t + dur, total)
        out.append((t, t1))
        t += dur
    return _keep(out)


def chunk_whole(audio, sr: int, max_dur: float = WHOLE_MAX) -> list:
    """The file as one piece — only when it is already piece-sized."""
    total = len(audio) / sr
    return [(0.0, total)] if MIN_DUR <= total <= max_dur else []


def chunk_onset(audio, sr: int, min_seg: float = ONSET_MIN,
                max_seg: float = MAX_SEG) -> list:
    """Cut at detected onsets (librosa, backtracked to the attack's
    valley). The event-aligned scissor: hits stay hits."""
    import librosa                                   # lazy: only this scissor
    mono = _mono(audio).astype(np.float32)
    total = len(mono) / sr
    if total < MIN_DUR:
        return []
    on = librosa.onset.onset_detect(
        y=mono, sr=sr, units="time", backtrack=True, hop_length=512)
    bounds = [0.0] + sorted(float(t) for t in on
                            if 0.05 < t < total - 0.05) + [total]
    segs = list(zip(bounds[:-1], bounds[1:]))
    segs = _merge_short(segs, min_seg)
    segs = _subdivide(segs, max_seg)
    return _keep(segs)


def chunk_silence(audio, sr: int, gate_db: float = SIL_GATE_DB,
                  min_sil: float = SIL_MIN_SIL, min_seg: float = SIL_MIN_SEG,
                  max_seg: float = MAX_SEG) -> list:
    """Cut where the loud falls silent: frame RMS, a gate at
    gate_db below the file's own peak, splits where quiet outlasts
    min_sil. Pure numpy."""
    mono = _mono(audio)
    total = len(mono) / sr
    win, hop = 1024, 512
    if total < MIN_DUR:
        return []
    if len(mono) <= win:                              # too short to gate
        return [(0.0, total)]
    n_frames = 1 + (len(mono) - win) // hop
    idx = np.arange(n_frames)[:, None] * hop + np.arange(win)[None, :]
    rms = np.sqrt((mono[idx] ** 2).mean(axis=1))
    db = 20.0 * np.log10(rms + 1e-12)
    peak = db.max()
    if peak < -60.0:                                  # the file is silence
        return []
    loud = db > (peak + gate_db)
    centers = (np.arange(n_frames) * hop + win / 2.0) / sr
    runs, t = [], None
    for f in range(n_frames):
        if loud[f] and t is None:
            t = f
        elif not loud[f] and t is not None:
            runs.append((centers[t], centers[f - 1]))
            t = None
    if t is not None:
        runs.append((centers[t], centers[n_frames - 1]))
    merged = []
    for s in runs:                                    # bridges over short
        if merged and s[0] - merged[-1][1] < min_sil:  # quiet gaps
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append((float(s[0]), float(s[1])))
    merged = [(max(0.0, a), min(total, b)) for a, b in merged]
    merged = _merge_short(merged, min_seg)
    merged = _subdivide(merged, max_seg)
    return _keep(merged)


# ------------------------------------------------------------------ registry

STRATEGIES = {
    "fixed": {
        "fn": chunk_fixed,
        "doc": f"uniform {FIXED_DUR:.0f}s windows — the metronome cut",
        "needs": None},
    "onset": {
        "fn": chunk_onset,
        "doc": "cut at detected onsets (librosa, backtracked) — event-aligned",
        "needs": "librosa"},
    "silence": {
        "fn": chunk_silence,
        "doc": (f"cut where loudness drops {SIL_GATE_DB:.0f} dB below the "
                f"file's own peak for {SIL_MIN_SIL * 1000:.0f} ms"),
        "needs": None},
    "whole": {
        "fn": chunk_whole,
        "doc": f"the whole file as one piece (files <= {WHOLE_MAX:.0f}s)",
        "needs": None},
}
DEFAULT_STRATEGIES = ("fixed", "onset", "silence", "whole")


def chunk(audio, sr: int, strategy: str, **kw) -> list:
    """audio -> [(t0, t1), ...] under the named scissor."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r} — "
                         f"one of {sorted(STRATEGIES)}")
    return STRATEGIES[strategy]["fn"](audio, sr, **kw)