"""scaffold: the spine — existing structure, encoded for a CPPN.

NEAT Drummer (Hoover, Rosario & Stanley, EvoMUSART 2008) feeds a
song's channels into a CPPN so the generated part is a FUNCTION of
them: the scaffold supplies every ounce of temporal structure and the
network only transforms it. Their subtle trick is the input encoding
(their fig. 2): each note becomes a spike that decays LINEARLY over
the note's duration, so the value the network sees at any tick says
where it is WITHIN the note. The decay is a temporal coordinate frame
— patterns can vary across a note's course, eighth notes read
differently from quarters, and 'answers' can live mid-decay. Timing
is kept; pitch is deliberately not (rhythm does not need it).

Audio is not MIDI: there are no note events to read, so prism derives
them. Four band envelopes (s1 low / s2 mid / s3 high / s4 air) plus the
full-band spectral flux (s0, the master rhythm) are peak-picked into
onset times, and every onset becomes a spike decaying until the NEXT
onset in its stream — the inter-onset interval is audio's note
duration. Three slow weather streams (live_rms, live_centroid,
live_flux) ride along at tick resolution: the spine's loudness,
brightness, and change.

Everything is measured once at the STFT frame rate (echo's window
sizes), then sampled onto the tick grid. Deterministic, offline, one
blockwise pass over the file — a 20-minute spine is as happy as a
4-second one.

    spine.wav ──STFT──> band envelopes + flux ──peak-pick──> onsets
                              │                                  │
                              │                    spike_train() (decays)
                              └── weather (rms/centroid/flux) ────┤
                                                                 ▼
                                        feeds(): per-tick senses for score.traverse
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np

from echo.probes import SR, decode

__all__ = ["Scaffold", "load_spine", "scaffold_from_array", "spike_train",
           "pick_onsets", "STREAMS", "WEATHER", "SENSES"]

WIN, HOP = 2048, 512                 # STFT sizes (echo's ear, 44.1 kHz)
_BLOCK = 1024                        # frames per pass — memory stays flat
_EPS = 1e-12
_LEAK_GUARD = 1e-4                   # band peak below this x global peak = silence

# the spike senses: s0 = full-band rhythm (spectral flux), s1..s4 = registers
BANDS = {"s1": (30.0, 200.0), "s2": (200.0, 1500.0),
         "s3": (1500.0, 5000.0), "s4": (5000.0, 17000.0)}
STREAMS = ("s0", "s1", "s2", "s3", "s4")
WEATHER = ("live_rms", "live_centroid", "live_flux")
SENSES = ("spikes", "weather", "position", "time")


# ------------------------------------------------------------------ onsets


def pick_onsets(novelty, frame_t, min_ioi_s: float = 0.08,
                rel_thr: float = 0.30) -> np.ndarray:
    """Peak-pick a novelty function: local maxima above a floor
    relative to the stream's own 95th percentile, with a refractory
    period. Scale-invariant (the floor normalizes), deterministic.
    Returns onset times in seconds."""
    nov = np.asarray(novelty, float)
    if len(nov) < 3:
        return np.zeros(0)
    p95 = float(np.quantile(nov, 0.95))
    if p95 < 1e-10:
        return np.zeros(0)
    n = nov / p95
    loc = (n[1:-1] >= n[:-2]) & (n[1:-1] > n[2:]) & (n[1:-1] >= rel_thr)
    cand = np.where(loc)[0] + 1
    out, last = [], -1e9
    for i in cand:
        t = float(frame_t[i])
        if t - last >= min_ioi_s:
            out.append(t)
            last = t
    return np.array(out)


def spike_train(onsets_s, n_ticks: int, tick_hz: float,
                min_decay_s: float = 0.05, max_decay_s: float = 2.0
                ) -> np.ndarray:
    """The paper's fig. 2, for audio. Each onset is a spike of height
    1.0 decaying LINEARLY to zero over its 'duration' — here the
    distance to the next onset in the same stream (the inter-onset
    interval is audio's note duration), clamped to
    [min_decay_s, max_decay_s]. Sampled at tick centers -> (n_ticks,).

    The value at a tick is where the event is in its course: 1.0 just
    attacked, 0.5 mid-event, ~0 the tail. That is the coordinate frame.
    """
    v = np.zeros(n_ticks, np.float32)
    on = np.sort(np.asarray(onsets_s, float))
    if len(on) == 0 or tick_hz <= 0 or n_ticks <= 0:
        return v
    dur = np.diff(on, append=on[-1] + max_decay_s)
    dur = np.clip(dur, min_decay_s, max_decay_s)
    ticks = np.arange(n_ticks) / tick_hz
    for t0, d in zip(on, dur):
        m = (ticks >= t0) & (ticks <= t0 + d)
        if not m.any():
            continue
        v[m] = np.maximum(v[m], 1.0 - (ticks[m] - t0) / d)
    return np.clip(v, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ analysis


def _analyze(x, sr: int):
    """One blockwise STFT pass -> frame times, per-band envelopes,
    weather (rms, centroid, flux). Float64 scalars per frame only —
    the magnitude matrix never materializes whole."""
    mid = x.mean(axis=1) if x.ndim == 2 else np.asarray(x, float)
    n = len(mid)
    T = max(4, 1 + max(0, n - WIN) // HOP)
    freqs = np.fft.rfftfreq(WIN, 1.0 / sr)
    hann = np.hanning(WIN)
    band_sel = {k: (freqs >= lo) & (freqs < hi) for k, (lo, hi) in BANDS.items()}
    env = {k: np.zeros(T) for k in BANDS}
    flux = np.zeros(T)
    rms = np.zeros(T)
    cent = np.zeros(T)
    prev_mag = None
    for s in range(0, T, _BLOCK):
        want = min(_BLOCK, T - s)
        s0 = s * HOP
        s1 = s * HOP + (want - 1) * HOP + WIN
        seg = mid[s0:s1]
        if len(seg) < WIN:
            seg = np.pad(seg, (0, WIN - len(seg)))
        fr = np.lib.stride_tricks.sliding_window_view(seg, WIN)[::HOP]
        if len(fr) > want:
            fr = fr[:want]
        stop = s + len(fr)
        mag = np.abs(np.fft.rfft(fr * hann[None, :], axis=1)) + _EPS
        ref = mag[:1] if prev_mag is None else prev_mag[None, :]
        d = np.vstack([ref, mag[:-1]])
        flux[s: stop] = np.maximum(0.0, mag - d).sum(axis=1)
        prev_mag = mag[-1]
        rms[s: stop] = np.sqrt((fr ** 2).mean(axis=1))
        tot = mag.sum(axis=1) + _EPS
        cent[s: stop] = (mag @ freqs) / tot
        for k, sel in band_sel.items():
            env[k][s: stop] = mag[:, sel].mean(axis=1)
    frame_t = (np.arange(T) * HOP + WIN / 2) / sr
    return frame_t, env, rms, cent, flux


def _robust01(arr) -> np.ndarray:
    """Min-max on [1st, 99th] percentile -> [0, 1]; flat input -> zeros."""
    a = np.asarray(arr, float)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


# ------------------------------------------------------------------ scaffold


@dataclass
class Scaffold:
    """A spine, measured. streams: per-tick spike trains (the senses
    s0..s4); weather: live_rms/live_centroid/live_flux on the same
    grid; onsets: the detected event times (provenance — every spike
    is auditable)."""

    path: str
    sha256: str
    sr: int
    spine_length_s: float
    bpm: float | None
    ticks_per_beat: int
    tick_hz: float
    n_ticks: int
    streams: dict = field(default_factory=dict)
    weather: dict = field(default_factory=dict)
    onsets: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def length_s(self) -> float:
        return self.n_ticks / self.tick_hz

    def feeds(self, senses=("spikes", "weather")) -> dict:
        """The organism's senses as evaluate() feeds, all (n_ticks,).
        spikes: the s0..s4 trains (the scaffold — default).
        weather: rms/centroid/flux (default).
        position: x = place in the piece [0, 1] — impure (structure
          not inherited from the spine), opt in deliberately.
        time: t = tick time in seconds — impure, same warning.
        """
        senses = tuple(senses)
        bad = [s for s in senses if s not in SENSES]
        if bad:
            raise ValueError(f"unknown senses {bad}; valid: {SENSES}")
        f: dict = {"b": np.float32(1.0), "x": np.float32(0.0),
                   "y": np.float32(0.0), "d": np.float32(0.0),
                   "t": np.float32(0.0)}
        if "spikes" in senses:
            f.update(self.streams)
        if "weather" in senses:
            f.update(self.weather)
        if "position" in senses:
            f["x"] = np.linspace(0.0, 1.0, self.n_ticks, dtype=np.float32)
        if "time" in senses:
            f["t"] = (np.arange(self.n_ticks) / self.tick_hz).astype(np.float32)
        return f

    def info(self) -> str:
        L = [f"spine   : {self.path or '<array>'}",
             f"grid    : {self.n_ticks} ticks @ {self.tick_hz:g} Hz "
             f"({self.length_s:.1f}s"
             + (f", bpm {self.bpm:g} x {self.ticks_per_beat}" if self.bpm
                else "") + ")",
             "onsets  : " + ", ".join(
                 f"{k}={len(self.onsets.get(k, []))}" for k in STREAMS),
             f"weather : rms {self.weather['live_rms'].mean():.2f} "
             f"centroid {self.weather['live_centroid'].mean():.2f} "
             f"flux {self.weather['live_flux'].mean():.2f} (means)"]
        return "\n".join(L)


def scaffold_from_array(x, sr: int = SR, bpm: float | None = 120.0,
                        ticks_per_beat: int = 4, tick_hz: float | None = None,
                        length_s: float | None = None, min_ioi_s: float = 0.08,
                        min_decay_s: float = 0.05, max_decay_s: float = 2.0,
                        rel_thr: float = 0.30) -> Scaffold:
    """Measure a spine held in memory (tests, live arrays)."""
    if tick_hz is None:
        if bpm is None:
            raise ValueError("give bpm+ticks_per_beat or tick_hz")
        tick_hz = bpm / 60.0 * ticks_per_beat
    spine_len = len(np.atleast_1d(x)) / float(sr)
    total = spine_len if length_s is None else float(length_s)
    n_ticks = max(4, int(np.ceil(total * tick_hz)))

    frame_t, env, rms, cent, flux = _analyze(x, sr)
    gpeak = max(float(e.max()) for e in env.values()) + _EPS

    streams, onsets = {}, {}
    # s0: the master rhythm from full-band spectral flux
    on = pick_onsets(flux, frame_t, min_ioi_s, rel_thr)
    onsets["s0"] = [float(t) for t in on]
    streams["s0"] = spike_train(on, n_ticks, tick_hz, min_decay_s, max_decay_s)
    # s1..s4: per-register trains, with a spectral-leakage guard
    for k in BANDS:
        if env[k].max() <= _LEAK_GUARD * gpeak:
            onsets[k], streams[k] = [], np.zeros(n_ticks, np.float32)
            continue
        e = env[k] + 1e-6 * gpeak + _EPS
        nov = np.maximum(0.0, np.diff(np.log(e)))
        on = pick_onsets(np.concatenate([[0.0], nov]), frame_t,
                         min_ioi_s, rel_thr)
        onsets[k] = [float(t) for t in on]
        streams[k] = spike_train(on, n_ticks, tick_hz, min_decay_s, max_decay_s)

    # weather on the tick grid
    tick_t = np.arange(n_ticks) / tick_hz
    weather = {
        "live_rms": np.interp(tick_t, frame_t, _robust01(rms)).astype(np.float32),
        "live_centroid": np.interp(tick_t, frame_t,
                                   _robust01(cent)).astype(np.float32),
        "live_flux": np.interp(tick_t, frame_t,
                               _robust01(flux)).astype(np.float32),
    }
    return Scaffold(path="", sha256="", sr=int(sr), spine_length_s=spine_len,
                    bpm=bpm, ticks_per_beat=int(ticks_per_beat),
                    tick_hz=float(tick_hz), n_ticks=n_ticks,
                    streams=streams, weather=weather, onsets=onsets,
                    params={"win": WIN, "hop": HOP, "min_ioi_s": min_ioi_s,
                            "min_decay_s": min_decay_s,
                            "max_decay_s": max_decay_s, "rel_thr": rel_thr,
                            "leak_guard": _LEAK_GUARD})


def load_spine(path, sr: int = SR, **kw) -> Scaffold:
    """Decode any audio file (ffmpeg, like every decoder here) and
    measure it. The file's sha256 travels inside the scaffold so every
    score names the exact spine it grew from."""
    path = Path(path)
    x = decode(path, sr=sr)
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    scaf = scaffold_from_array(x, sr=sr, **kw)
    scaf.path = str(path)
    scaf.sha256 = sha.hexdigest()
    return scaf