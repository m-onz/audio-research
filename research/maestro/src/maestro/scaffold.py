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

Audio is not MIDI: there are no note events to read, so maestro
derives them. Four band envelopes (s1 low / s2 mid / s3 high / s4
air) plus the full-band spectral flux (s0, the master rhythm) are
peak-picked into onset times, and every onset becomes a spike decaying
until the NEXT onset in its stream — the inter-onset interval is
audio's note duration. Three slow weather streams (live_rms,
live_centroid, live_flux) ride along at tick resolution: the spine's
loudness, brightness, and change.

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
           "pick_onsets", "estimate_bpm", "STREAMS", "WEATHER", "SENSES"]

WIN, HOP = 2048, 512                 # STFT sizes (echo's ear, 44.1 kHz)
_BLOCK = 1024                        # frames per pass — memory stays flat
_EPS = 1e-12
_LEAK_GUARD = 1e-4                   # band peak below this x global peak = silence
_LOG_FLOOR = 1e-3                    # log-domain floor: -60 dB of the global
                                     # peak — leakage rises stay invisible

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
    relative to the stream's own scale, with a refractory period.
    The scale is max(95th percentile, quarter of the peak) — the
    quantile alone collapses to zero on sparse trains (a click spine
    is 1% onset frames) and would declare everything novel; the peak
    term keeps sparse material normalized. Scale-invariant,
    deterministic. Returns onset times in seconds."""
    nov = np.asarray(novelty, float)
    if len(nov) < 3:
        return np.zeros(0)
    scale = max(float(np.quantile(nov, 0.95)), 0.25 * float(nov.max()), 1e-12)
    n = nov / scale
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


def estimate_bpm(onsets_s, lo: float = 60.0, hi: float = 200.0
                  ) -> float | None:
    """Tempo from onset times — the tick grid need not be guessed.
    Scan beat periods; score each by how well the inter-onset
    intervals fit: an interval k beats long counts 1/k (long spans
    are strong evidence), an interval that SUBDIVIDES the beat (m
    hits per beat) counts 1/m (hi-hat weather is weak evidence) —
    so beats outrank bars and subdivisions both. Refine on the
    matched multiples' median period. Returns bpm snapped to 0.1,
    or None when there are too few intervals to read. Honest limit:
    IOIs alone cannot resolve tempo octaves (uniform 16ths are
    360bpm quarters to any meter); accents would be needed."""
    on = np.sort(np.asarray(onsets_s, float))
    if len(on) < 4:
        return None
    ioi = np.diff(on)
    ioi = ioi[(ioi >= 0.10) & (ioi <= 2.5)]
    if len(ioi) < 3:
        return None
    Ts = np.arange(60.0 / hi, 60.0 / lo + 1e-9, 0.002)
    scores = np.zeros(len(Ts))
    for i, T in enumerate(Ts):
        r = ioi / T
        k = np.maximum(np.round(r), 1.0)
        hit_long = (np.abs(r - k) <= 0.10) & (r >= 1.0)
        m = np.maximum(np.round(1.0 / r), 2.0)      # m=1 is the k=1 hit
        hit_sub = (np.abs(1.0 / r - m) <= 0.10) & (r < 1.0) & (m >= 2.0)
        scores[i] = float(np.sum(1.0 / k[hit_long])
                          + np.sum(1.0 / m[hit_sub]))
    if scores.max() <= 0.0:
        return None
    # among near-ties prefer the SLOWER period: a beat and its
    # subdivision often explain the iois equally well (alternating
    # quarter/eighth is symmetric evidence) — the beat is the honest
    # reading; only a real preference breaks real ties.
    top = np.where(scores >= 0.95 * scores.max())[0]
    T = float(Ts[top[int(np.argmax(Ts[top]))]])
    # refine on long-span matches: median of ioi/k
    r = ioi / T
    k = np.maximum(np.round(r), 1.0)
    sel = (np.abs(r - k) <= 0.10) & (r >= 1.0)
    if sel.any():
        T = float(np.median(ioi[sel] / k[sel]))
    return round(60.0 / T, 1)


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
    is auditable). The frame-domain measurement (frame_t,
    weather_frames) is retained so the scaffold can be RE-SAMPLED
    onto any tick grid — a Form's non-uniform one — with sample()."""

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
    frame_t: np.ndarray | None = None          # measurement clock
    weather_frames: dict = field(default_factory=dict)
    tick_times: np.ndarray | None = None       # non-uniform grids (forms)

    @property
    def length_s(self) -> float:
        if self.tick_times is not None and len(self.tick_times) > 1:
            tt = self.tick_times
            return float(tt[-1] + (tt[-1] - tt[-2]))
        return self.n_ticks / self.tick_hz

    def sample(self, tick_times, loop: bool = False) -> "Scaffold":
        """Re-sample the senses onto an arbitrary tick-TIME grid —
        a Form's tempo map. loop=True tiles the spine cyclically so a
        short spine can drive a long piece (each loop lands on a
        different tick phase under a changing tempo, so the repeats
        vary for free). Spikes recompute from their onsets; weather
        interpolates the retained frames. Returns a NEW Scaffold."""
        tt = np.asarray(tick_times, np.float64)
        if len(tt) < 4 or np.any(np.diff(tt) <= 0):
            raise ValueError("tick_times must be >=4 and strictly increasing")
        n = len(tt)
        span = float(tt[-1] - tt[0])
        streams, onsets = {}, {}
        for k in STREAMS:
            base = np.asarray(self.onsets.get(k, []), np.float64)
            if loop and len(base) and span > self.spine_length_s:
                reps = int(np.ceil(span / self.spine_length_s)) + 1
                base = (base[None, :]
                        + np.arange(reps)[:, None] * self.spine_length_s
                        ).reshape(-1)
            onsets[k] = [float(t) for t in base if t <= tt[-1]]
            streams[k] = _spikes_at(base, tt, self.params.get(
                "min_decay_s", 0.05), self.params.get("max_decay_s", 2.0))
        weather = {}
        for k in WEATHER:
            src = self.weather_frames[k]
            tq = np.mod(tt - tt[0], self.spine_length_s) if loop else tt
            # the measurement clock starts at WIN/2; keep 0 in range
            tq = np.clip(tq, 0.0, self.frame_t[-1])
            weather[k] = np.interp(tq, self.frame_t, src).astype(np.float32)
        out = Scaffold(path=self.path, sha256=self.sha256, sr=self.sr,
                       spine_length_s=self.spine_length_s, bpm=None,
                       ticks_per_beat=self.ticks_per_beat,
                       tick_hz=float(n / max(span, 1e-9)), n_ticks=n,
                       streams=streams, weather=weather, onsets=onsets,
                       params=dict(self.params), frame_t=self.frame_t,
                       weather_frames=self.weather_frames,
                       tick_times=tt.astype(np.float64))
        return out

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
            if self.tick_times is not None:
                f["t"] = self.tick_times.astype(np.float32)
            else:
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


def _spikes_at(onsets_s, tick_times, min_decay_s: float = 0.05,
               max_decay_s: float = 2.0) -> np.ndarray:
    """spike_train for an ARBITRARY tick-time grid (a form's tempo
    map): the decay of each event still spans its inter-onset
    interval; the grid only decides WHERE we look."""
    tt = np.asarray(tick_times, np.float64)
    v = np.zeros(len(tt), np.float32)
    on = np.sort(np.asarray(onsets_s, np.float64))
    if len(on) == 0:
        return v
    dur = np.clip(np.diff(on, append=on[-1] + max_decay_s),
                  min_decay_s, max_decay_s)
    for t0, d in zip(on, dur):
        m = (tt >= t0) & (tt <= t0 + d)
        if not m.any():
            continue
        v[m] = np.maximum(v[m], 1.0 - (tt[m] - t0) / d)
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def scaffold_from_array(x, sr: int = SR, bpm: float | None = 120.0,
                        ticks_per_beat: int = 4, tick_hz: float | None = None,
                        length_s: float | None = None, min_ioi_s: float = 0.08,
                        min_decay_s: float = 0.05, max_decay_s: float = 2.0,
                        rel_thr: float = 0.30) -> Scaffold:
    """Measure a spine held in memory (tests, live arrays) and sample
    it onto a uniform tick grid. The frame-domain measurement is
    retained — call .sample(tick_times, loop) for forms."""
    if tick_hz is None:
        if bpm is None:
            raise ValueError("give bpm+ticks_per_beat or tick_hz")
        tick_hz = bpm / 60.0 * ticks_per_beat
    spine_len = len(np.atleast_1d(x)) / float(sr)
    total = spine_len if length_s is None else float(length_s)
    n_ticks = max(4, int(np.ceil(total * tick_hz)))

    frame_t, env, rms, cent, flux = _analyze(x, sr)
    gpeak = max(float(e.max()) for e in env.values()) + _EPS

    onsets = {}
    # s0: the master rhythm from full-band spectral flux
    on = pick_onsets(flux, frame_t, min_ioi_s, rel_thr)
    onsets["s0"] = [float(t) for t in on]
    # s1..s4: per-register trains, with a spectral-leakage guard
    for k in BANDS:
        if env[k].max() <= _LEAK_GUARD * gpeak:
            onsets[k] = []
            continue
        e = env[k] + _LOG_FLOOR * gpeak + _EPS
        nov = np.maximum(0.0, np.diff(np.log(e)))
        on = pick_onsets(np.concatenate([[0.0], nov]), frame_t,
                         min_ioi_s, rel_thr)
        onsets[k] = [float(t) for t in on]

    weather_frames = {"live_rms": _robust01(rms), "live_centroid": _robust01(cent),
                      "live_flux": _robust01(flux)}
    # loudness weather is PASSAGE-slow: ~0.75 s of smoothing, then
    # re-anchored between the 10th and 99th percentiles — "how loud
    # is this passage relative to this spine's own dynamic range".
    # Material with no dynamics (< 15% range) reads FLAT ZERO: a bed
    # that never breathes has no loud passages, and gates keyed to
    # live_rms (the kit's openhat) stay honestly shut.
    k = max(3, int(round(0.75 * sr / HOP)) | 1)
    win = np.hanning(k)
    src = weather_frames["live_rms"]
    k2 = k // 2
    padded = np.pad(src, (k2, k2), mode="edge")   # no edge ramps
    smoothed = np.convolve(padded, win / win.sum(), mode="same")[k2:k2 + len(src)]
    lo, hi = np.percentile(smoothed, [10, 99])
    if hi - lo < 0.15 * max(hi, 1e-9):
        weather_frames["live_rms"] = np.zeros_like(smoothed)
    else:
        weather_frames["live_rms"] = np.clip((smoothed - lo) / (hi - lo),
                                             0.0, 1.0)
    scaf = Scaffold(path="", sha256="", sr=int(sr), spine_length_s=spine_len,
                    bpm=bpm, ticks_per_beat=int(ticks_per_beat),
                    tick_hz=float(tick_hz), n_ticks=n_ticks, onsets=onsets,
                    params={"win": WIN, "hop": HOP, "min_ioi_s": min_ioi_s,
                            "min_decay_s": min_decay_s,
                            "max_decay_s": max_decay_s, "rel_thr": rel_thr,
                            "leak_guard": _LEAK_GUARD},
                    frame_t=frame_t, weather_frames=weather_frames)
    # the uniform grid is just sample() with evenly spaced tick times
    return scaf.sample(np.arange(n_ticks) / tick_hz, loop=False)


def load_spine(path, sr: int = SR, **kw) -> Scaffold:
    """Decode any audio file (ffmpeg, like every decoder in this
    ecosystem) and measure it. The file's sha256 travels inside the
    scaffold so every score names the exact spine it grew from."""
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