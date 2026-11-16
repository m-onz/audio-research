"""sonify: hearing video (the inverse channel — womb's flagship).

doom's `spectro` scene turns audio into a scrolling spectrogram; `spectra`
here turns frames *back* into sound. doom's `cloud` scatters audio grains in
time; `grains` scatters corpus grains where the *video* moves. loom leaves
silence where nothing happens; `tones` leaves silence where the video is
still. The three modes are the mirrors of the three libraries' gestures —
that symmetry is the point.

All modes are pure numpy + ffmpeg, deterministic from `seed`, fully offline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .video import decode_luma

__all__ = ["sonify", "SONIFY_MODES"]

SONIFY_MODES = ("spectra", "grains", "tones")
SR = 44100


# ------------------------------------------------------------------ shared

def _load_corpus_audio(corpus, sr: int = SR) -> np.ndarray:
    """Decode any audio file to mono float32 (the grain source)."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(corpus),
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32)


# ------------------------------------------------------------------ mode 1: spectra

def _sonify_spectra(path, seed, length=None, partials=256, gamma=1.6, sr=SR):
    """Frames as spectrogram columns: log-spaced partial bank, additive
    synthesis with phase continuation across frames (no clicks)."""
    rng = np.random.default_rng(seed)
    frames, fps = decode_luma(path, scale_w=180, scale_h=90)
    n = len(frames)
    if n == 0:
        raise ValueError(f"no frames decoded from {path}")
    if length is not None:
        n_keep = max(1, int(length * fps))
        frames = frames[:n_keep]
        n = len(frames)
    hop = int(round(sr / fps))

    # log-spaced partial frequencies 30 Hz .. 16 kHz
    freqs = np.geomspace(30.0, 16000.0, partials)
    # height fraction per partial: 0 = bottom of frame = low frequency
    heights = np.linspace(0.0, 1.0, partials)
    row_axis = np.linspace(0.0, 1.0, frames.shape[1])

    phase = rng.uniform(0, 2 * np.pi, partials)
    out = np.zeros((n - 1) * hop + sr // 4, dtype=np.float32)
    t_frame = np.arange(hop) / sr
    for i in range(n):
        # frame rows -> partial magnitudes (bottom-first, matching heights)
        rows = frames[i].mean(axis=1)[::-1]
        m = np.clip(np.interp(heights, row_axis, rows), 0.0, 1.0)
        m = (m ** gamma) * (24.0 / partials)
        phase = phase + 2 * np.pi * freqs * (hop / sr)
        seg = (np.sin(phase[:, None] + 2 * np.pi * freqs[:, None] * t_frame[None, :])
               * m[:, None]).sum(axis=0)
        out[i * hop: (i + 1) * hop] = seg.astype(np.float32)
    return _normalize(out), sr


# ------------------------------------------------------------------ mode 2: grains

def _sonify_grains(path, seed, corpus, grain_s=0.14, length=None, sr=SR):
    """Motion-gated granular resynthesis: the video's frame-difference
    energy scatters corpus grains — doom's cloud arranger, in audio."""
    rng = np.random.default_rng(seed)
    frames, fps = decode_luma(path, scale_w=120, scale_h=68)
    if length is not None:
        frames = frames[: max(1, int(length * fps))]
    n = len(frames)
    src = _load_corpus_audio(corpus, sr)
    if len(src) < sr // 2:
        raise ValueError("corpus audio too short for grain synthesis")

    motion = np.zeros(n)
    if n > 1:
        motion = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
        motion = np.concatenate([[motion[0]], motion])
    # motion curve -> grain density per frame (poisson, like doom's cloud)
    rate = np.clip(motion * 60.0, 0.0, 30.0)  # grains/frame at full motion

    grain_n = int(grain_s * sr)
    win = np.hanning(grain_n).astype(np.float32)
    total = int(n / fps * sr) + sr // 2
    out = np.zeros(total, dtype=np.float32)
    starts = []
    for i in range(n):
        k = rng.poisson(max(0.02, rate[i]))
        for _ in range(k):
            g0 = rng.integers(0, len(src) - grain_n)
            grain = src[g0: g0 + grain_n] * win
            at = int(i / fps * sr) + rng.integers(0, max(1, int(sr / fps) - grain_n))
            if at + grain_n <= total:
                out[at: at + grain_n] += grain * (0.4 + 0.8 * motion[i])
                starts.append(at)
    return _normalize(out), sr


# ------------------------------------------------------------------ mode 3: tones

def _sonify_tones(path, seed, bank=24, length=None, sr=SR):
    """Block-displacement drone: coarse horizontal flow of the video drives
    detune, panning and level of a sustained oscillator bank. Stillness is
    silence (loom's gap philosophy, video-driven)."""
    rng = np.random.default_rng(seed)
    frames, fps = decode_luma(path, scale_w=160, scale_h=90)
    if length is not None:
        frames = frames[: max(1, int(length * fps))]
    n = len(frames)

    # coarse block flow (vectorized over blocks and shifts), every 2nd frame
    bs, shifts = 15, np.arange(-5, 6)
    H, W = frames.shape[1:]
    flow_mag = np.zeros(n)
    flow_dir = np.zeros(n)
    for i in range(2, n, 2):
        a, b = frames[i - 2], frames[i]
        best_d = np.full(((H - bs) // bs, (W - bs) // bs), 1e9)
        best_s = np.zeros_like(best_d)
        for s in shifts:
            rolled = np.roll(a, s, axis=1)
            for iy in range(best_d.shape[0]):
                for ix in range(best_d.shape[1]):
                    y0, x0 = iy * bs, ix * bs
                    d = np.abs(b[y0:y0 + bs, x0:x0 + bs] - rolled[y0:y0 + bs, x0:x0 + bs]).mean()
                    if d < best_d[iy, ix]:
                        best_d[iy, ix], best_s[iy, ix] = d, s
        flow_mag[i - 1: i + 1] = np.abs(best_s).mean()
        flow_dir[i - 1: i + 1] = best_s.mean()
    # luma -> amplitude so dark scenes are quiet too
    luma = frames.mean(axis=(1, 2))
    level = np.clip((flow_mag / 3.0) * (0.3 + 0.7 * luma), 0, 1)

    # oscillator bank: pentatonic-ish partials, seeded
    scale = np.array([1, 9 / 8, 5 / 4, 3 / 2, 5 / 3, 2])
    base = 55.0 * (2 ** rng.integers(0, 3))
    freqs = base * scale[rng.integers(0, len(scale), bank)]

    hop = int(round(sr / fps))
    total = (n - 1) * hop
    out = np.zeros(total, dtype=np.float64)
    phase = rng.uniform(0, 2 * np.pi, bank)
    t_frame = np.arange(hop) / sr
    for i in range(1, n):
        detune = 2 ** (flow_dir[i] / 24.0)     # flow direction -> pitch drift
        amp = level[i]
        f = freqs * detune
        phase = phase + 2 * np.pi * f * (hop / sr)
        wave = (np.sin(phase[:, None] + 2 * np.pi * f[:, None] * t_frame[None, :])).sum(axis=0)
        seg = wave / bank * amp
        out[(i - 1) * hop: i * hop] += seg
    return _normalize(out.astype(np.float32)), sr


# ------------------------------------------------------------------ api

def _normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.abs(x).max()) if x.size else 0.0
    return x if peak < 1e-6 else (x / peak * 0.9).astype(np.float32)


def sonify(video, mode: str = "spectra", seed: int = 0,
           corpus=None, length: float | None = None, out=None):
    """Hear a video. mode: 'spectra' (frames->partials), 'grains' (motion
    scatters corpus grains — requires `corpus` audio), 'tones' (flow drone).

    Returns (audio float32 mono, sr). Writes a WAV if `out` is given.
    """
    mode = mode.lower()
    if mode not in SONIFY_MODES:
        raise ValueError(f"unknown sonify mode {mode!r}; have {SONIFY_MODES}")
    if mode == "grains":
        if corpus is None:
            raise ValueError("grains mode needs corpus= audio (the grain source)")
        audio, sr = _sonify_grains(video, seed, corpus, length=length)
    elif mode == "tones":
        audio, sr = _sonify_tones(video, seed, length=length)
    else:
        audio, sr = _sonify_spectra(video, seed, length=length)
    if out:
        write_wav(out, audio, sr)
    return audio, sr


def write_wav(path, audio: np.ndarray, sr: int) -> str:
    """Write float audio (mono (n,) or stereo (n, 2)) as 16-bit WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, np.float32)
    if x.ndim == 2 and x.shape[1] == 2:
        inter = np.empty(len(x) * 2, np.float32)
        inter[0::2], inter[1::2] = x[:, 0], x[:, 1]
        x, channels = inter, 2
    else:
        channels = 1
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le",
         "-ar", str(sr), "-ac", str(channels), "-i", "pipe:0", str(path)],
        input=pcm.tobytes(), check=True,
    )
    return str(path)