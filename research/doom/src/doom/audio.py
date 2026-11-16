"""audio: the sync clock.

doom's synchronization story is structural, not tuned: before any frames are
rendered, the whole audio file is decoded (streamed through ffmpeg in blocks)
and reduced to *per-video-frame features* — one dict per output frame:

    rms, bass, mid, treble, centroid, flux, onset (0/1), and the raw
    lowpassed mono samples that "live" inside that frame's time window.

Scenes and FX read `feats.rms` etc. for frame i; the frame is rendered for
time t = i / fps, and the audio a viewer hears at t is exactly the audio
those features describe. Sync is exact by construction — there is no
"reacting" latency because nothing reacts: both streams are generated from
the same clock.

The mono waveform is also kept decimated (~8 kHz) so oscilloscope/spectrogram
scenes can draw the actual samples.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["AudioFeatures", "analyze_audio", "probe_duration"]

EPS = 1e-9
DECIM_SR = 8000  # decimated mono rate kept for waveform-drawing scenes


@dataclass
class AudioFeatures:
    fps: float
    sr: int
    duration: float
    n_frames: int
    rms: np.ndarray          # (n_frames,) loudness 0..1-ish
    bass: np.ndarray         # low-band energy share 0..1
    mid: np.ndarray
    treble: np.ndarray
    centroid: np.ndarray     # Hz
    flux: np.ndarray         # spectral flux 0..1-ish
    onset: np.ndarray        # (n_frames,) bool — detected transients
    wave: np.ndarray         # decimated mono waveform @ DECIM_SR
    wave_sr: int = DECIM_SR
    t0: float = 0.0          # absolute time of frame 0 (windowed renders)

    def wave_window(self, i: int, seconds: float = 1.0) -> np.ndarray:
        """The decimated mono samples heard during output frame i."""
        t0 = max(0.0, self.t0 + i / self.fps)
        t1 = t0 + seconds
        a, b = int(t0 * self.wave_sr), min(len(self.wave), int(t1 * self.wave_sr))
        return self.wave[a:b] if b > a else np.zeros(8, dtype=np.float32)

    def window(self, start: float, length: float) -> "AudioFeatures":
        """A feature VIEW for [start, start+length) — for chunk renders."""
        i0 = int(round(start * self.fps))
        n = int(min(self.n_frames - i0, np.ceil(length * self.fps)))
        n = max(0, n)
        return AudioFeatures(
            fps=self.fps, sr=self.sr, duration=length, n_frames=n,
            rms=self.rms[i0: i0 + n], bass=self.bass[i0: i0 + n],
            mid=self.mid[i0: i0 + n], treble=self.treble[i0: i0 + n],
            centroid=self.centroid[i0: i0 + n], flux=self.flux[i0: i0 + n],
            onset=self.onset[i0: i0 + n], wave=self.wave, wave_sr=self.wave_sr,
            t0=start,
        )


def probe_duration(path) -> float:
    """Audio/video duration via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _decode_mono_blocks(path, block_seconds: float = 10.0):
    """Stream-decode any audio file to float32 mono at native rate via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", "48000", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    sr = 48000
    block_bytes = int(block_seconds * sr * 4)
    while True:
        raw = proc.stdout.read(block_bytes)
        if not raw:
            break
        yield np.frombuffer(raw, dtype=np.float32)
    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}")


def analyze_audio(path, fps: float = 24.0) -> AudioFeatures:
    """Decode `path` and reduce it to per-frame features at `fps`.

    Streams in 10 s blocks — hour-long files analyze in one pass without
    holding the decode in memory (features only).
    """
    path = Path(path)
    duration = probe_duration(path)
    n_frames = max(1, int(np.ceil(duration * fps)))

    hop = 48000 / fps  # samples per output frame
    win = int(hop)
    nfft = 2048

    acc = {k: [] for k in ("rms", "bass", "mid", "treble", "centroid", "flux")}
    wave_chunks = []
    prev_spec = None
    carry = np.zeros(0, dtype=np.float32)
    wave_carry = np.zeros(0, dtype=np.float32)
    dec = int(48000 / DECIM_SR)
    frame_i = 0
    hop_f = float(hop)

    for block in _decode_mono_blocks(path):
        # keep a decimated copy for waveform scenes (avg-decimation with a
        # carry: ffmpeg pipe reads are not guaranteed block-aligned)
        wave_carry = np.concatenate([wave_carry, block]) if len(wave_carry) else block
        use = len(wave_carry) // dec * dec
        if use:
            wave_chunks.append(wave_carry[:use].reshape(-1, dec).mean(axis=1).astype(np.float32))
            wave_carry = wave_carry[use:]

        carry = np.concatenate([carry, block]) if len(carry) else block
        # process complete frames worth of samples
        end = int((len(carry) - win) / hop_f)
        for k in range(end):
            seg = carry[int(k * hop_f): int(k * hop_f) + win]
            prev_spec = _push_features(seg, acc, prev_spec, nfft)
            frame_i += 1
        keep = int(end * hop_f) if end > 0 else 0
        carry = carry[keep:]

    if len(carry) >= win // 2:  # trailing partial frame
        seg = carry[:win]
        if len(seg) < win:
            seg = np.concatenate([seg, np.zeros(win - len(seg), np.float32)])
        _push_features(seg, acc, prev_spec, nfft)
        frame_i += 1
    # pad features to n_frames (edge-replicate) — audio may be shorter
    def padded(k):
        a = np.asarray(acc[k], dtype=np.float64)
        if len(a) == 0:
            return np.zeros(n_frames)
        if len(a) < n_frames:
            a = np.concatenate([a, np.full(n_frames - len(a), a[-1])])
        return a[:n_frames]

    rms = padded("rms")
    bass = padded("bass")
    mid = padded("mid")
    treble = padded("treble")
    centroid = padded("centroid")
    flux = padded("flux")

    # onset detection: locally adaptive peak-picking on spectral flux.
    # The threshold tracks a moving +/-1 s window (median + half the local
    # 90th-percentile spread), so quiet click passages and loud noisy ones
    # both detect their own transients.
    k = max(3, int(fps))
    thr = np.zeros(n_frames)
    for j in range(n_frames):
        w = flux[max(0, j - k): j + k + 1]
        m = np.median(w)
        thr[j] = m + 0.5 * max(0.0, np.percentile(w, 90) - m) + 1e-6
    onset = (flux > thr) & (flux > np.roll(flux, 1)) & (flux >= np.roll(flux, -1))
    onset[0] = onset[-1] = False
    # enforce minimum spacing of 2 frames
    for j in np.nonzero(onset)[0]:
        if onset[j]:
            onset[max(0, j - 2): j] = False

    wave = np.concatenate(wave_chunks) if wave_chunks else np.zeros(0, np.float32)

    return AudioFeatures(
        fps=fps, sr=48000, duration=duration, n_frames=n_frames,
        rms=rms.astype(np.float32), bass=bass.astype(np.float32),
        mid=mid.astype(np.float32), treble=treble.astype(np.float32),
        centroid=centroid.astype(np.float32), flux=flux.astype(np.float32),
        onset=onset, wave=wave,
    )


def _push_features(seg, acc, prev_spec, nfft):
    """Append one frame's features; returns the sqrt-spectrum for flux."""
    seg = seg.astype(np.float64)
    rms = float(np.sqrt(np.mean(seg * seg)) + EPS)
    w = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * w, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / 48000)
    tot = spec.sum() + EPS
    centroid = float((freqs * spec).sum() / tot)
    b = float(spec[freqs < 200].sum() / tot)
    m = float(spec[(freqs >= 200) & (freqs < 2000)].sum() / tot)
    t = float(spec[freqs >= 2000].sum() / tot)
    sq = np.sqrt(spec)
    if prev_spec is not None:
        flux = float(np.mean(np.abs(sq[: len(prev_spec)] - prev_spec)))
    else:
        flux = 0.0
    acc["rms"].append(min(1.0, rms * 4.0))
    acc["bass"].append(b)
    acc["mid"].append(m)
    acc["treble"].append(t)
    acc["centroid"].append(centroid)
    acc["flux"].append(flux)
    return sq