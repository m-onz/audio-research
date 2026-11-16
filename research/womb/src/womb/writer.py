"""writer: the joint A/V output pipe.

womb generates sound and picture in the same loop, so the MP4 is assembled
in two passes: frames stream to ffmpeg while audio accumulates in memory,
then the silent video is muxed with the audio (stream-copied, no re-encode).
Sync is exact by construction — frame i *is* audio hop i.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["AVWriter"]


class AVWriter:
    def __init__(self, out, width, height, fps, sr=44100, crf=18):
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h, self.fps, self.sr = int(width), int(height), float(fps), int(sr)
        self.hop = int(round(self.sr / self.fps))
        self.crf = int(crf)
        self.audio = np.zeros((0, 2), np.float32)   # always stereo inside
        self._stereo = False                        # ... but mono out if never fed 2D
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="womb-"))
        self._silent = self._tmp_dir / "silent.mp4"
        self._proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.w}x{self.h}", "-r", str(self.fps), "-i", "pipe:0",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(self.crf),
             "-pix_fmt", "yuv420p", str(self._silent)],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, frame: np.ndarray, hop_audio: np.ndarray | None = None) -> None:
        """One frame + its hop of audio (mono (n,) or stereo (n, 2);
        exact hop length; padded/truncated)."""
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if frame.shape != (self.h, self.w, 3):
            raise ValueError(f"frame {frame.shape} != {(self.h, self.w, 3)}")
        self._proc.stdin.write(frame.tobytes())
        self.frames += 1
        if hop_audio is None:
            hop_audio = np.zeros(self.hop, np.float32)
        hop_audio = np.asarray(hop_audio, np.float32)
        if hop_audio.ndim == 2 and hop_audio.shape[1] == 2:
            self._stereo = True
        else:
            hop_audio = np.stack([hop_audio[: self.hop]] * 2, axis=1)
        n = hop_audio.shape[0]
        if n < self.hop:
            hop_audio = np.pad(hop_audio, ((0, self.hop - n), (0, 0)))
        self.audio = np.concatenate(
            [self.audio, hop_audio[: self.hop].astype(np.float32)], axis=0)

    def close(self) -> dict:
        self._proc.stdin.close()
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg video pass exited {rc}")
        # audio pass: buffer -> wav (mono path is column 0: bit-identical)
        wav = self._tmp_dir / "audio.wav"
        if self._stereo:
            inter = np.empty(len(self.audio) * 2, np.float32)
            inter[0::2], inter[1::2] = self.audio[:, 0], self.audio[:, 1]
            pcm = (np.clip(inter, -1, 1) * 32767).astype(np.int16)
            channels = 2
        else:
            pcm = (np.clip(self.audio[:, 0], -1, 1) * 32767).astype(np.int16)
            channels = 1
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le",
             "-ar", str(self.sr), "-ac", str(channels), "-i", "pipe:0", str(wav)],
            input=pcm.tobytes(), check=True,
        )
        # mux pass (video stream-copied)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(self._silent), "-i", str(wav),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", "-movflags", "+faststart", str(self.out)],
            check=True,
        )
        import shutil

        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        return {"path": str(self.out), "frames": self.frames,
                "seconds": self.frames / self.fps}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()