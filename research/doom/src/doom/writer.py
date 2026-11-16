"""writer: the ffmpeg pipe.

One ffmpeg process, two inputs: raw RGB24 frames on stdin and the audio
file from disk. Video is generated for exactly ceil(duration * fps) frames
and `-shortest` lets the audio define the container length — frame-accurate
sync with no post-mux step.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

__all__ = ["FFmpegWriter", "verify_mp4"]


class FFmpegWriter:
    """Pipe RGB frames; mux audio in the same pass."""

    def __init__(self, out, width, height, fps, audio_path,
                 crf: int = 18, preset: str = "medium", audio_codec: str = "aac",
                 audio_start: float = 0.0):
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h, self.fps = int(width), int(height), float(fps)
        audio_args = ["-i", str(audio_path)]
        if audio_start > 0:
            audio_args = ["-ss", f"{audio_start:.6f}"] + audio_args  # input seek
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            # input 0: raw frames from stdin
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.w}x{self.h}", "-r", str(fps), "-i", "pipe:0",
            # input 1: the audio, played from the chunk start
            *audio_args,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", audio_codec, "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(self.out),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        """frame: (H, W, 3) uint8 (or float 0..1, auto-converted)."""
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        if frame.shape != (self.h, self.w, 3):
            raise ValueError(f"frame {frame.shape} != expected {(self.h, self.w, 3)}")
        self.proc.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> dict:
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with {rc}")
        return {"path": str(self.out), "frames": self.frames,
                "seconds": self.frames / self.fps}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.wait()


def verify_mp4(path) -> dict:
    """Return stream summary via ffprobe (duration, codecs, size)."""
    import json

    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    streams = {}
    for s in info["streams"]:
        streams[s["codec_type"]] = {
            "codec": s["codec_name"],
            "duration": float(s.get("duration", 0) or 0),
            "nb_frames": s.get("nb_frames"),
        }
    return {
        "path": str(path),
        "duration": float(info["format"]["duration"]),
        "size_bytes": int(info["format"]["size"]),
        "streams": streams,
    }