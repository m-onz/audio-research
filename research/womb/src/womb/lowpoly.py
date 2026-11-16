"""lowpoly: the PS2 reprocessing pass — womb's visual mutation layer.

Every frame becomes a field of jittered quads: vertex-snap wobble, flat
per-cell shading, RGB555 ordered dither, horizontal tears. The pass is
part of womb's *evolving genome* — novelty search mutates its parameters
along with the organ's — and it is audio-reactive when used in a piece:
the current hop's energy scales the wobble and tear.

    lowpoly_frame(frame, grid=(24, 42), jitter=2.0, tear=0.2, levels=16,
                  energy=0.5, rng=...)

    womb reprocess any_video.mp4 -o out.mp4     # re-lens existing artifacts
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from doom.palettes import psx_quantize

__all__ = ["lowpoly_frame", "reprocess", "LOWPOLY_DEFAULTS"]

LOWPOLY_DEFAULTS = dict(grid=(24, 42), jitter=2.0, tear=0.15, levels=16, edges=0.35)


def lowpoly_frame(frame: np.ndarray, grid=(24, 42), jitter=2.0, tear=0.15,
                  levels=16, edges=0.35, energy: float = 0.5,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Reprocess one frame into flat-shaded, vertex-jittered quads.

    energy (0..1, typically the current audio hop's rms) scales the
    wobble and the tear probability — loud = unstable geometry.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    h, w = frame.shape[:2]
    gw, gh = int(grid[0]), int(grid[1])
    gw, gh = max(4, min(gw, w // 4)), max(4, min(gh, h // 4))

    # jittered vertex lattice (shared edges tile correctly)
    energy = float(np.clip(energy, 0, 1)) if np.isfinite(energy) else 0.5
    amp = float(jitter) * (0.35 + 1.3 * energy)
    if not np.isfinite(amp) or amp < 0:
        amp = 1.0
    vy = np.clip(np.linspace(0, h, gh + 1)[:, None] + rng.normal(0, amp, (gh + 1, 1)), 0, h)  # (gh+1, 1): rows
    vx = np.clip(np.linspace(0, w, gw + 1)[None, :] + rng.normal(0, amp, (1, gw + 1)), 0, w)  # (1, gw+1): cols
    vy = np.nan_to_num(vy, nan=0.0, posinf=h, neginf=0.0)
    vx = np.nan_to_num(vx, nan=0.0, posinf=w, neginf=0.0)

    xs = np.linspace(0, w, gw + 1).astype(int)
    ys = np.linspace(0, h, gh + 1).astype(int)
    # cell mean colors (flat shading)
    colors = np.zeros((gh, gw, 3), np.float32)
    for gy in range(gh):
        row = frame[ys[gy]: ys[gy + 1]]
        for gx in range(gw):
            colors[gy, gx] = row[:, xs[gx]: xs[gx + 1]].mean(axis=(0, 1))

    out = np.zeros_like(frame)
    tear_rows = set()
    if rng.random() < tear * (0.3 + energy):
        tear_rows = {int(rng.integers(0, h)) for _ in range(int(1 + 3 * energy))}
    for gy in range(gh):
        y0 = max(0, min(int(vy[gy, 0]), h - 2))
        y1 = max(y0 + 1, min(int(vy[gy + 1, 0]), h))
        span = y1 - y0
        for gx in range(gw):
            x0 = max(0, min(int(vx[0, gx]), w - 2))
            x1 = max(x0 + 1, min(int(vx[0, gx + 1]), w))
            c = colors[gy, gx]
            shear = rng.normal(0, amp * 0.6)  # per-cell parallelogram warp
            for dy in range(span):
                y = y0 + dy
                if y in tear_rows:
                    out[y] = 0
                    continue
                s = shear * dy / span
                xl = max(0, min(int(x0 + s), w - 1))
                xr = max(xl + 1, min(int(x1 + s), w))
                out[y, xl:xr] = c
    # cell edge darkening first (on the flat field), THEN the RGB555 dither
    out = np.clip(out, 0, 255).astype(np.float32)
    if edges > 0:
        lum = out.mean(axis=2)
        gxk = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
        gyk = np.abs(np.diff(lum, axis=0, prepend=lum[:1]))
        mask = (gxk + gyk) > 26
        out[mask] *= (1.0 - float(np.clip(edges, 0, 0.9)))
    return psx_quantize(np.clip(out, 0, 255).astype(np.uint8), levels=int(levels))


def reprocess(video, out, grid=(24, 42), jitter=2.0, tear=0.15, levels=16,
              edges=0.35, seed=0, crf=19) -> str:
    """Re-lens an existing video (doom EP videos, womb pieces, anything)
    through the lowpoly pass — audio stream preserved untouched."""
    from .video import decode_frames, probe_video

    import json as _json

    info = probe_video(video)
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    has_audio = any(s["codec_type"] == "audio" for s in _json.loads(probe)["streams"])
    w, h = min(720, info["width"]), int(min(720, info["width"]) * info["height"] / info["width"])
    w -= w % 2; h -= h % 2
    size = (w, h)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(info["fps"]), "-i", "pipe:0"]
    if has_audio:
        cmd += ["-i", str(video), "-map", "0:v:0", "-map", "1:a:0",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rng = np.random.default_rng(seed)
    frames = decode_frames(video, size=size, block=24)
    for i, f in enumerate(frames):
        # energy from the frame's own motion (audio-independent lens)
        e = float(np.abs(f[1:] - f[:-1]).mean() * 20) if len(f) > 1 else 0.0
        pf = lowpoly_frame(f, grid=grid, jitter=jitter, tear=tear,
                           levels=levels, edges=edges, energy=e, rng=rng)
        proc.stdin.write(pf.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited {rc}")
    return str(out)