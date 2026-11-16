"""Pure-numpy additive glow renderer (no pygame required).

A float framebuffer with gaussian splats and per-frame decay: identical in
spirit to the paper's billboarded translucent particles, and it gives free
motion trails.  The viewer wraps it in pygame surfaces; the movie pipeline
pipes `frame()` bytes straight into ffmpeg.
"""

from __future__ import annotations

import numpy as np

from .sim import SPACE


class Canvas:
    """Float framebuffer + additive gaussian splats + exponential decay."""

    def __init__(self, w: int, h: int, spr_r: int, fade=0.90, exposure=1.4,
                 max_splats=100000):
        self.w, self.h = w, h
        self.fade, self.exposure = fade, exposure
        self.max_splats = max_splats
        self.fb = np.zeros((h, w, 3), dtype=np.float32)
        self.r = spr_r
        yy, xx = np.mgrid[-spr_r:spr_r + 1, -spr_r:spr_r + 1].astype(np.float32)
        sig = spr_r / 1.7
        spr = np.exp(-(xx * xx + yy * yy) / (2 * sig * sig))[:, :, None]
        self.spr = (spr / spr.max()).astype(np.float32)
        self.kx = w / (2 * SPACE)
        self.ky = h / (2 * SPACE)

    def clear(self):
        self.fb[:] = 0.0

    def splat(self, xs, ys, cols, gains):
        n = len(xs)
        if n == 0:
            return
        if n > self.max_splats:                      # subsample heavy trails
            keep = np.random.default_rng(7).choice(n, self.max_splats, replace=False)
            xs, ys, cols, gains = xs[keep], ys[keep], cols[keep], gains[keep]
            n = self.max_splats
        cg = cols * gains[:, None]
        fb, spr, r = self.fb, self.spr, self.r
        w, h, kx, ky = self.w, self.h, self.kx, self.ky
        for i in range(n):
            cx = int(xs[i] * kx + w * 0.5)
            cy = int(ys[i] * ky + h * 0.5)
            x0, y0 = cx - r, cy - r
            sx0, sy0 = max(0, -x0), max(0, -y0)
            sx1 = min(2 * r + 1, w - x0)
            sy1 = min(2 * r + 1, h - y0)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            fb[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1] += spr[sy0:sy1, sx0:sx1] * cg[i]

    def tick(self):
        self.fb *= self.fade

    def frame(self) -> np.ndarray:
        """uint8 (h, w, 3) frame ready for piping/rawvideo."""
        return (np.clip(self.fb * self.exposure, 0, 1) * 255).astype(np.uint8)

    def rgb_float(self) -> np.ndarray:
        """clipped float frame (h, w, 3) in [0,1] — for compositing."""
        return np.clip(self.fb * self.exposure, 0, 1)


def draw_step(canvas: Canvas, st: dict):
    """Splat one Sim.step() result (particles + trails) onto a canvas."""
    if st["n_alive"]:
        canvas.splat(st["px"], st["py"], st["col"], 0.35 + 1.25 * st["size"])
    if len(st["tpx"]):
        canvas.splat(st["tpx"], st["tpy"], st["tcol"],
                     np.full(len(st["tpx"]), 0.6))