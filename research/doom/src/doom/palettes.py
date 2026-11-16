"""palettes + retro quantization.

Named color ramps (demoscene/PS2/VHS flavored), palette mapping, RGB555
(15-bit) quantization, and 4x4 Bayer ordered dithering — the trio that
makes a smooth numpy field look like period hardware.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PALETTES", "get_palette", "apply_palette", "rgb555", "bayer4", "psx_quantize"]

# each palette: list of (r, g, b) 0..255 anchor colors (interpolated)
PALETTES: dict[str, list] = {
    "black_bronze": [(0, 0, 0), (16, 11, 7), (44, 30, 18), (92, 62, 34), (156, 110, 62), (214, 176, 122), (255, 242, 205)],
    "verdigris": [(0, 0, 0), (8, 18, 14), (20, 46, 36), (44, 92, 70), (96, 160, 120), (178, 220, 190), (240, 255, 240)],
    "inferno": [(0, 0, 20), (60, 15, 90), (180, 40, 60), (255, 120, 20), (255, 240, 160)],
    "ps2_dusk": [(10, 8, 24), (40, 30, 80), (110, 60, 120), (230, 120, 110), (255, 220, 170)],
    "vaporwave": [(20, 5, 45), (120, 30, 130), (255, 80, 180), (90, 220, 255), (255, 245, 235)],
    "green_terminal": [(2, 8, 4), (10, 60, 20), (30, 200, 80), (160, 255, 180), (240, 255, 240)],
    "amber": [(15, 8, 2), (70, 35, 5), (180, 110, 20), (255, 190, 80), (255, 240, 200)],
    "ice": [(5, 8, 20), (20, 50, 110), (60, 140, 220), (140, 210, 255), (240, 250, 255)],
    "gameboy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "vhs_night": [(8, 6, 14), (30, 24, 60), (90, 70, 130), (200, 150, 190), (250, 230, 240)],
    "blood": [(10, 2, 4), (60, 10, 16), (140, 25, 30), (230, 90, 60), (255, 210, 170)],
    "toxic": [(4, 12, 6), (18, 60, 20), (80, 160, 40), (190, 255, 90), (255, 255, 220)],
    "sunset_hdmi": [(20, 10, 40), (120, 40, 60), (240, 100, 40), (255, 200, 60), (255, 255, 210)],
    "mono": [(0, 0, 0), (255, 255, 255)],
}


def get_palette(name: str) -> np.ndarray:
    """-> (256, 3) uint8 LUT interpolated from the palette anchors."""
    anchors = np.asarray(PALETTES[name], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, len(anchors))
    xi = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(xi, xs, anchors[:, c]) for c in range(3)], axis=1)
    return np.clip(lut, 0, 255).astype(np.uint8)


def apply_palette(field01: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """field01: (H, W) float 0..1 -> (H, W, 3) uint8 via 256-entry LUT."""
    field01 = np.nan_to_num(np.asarray(field01, np.float64), nan=0.0,
                            posinf=1.0, neginf=0.0)
    idx = np.clip((field01 * 255.0).astype(np.int16), 0, 255)
    return lut[idx]


def rgb555(frame: np.ndarray) -> np.ndarray:
    """Quantize to 15-bit color (5 bits/channel) — the PS1 display depth."""
    return (frame & 0xF8).astype(frame.dtype)


_BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float64) / 16.0


def bayer4(shape: tuple) -> np.ndarray:
    """Tiled 4x4 Bayer threshold map matching (H, W)."""
    h, w = shape
    return np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]


def psx_quantize(frame: np.ndarray, levels: int = 32, dither: bool = True) -> np.ndarray:
    """The PSX look in one call: ordered dither into the RGB555 mask (0xF8)."""
    x = frame.astype(np.float64)
    if dither:
        x = x + (bayer4(frame.shape[:2]) - 0.5)[..., None] * (255.0 / levels)
    return (np.clip(x, 0, 255).astype(np.uint8)) & np.uint8(0xF8)