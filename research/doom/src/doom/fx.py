"""fx: audio-reactive post-processing.

An FX is `apply(frame, feats, i, rng, **params) -> frame`. Chains run after
the scene each frame. Several are onset-triggered (glitch bursts land ON
transients — that is what makes the video feel cut to the music), and the
PS2/CRT family (psx dither, scanlines, rgb split) runs continuously.

Register your own with @register_fx.
"""

from __future__ import annotations

import numpy as np

from .palettes import psx_quantize

__all__ = ["FX", "register_fx", "build_fx_chain", "list_fx"]

FX: dict[str, callable] = {}


def register_fx(fn):
    """Register an FX. `fn` may be a plain function (stateless) or a CLASS
    with __call__ — classes are instantiated fresh at chain-build time so
    stateful FX (trails, feedback) get clean, deterministic per-video state."""
    FX[fn.__name__.rstrip("_") if hasattr(fn, "__name__") else fn.__class__.__name__] = fn
    return fn


def list_fx() -> list[str]:
    return sorted(FX)


def build_fx_chain(specs) -> list[callable]:
    """specs: None | "name" | ["name", ["name", {params}], "name(kw=v)"]..."""
    if specs is None:
        return []
    if isinstance(specs, str):
        specs = [specs]
    chain = []
    for spec in specs:
        params = {}
        if isinstance(spec, (list, tuple)) and len(spec) == 2 and isinstance(spec[1], dict):
            spec, params = spec[0], dict(spec[1])
        name = str(spec)
        if "(" in name and name.endswith(")"):
            head, rest = name.split("(", 1)
            name = head.strip()
            import ast

            call = ast.parse(f"f({rest}", mode="eval").body
            for kw in call.keywords:
                try:
                    params[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    pass
        if name not in FX:
            raise ValueError(f"unknown fx {name!r}; have {list_fx()}")
        entry = FX[name]
        fn = entry() if isinstance(entry, type) else entry  # class -> fresh state
        chain.append((fn, params))
    return chain


def run_chain(chain, frame, feats, i, rng):
    for fn, params in chain:
        frame = fn(frame, feats, i, rng, **params)
    return frame


# --------------------------------------------------------------------------- glitch family

@register_fx
def slices_(frame, feats, i, rng, amount=0.5, max_bands=10):
    """Torn-transmission: horizontal band displacement, triggered by onsets,
    scaled by flux."""
    if not feats.onset[i] and rng.random() > 0.06 * feats.flux[i] * amount:
        return frame
    out = frame.copy()
    n = rng.integers(2, max_bands + 1)
    h = frame.shape[0]
    for _ in range(n):
        y0 = rng.integers(0, h - 4)
        bh = rng.integers(2, max(3, h // 24))
        dx = int(rng.integers(-frame.shape[1], frame.shape[1]) * 0.14 * amount)
        out[y0: y0 + bh] = np.roll(out[y0: y0 + bh], dx, axis=0)
    return out


@register_fx
def corrupt_(frame, feats, i, rng, amount=0.001, trigger=True):
    """Byte-level databending of the frame buffer — zero-runs and swaps."""
    if trigger and not feats.onset[i] and rng.random() > 0.15:
        return frame
    buf = bytearray(frame.tobytes())
    hits = int(len(buf) * amount * (1.0 + 4.0 * feats.flux[i]))
    for _ in range(hits):
        k = rng.integers(len(buf) - 1)
        mode = rng.integers(3)
        if mode == 0:
            buf[k] = 0
        elif mode == 1:
            buf[k], buf[k + 1] = buf[k + 1], buf[k]
        else:
            buf[k] ^= 1 << rng.integers(8)
    out = np.frombuffer(bytes(buf), np.uint8).reshape(frame.shape)
    return out


@register_fx
def pixel_sort_(frame, feats, i, rng, threshold=0.45, vertical=False):
    """Asendorf-style pixel sorting: sort bright runs within rows/columns.
    Runs a slice on every onset, lightly on high-flux frames."""
    if not feats.onset[i] and feats.flux[i] < 0.08:
        return frame
    out = frame.copy()
    lum = out.mean(axis=2, keepdims=True) / 255.0
    mask = lum[..., 0] > threshold
    if not vertical:
        for y in range(out.shape[0]):
            row = mask[y]
            if not row.any():
                continue
            seg = out[y][row]
            out[y][row] = seg[np.argsort(seg.mean(axis=1))]
    else:
        for x in range(out.shape[1]):
            col = mask[:, x]
            if not col.any():
                continue
            seg = out[:, x][col]
            out[:, x][col] = seg[np.argsort(seg.mean(axis=1))]
    return out


@register_fx
def rgb_split_(frame, feats, i, rng, px=3):
    """Chromatic aberration: channels pulled apart; distance rides treble."""
    d = int(max(1, px * (0.4 + 1.6 * feats.treble[i])))
    out = frame.copy()
    out[..., 0] = np.roll(frame[..., 0], d, axis=1)
    out[..., 2] = np.roll(frame[..., 2], -d, axis=1)
    return out


@register_fx
def block_glitch_(frame, feats, i, rng, size=24, amount=0.6):
    """Macroblock shuffle — datamosh-flavored block displacement on onsets."""
    if not feats.onset[i] and rng.random() > 0.1:
        return frame
    out = frame.copy()
    h, w = frame.shape[:2]
    s = max(4, int(size))
    gy, gx = h // s, w // s
    if gy < 2 or gx < 2:
        return out
    blocks = out[: gy * s, : gx * s].reshape(gy, s, gx, s, 3)
    n_swaps = max(1, int(gy * gx * 0.08 * amount))
    for _ in range(n_swaps):
        y1, x1 = rng.integers(gy), rng.integers(gx)
        y2, x2 = rng.integers(gy), rng.integers(gx)
        blocks[y1, :, x1], blocks[y2, :, x2] = blocks[y2, :, x2], blocks[y1, :, x1].copy()
    out[: gy * s, : gx * s] = blocks.reshape(gy * s, gx * s, 3)
    return out


@register_fx
def strobe_(frame, feats, i, rng, prob=0.06):
    """Frame dropouts to black/white — rhythmic on dense onsets."""
    if feats.onset[i] and rng.random() < prob * 3:
        return (frame * 0).astype(np.uint8) if rng.random() < 0.7 else \
            np.full_like(frame, 255)
    if rng.random() < prob * 0.25:
        return np.roll(frame, int(rng.integers(-8, 8)), axis=0)
    return frame


# --------------------------------------------------------------------------- screen & film family

@register_fx
def scanlines_(frame, feats, i, rng, strength=0.25, thickness=2):
    out = frame.astype(np.float32)
    out[::thickness] *= (1.0 - strength)
    return np.clip(out, 0, 255).astype(np.uint8)


@register_fx
def crt_(frame, feats, i, rng, curve=0.18, vignette=0.35):
    """Barrel warp + corner darkening + subtle phosphor mask."""
    h, w = frame.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    nx = (xs / w * 2 - 1).astype(np.float32)
    ny = (ys / h * 2 - 1).astype(np.float32)
    r2 = nx * nx + ny * ny
    ux = np.clip((nx * (1 + curve * r2) * 0.5 + 0.5) * (w - 1), 0, w - 1).astype(np.int32)
    uy = np.clip((ny * (1 + curve * r2) * 0.5 + 0.5) * (h - 1), 0, h - 1).astype(np.int32)
    out = frame[uy, ux].astype(np.float32)
    vig = 1.0 - vignette * np.clip(r2 - 0.32, 0, 2) ** 1.5
    out *= vig[..., None]
    # faint aperture-grille tint on every 3rd pixel column
    out[:, ::3, 0] *= 1.04
    out[:, 1::3, 1] *= 1.04
    return np.clip(out, 0, 255).astype(np.uint8)


@register_fx
def vhs_(frame, feats, i, rng, jitter=0.4, noise=0.10):
    """Head-switching jitter line, chroma bleed, tape noise."""
    out = frame.astype(np.float32)
    h = out.shape[0]
    # a travelling dropout band
    band_y = int((i * 3.7 + rng.random() * 4) % h)
    bh = max(2, h // 90)
    band = slice(band_y, min(h, band_y + bh))
    out[band] = np.roll(out[band], int(rng.integers(-30, 30) * jitter), axis=1)
    # chroma bleed: blur chroma horizontally (cheap box)
    chroma = out.copy()
    chroma[:, :-2] = (chroma[:, :-2] + 2 * chroma[:, 1:-1] + chroma[:, 2:]) * 0.25
    luma = out.mean(axis=2, keepdims=True)
    out = np.clip(luma + (chroma - luma) * 1.7, 0, 255)
    if rng.random() < 0.25:
        out += rng.normal(0, 255 * noise * 0.4, out.shape[1]).astype(np.float32)[None, :, None]
    out += rng.normal(0, 255 * noise * 0.15, out.shape) * (0.5 + feats.flux[i])
    return np.clip(out, 0, 255).astype(np.uint8)


@register_fx
def grain_(frame, feats, i, rng, amount=0.06):
    out = frame.astype(np.float32)
    out += rng.normal(0, 255 * amount * (0.5 + feats.rms[i]), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


@register_fx
def glow_(frame, feats, i, rng, cutoff=180, strength=0.5):
    """Cheap bloom: blur the highlights with an integral-image box blur, add back."""
    luma = frame.mean(axis=2)
    hi = np.clip(luma - cutoff, 0, 255) / 255.0

    def box(x, k):
        # windowed mean via a padded integral image — O(H*W), no loops
        p = np.pad(x, k, mode="edge")
        c = np.cumsum(np.cumsum(p, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        n = 2 * k + 1
        s = (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / float(n * n)
        return s[: x.shape[0], : x.shape[1]]

    hi = box(hi, 2) + 0.6 * box(hi, 5)
    out = frame.astype(np.float32)
    out += hi[..., None] * 255.0 * strength * (0.6 + 0.8 * feats.rms[i])
    return np.clip(out, 0, 255).astype(np.uint8)


@register_fx
def shake_(frame, feats, i, rng, px=6):
    """Impact shake on onsets, decaying with the frame's flux."""
    amp = px * (1.0 if feats.onset[i] else 0.12 * min(1.0, feats.flux[i] * 8))
    if amp < 0.5:
        return frame
    dx, dy = int(rng.integers(-amp, amp)), int(rng.integers(-amp, amp))
    out = np.roll(np.roll(frame, dx, axis=1), dy, axis=0)
    return out


@register_fx
def hue_cycle_(frame, feats, i, rng, rate=30.0):
    """Rotate hue over time (degrees per second * (1 + rms))."""
    deg = (i / feats.fps * rate * (0.3 + feats.rms[i])) % 360.0
    r, g, b = frame[..., 0] / 255.0, frame[..., 1] / 255.0, frame[..., 2] / 255.0
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = mx - mn + 1e-9
    h = np.zeros_like(mx)
    m = (mx == r)
    h[m] = ((g - b)[m] / d[m]) % 6
    m = (mx == g)
    h[m] = (b - r)[m] / d[m] + 2
    m = (mx == b)
    h[m] = (r - g)[m] / d[m] + 4
    h = (h * 60 + deg) % 360 / 60
    s = d / (mx + 1e-9)
    v = mx
    hh = np.clip(h, 0, 6)
    i5 = hh.astype(np.int32) % 6
    f = hh - np.floor(hh)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    out = np.zeros_like(frame, np.float32)
    conds = [(i5 == 0, (v, t, p)), (i5 == 1, (q, v, p)), (i5 == 2, (p, v, t)),
             (i5 == 3, (p, q, v)), (i5 == 4, (t, p, v)), (i5 == 5, (v, p, q))]
    for mask, (c1, c2, c3) in conds:
        out[..., 0][mask] = c1[mask] * 255
        out[..., 1][mask] = c2[mask] * 255
        out[..., 2][mask] = c3[mask] * 255
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- retro quantize family

@register_fx
def psx_(frame, feats, i, rng, levels=32, dither=True, downscale=0):
    """PS1 display chain: optional integer downscale + Bayer dither + 5-bit."""
    out = frame
    if downscale and downscale > 1:
        h, w = frame.shape[:2]
        out = out[: h // downscale * downscale, : w // downscale * downscale]
        out = np.repeat(np.repeat(out.reshape(h // downscale, downscale,
                                              w // downscale, downscale, 3).mean(axis=(1, 3)),
                                  downscale, 0), downscale, 1)
    return psx_quantize(out, levels=levels, dither=dither)


@register_fx
def posterize_(frame, feats, i, rng, levels=5):
    q = 255.0 / max(1, levels - 1)
    out = (np.round(frame / q) * q).astype(np.uint8)
    return out


@register_fx
def invert_pulse_(frame, feats, i, rng, prob=0.25):
    """On some onsets, flash the negative."""
    if feats.onset[i] and rng.random() < prob:
        return (255 - frame).astype(np.uint8)
    return frame


@register_fx
def kaleido_(frame, feats, i, rng, sides=6):
    """Mirror wedges around the vertical center — instant VJ symmetry."""
    h, w = frame.shape[:2]
    half = frame[:, w // 2:]
    out = np.concatenate([half[:, ::-1], half], axis=1)
    return out

# ===========================================================================
# GEOMETRY GLITCH — the lowpoly/mask chaos layer.
# Built to mangle structured scenes (mask, lowpoly, tunnel, plasma): the
# artifacts are geometric — blocks, bands, shear, wireframes, combs — and
# every one is audio-triggered: onsets tear, flux warps, bass breathes.
# ===========================================================================

@register_fx
class polytear_:
    """Polygon tearing: coarse block bands shear sideways, harder on onsets.
    Reads as torn polygon strips rather than scanline noise."""

    def __init__(self, block=16, amount=0.6):
        self.block, self.amount = int(block), float(amount)
        self.energy = 0.0

    def __call__(self, frame, feats, i, rng, block=None, amount=None):
        h, w = frame.shape[:2]
        b = int(block if block is not None else self.block)
        a = float(amount if amount is not None else self.amount)
        self.energy *= 0.90
        if feats.onset[i]:
            self.energy = 1.0
        elif rng.random() > 0.25 * feats.flux[i]:
            return frame
        out = frame.copy()
        for y0 in range(0, h - b, b):
            if rng.random() > 0.5 * (self.energy + a * 0.2):
                continue
            dx = int(rng.integers(-w, w) * 0.10 * a * self.energy)
            out[y0: y0 + b] = np.roll(out[y0: y0 + b], dx, axis=1)
        # occasional diagonal rip
        if self.energy > 0.6 and rng.random() < 0.5:
            k = rng.integers(-h // 3, h // 3)
            out = np.roll(out, k, axis=0)
        return out


@register_fx
class affine_warp_:
    """Per-band affine shear — PS1 texture-warp energy applied to the whole
    frame. Treble bends, onsets snap it hard."""

    def __init__(self):
        self.kick = 0.0

    def __call__(self, frame, feats, i, rng, strength=0.9):
        h, w = frame.shape[:2]
        self.kick *= 0.86
        if feats.onset[i]:
            self.kick = 1.0
        bend = strength * (0.15 * feats.treble[i] + 0.7 * self.kick)
        if bend < 0.03:
            return frame
        out = np.zeros_like(frame)
        for y in range(h):
            off = int(np.sin(y / h * np.pi * 3 + i * 0.31) * w * 0.06 * bend)
            out[y] = np.roll(frame[y], off)
        return out


@register_fx
class vertex_jitter_:
    """The PS1 wobble as an FX: quantize the frame to a coarse vertex grid
    and shake the blocks — harder with bass, spikes on onsets."""

    def __call__(self, frame, feats, i, rng, grid=24, amount=0.8):
        h, w = frame.shape[:2]
        g = max(4, int(grid))
        amp = int(2 + amount * (3 * feats.bass[i] + (10 if feats.onset[i] else 0)))
        out = frame.copy()
        for y0 in range(0, h, g):
            for x0 in range(0, w, g):
                if rng.random() < 0.25:
                    y1, x1 = min(h, y0 + g), min(w, x0 + g)
                    blk = out[y0:y1, x0:x1]
                    out[y0:y1, x0:x1] = np.clip(
                        blk.astype(np.int32) + rng.integers(-amp, amp + 1), 0, 255
                    ).astype(np.uint8)
        return out


@register_fx
class depth_corrupt_:
    """Perspective block corruption: rows shear proportionally to their
    distance from a moving vanishing line — the frame folds like a tunnel."""

    def __call__(self, frame, feats, i, rng, amount=0.5):
        h, w = frame.shape[:2]
        if not feats.onset[i] and rng.random() > 0.35 * feats.flux[i]:
            return frame
        out = frame.copy()
        vline = rng.uniform(0.2, 0.8) * h
        for y in range(h):
            d = abs(y - vline) / h
            if rng.random() < 0.22 + 0.5 * d:
                dx = int(rng.integers(-1, 2) * w * amount * d * 0.5)
                out[y] = np.roll(out[y], dx)
        return out


@register_fx
def wireframe_(frame, feats, i, rng, color=(120, 255, 180), glow=0.0, threshold=26):
    """Sobel edge overlay — the PS2 debug wireframe. Brightness follows
    flux; onsets flash the lines."""
    luma = frame.mean(axis=2).astype(np.float32)
    gx = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    gy = np.abs(np.diff(luma, axis=0, prepend=luma[:1]))
    edges = (gx + gy) > threshold
    out = frame.astype(np.float32)
    strength = (0.35 + 0.65 * feats.flux[i] * 3) * (1.5 if feats.onset[i] else 1.0)
    out[edges] = out[edges] * (1.0 - 0.6 * min(1.0, strength)) + \
        np.asarray(color, np.float32) * min(1.0, strength)
    return np.clip(out, 0, 255).astype(np.uint8)


@register_fx
def interlace_(frame, feats, i, rng, comb=2, strength=0.5):
    """PS2 interlacing artifact: darken/duplicate alternate rows, with a
    rolling sync error band driven by bass."""
    out = frame.astype(np.float32)
    out[::comb] *= (1.0 - strength * 0.5)
    band = int((i * (7 + 30 * feats.bass[i])) % frame.shape[0])
    out[band: band + 4] = np.roll(out[band: band + 4], 6, axis=1)
    return np.clip(out, 0, 255).astype(np.uint8)


@register_fx
class trail_:
    """Frame-difference trails (stateful): motion smears forward, decaying
    with the music's loudness — datamosh-adjacent without codec access."""

    def __init__(self, sustain=0.55):
        self.prev = None
        self.sustain = float(sustain)

    def __call__(self, frame, feats, i, rng, sustain=None):
        s = float(sustain if sustain is not None else self.sustain)
        if self.prev is None or self.prev.shape != frame.shape:
            self.prev = frame.astype(np.float32)
            return frame
        diff = np.abs(frame.astype(np.float32) - self.prev)
        keep = np.clip(s * (0.5 + 1.2 * feats.rms[i]), 0, 0.92)
        self.prev = frame.astype(np.float32) * (1 - keep) + self.prev * keep
        hot = diff.mean(axis=2) > 14
        out = frame.astype(np.float32)
        out[hot] = out[hot] * 0.4 + self.prev[hot] * 0.9
        return np.clip(out, 0, 255).astype(np.uint8)


@register_fx
def chroma_ripple_(frame, feats, i, rng, amp=3, freq=0.05):
    """Sinusoidal per-channel horizontal offsets — the signal hums."""
    h, w = frame.shape[:2]
    ys = np.arange(h)
    out = frame.copy()
    phase = i * 0.21 + feats.mid[i] * 6
    off_r = (np.sin(ys * freq + phase) * amp * (0.4 + 1.6 * feats.treble[i])).astype(int)
    off_b = (np.sin(ys * freq + phase + 2.1) * amp * (0.4 + 1.6 * feats.bass[i])).astype(int)
    for y in range(h):
        out[y, :, 0] = np.roll(frame[y, :, 0], off_r[y])
        out[y, :, 2] = np.roll(frame[y, :, 2], off_b[y])
    return out


@register_fx
class block_zoom_:
    """Onset-triggered macroblock punch-in: the frame tiles into blocks and
    zooms one frame per onset — hard cuts of geometry."""

    def __init__(self, size=48):
        self.zoom = 0.0
        self.size = int(size)

    def __call__(self, frame, feats, i, rng, size=None):
        if feats.onset[i]:
            self.zoom = 1.0
        self.zoom *= 0.82
        if self.zoom < 0.15:
            return frame
        s = int(size if size is not None else self.size)
        h, w = frame.shape[:2]
        bh = max(4, int(s * (1 - 0.3 * self.zoom)))
        bw = max(4, int(s * (1 - 0.3 * self.zoom)))
        gy, gx = h // bh, w // bw
        if gy < 2 or gx < 2:
            return frame
        small = frame[: gy * bh, : gx * bw].reshape(
            gy, bh, gx, bw, 3).mean(axis=(1, 3))
        small = np.clip(small * (1 + 0.25 * self.zoom), 0, 255).astype(np.uint8)
        yi = np.arange(h) * gy // h
        xi = np.arange(w) * gx // w
        return small[np.ix_(yi, xi)]
