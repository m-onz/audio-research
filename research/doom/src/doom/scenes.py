"""scenes: the visual strategy registry.

A Scene is a small stateful class: `render(feats, i) -> (H, W, 3) uint8`,
called once per output frame in order. Frame i is drawn from the audio
features of frame i — that is the whole synchronization mechanism.

Every scene is audio-reactive through the feature vector:
    feats.rms/.bass/.mid/.treble/.centroid/.flux  (floats for THIS frame)
    feats.onset                                   (bool — a transient)
    feats.wave_window(i)                          (the actual samples)

Register your own with @register_scene — same open-registry philosophy as
loom. Heavy scenes render at reduced internal resolution and upscale
NEAREST; that is both the speed trick and most of the retro charm.
"""

from __future__ import annotations

import numpy as np

from .palettes import apply_palette, get_palette

__all__ = ["SCENES", "register_scene", "make_scene", "list_scenes"]

SCENES: dict[str, type["Scene"]] = {}


def register_scene(cls):
    SCENES[cls.name] = cls
    return cls


def make_scene(name: str, w: int, h: int, seed: int, **params) -> "Scene":
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; have {sorted(SCENES)}")
    return SCENES[name](w, h, seed, **params)


def list_scenes() -> list[str]:
    return sorted(SCENES)


class Scene:
    name = "?"

    def __init__(self, w: int, h: int, seed: int, **params) -> None:
        self.w, self.h = w, h
        self.rng = np.random.default_rng(seed)
        self.params = params
        ys, xs = np.mgrid[0:h, 0:w]
        self.X = (xs / w * 2 - 1).astype(np.float32)  # -1..1
        self.Y = (ys / h * 2 - 1).astype(np.float32)
        self.setup(**params)

    def setup(self, **params) -> None:
        pass

    def render(self, feats, i: int) -> np.ndarray:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------
    def _down(self, div: int):
        """Internal low-res canvas + an upscale that always lands on the exact
        target size (nearest sampling) for any divisor."""
        w, h = max(8, self.w // div), max(8, self.h // div)

        def up(f):
            yi = np.arange(self.h) * h // self.h
            xi = np.arange(self.w) * w // self.w
            return f[np.ix_(yi, xi)]

        return w, h, up

    def _pal(self, name: str):
        return get_palette(self.params.get("palette", name))


# --------------------------------------------------------------------------- 1 plasma

@register_scene
class Plasma(Scene):
    """Domain-warped sine plasma — the demoscene classic. Palette cycles with
    loudness; onsets kick the palette and burst the warp."""

    name = "plasma"

    def setup(self, palette="inferno", speed=1.0):
        self.lut = self._pal(palette)
        self.phase = 0.0
        self.burst = 0.0

    def render(self, F, i):
        dt = 1.0 / F.fps
        self.phase += dt * (0.3 + 1.8 * F.rms[i]) * self.params.get("speed", 1.0)
        self.burst *= 0.94
        if F.onset[i]:
            self.burst = 1.0
            self.phase += 0.35
        p = self.phase
        w = 2.6 + 3.5 * self.burst
        f = (
            np.sin(self.X * w + p)
            + np.sin(self.Y * w - p * 0.7)
            + np.sin((self.X + self.Y) * w * 0.7 + p * 1.3)
            + np.sin(np.hypot(self.X, self.Y) * w * 1.4 - p)
        ) / 4.0
        field = 0.5 + 0.5 * f
        if self.burst > 0.01:
            field = np.clip(field + self.burst * 0.15 * np.sin(self.X * 30), 0, 1)
        return apply_palette(field, self.lut)


# --------------------------------------------------------------------------- 2 tunnel

@register_scene
class Tunnel(Scene):
    """Polar-UV tunnel: u=angle, v=1/radius; checker+stripes texture.
    Bass drives fly-speed; onsets flash an expanding shockwave ring."""

    name = "tunnel"

    def setup(self, palette="ps2_dusk"):
        self.lut = self._pal(palette)
        self.z = 0.0
        self.flash = -1.0

    def render(self, F, i):
        dt = 1.0 / F.fps
        self.z += dt * (0.4 + 2.2 * F.bass[i])
        r = np.hypot(self.X, self.Y) + 0.03
        a = np.arctan2(self.Y, self.X)
        u = (a / (2 * np.pi) + 1) * 8.0
        v = (1.0 / r + self.z) * 6.0
        checker = ((np.floor(u) + np.floor(v)) % 2).astype(np.float32)
        stripes = 0.5 + 0.5 * np.sin(u * np.pi * 2)
        tex = checker * 0.6 + stripes * 0.4
        fog = np.exp(-r * 1.9)
        field = np.clip(tex * fog, 0, 1) * (0.55 + 0.7 * F.rms[i])
        if F.onset[i]:
            self.flash = 0.0
        if self.flash >= 0.0:
            self.flash += dt * 2.5
            ring = np.abs(r - self.flash * 0.8) < 0.06
            field = np.clip(field + ring * (1.0 - self.flash), 0, 1)
            if self.flash > 1.4:
                self.flash = -1.0
        return apply_palette(field, self.lut)


# --------------------------------------------------------------------------- 3 starfield

@register_scene
class Starfield(Scene):
    """3D star streaks. Speed follows loudness; onsets spawn star bursts."""

    name = "starfield"

    def setup(self, n=400, palette="ice"):
        self.lut = self._pal(palette)
        self.n = int(n)
        self.stars = np.column_stack([
            self.rng.uniform(-1, 1, self.n),
            self.rng.uniform(-1, 1, self.n),
            self.rng.uniform(0.05, 4.0, self.n),
        ])

    def render(self, F, i):
        dt = 1.0 / F.fps
        speed = 0.4 + 3.5 * F.rms[i]
        self.stars[:, 2] -= dt * speed
        respawn = self.stars[:, 2] <= 0.05
        if F.onset[i]:  # burst: respawn a tenth at random depths
            respawn |= self.rng.random(self.n) < 0.1
        k = int(respawn.sum())
        if k:
            self.stars[respawn, 0] = self.rng.uniform(-1, 1, k)
            self.stars[respawn, 1] = self.rng.uniform(-1, 1, k)
            self.stars[respawn, 2] = self.rng.uniform(2.5, 4.0, k)
        # nebula background so quiet passages stay alive (never dead black)
        t = i / F.fps
        neb = (
            np.sin(self.X * 2.0 + t * 0.21)
            + np.sin(self.Y * 2.3 - t * 0.17)
            + np.sin((self.X + self.Y) * 1.6 + t * 0.1)
        ) / 6.0 + 0.5
        frame = apply_palette(neb * (0.10 + 0.35 * F.rms[i]), self.lut)
        frame = frame.astype(np.int32)
        z = self.stars[:, 2]
        px = (self.stars[:, 0] / z * 0.5 + 0.5) * (self.w - 1)
        py = (self.stars[:, 1] / z * 0.5 + 0.5) * (self.h - 1)
        shade = np.clip(1.4 - z * 0.33, 0, 1)
        ok = (px >= 0) & (px < self.w) & (py >= 0) & (py < self.h)
        xi, yi = px[ok].astype(int), py[ok].astype(int)
        np.add.at(frame, (yi, xi, 0), (shade[ok] * 235).astype(np.int32))
        np.add.at(frame, (yi, xi, 1), (shade[ok] * 245).astype(np.int32))
        np.add.at(frame, (yi, xi, 2), 255)
        return np.clip(frame, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- 4 scope

@register_scene
class Scope(Scene):
    """Oscilloscope: draws the ACTUAL samples playing this frame — the most
    literal sync possible. XY lissajous mode with phosphor persistence."""

    name = "scope"

    def setup(self, palette="green_terminal", persistence=0.82, xy=True):
        self.lut = self._pal(palette)
        self.pers = float(persistence)
        self.prev = np.zeros((self.h, self.w), np.float32)

    def render(self, F, i):
        wave = F.wave_window(i, seconds=1.2 / F.fps)
        if len(wave) < 16:
            wave = np.zeros(64, np.float32)
        canvas = self.prev * self.pers
        # horizontal trace
        x = np.linspace(0.05, 0.95, len(wave)) * (self.w - 1)
        y = (0.5 + wave * 0.42 * (0.5 + F.rms[i])) * (self.h - 1)
        xi = np.clip(x.astype(int), 0, self.w - 1)
        yi = np.clip(y.astype(int), 0, self.h - 1)
        np.add.at(canvas, (yi, xi), 1.0)
        if self.params.get("xy", True):  # XY overlay: wave vs shifted wave
            lag = max(1, len(wave) // 4)
            xx = (0.5 + wave * 0.44) * (self.w - 1)
            yy = (0.5 + np.roll(wave, lag) * 0.44) * (self.h - 1)
            xi = np.clip(xx.astype(int), 0, self.w - 1)
            yi = np.clip(yy.astype(int), 0, self.h - 1)
            np.add.at(canvas, (yi, xi), 0.7)
        canvas = np.clip(canvas, 0, 1)
        self.prev = canvas
        return apply_palette(canvas * 0.9 + 0.02, self.lut)


# --------------------------------------------------------------------------- 5 spectro

@register_scene
class Spectro(Scene):
    """Rolling spectrogram of the actual audio — the 'truth' scene. Columns
    scroll left; each new column is the FFT of this frame's window."""

    name = "spectro"

    def setup(self, palette="inferno", n_bands=96):
        self.lut = self._pal(palette)
        self.n_bands = int(n_bands)
        self.hist = np.zeros((self.n_bands, self.w), np.float32)

    def render(self, F, i):
        wave = F.wave_window(i, seconds=0.12)
        if len(wave) < 32:
            wave = np.zeros(64, np.float32)
        seg = wave[:512]
        seg = np.pad(seg, (0, max(0, 512 - len(seg))))[:512]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        # log-spaced bands
        edges = np.unique(np.geomspace(1, len(spec), self.n_bands + 1).astype(int))
        bands = np.array([spec[edges[k]: edges[k + 1]].mean() for k in range(len(edges) - 1)])
        bands = np.resize(bands, self.n_bands)  # unique() may collapse edges
        db = 20 * np.log10(bands * 40 + 1e-6) / 60.0 + 1.0  # ~0..1
        col = np.clip(db[::-1], 0, 1)[:, None] ** 1.5
        self.hist = np.concatenate([self.hist[:, 1:], col], axis=1)
        img = np.repeat(self.hist, max(1, self.h // self.n_bands), axis=0)[: self.h]
        img = np.resize(img, (self.h, self.w))
        if F.onset[i]:
            img[:, -1] = 1.0
        return apply_palette(img * (0.5 + 0.7 * F.rms[i]), self.lut)


F_WAVE_SR = 8000  # decimated wave sample rate from audio.py


# --------------------------------------------------------------------------- 6 boids

@register_scene
class Boids(Scene):
    """Flocking (Nature of Code). Bass = cohesion, treble = separation,
    onsets teleport the attractor and scatter the flock. Trails persist."""

    name = "boids"

    def setup(self, n=180, palette="vhs_night"):
        self.lut = self._pal(palette)
        self.n = int(n)
        self.pos = np.column_stack([self.rng.uniform(0, 1, self.n), self.rng.uniform(0, 1, self.n)])
        self.vel = self.rng.uniform(-0.3, 0.3, (self.n, 2))
        self.target = np.array([0.5, 0.5])
        self.trail = np.zeros((self.h, self.w), np.float32)

    def render(self, F, i):
        dt = 1.0 / F.fps
        if F.onset[i]:
            self.target = self.rng.uniform(0.2, 0.8, 2)
            self.vel += self.rng.uniform(-0.8, 0.8, self.vel.shape)
        p, v = self.pos, self.vel
        d = p[:, None, :] - p[None, :, :]
        dist2 = (d ** 2).sum(-1) + 1e-6
        cohesion = ((self.target - p) * (2.0 + 6.0 * F.bass[i])).mean(axis=1, keepdims=True) * dt
        # separation: push away from the closest neighbors (vectorized top-k by cap)
        near = dist2 < 0.01
        sep = (d * near[..., None]).sum(1) * (30.0 * F.treble[i] + 10.0) * dt
        v += cohesion - sep
        # steering toward target + wander
        v += (self.target - p) * dt * (1.5 + 5 * F.bass[i])
        v += self.rng.normal(0, 0.5 * F.flux[i] + 0.05, v.shape) * dt
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(speed, 1e-5) * np.clip(speed, 0.1, 1.6)
        p += v * dt
        p[:] = (p + 1.0) % 1.0
        # draw with trails
        self.trail *= 0.9
        xi = (p[:, 0] * (self.w - 1)).astype(int)
        yi = (p[:, 1] * (self.h - 1)).astype(int)
        bright = 0.6 + 0.5 * F.rms[i]
        np.add.at(self.trail, (yi, xi), bright)
        field = np.clip(self.trail, 0, 1)
        return apply_palette(field * 0.92 + 0.03, self.lut)


# --------------------------------------------------------------------------- 7 flow

@register_scene
class FlowField(Scene):
    """Particles surfing a curl-noise flow field; onsets reseed bursts."""

    name = "flow"

    def setup(self, n=700, palette="toxic", scale=3.0):
        self.lut = self._pal(palette)
        self.n = int(n)
        self.pos = self.rng.uniform(0, 1, (self.n, 2))
        self.trail = np.zeros((self.h, self.w), np.float32)
        self.s = float(scale)
        self.ph = self.rng.uniform(0, 100)

    def _angle(self, x, y, t):
        # smooth pseudo-noise angle field from layered sines
        a = (np.sin(x * self.s + t * 0.3 + self.ph)
             + np.cos(y * self.s * 1.3 - t * 0.21)
             + 0.6 * np.sin((x + y) * self.s * 0.7 + t * 0.11))
        return a * np.pi

    def render(self, F, i):
        dt = 1.0 / F.fps
        t = i / F.fps
        if F.onset[i]:
            k = self.n // 8
            self.pos[:k] = self.rng.uniform(0, 1, (k, 2))
        ang = self._angle(self.pos[:, 0], self.pos[:, 1], t)
        vel = np.column_stack([np.cos(ang), np.sin(ang)]) * (0.25 + 1.4 * F.rms[i])
        self.pos += vel * dt
        out = (self.pos > 1.0) | (self.pos < 0.0)
        k = out.any(axis=1).sum()
        if k:
            self.pos[out.any(axis=1)] = self.rng.uniform(0, 1, (k, 2))
        self.trail *= 0.88
        xi = (self.pos[:, 0] * (self.w - 1)).astype(int)
        yi = (self.pos[:, 1] * (self.h - 1)).astype(int)
        np.add.at(self.trail, (yi, xi), 1.0)
        field = np.clip(self.trail, 0, 1)
        return apply_palette(field * 0.9 + 0.02, self.lut)


# --------------------------------------------------------------------------- 8 fire

@register_scene
class Fire(Scene):
    """The PSX-Doom fire cellular automaton. Fuel = loudness; wind = flux."""

    name = "fire"

    def setup(self, palette="blood", div=2):
        self.lut = self._pal(palette)
        w, h, up = self._down(div)
        self.dw, self.dh, self.up = w, h, up
        self.buf = np.zeros((h, w), np.float32)

    def render(self, F, i):
        h, w = self.dh, self.dw
        fuel = 0.35 + 0.75 * F.rms[i]
        self.buf[-1, :] = np.clip(self.buf[-1, :] * 0.4 + fuel * self.rng.uniform(0.7, 1.0, w), 0, 1.4)
        # propagate upward with random lateral jitter (classic 4-neighbor avg)
        src = self.buf
        rnd = self.rng.integers(-2, 3, (h, w))
        cols = np.clip(np.arange(w)[None, :] + rnd, 0, w - 1)
        shifted = src[np.arange(h)[:, None], cols]
        avg = (shifted + np.roll(shifted, 1, 0) + np.roll(shifted, -1, 0) + np.roll(shifted, 1, 1)) * 0.25
        self.buf = np.roll(avg * 0.965, 1, axis=0)
        field = np.clip(self.buf, 0, 1)
        frame = apply_palette(field, self.lut)
        return self.up(frame)


# --------------------------------------------------------------------------- 9 reaction-diffusion

@register_scene
class RD(Scene):
    """Gray-Scott reaction-diffusion. Bass modulates feed, treble kill;
    onsets inject chemical splats. Runs at quarter res, upscaled."""

    name = "rd"

    def setup(self, f=0.055, k=0.062, palette="ice", div=4, steps=8):
        self.lut = self._pal(palette)
        w, h, up = self._down(div)
        self.dw, self.dh, self.up = w, h, up
        self.A = np.ones((h, w), np.float32)
        self.B = np.zeros((h, w), np.float32)
        y, x = np.mgrid[0:h, 0:w]
        self.B[(x - w // 2) ** 2 + (y - h // 2) ** 2 < 16] = 1.0
        self.f0, self.k0, self.steps = float(f), float(k), int(steps)

    def _lap(self, u):
        return (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u)

    def render(self, F, i):
        h, w = self.dh, self.dw
        f = self.f0 + 0.012 * (F.bass[i] - 0.3)
        k = self.k0 + 0.010 * (F.treble[i] - 0.2)
        if F.onset[i]:
            cy, cx = self.rng.integers(4, h - 4), self.rng.integers(4, w - 4)
            yy, xx = np.mgrid[max(0, cy - 3):cy + 4, max(0, cx - 3):cx + 4]
            self.B[yy, xx] = 1.0
        for _ in range(self.steps):
            la, lb = self._lap(self.A), self._lap(self.B)
            reaction = self.A * self.B * self.B
            # classic Gray-Scott: Du=0.16, Dv=0.08, dt=1 (stable 5-point Laplacian)
            self.A += 0.16 * la - reaction + f * (1 - self.A)
            self.B += 0.08 * lb + reaction - (k + f) * self.B
        self.A = np.clip(self.A, 0, 1)
        self.B = np.clip(self.B, 0, 1)
        field = np.clip(1.15 - self.B * 1.6, 0, 1)
        return self.up(apply_palette(field, self.lut))


# --------------------------------------------------------------------------- 10 lenia

@register_scene
class Lenia(Scene):
    """Lenia-style continuous CA (the self-organizing 'creatures' family).
    Audio modulates the time step; onsets poke the medium."""

    name = "lenia"

    def setup(self, palette="sunset_hdmi", div=4, dt=0.12, r=10.0):
        self.lut = self._pal(palette)
        w, h, up = self._down(div)
        self.dw, self.dh, self.up = w, h, up
        y, x = np.mgrid[0:h, 0:w]
        d = np.hypot(x - w / 2, y - h / 2)
        self.kernel = np.clip(np.exp(-((d - r) ** 2) / (2 * (r / 2.2) ** 2)), 0, 1)
        self.kernel /= self.kernel.sum() + 1e-9
        self.state = np.zeros((h, w), np.float32)
        self.state[h // 2 - 5: h // 2 + 5, w // 2 - 5: w // 2 + 5] = \
            self.rng.uniform(0.6, 1.0, (10, 10))
        self.dt0 = float(dt)

    def render(self, F, i):
        if F.onset[i]:
            cy, cx = self.rng.integers(6, self.dh - 6), self.rng.integers(6, self.dw - 6)
            yy, xx = np.mgrid[cy - 3: cy + 4, cx - 3: cx + 4]
            self.state[yy % self.dh, xx % self.dw] += 0.8
        dt = self.dt0 * (0.6 + 1.6 * F.rms[i])
        # convolve with the ring kernel via FFT (cheap at this size)
        s = self.state
        Ff = np.fft.rfft2(s) * np.fft.rfft2(self.kernel, s.shape)
        pot = np.fft.irfft2(Ff, s.shape)
        growth = np.clip(
            2.0 * np.exp(-((pot - 0.135) ** 2) / (2 * 0.015 ** 2)) - 1.0, -1, 1
        )
        s = np.clip(s + dt * growth, 0, 1)
        # slight damping toward quiet passages keeps it alive
        s *= 0.999
        self.state = s
        return self.up(apply_palette(s, self.lut))


# --------------------------------------------------------------------------- 11 lowpoly (PS2)

@register_scene
class LowPoly(Scene):
    """PS2-era terrain flyover, software-rendered: vertex SNAPPING to a coarse
    screen grid (the wobble), flat shading, depth-cue fog."""

    name = "lowpoly"

    def setup(self, palette="ps2_dusk", grid=56, fog=(24, 20, 40), snap=6):
        self.lut = self._pal(palette)
        self.g = int(grid)
        self.snap = int(snap)
        self.fog = np.array(fog, np.float32)
        self.z = 0.0
        self.heights = self._value_noise()

    def _value_noise(self):
        g = self.g
        base = self.rng.uniform(0, 1, (8, 8))
        big = np.kron(base, np.ones((g // 8 + 1, g // 8 + 1)))[: g + 1, : g + 1]
        big = np.resize(big, (g + 1, g + 1))
        big += 0.4 * np.resize(np.kron(self.rng.uniform(0, 1, (16, 16)),
                                       np.ones((g // 16 + 1, g // 16 + 1))), (g + 1, g + 1))
        return big

    def render(self, F, i):
        dt = 1.0 / F.fps
        self.z += dt * (3.0 + 9.0 * F.bass[i])
        g, snap = self.g, self.snap
        zi = self.z
        # world grid, scrolled by z
        ix = np.arange(g + 1)
        wx = ix / g * 2.2 - 1.1
        wz = ix / g * 6.0 + zi % 6.0 - 3.0
        WZ, WX = np.meshgrid(wz, wx, indexing="ij")
        hrow = np.resize(self.heights, (g + 1, g + 1))
        bumps = 0.25 + 0.55 * F.bass[i]
        H = np.sin(WZ * 0.9) * 0.22 * bumps + np.sin(WX * 2.1 + WZ * 0.4) * 0.12 + hrow * 0.18 * bumps
        if F.onset[i]:  # onset: carve a spike near the camera
            H[g // 2, :] += 0.5
        # camera at origin looking down +z, slight pitch
        cam_y, pitch = 0.55, 0.35
        depth = np.maximum(WZ - 0.4, 0.15)
        sx = (WX / depth) * 1.2
        sy = (cam_y - H - WZ * np.tan(pitch * 0.18)) / depth * 1.2 + 0.18
        px = ((sx * 0.5 + 0.5) * (self.w - 1)).astype(np.int32)
        py = ((0.5 - sy * 0.5) * (self.h - 1)).astype(np.int32)
        # THE PS1 TRICK: snap vertices to a coarse grid
        px = (px // snap) * snap
        py = (py // snap) * snap
        frame = np.zeros((self.h, self.w, 3), np.float32)
        frame[:] = self.fog
        for yy in range(g - 1, -1, -1):  # far -> near (painter's algorithm)
            for xx in range(g):
                pts = [(px[yy, xx], py[yy, xx]), (px[yy + 1, xx], py[yy + 1, xx]),
                       (px[yy + 1, xx + 1], py[yy + 1, xx + 1]), (px[yy, xx + 1], py[yy, xx + 1])]
                if max(p[1] for p in pts) < 0 or min(p[1] for p in pts) >= self.h:
                    continue
                if max(p[0] for p in pts) < 0 or min(p[0] for p in pts) >= self.w:
                    continue
                shade = 0.45 + 0.55 * abs(H[yy, xx] - H[yy + 1, xx + 1]) * 2.2
                shade = np.clip(shade * (0.6 + 0.8 * F.rms[i]), 0, 1)
                color = np.asarray(self.lut[int(shade * 255)], np.float32)
                fogmix = np.clip((depth[yy, xx] - 0.4) / 3.2, 0, 0.92)
                color = color * (1 - fogmix) + self.fog * fogmix
                x0, x1 = max(0, min(p[0] for p in pts)), min(self.w - 1, max(p[0] for p in pts))
                y0, y1 = max(0, min(p[1] for p in pts)), min(self.h - 1, max(p[1] for p in pts))
                if x1 > x0 and y1 > y0:
                    frame[y0:y1, x0:x1] = color
        return np.clip(frame, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- 12 mandelbrot

@register_scene
class Mandel(Scene):
    """Mandelbrot zoom at quarter res. Zoom accelerates with loudness;
    onsets snap back out. Palette by smooth escape count."""

    name = "mandel"

    def setup(self, palette="inferno", div=4, iters=48, zoom_rate=0.010):
        self.lut = self._pal(palette)
        w, h, up = self._down(div)
        self.dw, self.dh, self.up = w, h, up
        w2, h2 = self.w // div, self.h // div
        ys, xs = np.mgrid[0:h2, 0:w2]
        self.cx0 = (xs / w2 * 3.5 - 2.5).astype(np.float64)
        self.cy0 = (ys / h2 * 2.0 - 1.0).astype(np.float64)
        self.iters = int(iters)
        self.zr = 0.010
        self.logz = np.log(0.9)
        self.target = (-0.7436438870371587, 0.13182590420531197)  # deep seahorse valley

    def render(self, F, i):
        if F.onset[i]:
            self.logz -= 0.9  # snap back out
        self.logz += (0.004 + 0.030 * F.rms[i]) * self.params.get("zoom_rate", 1.0) / 0.010
        scale = np.exp(self.logz)
        cx, cy = self.target
        zr = self.cx0 * scale + cx
        zi = self.cy0 * scale + cy
        Zr = np.zeros_like(zr)
        Zi = np.zeros_like(zi)
        out = np.zeros(zr.shape, np.float64)
        for n in range(self.iters):
            alive = (Zr * Zr + Zi * Zi) < 4.0
            # pin escaped points to the target so inf/nan never propagate
            t = np.where(alive, Zr * Zr - Zi * Zi + zr, zr)
            Zi = np.where(alive, 2 * Zr * Zi + zi, 0.0)
            Zr = t
            out += alive
        out /= self.iters
        smooth = np.sqrt(out) * (0.75 + 0.5 * np.sin(i * 0.02 + F.centroid[i] * 0.002))
        field = np.clip(smooth, 0, 1)
        return self.up(apply_palette(field, self.lut))


# --------------------------------------------------------------------------- 13 worms

@register_scene
class Worms(Scene):
    """Minsky-oscillator harmonographs drawing onto a decaying canvas.
    Treble bends the feedback; onsets re-seed the oscillator pairs."""

    name = "worms"

    def setup(self, palette="vaporwave", n=14):
        self.lut = self._pal(palette)
        self.n = int(n)
        self.canvas = np.zeros((self.h, self.w), np.float32)
        self._seed()

    def _seed(self):
        self.state = self.rng.uniform(-2, 2, (self.n, 4))  # x, y, dx, dy

    def render(self, F, i):
        if F.onset[i]:
            self._seed()
        damp = 0.9985 - 0.004 * F.treble[i]
        s = self.state
        for k in range(self.n):
            x, y, dx, dy = s[k]
            for _ in range(12):  # minsky integration steps
                dx += y * (0.06 + 0.05 * F.bass[i])
                x += dx
                dy -= x * (0.06 + 0.02 * F.treble[i])
                y += dy
            s[k] = (x * damp, y * damp, dx * damp, dy * damp)
            xi = np.clip(int((x * 0.22 + 0.5) * (self.w - 1)), 0, self.w - 1)
            yi = np.clip(int((0.5 - y * 0.22) * (self.h - 1)), 0, self.h - 1)
            self.canvas[yi, xi] = 1.0
        self.canvas *= 0.985
        field = np.clip(self.canvas * (0.6 + 0.8 * F.rms[i]), 0, 1)
        return apply_palette(field * 0.95 + 0.02, self.lut)


# --------------------------------------------------------------------------- 14 metaballs

@register_scene
class Metaballs(Scene):
    """Isosurface blobs from the set's own energy: radius = bass, count
    jumps on onsets. Quarter-res field, palette-mapped."""

    name = "metaballs"

    def setup(self, palette="blood", n=8, div=3):
        self.lut = self._pal(palette)
        w, h, up = self._down(div)
        self.dw, self.dh, self.up = w, h, up
        self.n = int(n)
        self.balls = self.rng.uniform(0.15, 0.85, (self.n, 2))
        self.vel = self.rng.uniform(-0.05, 0.05, (self.n, 2))
        self.active = max(3, self.n // 2)

    def render(self, F, i):
        dt = 1.0 / F.fps
        if F.onset[i]:
            self.active = min(self.n, self.active + 1)
            self.balls = self.rng.uniform(0.2, 0.8, (self.n, 2))
            self.vel = self.rng.uniform(-0.08, 0.08, (self.n, 2))
        self.vel += (self.rng.uniform(-1, 1, self.vel.shape) * 0.4 * F.flux[i] * dt)
        self.balls += self.vel * (0.5 + 2.0 * F.rms[i])
        self.balls[:] = (self.balls + 0.2) % 1.4 - 0.2
        ys, xs = np.mgrid[0:self.dh, 0:self.dw]
        field = np.zeros((self.dh, self.dw), np.float32)
        rad = (0.10 + 0.09 * F.bass[i]) ** 2
        for k in range(self.active):
            bx, by = self.balls[k]
            field += rad / ((xs / self.dw - bx) ** 2 + (ys / self.dh - by) ** 2 + 0.003)
        field = np.clip(field * (0.7 + 0.9 * F.rms[i]) - 0.35, 0, 1)
        return self.up(apply_palette(field, self.lut))


# --------------------------------------------------------------------------- 15 feedback

@register_scene
class Feedback(Scene):
    """Video feedback: the previous frame zoomed, rotated and re-graded every
    step — demoscene tunnel material. Bass drives the zoom, treble the hue."""

    name = "feedback"

    def setup(self, palette="vhs_night", zoom=1.012, rot=0.006, sustain=0.93):
        self.lut = self._pal(palette)
        self.buf = None
        self.z0, self.r0, self.s0 = float(zoom), float(rot), float(sustain)

    def render(self, F, i):
        h, w = self.h, self.w
        if self.buf is None:
            seed = apply_palette(self.rng.uniform(0, 1, (h, w)), self.lut)
            self.buf = seed.astype(np.float32)
        zoom = self.z0 + 0.02 * F.bass[i]
        rot = self.r0 + 0.004 * F.treble[i]
        ang = np.arctan2(self.Y, self.X) + rot
        rr = np.hypot(self.X, self.Y) * zoom
        ux = np.clip((rr * np.cos(ang) * 0.5 + 0.5) * (w - 1), 0, w - 1)
        uy = np.clip((0.5 - rr * np.sin(ang) * 0.5) * (h - 1), 0, h - 1)
        xi, yi = ux.astype(np.int32), uy.astype(np.int32)
        nxt = self.buf[yi, xi] * (self.s0 - 0.25 * F.rms[i])
        # keep it alive: inject a bit of fresh palette noise, more on onsets
        inject = 0.02 + (0.35 if F.onset[i] else 0.0) + 0.08 * F.flux[i]
        grain = apply_palette(self.rng.uniform(0, 1, (h, w)), self.lut).astype(np.float32)
        nxt = np.clip(nxt + grain * inject, 0, 255)
        self.buf = nxt
        # hue drift via palette rotation of the luminance
        lum = nxt.mean(axis=2) / 255.0
        shift = (i * (0.2 + F.treble[i] * 2.0)) % 256
        lut = np.roll(self.lut, int(shift), axis=0)
        return apply_palette(np.clip(lum, 0, 1), lut)

# --------------------------------------------------------------------------- 16 death mask

@register_scene
class DeathMask(Scene):
    """Newton's black death mask: a symmetric bas-relief face lit by a
    rotating raking light — chiaroscuro in the bronze. Fully vectorized:
    the heightfield's per-frame weights breathe with the music."""

    name = "mask"

    def setup(self, palette="black_bronze", div=2, drift=0.06,
             mode="relief", contour_levels=14):
        self.lut = self._pal(palette)
        self.mode = mode
        self.contour_levels = float(contour_levels)
        w, h, up = self._down(div)
        self.dw, self.dh, self.up = w, h, up
        ys, xs = np.mgrid[0:h, 0:w]
        X = xs / w          # 0..1 left-right
        Y = ys / h          # 0..1 top-bottom
        cx = X - 0.5

        def g(x0, y0, sx, sy):
            return np.exp(-(((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2))

        def sym(x0, y0, sx, sy):
            # symmetric pair about the center line
            return g(0.5 - abs(x0 - 0.5) if x0 < 0.5 else x0, y0, sx, sy) + \
                   g(0.5 + abs(x0 - 0.5) if x0 > 0.5 else x0, y0, sx, sy)

        # face components (all symmetric, mirror the left half)
        left = lambda x0, y0, sx, sy: g(x0, y0, sx, sy) + g(1.0 - x0, y0, sx, sy)
        self.parts = {
            "brow":  left(0.28, 0.415, 0.16, 0.020) * 0.9,
            "socket": -left(0.31, 0.455, 0.055, 0.032) * 1.6,
            "nose": g(0.5, 0.55, 0.028, 0.085) * 1.35,
            "nosetip": g(0.5, 0.615, 0.032, 0.028) * 1.1,
            "cheek": left(0.255, 0.565, 0.075, 0.055) * 0.75,
            "mouth": -g(0.5, 0.745, 0.085, 0.012) * 0.9,
            "chin": g(0.5, 0.825, 0.075, 0.045) * 0.8,
            "forehead": g(0.5, 0.33, 0.19, 0.10) * 0.5,
        }
        # skin + sculpt noise, mirrored for symmetry
        noise = (np.sin(X * 19 + Y * 7) * np.sin(Y * 23 - X * 11)
                 + 0.5 * np.sin((X + Y) * 31)) * 0.03
        self.skin = noise + np.fliplr(noise)  # mirrored -> symmetric relief
        # oval mask silhouette: everything outside the oval drops away
        oval = (((cx) / 0.40) ** 2 + ((Y - 0.50) / 0.50) ** 2)
        self.inside = np.clip(1.25 - oval, 0, 1) ** 0.6
        self.base = np.zeros((h, w), np.float32)
        self.ang = self.rng.uniform(0, 6.28)
        self.flash = 0.0
        self.zoom = 1.0
        self.drift = float(drift)

    def render(self, F, i):
        dt = 1.0 / F.fps
        # the mask breathes: features swell with loudness, sockets deepen
        breathe = 0.85 + 0.30 * F.rms[i]
        h = self.base + self.skin * 0.5
        h = (h + breathe * sum(self.parts.values())) * self.inside
        # light: raking, rotating slowly; onsets flash and swing it
        self.ang += dt * (0.10 + 0.25 * F.bass[i])
        self.flash *= 0.88
        if F.onset[i]:
            self.flash = 1.0
        L = np.array([np.cos(self.ang), np.sin(self.ang) * 0.5 + 0.3, 0.62 + 0.38 * self.flash])
        L = L / np.linalg.norm(L)
        gy, gx = np.gradient(h)
        gx, gy = -gx * 60.0, -gy * 60.0  # exaggerated relief normals
        nz = np.ones_like(h)
        norm = np.sqrt(gx * gx + gy * gy + 1.0)
        shade = np.clip((gx * L[0] + gy * L[1] + nz * L[2]) / norm, 0, 1)
        # dramatic tone: crush lows, hot speculars
        field = shade ** (2.2 - 0.7 * self.flash)
        field = field * (0.75 + 0.5 * F.rms[i]) + 0.02
        if self.flash > 0.05:  # rim-light kick on onsets
            rim = np.clip(np.abs(gx * L[0] + gy * L[1]) / norm * 1.6, 0, 1)
            field = np.clip(field + self.flash * 0.35 * rim, 0, 1)
        if self.mode in ("contour", "engrave"):
            # topographic engraving lines of the heightfield
            v = (h * self.contour_levels) % 1.0
            lines = np.clip(1.0 - np.abs(v - 0.5) * 2.6, 0, 1) ** 2
            field = np.clip(0.10 + 0.75 * lines + 0.35 * field, 0, 1)
        elif self.mode == "both":
            v = (h * self.contour_levels) % 1.0
            lines = np.clip(1.0 - np.abs(v - 0.5) * 2.6, 0, 1) ** 2
            field = np.clip(field * 0.6 + lines * 0.45, 0, 1)
        return self.up(apply_palette(np.clip(field, 0, 1), self.lut))


# --------------------------------------------------------------------------- 17 newton fractal

@register_scene
class NewtonFractal(Scene):
    """Newton-Raphson basins of attraction for z^3 - 1: the fractal literally
    named after the man. Zoom rides the bass; onsets jolt the basin hues."""

    name = "newton"

    def setup(self, palette="verdigris", div=3, iters=20, zoom_rate=0.006):
        self.lut = self._pal(palette)
        w, h, up = self._down(div)
        self.dw, self.dh, self.up = w, h, up
        ys, xs = np.mgrid[0:h, 0:w]
        self.X0 = (xs / w * 2 - 1).astype(np.complex128) * (w / h)
        self.Y0 = (ys / h * 2 - 1).astype(np.complex128) * 1j
        self.iters = int(iters)
        self.logz = np.log(0.75)
        self.hue_shift = 0
        self.zr_rate = float(zoom_rate)
        self.rot = 0.0

    def render(self, F, i):
        if F.onset[i]:
            self.hue_shift = (self.hue_shift + 85) % 256
            self.logz -= 0.22  # snap the zoom back on transients
        self.logz += self.zr_rate * (0.4 + 2.2 * F.bass[i])
        self.rot += 0.0025 * (1.0 + 3.0 * F.treble[i] * 0.0 + F.mid[i])
        scale = np.exp(self.logz)
        c = np.exp(1j * self.rot)
        z = (self.X0 * 1.0 + self.Y0) * scale * c
        steps = np.zeros(z.shape, np.float32)
        frozen = np.zeros(z.shape, dtype=bool)  # points that blew up: park them
        for n in range(self.iters):
            dz = (z * z * z - 1.0) / (3.0 * z * z + 1e-12)
            z = z - dz
            big = np.abs(z) > 1e8
            frozen |= big
            if frozen.any():
                z = np.where(frozen, 0.0, z)  # 0 sits at a basin center
            steps += ((np.abs(z * z * z - 1.0) > 0.004) & ~frozen)
        # which of the three roots?
        roots = np.exp(2j * np.pi * np.arange(3) / 3)
        dist = np.stack([np.abs(z - r) for r in roots])
        which = dist.argmin(axis=0)
        basin = which / 3.0
        speed = 1.0 - steps / self.iters  # fast convergence = bright
        field = (0.15 + 0.85 * speed) * (0.35 + 0.65 * basin)
        field = np.clip(field * (0.7 + 0.6 * F.rms[i]), 0, 1)
        lut = np.roll(self.lut, self.hue_shift, axis=0)
        return self.up(apply_palette(field, lut))
