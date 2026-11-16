"""organs: joint audiovisual generators — womb's core.

An organ is ONE evolving system heard and seen simultaneously. Its state is
projected to both senses each frame: `render()` is the picture, `emit()` is
the sound of the SAME step. Sync is not feature-driven (doom) or
post-hoc (muxing) — it is structural: the standing wave you see IS the
tone you hear; the cell births you see ARE the grains you hear.

This is womb's answer to "generate original audiovisuals by novel
approaches": five substrate families, each a different physics of seeing
and hearing:

  waveplate   2D wave equation on a plate; audio = real probe readouts
              (Chladni physics: the visible modes are the audible tones)
  automata    continuous cellular automaton; births trigger corpus grains,
              column density plays as a live spectrum of the picture
  swarm       agents on a region grid; occupancy = voice gains, boundary
              crossings = grains, positions = pan/pitch of the voices
  kuramoto    coupled phase oscillators; phase field seen AND summed as
              the literal waveform of the grid's voices
  territory   the corpus itself as a 2D territory (feature map); a walker
              tours YOUR material — the trajectory is the piece, heard as
              corpus chunks and seen as the map being traversed

Corpus grains: any folder of wavs (loom output, the EP, anything) becomes
the organ's voice material. No corpus = organs fall back to their own
synthesis. Register your own with @register_organ.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from doom.palettes import apply_palette, get_palette

__all__ = ["ORGANS", "register_organ", "make_organ", "list_organs", "load_corpus_bank"]

SR = 44100
ORGANS: dict[str, type["Organ"]] = {}


def register_organ(cls):
    ORGANS[cls.name] = cls
    return cls


def list_organs() -> list[str]:
    return sorted(ORGANS)


def make_organ(name, w, h, fps, seed, corpus=None, **params) -> "Organ":
    if name not in ORGANS:
        raise ValueError(f"unknown organ {name!r}; have {list_organs()}")
    if corpus is not None and not isinstance(corpus, CorpusBank):
        corpus = load_corpus_bank(corpus, seed=seed)   # str/Path -> bank
    return ORGANS[name](w, h, fps, seed, corpus=corpus, **params)


# ------------------------------------------------------------------ corpus

class CorpusBank:
    """Grain material from a folder of audio files (loom output, the EP...)."""

    def __init__(self, paths, sr=SR, max_files=24, max_seconds=20.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.clips = []
        for p in paths[: max_files]:
            try:
                out = subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(p),
                     "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
                    capture_output=True, check=True,
                ).stdout
                x = np.frombuffer(out, np.float32)
                if len(x) > int(max_seconds * sr):
                    x = x[: int(max_seconds * sr)]
                if len(x) > sr // 2:
                    self.clips.append(x)
            except Exception:
                continue
        if not self.clips:
            raise ValueError(f"no usable audio in corpus ({len(paths)} paths tried)")

    def grain(self, seconds: float, sr: int = SR) -> np.ndarray:
        """A random corpus grain with an edge-safe cosine fade."""
        c = self.clips[int(self.rng.integers(len(self.clips)))]
        n = max(256, int(seconds * sr))
        start = int(self.rng.integers(0, max(1, len(c) - n)))
        g = c[start: start + n].copy()
        if len(g) < n:
            g = np.pad(g, (0, n - len(g)))
        f = min(len(g) // 6, 256)
        if f > 8:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(f) / f)
            g[:f] *= ramp
            g[-f:] *= ramp[::-1]
        return g


def load_corpus_bank(corpus, sr=SR, seed=0) -> CorpusBank | None:
    if corpus is None:
        return None
    p = Path(corpus)
    if p.is_dir():
        paths = sorted(q for q in p.iterdir() if q.suffix.lower() == ".wav")
    elif p.is_file():
        paths = [p]
    else:
        return None
    if not paths:
        return None
    return CorpusBank(paths, sr=sr, seed=seed)


# ------------------------------------------------------------------ base

class Organ:
    name = "?"

    def __init__(self, w, h, fps, seed, corpus=None, palette="inferno", **params):
        self.w, self.h, self.fps = int(w), int(h), float(fps)
        self.hop = int(round(SR / fps))
        self.rng = np.random.default_rng(seed)
        self.params = dict(params or {})
        self.lut = get_palette(self.params.get("palette", palette))
        self.bank = corpus if isinstance(corpus, CorpusBank) else None
        self._pending: list[tuple[int, float, float]] = []  # (in_hop_offset, gain, pitch)
        self.setup(**params)

    def setup(self, **params) -> None:
        pass

    def step(self, i: int) -> None:
        pass

    def render(self) -> np.ndarray:
        raise NotImplementedError

    def emit(self) -> np.ndarray:
        """The frame's audio hop: the organ's own voices + queued grains."""
        hop = self.hop
        out = self.voices(hop).astype(np.float32)
        for offset, gain, pitch in self._pending:
            g = self._grain(pitch)
            at = int(offset)
            if at >= hop:
                continue
            seg = g[: hop - at]
            out[at: at + len(seg)] += seg * gain
        self._pending = []
        return out

    def voices(self, hop: int) -> np.ndarray:
        return np.zeros(hop, np.float32)

    def _grain(self, pitch: float) -> np.ndarray:
        """A corpus grain pitch-shifted by `pitch` (1.0 = original)."""
        seconds = 0.09 + 0.14 / max(0.25, pitch)
        if self.bank is not None:
            g = self.bank.grain(seconds)
        else:  # no corpus: a synthetic ping so events still sound
            n = int(seconds * SR)
            t = np.arange(n) / SR
            g = (np.exp(-t * 9) * np.sin(2 * np.pi * 320 * t) * 0.4).astype(np.float32)
        if abs(pitch - 1.0) > 1e-3:
            n_new = max(64, int(len(g) / pitch))
            pos = np.linspace(0, len(g) - 1, n_new)
            g = np.interp(pos, np.arange(len(g)), g).astype(np.float32)
        return g

    def event(self, at_frac: float, gain: float, pitch: float) -> None:
        """Queue a grain for this frame at a fraction of the hop."""
        self._pending.append((int(at_frac * self.hop), gain, pitch))

    # -- visual helpers ---------------------------------------------------
    def _up(self, field_hw: np.ndarray) -> np.ndarray:
        """Nearest-upscale a low-res (h, w) field to the organ's canvas."""
        H, W = field_hw.shape
        yi = np.arange(self.h) * H // self.h
        xi = np.arange(self.w) * W // self.w
        big = field_hw[np.ix_(yi, xi)]
        return apply_palette(np.clip(big, 0, 1), self.lut)


# ------------------------------------------------------------------ 1 waveplate

@register_organ
class WavePlate(Organ):
    """A struck plate: 2D wave equation (FDL) with damping. The picture is
    the displacement field; the sound is the plate read at probe points —
    the visible modes are the audible ones. Onsets strike the plate."""

    name = "waveplate"

    def setup(self, grid=96, damp=0.9985, strike_every=12.0):
        self.n = int(grid)
        self.damp = float(damp)
        self.u = np.zeros((self.n, self.n), np.float32)
        self.u_prev = np.zeros_like(self.u)
        self.k = 0
        self.strike_every = float(strike_every)
        # probe points (read out into audio)
        r = np.linspace(0.2, 0.8, 5)
        self.probes = [(int(a * self.n), int(b * self.n)) for a, b in zip(r, r[::-1])]

    def step(self, i: int):
        lap = (np.roll(self.u, 1, 0) + np.roll(self.u, -1, 0)
               + np.roll(self.u, 1, 1) + np.roll(self.u, -1, 1) - 4 * self.u)
        nxt = (2 * self.u - self.u_prev + 0.42 * lap) * self.damp
        self.u_prev, self.u = self.u, nxt.astype(np.float32)
        self.k += 1
        if i % max(1, int(self.strike_every)) == 0:
            cy, cx = self.rng.integers(self.n // 4, 3 * self.n // 4, 2)
            yy, xx = np.mgrid[cy - 3: cy + 4, cx - 3: cx + 4]
            self.u[yy, xx] += self.rng.uniform(0.8, 1.4)
        if self.rng.random() < 0.01:  # soft rain of small strikes
            cy, cx = self.rng.integers(0, self.n, 2)
            self.u[cy, cx] += 0.3

    def voices(self, hop: int) -> np.ndarray:
        t = np.arange(hop) / SR
        out = np.zeros(hop, np.float32)
        for k, (py, px) in enumerate(self.probes):
            v = float(self.u[max(0, py - 1): py + 2, max(0, px - 1): px + 2].mean())
            if abs(v) < 1e-5:
                continue
            # the probe value drives BOTH a direct contact thump and a
            # resonant partial whose frequency tracks the probe height
            freq = 180 + 900 * abs(v)
            out += (np.sin(2 * np.pi * freq * t) * np.exp(-t * 18) * v * 0.22).astype(np.float32)
        return out

    def render(self) -> np.ndarray:
        f = 0.5 + 0.5 * np.tanh(self.u * 2.5)
        return self._up(f)


# ------------------------------------------------------------------ 2 automata

@register_organ
class Automata(Organ):
    """A smooth continuous CA. Cell births fire corpus grains (pitch = row);
    the column histogram of the state plays as a live spectrum — the
    picture's own shape is the sound's own shape."""

    name = "automata"

    def setup(self, grid=110, feed=0.045, kill=0.065):
        self.n = int(grid)
        y, x = np.mgrid[0: self.n, 0: self.n]
        self.state = np.clip(np.exp(-(((x - self.n / 2) ** 2 + (y - self.n / 2) ** 2)) / 300), 0, 1).astype(np.float32)
        self.prev_state = self.state.copy()
        self.freqs = np.geomspace(60, 4000, 48)
        self.phase = np.zeros(48, np.float64)
        self.f0, self.k0 = float(feed), float(kill)

    def step(self, i: int):
        s = self.state
        lap = (np.roll(s, 1, 0) + np.roll(s, -1, 0)
               + np.roll(s, 1, 1) + np.roll(s, -1, 1) - 4 * s) * 0.16
        ds = np.clip(s + lap * 1.2 - s * s * (0.9 + 0.05 * np.sin(i * 0.01)), 0, 1)
        ds += (self.rng.random(s.shape) < 0.00008) * self.rng.uniform(0.4, 0.9)
        births = (ds > 0.55) & (s <= 0.55)
        self.prev_state = s
        self.state = ds.astype(np.float32)
        # births -> grains (a few per frame max)
        ys, xs = np.nonzero(births)
        for k in range(min(3, len(ys))):
            row = ys[k] / self.n
            self.event(self.rng.random(), 0.5, 0.5 + 1.5 * (1 - row))

    def voices(self, hop: int) -> np.ndarray:
        # 48-band spectrum = mean of the state by row bands (the picture's shape)
        n = self.n
        edges = np.linspace(0, n, len(self.freqs) + 1).astype(int)
        band = np.array([self.state[a: b].mean() if b > a else 0.0
                         for a, b in zip(edges[:-1], edges[1:])])
        mags = np.clip(band * 2.2, 0, 1) * 0.35
        t = np.arange(hop) / SR
        self.phase += 2 * np.pi * self.freqs * (hop / SR)
        return (np.sin(self.phase[:, None] + 2 * np.pi * self.freqs[:, None] * t[None, :])
                * mags[:, None]).sum(axis=0).astype(np.float32)

    def render(self) -> np.ndarray:
        return self._up(self.state)


# ------------------------------------------------------------------ 3 swarm

@register_organ
class Swarm(Organ):
    """Agents on a region grid (6x4). Occupancy of a region = its voice's
    gain (pentatonic by region); a boundary crossing fires a grain pitched
    by the region. You hear the traffic you see."""

    name = "swarm"

    def setup(self, agents=140, cols=6, rows=4):
        self.n_agents = int(agents)
        self.cols, self.rows = int(cols), int(rows)
        self.pos = self.rng.uniform(0.05, 0.95, (self.n_agents, 2))
        self.vel = self.rng.uniform(-0.15, 0.15, (self.n_agents, 2))
        self.trail = np.zeros((int(self.h * 0.5), int(self.w * 0.5)), np.float32)
        self.freqs = self._pentatonic(self.cols * self.rows)
        self.phase = np.zeros(self.cols * self.rows, np.float64)
        self.occ = np.zeros(self.cols * self.rows, np.float32)

    @staticmethod
    def _pentatonic(n, base=110.0):
        scale = np.array([1, 9 / 8, 5 / 4, 3 / 2, 5 / 3])
        return base * scale[np.arange(n) % 5] * (2 ** (np.arange(n) // 5))

    def step(self, i: int):
        p, v = self.pos, self.vel
        # flocking-lite: pull toward center + avoid crowding
        center = p.mean(axis=0)
        v += (center - p) * 0.9 / self.fps
        dist2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1) + 1e-6
        close = dist2 < 0.004
        push = ((p[:, None, :] - p[None, :, :]) * close[..., None]).sum(1)
        v += push * 4.0 / self.fps
        v += self.rng.normal(0, 0.25 / self.fps, v.shape)
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(speed, 1e-5) * np.clip(speed, 0.12, 0.5)
        old_region = self._region(p)
        p += v / self.fps
        p[:] = np.clip(p, 0.02, 0.98)
        new_region = self._region(p)
        for k in np.nonzero(new_region != old_region)[0][:4]:
            r = int(new_region[k])
            self.event(self.rng.random(), 0.35, self.freqs[r] / 440.0)
        self.occ = np.bincount(new_region, minlength=self.cols * self.rows).astype(np.float32)

    def _region(self, p):
        cx = np.clip((p[:, 0] * self.cols).astype(int), 0, self.cols - 1)
        cy = np.clip((p[:, 1] * self.rows).astype(int), 0, self.rows - 1)
        return cy * self.cols + cx

    def voices(self, hop: int) -> np.ndarray:
        t = np.arange(hop) / SR
        gains = np.sqrt(self.occ / max(1, self.occ.max() + 1e-9)) * 0.18
        self.phase += 2 * np.pi * self.freqs * (hop / SR)
        return (np.sin(self.phase[:, None] + 2 * np.pi * self.freqs[:, None] * t[None, :])
                * gains[:, None]).sum(axis=0).astype(np.float32)

    def render(self) -> np.ndarray:
        th, tw = self.trail.shape
        self.trail *= 0.90
        xi = (self.pos[:, 0] * (tw - 1)).astype(int)
        yi = (self.pos[:, 1] * (th - 1)).astype(int)
        np.add.at(self.trail, (yi, xi), 0.8)
        field = np.clip(self.trail, 0, 1)
        big = apply_palette(field, self.lut)
        return np.asarray(np.clip(np.repeat(np.repeat(big, 2, 0), 2, 1)[: self.h, : self.w], 0, 255), np.uint8)


# ------------------------------------------------------------------ 4 kuramoto

@register_organ
class Kuramoto(Organ):
    """A grid of coupled phase oscillators. The phase field is seen directly,
    and the SAME phases are summed as voices — synchronization waves become
    audible beating, drift becomes shimmer."""

    name = "kuramoto"

    def setup(self, grid=64, coupling=1.4, noise=0.02):
        self.n = int(grid)
        self.phase = self.rng.uniform(0, 2 * np.pi, (self.n, self.n)).astype(np.float32)
        self.omega = (np.sin(np.linspace(0, np.pi, self.n))[None, :]
                      + np.cos(np.linspace(0, np.pi, self.n)[:, None])).astype(np.float32) * 2.0 + 1.0
        self.K, self.noise = float(coupling), float(noise)
        # one voice per row band, frequency laid out low rows = low tones
        self.freqs = np.geomspace(70, 900, 12)
        self.vphase = np.zeros(12, np.float64)

    def step(self, i: int):
        mean_n = np.roll(self.phase, 1, 0) + np.roll(self.phase, -1, 0) \
                 + np.roll(self.phase, 1, 1) + np.roll(self.phase, -1, 1)
        d = (self.K / 4.0) * np.sin(mean_n - 4 * self.phase)
        self.phase = (self.phase + (self.omega * 0.016 + d * 0.045
                                    + self.noise * self.rng.normal(size=self.phase.shape))) % (2 * np.pi)

    def voices(self, hop: int) -> np.ndarray:
        t = np.arange(hop) / SR
        edges = np.linspace(0, self.n, 13).astype(int)
        bands = np.array([self.phase[a: b].mean(axis=0)
                          for a, b in zip(edges[:-1], edges[1:])])  # (12, n)
        coherent = np.abs(np.exp(1j * bands).mean(axis=1))              # sync per band
        spread = bands.std(axis=1)
        gains = (0.05 + 0.16 * coherent) * 1.0
        self.vphase += 2 * np.pi * self.freqs * (hop / SR) * (1 + spread * 0.02)
        out = (np.sin(self.vphase[:, None] + 2 * np.pi * self.freqs[:, None] * t[None, :])
               * gains[:, None]).sum(axis=0)
        # faint raw-field term: the grid's phase sum as audible shimmer
        if hop <= self.phase.shape[1]:
            out = out + 0.015 * np.sin(self.phase[0, : hop].astype(np.float64) * 3.0)
        return out.astype(np.float32)

    def render(self) -> np.ndarray:
        f = (self.phase % (2 * np.pi)) / (2 * np.pi)
        return self._up(f)


# ------------------------------------------------------------------ 5 territory

@register_organ
class Territory(Organ):
    """The corpus as territory: every corpus window lands on a 2D map of
    its own features (brightness vs loudness). A walker tours the map; at
    each stop it plays the chunk it stands on and lights the map around
    itself. The trajectory through YOUR material is the piece."""

    name = "territory"

    def setup(self, window=2.0, step_s=1.0, hop_len=0.25):
        self.window = int(window * SR)
        self.chunks, self.pts = self._build_map()
        self.walker = self._seed_walker()
        self.step_s = float(step_s)
        self.hop_len = float(hop_len)
        self.frame = 0
        self.glow = np.zeros_like(self.pts, np.float32)
        self.playing: np.ndarray | None = None
        self.play_pos = 0
        self.pos = (0.5, 0.5)
        self._seg = np.zeros(0, np.float32)

    def _build_map(self):
        if self.bank is None or not self.bank.clips:
            raise ValueError("territory organ requires a corpus")
        X = np.concatenate(self.bank.clips)
        n_chunks = max(4, min(240, len(X) // self.window))
        stride = max(self.window, (len(X) - self.window) // max(1, n_chunks))
        chunks, pts = [], []
        for k in range(n_chunks):
            c = X[k * stride: k * stride + self.window]
            if len(c) < self.window // 2:
                break
            rms = float(np.sqrt(np.mean(c * c)) + 1e-9)
            seg = c[:1024]
            if len(seg) < 1024:
                seg = np.pad(seg, (0, 1024 - len(seg)))
            S = np.abs(np.fft.rfft(seg * np.hanning(1024)))
            fr = np.fft.rfftfreq(1024, 1 / SR)
            cent = float((fr * S).sum() / (S.sum() + 1e-9))
            chunks.append(c)
            pts.append((min(1.0, rms * 3), min(1.0, cent / 8000)))
        return chunks, np.asarray(pts, np.float32)

    def _seed_walker(self):
        # Lissajous + noise walk over the map
        a, b = self.rng.uniform(1, 3, 2)
        return dict(t=0.0, speed=self.rng.uniform(0.05, 0.11), a=a, b=b,
                    drift=self.rng.uniform(-0.3, 0.3))

    def step(self, i: int):
        self.frame += 1
        w = self.walker
        w["t"] += w["speed"] / self.fps
        x = 0.5 + 0.42 * np.sin(2 * np.pi * w["a"] * w["t"] * 6.28)
        y = 0.5 + 0.42 * np.sin(2 * np.pi * w["b"] * w["t"] * 6.28 + w["drift"])
        self.pos = (np.clip(x, 0, 1), np.clip(y, 0, 1))
        # nearest chunk -> start playing when the previous one ended
        if self.playing is None or self.play_pos >= len(self.playing):
            d = np.abs(self.pts - np.array(self.pos)).sum(axis=1)
            k = int(np.argmin(d))
            self.playing = self.chunks[k].astype(np.float32) * 0.9
            self.play_pos = 0
            self.glow *= 0.0
            self.glow[k] = 1.0
        take = int(self.hop_len * SR)
        seg = self.playing[self.play_pos: self.play_pos + take]
        self._seg = seg
        self.play_pos += self.hop

    def voices(self, hop: int) -> np.ndarray:
        seg = getattr(self, "_seg", np.zeros(0, np.float32))
        out = np.zeros(hop, np.float32)
        if len(seg):
            out[: len(seg[: hop])] += seg[: hop]
        return out

    def render(self) -> np.ndarray:
        H, W = self.h, self.w
        canvas = np.zeros((H, W), np.float32)
        # map points
        px = (self.pts[:, 0] * (W - 1)).astype(int)
        py = ((1 - self.pts[:, 1]) * (H - 1)).astype(int)
        canvas[py, px] = 0.55
        # glow around the current chunk
        gy = int((1 - self.pos[1]) * (H - 1))
        gx = int(self.pos[0] * (W - 1))
        ys, xs = np.mgrid[max(0, gy - 30): min(H, gy + 31), max(0, gx - 30): min(W, gx + 31)]
        r2 = (ys - gy) ** 2 + (xs - gx) ** 2
        canvas[ys, xs] = np.maximum(canvas[ys, xs], np.exp(-r2 / 90.0))
        # walker + trail
        canvas[gy, gx] = 1.0
        return apply_palette(np.clip(canvas, 0, 1), self.lut)