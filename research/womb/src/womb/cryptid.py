"""cryptid: sound-objects — born, not designed (CRYPTID.md, phase A).

The thesis: traditional synthesis is *designed oscillation plus designed
shaping*, so it is bounded by what a designer can specify. A cryptid is
the opposite: an unbounded network of dynamical laws whose states are
simulated, and an *ear* — an evolved measurement operator — that reads
them out. Sound = observation of a natural process, the way a seismograph
is the "synthesizer" of plate tectonics.

The invariant (enforced by construction, not discipline):

    the only signals that exist are states of evolved processes. No
    oscillators, no notes, no scales, no filters-as-tone, no instrument
    samples, no corpus playback. The ear selects, gates, differences,
    delay-embeds, packs and projects — it never adds energy. The sole
    designed stage is the safety limiter.

Laws (the zoo — @law-registered, open like everything in womb):

    jerk       evolved Sprott-style chaos x''' = a x'' + b x' + c x + d x^3
    osc_pop    a population of Kuramoto units (ring / global kernels,
               phase lag) — chimeras live here
    map        chaotic logistic map, slow deterministic driver
    sand       BTW sandpile: self-organized criticality, power-law weather
    excitable  FitzHugh-Nagumo cousin: rest / spike / refractory
    fire       integrate-and-fire unit with refractory period
    bool       Kauffman Boolean regulatory unit (non-metric gates)
    delay      delayed state feedback ring (echo-chaos)
    clamp      nonlinear quantizer (aliasing as a component, not a post)
    bytefeed   any file's bytes as a state stream (databending species)

Ears are measurement ops: tap / diff / embed (Takens delay-difference) /
events (threshold crossings as impulses) / chimera (coherence-blended
window readout of an osc_pop). Edges wire node channels into node inputs,
with slots so a law can tell its inputs apart.

Three hand-built champions prove the physics (params found by search on
this machine, see notes in each genome):

    chimera      ring of 128+128 Kuramoto units — a coherence boundary
                 that wanders the ring; tonal where coherent, drifting
                 whistles where not
    clickstorm   a 40x40 sandpile driven by chaotic maps — 1/f click
                 weather with power-law amplitudes and causal pan
    jerk_voice   an evolved jerk attractor whose stiffness drifts along
                 a slow chaotic map — an inharmonic voice that never
                 settles, doubled by a quantizer shadow

    womb cryptid list
    womb cryptid render chimera -l 60 -o womb_out/cryptid_chimera.wav
    womb cryptid info jerk_voice

Fully offline, pure numpy + ffmpeg, deterministic from the genome seed.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np

__all__ = ["LAWS", "make_law", "CHAMPIONS", "build_champion", "render",
           "simulate", "viability", "describe_genome", "save_specimen",
           "load_specimen"]

SR = 44100


# ------------------------------------------------------------------ laws

class Law:
    """One dynamical unit: a node in a cryptid's process network."""

    name = "?"
    channels = 1          # size of the out() vector (breeding uses this)

    def __init__(self, params, seed, sim_rate):
        self.params = dict(params)
        self.seed = int(seed)
        self.sim_rate = float(sim_rate)
        self._ev: list[tuple[float, float]] = []
        self.setup(**self.params)

    def setup(self, **kw) -> None:
        pass

    def step(self, t: int, dt: float, inputs: list[float]) -> None:
        raise NotImplementedError

    def out(self) -> np.ndarray:
        raise NotImplementedError

    def take_events(self) -> list[tuple[float, float]]:
        """(amplitude, pan) events born this step; consumed on read."""
        ev = self._ev
        self._ev = []
        return ev


LAWS: dict[str, type[Law]] = {}


def law(cls):
    LAWS[cls.name] = cls
    return cls


def law_channels(name: str) -> int:
    """Out-channel count of a law (breeding wires edges against this)."""
    if name not in LAWS:
        raise ValueError(f"unknown law {name!r}; have {sorted(LAWS)}")
    return LAWS[name].channels


def make_law(name, params, seed, sim_rate) -> Law:
    if name not in LAWS:
        raise ValueError(f"unknown law {name!r}; have {sorted(LAWS)}")
    return LAWS[name](params, seed, sim_rate)


@law
class JerkLaw(Law):
    """Evolved Sprott-style chaos, the full sparse family:
        x''' = a x'' + b x' + c x + d x^3 + e x^2 + f x x' + g x'^2
    Channels out: [x, v, a]. `mod` lets a slow input breathe on the
    damping a. `speed` dilates ODE time to audio rate (lambda):
    each sim step integrates `speed/ode_rate`-scaled substeps of the
    ODE at its own comfortable dt — chaos found at O(1) coefficients
    lands at audible frequencies. Found by search; see CRYPTID.md."""

    name = "jerk"
    channels = 3

    def setup(self, a=-1.0, b=-0.5, c=-1.0, d=-1.0, e=0.0, f=0.0, g=0.0,
               mod=0.0, speed=110.0, ode_rate=60.0,
               x0=0.4, v0=0.0, acc0=0.0):
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.d, self.e, self.f, self.g = float(d), float(e), float(f), float(g)
        self.mod = float(mod)
        self.speed = float(speed)
        self.dt_ode = 1.0 / float(ode_rate)
        # cap substeps so mutated speeds can't blow the render budget;
        # excess speed folds back (the law prices its own energy)
        ksub = self.speed * (1.0 / self.sim_rate) / self.dt_ode
        self.ksub = max(1, int(round(min(ksub, 10))))
        self.x, self.v, self.acc = float(x0), float(v0), float(acc0)
        self._o = np.zeros(3)

    def step(self, t, dt, inputs):
        I = inputs[0] if inputs else 0.0
        a_eff = self.a * (1.0 + self.mod * (2.0 * I - 1.0)) if self.mod else self.a
        b_, c_, d_, e_, f_, g_ = self.b, self.c, self.d, self.e, self.f, self.g
        h = self.dt_ode
        x, v, ac = self.x, self.v, self.acc
        for _ in range(self.ksub):
            k1a = a_eff*ac + b_*v + c_*x + d_*x*x*x + e_*x*x + f_*x*v + g_*v*v
            xp, vp, ap = x + v*h, v + ac*h, ac + k1a*h
            k2a = a_eff*ap + b_*vp + c_*xp + d_*xp*xp*xp + e_*xp*xp + f_*xp*vp + g_*vp*vp
            x = x + 0.5*h*(v + vp)
            v = v + 0.5*h*(ac + ap)
            ac = ac + 0.5*h*(k1a + k2a)
        self.x, self.v, self.acc = x, v, ac

    def out(self):
        self._o[0], self._o[1], self._o[2] = self.x, self.v, self.acc
        return self._o


@law
class OscPop(Law):
    """A population of Kuramoto phase units. Coupling kernel: 'ring'
    (nonlocal, linear-decay, phase-lagged — where chimeras live) or
    'global' (mean-field). External mean-field drive: two slot inputs
    (Re, Im of another population's mean field).

    Channels out: [ReS, ImS, |S|] (population mean field). Window stats
    (per `windows` slices of the ring) are cached each step for the
    chimera ear: local coherence, its projection, and one center-unit
    phase projection per window."""

    name = "osc_pop"
    channels = 3

    def setup(self, n=128, radius=34, k=0.1, alpha=1.46, freq=110.0,
              spread=0.0, windows=16, kernel="ring", ext_gain=1.0,
              ic="half", seed=0):
        self.n = int(n)
        self.W = int(windows)
        if self.n % self.W:
            raise ValueError(f"windows {self.W} must divide n {self.n}")
        self.k, self.alpha, self.ext_gain = float(k), float(alpha), float(ext_gain)
        if kernel == "ring":
            # nonlocal linear-decay kernel on the ring (circulant -> FFT coupling)
            d0 = np.minimum(np.arange(self.n), self.n - np.arange(self.n))
            ker = self.k * np.maximum(0.0, 1.0 - d0 / float(radius))
            ker[0] = 0.0
            self.Fker = np.fft.fft(ker)          # circulant eigenvalues
            self._coup = self._coup_ring
        elif kernel == "global":
            self.gscale = self.k / self.n       # mean-field (diagonal zero)
            self._coup = self._coup_global
        else:
            raise ValueError(f"unknown kernel {kernel!r}")
        rng = np.random.default_rng(seed)
        w = 2 * np.pi * float(freq)
        sig = 2 * np.pi * float(spread)
        self.omega = w + (rng.normal(0.0, sig, self.n) if sig > 0 else np.zeros(self.n))
        th = rng.uniform(0, 2 * np.pi, self.n)
        if ic == "half":        # half coherent, half incoherent (chimera seed)
            th[: self.n // 2] = rng.normal(0.0, 0.05, self.n // 2) % (2 * np.pi)
        self.th = th
        self.m = self.n // self.W                      # units per window
        self.uidx = np.arange(self.W) * self.m + self.m // 2   # center units
        self._o = np.zeros(3)
        self.win = {"coh": np.zeros(self.W), "re": np.zeros(self.W),
                    "im": np.zeros(self.W), "ucos": np.zeros(self.W)}

    def _coup_ring(self, s, c, ca, sa):
        """G@s and G@c in one complex FFT round-trip (circulant kernel)."""
        z = np.fft.ifft(self.Fker * np.fft.fft(np.exp(1j * self.th)))
        return ca * z.imag - sa * z.real     # (Gs)*ca_i - (Gc)*sa_i

    def _coup_global(self, s, c, ca, sa):
        g = self.gscale
        return ca * g * (s.sum() - s) - sa * g * (c.sum() - c)

    def step(self, t, dt, inputs):
        th = self.th
        s = np.sin(th)
        c = np.cos(th)
        sa = np.sin(th + self.alpha)
        ca = np.cos(th + self.alpha)
        coup = self._coup(s, c, ca, sa)
        if inputs:  # external mean-field drive (slot0=Re, slot1=Im)
            ire = inputs[0] if len(inputs) > 0 else 0.0
            iim = inputs[1] if len(inputs) > 1 else 0.0
            A = ire * np.cos(self.alpha) + iim * np.sin(self.alpha)
            B = iim * np.cos(self.alpha) - ire * np.sin(self.alpha)
            coup = coup + self.ext_gain * (B * c - A * s)
        self.th = th + (self.omega + coup) * dt
        # post-update mean fields + window stats (the state the ear hears)
        s2 = np.sin(self.th)
        c2 = np.cos(self.th)
        sw = s2.reshape(self.W, self.m).mean(axis=1)
        cw = c2.reshape(self.W, self.m).mean(axis=1)
        self.win["coh"][:] = np.hypot(cw, sw)
        self.win["re"][:] = cw
        self.win["im"][:] = sw
        self.win["ucos"][:] = c2[self.uidx]
        re, im = float(cw.mean()), float(sw.mean())
        self._o[0], self._o[1], self._o[2] = re, im, float(np.hypot(re, im))

    def out(self):
        return self._o


@law
class MapLaw(Law):
    """Chaotic logistic map — the slow deterministic driver. Updates
    every `period_s`, holds its value between updates. Channel: [x]."""

    name = "map"
    channels = 1

    def setup(self, r=3.98, period_s=1.0, seed=0):
        self.r = float(r)
        self.period = max(1, int(round(float(period_s) * self.sim_rate)))
        self.x = float(np.random.default_rng(seed).uniform(0.1, 0.9))
        self.count = self.period
        self._o = np.zeros(1)

    def step(self, t, dt, inputs):
        self.count -= 1
        if self.count <= 0:
            self.x = self.r * self.x * (1.0 - self.x)
            self.count = self.period

    def out(self):
        self._o[0] = self.x
        return self._o


@law
class Sandpile(Law):
    """BTW sandpile: self-organized criticality. Grains drop (cell chosen
    by drive inputs slot0/slot1, else seeded rng; interval jittered by
    slot0), topple at height>=4 into open boundaries, and every avalanche
    becomes an event: amplitude log1p(size), pan from the toppled cells'
    mean position. Channels: [activity EMA, log1p(last size)]."""

    name = "sand"
    channels = 2

    def setup(self, w=40, h=40, interval_s=0.022, jitter=0.85, warmup=3000,
              act_tau=0.15, seed=0):
        self.w, self.h = int(w), int(h)
        self.cells = self.w * self.h
        self.interval = max(1, int(round(float(interval_s) * self.sim_rate)))
        self.jitter = float(jitter)
        self.hgrid = np.zeros((self.h, self.w), np.int32)
        self.rng = np.random.default_rng(seed)
        self.act = 0.0
        self.last_log = 0.0
        self.count = self.interval
        self.act_k = float(1.0 - np.exp(- (1.0 / self.sim_rate) / max(1e-4, act_tau)))
        self._o = np.zeros(2)
        for _ in range(int(warmup)):   # pre-drive to criticality
            self._drop(float(self.rng.random()), float(self.rng.random()))

    def _drop(self, x, y):
        cx = min(self.w - 1, max(0, int(x * self.w)))
        cy = min(self.h - 1, max(0, int(y * self.h)))
        self.hgrid[cy, cx] += 1
        size, sx, nt = 0, 0, 0
        guard = 0
        while True:
            mask = self.hgrid >= 4
            if not mask.any():
                break
            n = int(mask.sum())
            size += n
            ys, xs = np.nonzero(mask)
            sx += int(xs.sum())
            nt += n
            self.hgrid[mask] -= 4
            m = mask
            self.hgrid[1:, :] += m[:-1, :]
            self.hgrid[:-1, :] += m[1:, :]
            self.hgrid[:, 1:] += m[:, :-1]
            self.hgrid[:, :-1] += m[:, 1:]
            guard += 1
            if guard > 50000:
                break
        if size > 0:
            pan = 2.0 * (sx / (nt * self.w) - 0.5)
            amp = min(1.2, float(np.log1p(size) / np.log1p(4096.0)))
            self._ev.append((amp, pan))
        return size

    def step(self, t, dt, inputs):
        size = 0
        if self.count <= 0:
            x = float(inputs[0]) if len(inputs) > 0 else float(self.rng.random())
            y = float(inputs[1]) if len(inputs) > 1 else float(self.rng.random())
            size = self._drop(x, y)
            ival = self.interval * (1.0 + self.jitter * (2.0 * x - 1.0))
            self.count = max(1, int(round(ival)))
        else:
            self.count -= 1
        self.act += (min(1.0, size / 40.0) - self.act) * self.act_k
        self.last_log = float(np.log1p(size))

    def out(self):
        o = self._o
        o[0], o[1] = self.act, self.last_log
        return o


@law
class Excitable(Law):
    """FitzHugh-Nagumo cousin: rest / spike / refractory. Channels
    [v, w]; a rising zero-crossing of v is an event."""

    name = "excitable"
    channels = 2

    def setup(self, a=0.7, b=0.8, eps=0.02, z0=0.4, v0=0.0, w0=0.0):
        self.a, self.b, self.eps, self.z0 = float(a), float(b), float(eps), float(z0)
        self.v, self.w, self._pv = float(v0), float(w0), float(v0)
        self._o = np.zeros(2)

    def step(self, t, dt, inputs):
        I = inputs[0] if inputs else 0.0
        self._pv = self.v
        dv = (self.v - self.v ** 3 / 3.0 - self.w + self.z0 + I) / self.eps
        dw = self.v + self.a - self.b * self.w
        self.v += dv * dt
        self.w += dw * dt
        if self._pv <= 0.0 < self.v:
            self._ev.append((1.0, 0.0))

    def out(self):
        o = self._o
        o[0], o[1] = self.v, self.w
        return o


@law
class Fire(Law):
    """Integrate-and-fire: u charges from input+bias, spikes at theta,
    then refractory. Channels [u, spike]; each spike is an event."""

    name = "fire"
    channels = 2

    def setup(self, bias=0.5, theta=1.0, refr_s=0.01, seed=0):
        self.bias, self.theta = float(bias), float(theta)
        self.refr = max(1, int(round(float(refr_s) * self.sim_rate)))
        self.u, self.rest = 0.0, 0
        self._o = np.zeros(2)

    def step(self, t, dt, inputs):
        I = inputs[0] if inputs else 0.0
        if self.rest > 0:
            self.rest -= 1
        else:
            self.u += (I + self.bias) * dt
            if self.u >= self.theta:
                self._ev.append((0.9, 0.0))
                self.u = 0.0
                self.rest = self.refr
        o = self._o
        o[0], o[1] = self.u, 1.0 if self.rest == self.refr else 0.0

    def out(self):
        return self._o


@law
class BoolNet(Law):
    """Kauffman Boolean unit: thresholded slot inputs (k of them) index
    a Boolean table; the bit updates every period_s and a 0->1 flip is
    an event. Channel [bit]."""

    name = "bool"
    channels = 1

    def setup(self, k=2, table=(0, 1, 1, 0), period_s=0.25):
        self.k = int(k)
        if len(table) != 2 ** self.k:
            raise ValueError("table must have 2^k entries")
        self.table = [int(v) for v in table]
        self.period = max(1, int(round(float(period_s) * self.sim_rate)))
        self.bit, self.count = 0, self.period
        self._o = np.zeros(1)

    def step(self, t, dt, inputs):
        self.count -= 1
        if self.count <= 0:
            idx = 0
            for j in range(self.k):
                if len(inputs) > j and inputs[j] > 0.5:
                    idx |= 1 << j
            nb = self.table[idx]
            if nb and not self.bit:
                self._ev.append((1.0, 0.0))
            self.bit = nb
            self.count = self.period

    def out(self):
        self._o[0] = float(self.bit)
        return self._o


@law
class DelayLine(Law):
    """Delayed state feedback ring: pushes its input, reads length_s
    back. `nl` > 0 folds it through tanh (echo-chaos). Channel [value]."""

    name = "delay"
    channels = 1

    def setup(self, length_s=0.5, gain=1.0, nl=2.0):
        n = max(1, int(round(float(length_s) * self.sim_rate)))
        self.buf = np.zeros(n)
        self.ptr = 0
        self.gain, self.nl = float(gain), float(nl)
        self.val = 0.0
        self._o = np.zeros(1)

    def step(self, t, dt, inputs):
        I = inputs[0] if inputs else 0.0
        v = self.buf[self.ptr]
        self.buf[self.ptr] = I * self.gain
        self.ptr = (self.ptr + 1) % len(self.buf)
        self.val = float(np.tanh(self.nl * v) / self.nl) if self.nl else v

    def out(self):
        self._o[0] = self.val
        return self._o


@law
class Clamp(Law):
    """Nonlinear quantizer: tanh then round to `steps` levels — the
    aliasing aesthetic as a component inside the process, not a post.
    Channel [value]."""

    name = "clamp"
    channels = 1

    def setup(self, gain=1.5, steps=4):
        self.gain, self.steps = float(gain), max(1, int(steps))
        self.val = 0.0
        self._o = np.zeros(1)

    def step(self, t, dt, inputs):
        I = inputs[0] if inputs else 0.0
        v = float(np.tanh(self.gain * I))
        self.val = round(v * self.steps) / self.steps * 0.95

    def out(self):
        self._o[0] = self.val
        return self._o


@law
class ByteFeed(Law):
    """Databending species: a file's bytes as a state stream (never
    playback — the bytes ARE a state variable). b64 keeps specimens
    frozen; `path` is read once at build. Channel [byte value]."""

    name = "bytefeed"
    channels = 1

    def setup(self, b64=None, path=None, rate=4410, seed=0):
        raw = base64.b64decode(b64) if b64 else Path(path).read_bytes()
        if not raw:
            raise ValueError("bytefeed needs bytes")
        self.data = np.frombuffer(raw, np.uint8).astype(np.float64) / 255.0
        self.rate = float(rate)
        self.pos = 0.0
        self.val = 0.0
        self._o = np.zeros(1)

    def step(self, t, dt, inputs):
        self.pos += self.rate / self.sim_rate
        i = int(self.pos) % len(self.data)
        self.val = float(self.data[i])

    def out(self):
        self._o[0] = self.val
        return self._o


# ------------------------------------------------------------------ ear

def _pan_pair(pan: float) -> tuple[float, float]:
    """Equal-power pan: pan in [-1, 1] -> (left, right) weights."""
    th = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * np.pi / 4.0
    return float(np.cos(th)), float(np.sin(th))


class _Op:
    def __init__(self, spec):
        self.node = int(spec["node"])

    def step(self, outs, events, t) -> tuple[float, float]:
        raise NotImplementedError


class _TapOp(_Op):
    """Direct state read: out[node][ch] * gain, panned (fixed pan, or
    pan_node's channel mapped through [pan_lo, pan_hi])."""

    def __init__(self, spec):
        super().__init__(spec)
        self.ch = int(spec.get("ch", 0))
        self.gain = float(spec.get("gain", 0.5))
        self.pan = float(spec.get("pan", 0.0))
        self.pan_node = spec.get("pan_node")
        self.pan_ch = int(spec.get("pan_ch", 0))
        self.pan_lo, self.pan_hi = float(spec.get("pan_lo", -0.7)), float(spec.get("pan_hi", 0.7))
        self._pl, self._pr = _pan_pair(self.pan)

    def step(self, outs, events, t):
        o = outs[self.node] if self.node < len(outs) else None
        if o is None or self.ch >= len(o):
            return 0.0, 0.0                      # node/channel drifted away: silence
        v = float(o[self.ch]) * self.gain
        if self.pan_node is not None:
            if self.pan_node < len(outs) and self.pan_ch < len(outs[self.pan_node]):
                p = outs[self.pan_node][self.pan_ch]
                pan = self.pan_lo + (self.pan_hi - self.pan_lo) * float(np.clip(p, 0, 1))
            else:
                pan = self.pan
            l, r = _pan_pair(pan)
            return v * l, v * r
        return v * self._pl, v * self._pr


class _DiffOp(_Op):
    """First difference of a channel — the velocity of a measurement."""

    def __init__(self, spec):
        super().__init__(spec)
        self.ch = int(spec.get("ch", 0))
        self.gain = float(spec.get("gain", 0.25))
        self.pan = float(spec.get("pan", 0.0))
        self.last = 0.0
        self._pl, self._pr = _pan_pair(self.pan)

    def step(self, outs, events, t):
        o = outs[self.node] if self.node < len(outs) else None
        if o is None or self.ch >= len(o):
            return 0.0, 0.0
        v = float(o[self.ch])
        d = (v - self.last) * self.gain
        self.last = v
        return d * self._pl, d * self._pr


class _EmbedOp(_Op):
    """Takens-flavored observation: x(t) - x(t - delay samples)."""

    def __init__(self, spec, sim_rate):
        super().__init__(spec)
        self.ch = int(spec.get("ch", 0))
        self.gain = float(spec.get("gain", 0.3))
        self.delay = max(1, int(spec.get("delay", 97)))
        self.pan = float(spec.get("pan", 0.0))
        self.buf = np.zeros(self.delay)
        self.ptr = 0
        self._pl, self._pr = _pan_pair(self.pan)

    def step(self, outs, events, t):
        o = outs[self.node] if self.node < len(outs) else None
        if o is None or self.ch >= len(o):
            return 0.0, 0.0
        v = float(o[self.ch])
        old = self.buf[self.ptr]
        self.buf[self.ptr] = v
        self.ptr = (self.ptr + 1) % self.delay
        d = (v - old) * self.gain
        return d * self._pl, d * self._pr


class _EventsOp(_Op):
    """Threshold events (law-born) as impulses: amplitude from the law,
    pan from the law's event pan hint."""

    def __init__(self, spec):
        super().__init__(spec)
        self.gain = float(spec.get("gain", 0.8))
        self.pan_spread = float(spec.get("pan_spread", 0.9))
        self.refractory = int(spec.get("refractory", 0))
        self.last_t = -10 ** 9

    def step(self, outs, events, t):
        l = r = 0.0
        for (nid, amp, pan) in events:
            if nid != self.node:
                continue
            if self.refractory and t - self.last_t < self.refractory:
                continue
            self.last_t = t
            wl, wr = _pan_pair(pan * self.pan_spread)
            l += amp * self.gain * wl
            r += amp * self.gain * wr
        return l, r


class _ChimeraOp(_Op):
    """osc_pop readout: per ring window, a coherence-blended blend —
    the coherent part sings (mean-field projection), the incoherent
    part leaks a drifting center-unit whistle. Pan sweeps the ring."""

    def __init__(self, spec, nodes):
        super().__init__(spec)
        nd = nodes[self.node]
        if not isinstance(nd, OscPop):
            raise ValueError("chimera ear needs an osc_pop node")
        W = nd.W
        self.tone = float(spec.get("tone", 0.4))
        self.noise = float(spec.get("noise", 0.3))
        lo, hi = float(spec.get("pan_lo", -0.9)), float(spec.get("pan_hi", 0.9))
        pans = np.linspace(lo, hi, W)
        th = (pans + 1.0) * np.pi / 4.0
        self.wL = np.cos(th)
        self.wR = np.sin(th)

    def step(self, outs, events, t):
        # law-specific window stats (measurement of the SAME state)
        nd = self._nodes[self.node]
        w = nd.win
        y = self.tone * w["coh"] * w["re"] + self.noise * (1.0 - w["coh"]) * w["ucos"]
        return float(self.wL @ y), float(self.wR @ y)


_OPS = {"tap": _TapOp, "diff": _DiffOp, "embed": _EmbedOp,
        "events": _EventsOp, "chimera": _ChimeraOp}


class Ear:
    """The measurement genome: a list of ops, summed into L/R."""

    def __init__(self, specs, nodes, sim_rate):
        self.ops = []
        for spec in specs:
            kind = spec["op"]
            if kind not in _OPS:
                raise ValueError(f"unknown ear op {kind!r}; have {sorted(_OPS)}")
            if kind == "embed":
                self.ops.append(_EmbedOp(spec, sim_rate))
            elif kind == "chimera":
                op = _ChimeraOp(spec, nodes)
                op._nodes = nodes           # needs live law access
                self.ops.append(op)
            else:
                self.ops.append(_OPS[kind](spec))

    def step(self, outs, events, t):
        l = r = 0.0
        for op in self.ops:
            dl, dr = op.step(outs, events, t)
            l += dl
            r += dr
        return l, r


# ------------------------------------------------------------------ simulate

def simulate(genome: dict, length: float):
    """Run a cryptid. Returns (L, R, sim_rate, nodes) — raw ear readout
    at the genome's sim rate (pre-limiter, pre-resample)."""
    sim_rate = float(genome.get("sim_rate", 11025))
    dt = 1.0 / sim_rate
    seed = int(genome.get("seed", 0))
    nodes = [make_law(n["law"], n.get("params", {}), seed + 997 * i, sim_rate)
             for i, n in enumerate(genome["nodes"])]
    edges = [(int(e["src"]), int(e.get("ch", 0)), int(e["dst"]),
              float(e.get("gain", 1.0)), int(e.get("slot", 0)))
             for e in genome.get("edges", [])]
    # group edges by (dst, slot) for cheap per-step gathering
    groups: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for src, ch, dst, gain, slot in edges:
        groups.setdefault((dst, slot), []).append((src, ch, gain))
    ear = Ear(genome.get("ears", []), nodes, sim_rate)

    n_steps = int(round(length * sim_rate))
    L = np.zeros(n_steps)
    R = np.zeros(n_steps)
    outs = [nd.out() for nd in nodes]
    for t in range(n_steps):
        ins = [{} for _ in nodes]  # per node: {slot: summed value}
        for (dst, slot), terms in groups.items():
            s = 0.0
            for src, ch, gain in terms:
                s += gain * float(outs[src][ch])
            ins[dst][slot] = s
        events = []
        for d, nd in enumerate(nodes):
            slots = ins[d]
            inputs = [slots[j] for j in sorted(slots)] if slots else []
            nd.step(t, dt, inputs)
            for amp, pan in nd.take_events():
                events.append((d, amp, pan))
        outs = [nd.out() for nd in nodes]
        l, r = ear.step(outs, events, t)
        L[t] = l
        R[t] = r
    return L, R, sim_rate, nodes


# ------------------------------------------------------------------ render

def _upsample(x: np.ndarray, sim_rate: float, sr: int) -> np.ndarray:
    f = sr / sim_rate
    if abs(f - round(f)) > 1e-9:
        raise ValueError(f"sim_rate {sim_rate} must divide {sr}")
    f = int(round(f))
    n = len(x)
    t_new = np.arange(n * f) / f
    return np.interp(t_new, np.arange(n), x)


def _finish(L: np.ndarray, R: np.ndarray, sim_rate: float, sr: int) -> np.ndarray:
    """The safety stage (the only designed stage): sanitize, resample,
    de-DC, fades, soft limit, normalize. Stereo (n, 2) float32 out.
    Blown-up simulations (Inf/NaN) die here — the MC sees the corpse."""
    a = _upsample(L, sim_rate, sr)
    b = _upsample(R, sim_rate, sr)
    out = np.stack([a, b], axis=1)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    out -= out.mean(axis=0)                       # seismograph removes DC
    n = len(out)
    fade = min(n // 8, int(0.06 * sr))            # 60 ms in/out
    if fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
        out[:fade] *= ramp[:, None]
        out[-fade:] *= ramp[::-1][:, None]
    out = np.tanh(1.35 * out) / np.tanh(1.35)      # soft knee
    peak = float(np.abs(out).max())
    if peak > 1e-9:
        out = out / peak * 0.92
    return out.astype(np.float32)


def render(genome: dict, length: float = 60.0, out=None, sr: int = SR,
           hi: bool = False, verbose: bool = False) -> dict:
    """Render a cryptid to finished stereo audio. `hi` simulates at the
    output rate directly (no decimation aliasing; slower). Deterministic
    from genome+seed."""
    g = dict(genome)
    if hi:
        g["sim_rate"] = sr
    t0 = time.time()
    L, R, sim_rate, nodes = simulate(g, length)
    audio = _finish(L, R, sim_rate, sr)
    stats = {"audio": audio, "sr": sr, "sim_rate": sim_rate,
             "seconds": length, "nodes": len(nodes),
             "elapsed": round(time.time() - t0, 1)}
    if out:
        from .sonify import write_wav

        stats["path"] = write_wav(out, audio, sr)
    if verbose:
        v = viability(audio, sr)
        print(f"[cryptid] {genome.get('name','?')}: {length:.0f}s at sim {sim_rate:.0f}Hz "
              f"in {stats['elapsed']}s — rms {v['rms']:.3f}, flatness {v['flatness']:.2f}")
    return stats


# ------------------------------------------------------------------ viability

def viability(audio: np.ndarray, sr: int = SR, strictness: float = 0.5) -> dict:
    """The minimal-criterion measurement (MCC fuel): audible, alive
    (behavior varies across blocks, in ANY modality of the measurement
    — loudness, spectrum, or stereo image), finite. Never 'good': no
    descriptor here ranks quality. `strictness` (Soros' dial, 0.1..1)
    scales the bands — calibrated at 0.5; both extremes are meant to
    stagnate."""
    x = np.asarray(audio, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    finite = bool(np.isfinite(x).all())
    nb = max(8, int(x.shape[0] // (2 * sr)) or 1)      # ~2 s blocks
    blocks = np.array_split(x, nb, axis=0)
    rms_b, cent_b, side_b = [], [], []
    nfft = 4096
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    for b in blocks:
        rms_b.append(np.sqrt((b ** 2).mean() + 1e-12))
        m = b.mean(axis=1)
        m = m[: len(m) // nfft * nfft]
        if len(m) < nfft:
            m = np.pad(m, (0, nfft - len(m)))
        S = np.abs(np.fft.rfft(m.reshape(-1, nfft) * np.hanning(nfft), axis=1)).mean(axis=0) + 1e-12
        cent_b.append((fr * S).sum() / S.sum())
        side_b.append(np.sqrt(((b[:, 0] - b[:, 1]) ** 2).mean() + 1e-12))
    rms_b, cent_b, side_b = np.array(rms_b), np.array(cent_b), np.array(side_b)
    rms = float(np.sqrt((x ** 2).mean()))
    alive = dict(rms=float(rms_b.std() / (rms_b.mean() + 1e-9)),
                 centroid=float(cent_b.std() / (cent_b.mean() + 1e-9)),
                 side=float(side_b.std() / (side_b.mean() + 1e-9)))
    seg = x[len(x) // 4: 3 * len(x) // 4].mean(axis=1)
    m = seg[: len(seg) // nfft * nfft]
    if len(m) < nfft:
        m = np.pad(m, (0, nfft - len(m)))
    S = np.abs(np.fft.rfft(m.reshape(-1, nfft) * np.hanning(nfft), axis=1)).mean(axis=0) + 1e-12
    flatness = float(np.exp(np.log(S).mean()) / S.mean())
    alive_ratio = max(alive.values())
    rms_lo = float(np.clip(0.006 * (strictness / 0.5) ** 2, 1e-5, 0.05))
    alive_lo = float(np.clip(0.02 * (strictness / 0.5) ** 2, 1e-4, 0.1))
    viable = bool(finite and rms_lo < rms < 0.9 and alive_ratio > alive_lo)
    return {"viable": viable, "rms": round(rms, 4), "peak": round(float(np.abs(x).max()), 4),
            "alive_ratio": round(alive_ratio, 4), "alive": {k: round(v, 4) for k, v in alive.items()},
            "flatness": round(flatness, 4), "finite": finite,
            "bands": {"rms_lo": round(rms_lo, 5), "alive_lo": round(alive_lo, 5)}}


# ------------------------------------------------------------------ champions

def _chimera_genome() -> dict:
    """A wandering coherence boundary on two rings (110 + 55 Hz): where
    the ring is coherent you hear the mean-field tone; where it is
    incoherent a drifting center-unit whistle leaks through; the pan
    position of the tonal cluster IS the boundary position."""
    def pop(freq, spread, seed, radius, k, alpha):
        return {"law": "osc_pop", "params": dict(
            n=128, radius=radius, k=k, alpha=alpha, freq=freq, spread=spread,
            windows=16, kernel="ring", ic="half", seed=seed)}
    return {
        "name": "chimera", "lineage": "phase-a/hand",
        "born": {"gen": 0, "parent": None, "mutation": "hand-built",
                 "search": "coherence-split sweep, this machine"},
        "seed": 7, "sim_rate": 2205,
        "nodes": [pop(110.0, 0.5, 101, 34, 1.6, 1.38),
                  pop(55.0, 1.5, 202, 48, 1.6, 1.46)],
        "edges": [],
        "ears": [
            {"op": "chimera", "node": 0, "tone": 0.40, "noise": 0.55,
             "pan_lo": -0.9, "pan_hi": 0.9},
            {"op": "chimera", "node": 1, "tone": 0.5, "noise": 0.6,
             "pan_lo": -0.55, "pan_hi": 0.55},
        ],
        "notes": "two ring-chimera populations; coherence split tuned on "
                 "this machine (98% sustained over 65 s at r=34 k=1.6 "
                 "a=1.38 spread=0.5); pitch lands near (not at) nominal "
                 "freq — entrainment drifts it; the boundary wander is "
                 "heard as stereo + spectral movement",
    }


def _clickstorm_genome() -> dict:
    """Sandpile weather: chaotic maps choose where and when grains fall;
    every avalanche is a click with power-law amplitude and causal pan;
    the activity EMA is the storm bed underneath."""
    return {
        "name": "clickstorm", "lineage": "phase-a/hand",
        "born": {"gen": 0, "parent": None, "mutation": "hand-built"},
        "seed": 11, "sim_rate": 11025,
        "nodes": [
            {"law": "map", "params": dict(r=3.993, period_s=0.02, seed=31)},
            {"law": "map", "params": dict(r=3.977, period_s=0.03, seed=57)},
            {"law": "sand", "params": dict(w=40, h=40, interval_s=0.022,
                                           jitter=0.85, warmup=3000, seed=97)},
        ],
        "edges": [
            {"src": 0, "ch": 0, "dst": 2, "gain": 1.0, "slot": 0},
            {"src": 1, "ch": 0, "dst": 2, "gain": 1.0, "slot": 1},
        ],
        "ears": [
            {"op": "events", "node": 2, "gain": 1.15, "pan_spread": 0.95},
            {"op": "tap", "node": 2, "ch": 0, "gain": 0.09, "pan": 0.0},
        ],
        "notes": "BTW sandpile at SOC; drive position and interval are "
                 "logistic-map chaos (no rng in the loop)",
    }


def _jerk_voice_genome() -> dict:
    """An evolved jerk attractor whose linear stiffness breathes along a
    slow chaotic map: an inharmonic voice that never settles. A quantizer
    node shadows it — the same voice, crushed. Pan drifts on a third map."""
    return {
        "name": "jerk_voice", "lineage": "phase-a/hand",
        "born": {"gen": 0, "parent": None, "mutation": "coefficient search"},
        "seed": 3, "sim_rate": 11025,
        "nodes": [
            {"law": "jerk", "params": dict(a=-0.44, b=-1.97, c=-2.97,
                                           d=0.68, e=0.63, f=-2.91, g=1.06,
                                           mod=0.10, speed=1500.0, ode_rate=60.0,
                                           x0=0.4)},
            {"law": "clamp", "params": dict(gain=1.6, steps=4)},
            {"law": "map", "params": dict(r=3.98, period_s=9.0, seed=11)},
            {"law": "map", "params": dict(r=3.91, period_s=13.0, seed=23)},
        ],
        "edges": [
            {"src": 0, "ch": 0, "dst": 1, "gain": 1.0},   # jerk x -> quantizer
            {"src": 2, "ch": 0, "dst": 0, "gain": 1.0},   # stiffness breathes
        ],
        "ears": [
            {"op": "tap", "node": 0, "ch": 0, "gain": 0.42, "pan_node": 3,
             "pan_lo": -0.8, "pan_hi": 0.8},
            {"op": "tap", "node": 0, "ch": 1, "gain": 0.08, "pan": -0.25},
            {"op": "embed", "node": 0, "ch": 0, "delay": 97, "gain": 0.3, "pan": 0.3},
            {"op": "tap", "node": 1, "ch": 0, "gain": 0.28, "pan": 0.45},
        ],
        "notes": "coefficients found by blind search over the sparse jerk "
                 "family (Sprott's method; CRYPTID.md sec. 1) — the only "
                 "Lyapunov-positive set in 8000 samples; damping breathes "
                 "+-10% on a 9 s chaotic map; speed 1500 dilates the ODE "
                 "clock to audio",
    }


CHAMPIONS: dict[str, callable] = {
    "chimera": _chimera_genome,
    "clickstorm": _clickstorm_genome,
    "jerk_voice": _jerk_voice_genome,
}


def build_champion(name: str) -> dict:
    if name not in CHAMPIONS:
        raise ValueError(f"unknown champion {name!r}; have {sorted(CHAMPIONS)}")
    return CHAMPIONS[name]()


# ------------------------------------------------------------------ specimens

def save_specimen(genome: dict, path) -> str:
    Path(path).write_text(json.dumps(genome, indent=1))
    return str(path)


def load_specimen(path) -> dict:
    g = json.loads(Path(path).read_text())
    for k in ("name", "nodes", "ears"):
        if k not in g:
            raise ValueError(f"specimen {path} lacks {k!r}")
    return g


def describe_genome(genome: dict) -> str:
    """The cryptid's biography: what it is, where it came from, how it
    is heard. The story is part of the artifact."""
    g = genome
    born = g.get("born", {})
    lines = [f"cryptid {g.get('name', '?')}  (lineage {g.get('lineage', '?')})",
             f"  born: gen {born.get('gen', '?')}, parent {born.get('parent')}, "
             f"mutation {born.get('mutation', '?')}",
             f"  seed {g.get('seed', 0)}, sim rate {g.get('sim_rate', 11025)} Hz"]
    for i, n in enumerate(g["nodes"]):
        p = ", ".join(f"{k}={v}" for k, v in n.get("params", {}).items())
        lines.append(f"  node {i}: {n['law']}({p})")
    for e in g.get("edges", []):
        lines.append(f"  edge {e['src']}[ch{e.get('ch', 0)}] -> {e['dst']} "
                     f"x{e.get('gain', 1.0)} (slot {e.get('slot', 0)})")
    for e in g.get("ears", []):
        bits = ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("op", "node"))
        lines.append(f"  ear: {e['op']} on node {e['node']} — {bits}")
    if g.get("notes"):
        lines.append(f"  notes: {g['notes']}")
    return "\n".join(lines)