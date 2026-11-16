"""Processors: composable DSP transforms for chunks.

A Processor is anything with a `name` and::

    process(audio (n,2) float32, sr, t, rng, **params) -> (m,2) float32

`t` is the event's normalized time in the piece [0, 1], so any parameter
can be a `Curve` (``gain=[1, 0.3]`` = fade out across the piece). The
registry is open: ``@register_processor`` — this is also where learned /
ML models plug in (see loom/ml.py for a worked example).

Processors may change length freely (speed, repeat, trim) — the arranger
and renderer track durations from the processed audio, never assume.

All of this is deliberately dependency-free and offline-first: filters are
FFT-domain with smooth raised-cosine edges, echo is a finite geometric tap
series, everything vectorized over numpy.
"""

from __future__ import annotations

import numpy as np

from .curves import Curve
from .io import resample
from .rng import Seeded

__all__ = [
    "Processor",
    "PROCESSORS",
    "register_processor",
    "build_chain",
    "Chain",
    "list_processors",
    "processor_help",
]

PROCESSORS: dict[str, type["Processor"]] = {}


class Processor:
    """Base class / protocol. Subclasses set `name` and implement process()."""

    name: str = "?"

    def process(self, audio: np.ndarray, sr: int, t: float, rng: Seeded, **params) -> np.ndarray:
        raise NotImplementedError


def register_processor(cls: type[Processor]) -> type[Processor]:
    PROCESSORS[cls.name] = cls
    return cls


def list_processors() -> list[str]:
    return sorted(PROCESSORS)


def processor_help() -> str:
    rows = [
        ("gain(amount=1.0)", "multiply by amount (curve-friendly)"),
        ("normalize()", "peak-normalize the chunk"),
        ("reverse()", "time-reverse"),
        ("speed(x=1.0, quality='linear')", "varispeed: 2.0 = octave up, 0.5 = octave down"),
        ("stretch(factor=2.0, grain_s=0.05)", "pitch-PRESERVING granular time-stretch"),
        ("stutter(count=5, step_semi=1.5, direction='down', slice_s=0.11)",
         "stutter-fill: re-triggered slice, pitch shifting each repeat"),
        ("chop(steps=4, pulses=2, rotate=0)", "Euclid gating inside the chunk (micro-silence)"),
        ("morph(processors=[...], weights=[...])", "diversity engine: one pooled effect per event (seeded)"),
        ("lowpass(cut_hz=2000, width_oct=0.25)", "FFT-domain lowpass with raised-cosine edge"),
        ("highpass(cut_hz=200, width_oct=0.25)", "FFT-domain highpass"),
        ("bandpass(lo_hz=200, hi_hz=2000)", "FFT-domain bandpass"),
        ("ring(freq_hz=220.0, mix=0.5)", "sine ring-modulation"),
        ("drive(amount=3.0)", "tanh waveshaping saturation"),
        ("crush(bits=6)", "bit-depth reduction"),
        ("corrupt(mode='zeros', amount=0.001)", "byte-level corruption (databending)"),
        ("echo(delay_s=0.25, feedback=0.45, mix=0.4)", "finite-tap feedback delay"),
        ("repeat(count=4, decay=0.75, vary=0.0)", "stutter: repeat with decay + optional pitch jitter"),
        ("fades(attack=0.005, release=0.005)", "cosine edge ramps (click safety)"),
        ("env(shape='perc', attack=0.1)", "envelope shape: perc|tri|hann"),
        ("gate(threshold_db=-40.0)", "zero out samples below threshold"),
        ("trim(threshold_db=-40.0, pad=0.01)", "trim near-silence at head/tail"),
    ]
    lines = [f"loom processors ({len(rows)}):"]
    for sig, desc in rows:
        lines.append(f"  {sig:<44} {desc}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- chain

Step = tuple[Processor | str, dict]


class Chain:
    """A sequence of processor steps with per-event resolved params."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps: list[tuple[Processor, dict]] = []
        for spec, params in steps:
            proc = _resolve(spec)
            if isinstance(proc, _Bound):  # unwrap inline "name(kw=v)" params
                proc, bound = proc.inner, proc.params
                params = {**bound, **dict(params)}
            resolved = {
                k: (v if k in _NON_CURVE_PARAMS else _coerce_param(v))
                for k, v in dict(params).items()
            }
            self.steps.append((proc, resolved))

    def run(self, audio: np.ndarray, sr: int, t: float, rng: Seeded) -> np.ndarray:
        for proc, params in self.steps:
            kw = {k: (v.at(t) if isinstance(v, Curve) else v) for k, v in params.items()}
            if audio.shape[0] == 0:
                return audio
            audio = proc.process(audio, sr, t, rng, **kw)
        return _finite(audio)

    def describe(self) -> list[dict]:
        out = []
        for proc, params in self.steps:
            out.append({"name": proc.name, "params": {k: (v.spec if isinstance(v, Curve) else v) for k, v in params.items()}})
        return out


def _coerce_param(v):
    """Any list/tuple/callable param becomes a Curve; the audio_lab idiom:
    constants are numbers, curves are lists. Structured params (morph's
    processor pool) are exempt via _NON_CURVE_PARAMS."""
    from .curves import Curve as _C

    if isinstance(v, _C):
        return v
    if isinstance(v, (list, tuple)):
        return _C(list(v))
    if callable(v) and not isinstance(v, str):
        return _C(v)
    return v


# params that hold structured data, not curves
_NON_CURVE_PARAMS = {"processors", "weights", "voices"}


def _resolve(spec: Processor | str) -> Processor:
    if isinstance(spec, Processor):
        return spec
    if isinstance(spec, str):
        name = spec.strip()
        if name in PROCESSORS:
            entry = PROCESSORS[name]  # may be a class *or* a fitted instance
            return entry if isinstance(entry, Processor) else entry()
        # allow "name(kw=v, ...)" inline specs
        if "(" in name and name.endswith(")"):
            head, rest = name.split("(", 1)
            head = head.strip()
            if head in PROCESSORS:
                import ast

                call = ast.parse(f"f({rest}", mode="eval").body  # Call node
                kwargs = {}
                for kw in call.keywords:
                    try:
                        kwargs[kw.arg] = ast.literal_eval(kw.value)
                    except Exception:
                        pass
                inner = PROCESSORS[head]
                inner = inner if isinstance(inner, Processor) else inner()
                return _Bound(inner, kwargs)
        raise ValueError(f"unknown processor {spec!r}; have {list_processors()}")
    raise TypeError(f"bad processor spec {spec!r}")


class _Bound(Processor):
    """A processor with pre-bound default params (from inline "name(kw=v)" specs)."""

    def __init__(self, inner: Processor, params: dict) -> None:
        self.inner = inner
        self.params = params
        self.name = inner.name

    def process(self, audio, sr, t, rng, **kw):
        merged = {**self.params, **kw}
        return self.inner.process(audio, sr, t, rng, **merged)


def build_chain(spec) -> Chain:
    """Build a Chain from:

    - None -> empty chain
    - "name" or "name(kw=v)" -> single step
    - ["lowpass", ["echo", {"delay_s": 0.3}], "crush(bits=4)"] -> mixed list;
      a [str, dict] pair is a parametrized step; a Processor instance is a
      learned step
    - {"name": ..., "params": {...}} entries (Chain.describe() round-trip)
    - {"lowpass": {...}, "echo": {...}} -> dict of steps
    """
    steps: list[Step] = []
    if spec is None:
        pass
    elif isinstance(spec, Processor):
        steps.append((spec, {}))
    elif isinstance(spec, str):
        steps.append((spec, {}))
    elif isinstance(spec, dict):
        _chain_dict_items(spec, steps)
    elif isinstance(spec, (list, tuple)):
        # a single [name|Processor, params-dict] pair is ONE parametrized step
        # (needed when specs are passed individually, e.g. morph pools);
        # anything else is a list of steps
        if len(spec) == 2 and isinstance(spec[0], (str, Processor)) and isinstance(spec[1], dict):
            steps.append((spec[0], dict(spec[1])))
        else:
            item: object
            for item in spec:
                if isinstance(item, dict):
                    _chain_dict_items(item, steps)
                elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], (str, Processor)):
                    steps.append((item[0], dict(item[1])))
                else:
                    steps.append((item, {}))
    else:
        raise TypeError(f"cannot build chain from {type(spec).__name__}")
    return Chain(steps)


def _chain_dict_items(d: dict, steps: list[Step]) -> None:
    """Handle both dict spellings: {name: params} and describe()'s
    {"name": n, "params": p}."""
    if set(d.keys()) == {"name", "params"}:
        steps.append((d["name"], dict(d.get("params") or {})))
        return
    for k, v in d.items():
        steps.append((k, dict(v or {})))


def _finite(x: np.ndarray) -> np.ndarray:
    """Guard: never let NaN/Inf reach the renderer."""
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(x, dtype=np.float32)


# --------------------------------------------------------------------------- built-ins

def _euclid_pattern_local(steps: int, pulses: int) -> list[int]:
    """Even `pulses` in `steps` (same math as the arrangers' grid)."""
    steps, pulses = int(steps), int(pulses)
    if steps <= 0 or pulses <= 0:
        return [0] * max(0, steps)
    pulses = min(pulses, steps)
    pat = [0] * steps
    for i in range(pulses):
        pat[(i * steps) // pulses] = 1
    return pat


def _mini_fades(seg: np.ndarray, fade_n: int) -> np.ndarray:
    """Short cosine edge ramps to keep gated segments click-free."""
    n = seg.shape[0]
    f = min(fade_n, n // 2)
    if f > 1:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(f) / f)
        seg = seg.copy()
        seg[:f] *= ramp[:, None]
        seg[-f:] *= ramp[::-1][:, None]
    return seg


@register_processor
class Stretch(Processor):
    name = "stretch"

    def process(self, audio, sr, t, rng, factor=2.0, grain_s=0.05):
        """Pitch-PRESERVING granular time-stretch (factor>1 = longer/slower).
        Synchronous overlap-add of hann grains: input grains advance at a
        fixed rate, output grains at rate x factor — duration changes, pitch
        doesn't. The texture drag companion to varispeed `speed`."""
        factor = float(factor)
        n = audio.shape[0]
        if abs(factor - 1.0) < 1e-6 or n < 512:
            return audio
        g = int(float(grain_s) * sr)
        g = max(256, min(g, n))
        win = np.hanning(g)
        # output grains always overlap 2x (hop g/2) so the OLA norm never
        # collapses at seams; the input hop stretches instead. The input is
        # zero-padded by g so the loop is driven purely by OUTPUT coverage:
        # stopping on input exhaustion leaves a single-coverage tail inside
        # the emitted region whose OLA normalization explodes.
        hop_out = max(8, g // 2)
        hop_in = max(1.0, hop_out / factor)
        out_n = int(n * factor)
        padded = np.zeros((n + g, audio.shape[1]), dtype=np.float64)
        padded[:n] = audio
        out = np.zeros((out_n + g, audio.shape[1]), dtype=np.float64)
        norm = np.zeros(out_n + g, dtype=np.float64)
        pos_in = 0.0
        pos_out = 0
        while pos_out < out_n:
            i0 = min(int(pos_in), n)
            out[pos_out : pos_out + g] += padded[i0 : i0 + g] * win[:, None]
            norm[pos_out : pos_out + g] += win * win
            pos_in += hop_in
            pos_out += hop_out
        good = norm > 1e-6
        out[good] /= norm[good][:, None]
        # head/tail single-coverage slivers are trimmed, not divided
        lo, hi = hop_out, max(hop_out + 1, out_n - hop_out)
        return out[lo:hi].astype(np.float32)


@register_processor
class Chop(Processor):
    name = "chop"

    def process(self, audio, sr, t, rng, steps=4, pulses=2, rotate=0, fade_s=0.004):
        """Rhythmic gating inside the chunk: keep only Euclid-selected steps,
        mute the rest — silence as a micro-structure. Click-free edge ramps."""
        n = audio.shape[0]
        steps = max(2, int(steps))
        pulses = min(steps, max(1, int(pulses)))
        pat = _euclid_pattern_local(steps, pulses)
        rot = int(rotate) % steps
        pat = pat[rot:] + pat[:rot]
        step_n = n // steps
        if step_n < 2:
            return audio
        fade_n = max(2, int(float(fade_s) * sr))
        out = np.zeros_like(audio)
        for i, keep in enumerate(pat):
            if keep:
                s, e = i * step_n, min(n, (i + 1) * step_n)
                out[s:e] = _mini_fades(audio[s:e], fade_n)
        return out


@register_processor
class Stutter(Processor):
    name = "stutter"

    def process(self, audio, sr, t, rng, count=5, step_semi=1.5, direction="down",
                slice_s=0.11, decay=0.8):
        """Stutter-fill: the same short slice re-triggered `count` times, each
        repeat pitch-shifted progressively by `step_semi` semitones
        (direction 'down' | 'up' | 'random' per event)."""
        n = audio.shape[0]
        slice_n = max(64, int(float(slice_s) * sr))
        if n < 256:
            return audio
        count = max(2, int(count))
        d = str(direction)
        if d == "random":
            d = "down" if rng.rng.random() < 0.5 else "up"
        sign = -1.0 if d == "down" else 1.0
        out = np.zeros((count * slice_n, audio.shape[1]), dtype=np.float32)
        for i in range(count):
            f = 2.0 ** (sign * float(step_semi) * i / 12.0)  # speed factor
            need = min(n, max(32, int(slice_n * f)))
            src = audio[:need]
            piece = resample(src, max(1, int(sr * f)), sr) if abs(f - 1.0) > 1e-6 else src.copy()
            piece = (piece * (float(decay) ** i)).astype(np.float32)
            m = min(piece.shape[0], slice_n)
            out[i * slice_n : i * slice_n + m] = piece[:m]
        return out


@register_processor
class Morph(Processor):
    """Diversity engine: holds a pool of processor specs and applies ONE per
        event, chosen by seeded rng — so a chain slot behaves differently
        every couple of seconds. Optional `weights` bias the pool.

        process(audio, sr, t, rng,
                processors=[["crush", {"bits": 4}], "reverse",
                           ["ring", {"freq_hz": 660, "mix": 0.4}], ...],
                weights=[...])
    """

    name = "morph"

    def __init__(self) -> None:
        self._cache_key = None
        self._chains = None
        self._weights = None

    def _resolve(self, specs, weights):
        key = repr(specs)
        if self._cache_key != key:
            chains = []
            for spec in specs if isinstance(specs, (list, tuple)) else [specs]:
                chains.append(build_chain(spec))
            self._chains = chains
            self._cache_key = key
            self._weights = weights
        return self._chains

    def process(self, audio, sr, t, rng, processors=None, weights=None, **_):
        if not processors:
            return audio
        chains = self._resolve(processors, weights)
        if len(chains) == 1:
            idx = 0
        elif weights:
            w = np.asarray(weights, dtype=np.float64)[: len(chains)]
            idx = int(rng.rng.choice(len(chains), p=w / w.sum()))
        else:
            idx = rng.ints(len(chains))
        return chains[idx].run(audio, sr, t, rng)


@register_processor
class Gain(Processor):
    name = "gain"

    def process(self, audio, sr, t, rng, amount=1.0):
        return audio * float(amount)


@register_processor
class Normalize(Processor):
    name = "normalize"

    def process(self, audio, sr, t, rng, headroom_db=0.3):
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-10:
            return audio * (10.0 ** (-float(headroom_db) / 20.0) / peak)
        return audio


@register_processor
class Reverse(Processor):
    name = "reverse"

    def process(self, audio, sr, t, rng):
        return audio[::-1].copy()


@register_processor
class Speed(Processor):
    name = "speed"

    def process(self, audio, sr, t, rng, x=1.0, quality="linear"):
        x = float(x)
        if abs(x - 1.0) < 1e-9 or audio.shape[0] < 4:
            return audio
        # play x times faster: n / x samples, pitch multiplied by x
        return resample(audio, max(1, int(sr * x)), sr, quality="linear" if quality == "linear" else "auto")


def _fft_mask(sr: int, n: int, lo=None, hi=None, width_oct: float = 0.25) -> np.ndarray:
    """Raised-cosine filter mask over rfft frequencies (transition `width_oct` octaves)."""
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = np.ones_like(freqs)
    if hi is not None:  # lowpass: 1 below lo_edge, cosine ramp down to 0 at hi
        hi = float(hi)
        lo_edge = hi / 2 ** width_oct
        t = np.clip((freqs - lo_edge) / max(1e-9, hi - lo_edge), 0.0, 1.0)
        mask *= 0.5 * (1.0 + np.cos(np.pi * t))
    if lo is not None:  # highpass: cosine ramp up from 0 at lo to 1 at hi_edge
        lo = float(lo)
        hi_edge = lo * 2 ** width_oct
        t = np.clip((freqs - lo) / max(1e-9, hi_edge - lo), 0.0, 1.0)
        mask *= 0.5 * (1.0 - np.cos(np.pi * t))
    return np.clip(mask, 0.0, 1.0)


class _FFTBand(Processor):
    def _apply(self, audio, sr, mask):
        n = audio.shape[0]
        if n < 16:
            return audio
        out = np.empty_like(audio)
        for c in range(audio.shape[1]):
            out[:, c] = np.fft.irfft(np.fft.rfft(audio[:, c]) * mask, n)
        return out


@register_processor
class Lowpass(_FFTBand):
    name = "lowpass"

    def process(self, audio, sr, t, rng, cut_hz=2000.0, width_oct=0.25):
        return self._apply(audio, sr, _fft_mask(sr, audio.shape[0], hi=float(cut_hz), width_oct=float(width_oct)))


@register_processor
class Highpass(_FFTBand):
    name = "highpass"

    def process(self, audio, sr, t, rng, cut_hz=200.0, width_oct=0.25):
        return self._apply(audio, sr, _fft_mask(sr, audio.shape[0], lo=float(cut_hz), width_oct=float(width_oct)))


@register_processor
class Bandpass(_FFTBand):
    name = "bandpass"

    def process(self, audio, sr, t, rng, lo_hz=200.0, hi_hz=2000.0, width_oct=0.25):
        m = _fft_mask(sr, audio.shape[0], lo=float(lo_hz), hi=float(hi_hz), width_oct=float(width_oct))
        return self._apply(audio, sr, m)


@register_processor
class Ring(Processor):
    name = "ring"

    def process(self, audio, sr, t, rng, freq_hz=220.0, mix=0.5):
        n = audio.shape[0]
        carrier = np.sin(2 * np.pi * float(freq_hz) * np.arange(n) / sr)[:, None]
        return (1.0 - float(mix)) * audio + float(mix) * audio * carrier


@register_processor
class Drive(Processor):
    name = "drive"

    def process(self, audio, sr, t, rng, amount=3.0):
        k = max(0.0, float(amount))
        if k < 1e-6:
            return audio
        return np.tanh(audio * k) / np.tanh(k)


@register_processor
class Crush(Processor):
    name = "crush"

    def process(self, audio, sr, t, rng, bits=6):
        levels = float(2 ** max(1, int(bits)))
        return np.round(audio * levels) / levels


@register_processor
class Corrupt(Processor):
    name = "corrupt"

    def process(self, audio, sr, t, rng, mode="zeros", amount=0.001):
        """Byte-level databending on the int16 stream: 'zeros' punches zero
        runs, 'swap' swaps byte pairs (harmonic shredding), 'flip' flips bits."""
        x16 = np.clip(audio, -1.0, 1.0)
        raw = (x16 * 32767.0).astype(np.int16).tobytes()
        buf = bytearray(raw)
        n_hits = int(len(buf) * max(0.0, float(amount)))
        if mode in ("zeros", "zero"):
            for _ in range(n_hits):
                i = rng.ints(len(buf))
                run = rng.ints(1, 9)
                for j in range(min(run, len(buf) - i)):
                    buf[i + j] = 0
        elif mode == "swap":
            for _ in range(n_hits):
                i = rng.ints(len(buf) - 1)
                buf[i], buf[i + 1] = buf[i + 1], buf[i]
        elif mode == "flip":
            for _ in range(n_hits):
                i = rng.ints(len(buf))
                buf[i] ^= 1 << rng.ints(8)
        else:
            raise ValueError(f"corrupt mode {mode!r} (zeros | swap | flip)")
        out = np.frombuffer(bytes(buf), np.int16).astype(np.float32) / 32768.0
        return out.reshape(-1, audio.shape[1])


@register_processor
class Echo(Processor):
    name = "echo"

    def process(self, audio, sr, t, rng, delay_s=0.25, feedback=0.45, mix=0.4, max_taps=24):
        """Finite geometric tap series (offline-friendly 'feedback' delay):
        dry + mix*(delayed copy) + mix*fb*(2x delayed) + ..."""
        n = audio.shape[0]
        d = max(1, int(float(delay_s) * sr))
        fb = min(0.95, abs(float(feedback)))
        out = audio.astype(np.float64)
        g = float(mix)
        for tap in range(1, int(max_taps) + 1):
            if g < 1e-4 or tap * d >= n:
                break
            y = np.zeros_like(out)
            y[tap * d :] = audio[: n - tap * d]
            out = out + g * y
            g *= fb
        return out.astype(np.float32)


@register_processor
class Repeat(Processor):
    name = "repeat"

    def process(self, audio, sr, t, rng, count=4, decay=0.75, vary=0.0):
        """Stutter: repeat the chunk `count` times with decay and optional
        random pitch jitter `vary` (semitones) per copy."""
        count = max(1, int(count))
        pieces = [audio]
        g = float(decay)
        for i in range(1, count):
            piece = audio * (g**i)
            if float(vary) > 0 and piece.shape[0] > 16:
                semi = rng.float(-float(vary), float(vary))
                piece = resample(piece, sr, int(sr * 2 ** (semi / 12.0)))
            pieces.append(piece)
        n = sum(p.shape[0] for p in pieces)
        out = np.zeros((n, audio.shape[1]), dtype=np.float32)
        at = 0
        for p in pieces:
            out[at : at + p.shape[0]] = p
            at += p.shape[0]
        return out


@register_processor
class Fades(Processor):
    name = "fades"

    def process(self, audio, sr, t, rng, attack=0.005, release=0.005):
        return _edge_fades(audio, sr, float(attack), float(release))


def _edge_fades(audio: np.ndarray, sr: int, attack: float, release: float) -> np.ndarray:
    n = audio.shape[0]
    out = audio.astype(np.float32, copy=True)
    na = min(n, max(0, int(attack * sr)))
    nr = min(n, max(0, int(release * sr)))
    if na > 0:
        out[:na] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na))[:, None]
    if nr > 0:
        out[-nr:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(nr) / nr))[:, None]
    return out


@register_processor
class Env(Processor):
    name = "env"

    def process(self, audio, sr, t, rng, shape="perc", attack=0.1):
        n = audio.shape[0]
        if shape == "perc":
            e = np.exp(-4.5 * np.linspace(0.0, 1.0, n))
            a = min(n - 1, max(1, int(float(attack) * n)))
            e[:a] *= np.linspace(0, 1, a)
        elif shape == "tri":
            e = 1.0 - np.abs(np.linspace(-1.0, 1.0, n))
        elif shape in ("hann", "hanning"):
            e = np.hanning(n)
        elif shape == "const":
            e = np.ones(n)
            f = min(64, n // 3)
            if f > 0:
                e[:f] = np.linspace(0, 1, f)
                e[-f:] = np.linspace(1, 0, f)
        else:
            raise ValueError(f"env shape {shape!r} (perc | tri | hann | const)")
        return audio * (e / max(1e-9, e.max()))[:, None]


@register_processor
class Gate(Processor):
    name = "gate"

    def process(self, audio, sr, t, rng, threshold_db=-40.0):
        m = audio.mean(axis=1)
        thresh = 10.0 ** (float(threshold_db) / 20.0)
        loud = np.abs(m) > thresh
        return audio * loud[:, None]


@register_processor
class Trim(Processor):
    name = "trim"

    def process(self, audio, sr, t, rng, threshold_db=-40.0, pad=0.01):
        m = np.abs(audio.mean(axis=1))
        thresh = 10.0 ** (float(threshold_db) / 20.0)
        loud = np.nonzero(m > thresh)[0]
        if len(loud) == 0:
            return audio
        pad_n = int(float(pad) * sr)
        a = max(0, int(loud[0]) - pad_n)
        b = min(audio.shape[0], int(loud[-1]) + pad_n)
        return audio[a:b].copy()