"""drums: a CUSTOM per-generation drum kit.

Each riff gets its OWN kit: for every voice (kick / snare / hihat / openhh /
perc) a sample is picked from a categorized drum library and then
TRANSFORMED (pitch / drive / bitcrush / ring-mod / filter / gate) with the
genome's own seed — so no two ideas share a kit. Voice names still match the
patch's alien voices, so the exported .pd messages drive them directly.

    kit, prov = drums.make_kit(np.random.default_rng(seed))
"""
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from . import core

# the drum library: one subfolder per category (kick/ snare/ hihat/ ...), each
# holding .wav hits. Defaults to ./sounds/drums; override with
# AUDIO_RESEARCH_DRUMS. If a category folder is missing the voice is simply
# omitted from the kit (riffs still render, just without that drum).
DRUMS = Path(os.environ.get("AUDIO_RESEARCH_DRUMS", "sounds/drums"))

DRUM_VOICES = ("kick", "snare", "hihat", "openhh", "perc")

# voice -> candidate category folders in the library (merged; first existing
# folders contribute, deeper corpus dirs are ignored — these are real drums)
_FOLDERS = {
    "kick":   ["kick", "kick2"],
    "snare":  ["snare2", "snare"],
    "hihat":  ["hihat2", "hihat"],
    "openhh": ["openhh", "openhihats"],
    "perc":   ["dnb-perc", "dnb-perc2", "perc", "rimshot", "shaker", "toms", "clap"],
}
_MAXDUR = {"kick": 0.9, "snare": 0.6, "hihat": 0.3, "openhh": 0.7, "perc": 0.7}

_CANDS = None                                  # cached file lists (cheap: just names)


# ----------------------------------------------------------------- load

def _load_one(path, voice, sr0=44100):
    try:
        x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception:
        return None
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    if sr != sr0:
        x = core.resample(x, sr / sr0)
    m = np.abs(x).mean(axis=1)
    pk = float(m.max())
    if pk < 1e-5:
        return None
    above = np.flatnonzero(m > 0.04 * pk)          # onset-trim to the attack
    if len(above):
        x = np.ascontiguousarray(x[max(0, int(above[0]) - int(0.001 * sr0)):])
    x = x[: int(_MAXDUR[voice] * sr0)]
    x = x / (float(np.abs(x).max()) + 1e-9) * 0.9
    f = min(len(x) // 8, int(0.004 * sr0))
    if f > 2:
        x = x.copy(); x[-f:] *= np.linspace(1, 0, f)[:, None]
    return x.astype(np.float32)


def _candidates():
    """{voice: [paths]} — scanned once from the library."""
    global _CANDS
    if _CANDS is not None:
        return _CANDS
    out = {}
    for voice, folders in _FOLDERS.items():
        paths = []
        for folder in folders:
            d = DRUMS / folder
            if d.is_dir():
                paths += [str(p) for p in sorted(d.glob("*.[wW]av"))]
        out[voice] = paths
    _CANDS = out
    return out


# ----------------------------------------------------------------- transform

def _transform_drum(x, sr, voice, rng):
    """1-2 light, transient-preserving transforms — gives each kit character
    without destroying the hit."""
    from scipy.signal import butter, lfilter
    y = np.asarray(x, np.float32).copy()
    ops = list(rng.permutation(["pitch", "drive", "crush", "ring", "filter", "gate"]))
    for op in ops[: int(rng.integers(1, 3))]:
        if op == "pitch":
            semi = rng.uniform(-3, 3) if voice == "kick" else rng.uniform(-5, 7)
            y = core.resample(y, 2.0 ** (semi / 12.0))
        elif op == "drive":
            d = float(rng.uniform(1.5, 4.0))
            y = (np.tanh(y * d) / np.tanh(d)).astype(np.float32)
        elif op == "crush":
            hold = int(rng.integers(2, 5))                 # sample-rate crush grit
            y = np.repeat(y[::hold], hold, axis=0)[: len(y)]
        elif op == "ring":
            f = (rng.uniform(40, 220) if voice in ("kick", "snare")
                 else rng.uniform(200, 1400))
            t = np.arange(len(y)) / sr
            y = (y * np.sin(2 * np.pi * f * t)[:, None]).astype(np.float32)
        elif op == "filter":
            if rng.random() < 0.5:
                b, a = butter(2, float(rng.uniform(1800, 7000)) / (sr / 2), "low")
            else:
                b, a = butter(2, float(rng.uniform(70, 400)) / (sr / 2), "high")
            y = lfilter(b, a, y, axis=0).astype(np.float32)
        elif op == "gate":
            env = np.exp(-np.linspace(0, float(rng.uniform(4, 12)), len(y)))
            y = (y * env[:, None]).astype(np.float32)
    y = core.safe(y)
    pk = float(np.abs(y).max())
    if pk > 1e-6:
        y = y / pk * 0.9
    return y.astype(np.float32)


# ----------------------------------------------------------------- build kit

def make_kit(rng, variants=2, transform=True):
    """A fresh kit for one generation. Returns ({voice: [buffers]}, provenance).
    `rng` seeded from the genome -> each idea gets a unique kit."""
    cands = _candidates()
    kit, prov = {}, {}
    for voice in DRUM_VOICES:
        paths = cands.get(voice) or []
        if not paths:
            continue
        # sample WITH replacement so even a thin folder yields `variants`
        # differently-transformed hits
        picks = rng.integers(0, len(paths), size=max(1, variants))
        buf, info = [], []
        for j in picks:
            y = _load_one(paths[int(j)], voice)
            if y is None:
                continue
            tag = "raw"
            if transform:
                yt = _transform_drum(y, 44100, voice, rng)
                if yt is not None and len(yt) >= 64:
                    y, tag = yt, "xform"
            buf.append(y)
            info.append((Path(paths[int(j)]).stem[:22], tag))
        if buf:
            kit[voice] = buf
            prov[voice] = info
    return kit, prov


def kit():
    """Backward-compatible default kit (unseeded)."""
    k, _ = make_kit(np.random.default_rng(0))
    return k


def spans():
    return {v: len(b) for v, b in kit().items()}
