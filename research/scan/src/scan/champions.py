"""champions: three hand-built known answers.

Each proves one region of the rig's grammar — the static additive
reference, the PPG sweep, and the Wavestation sequence — the way
prism's bellfield/glassvoice/tide proved their traversals. Params were
tuned by ear on this machine; the genome is the provenance.
"""

from __future__ import annotations

import numpy as np

from .organism import WtGenome

__all__ = ["CHAMPIONS", "champion", "blurb"]


def _organpipe() -> WtGenome:
    """The additive reference: one timbre, held. 12 harmonics at 1/k,
    four identical frames (the medium at rest), nothing moves but the
    slow decay breath. Known answer: a static reedy organ — and the
    viability gate's near-miss (rms moves only 1.1x across the clip)."""
    partials = [1.0 / k for k in range(1, 13)]
    frames = [{"partials": list(partials), "inharm": 0.0} for _ in range(4)]
    return WtGenome(
        f0=110.0,
        pitch={"type": "fixed", "rate": 0.5, "depth": 0.0, "end": 1.0},
        frames=frames,
        morph={"type": "lfo", "rate": 0.1, "depth": 0.0, "shape": "sin",
               "duty": 0.0, "bps": [0.5, 0.5, 0.5]},
        voices=[{"detune": 0.0, "level": 1.0, "pan": 0.0,
                 "morph_offset": 0.0, "attack": 0.01, "decay_tau": 8.0}],
        vector={"type": "fixed", "x0": 0.0, "y0": 0.0, "rate_x": 0.1,
                "rate_y": 0.13, "depth": 0.0, "shape_x": "sin",
                "shape_y": "tri"},
        seed=11,
    )


def _surfer() -> WtGenome:
    """The PPG sweep: 32 frames walking sine -> bright reeded stack,
    scanned by a slow sine LFO while the pitch glides an octave down.
    Timbre evolution with zero designed envelopes — the morph path IS
    the envelope. Known answer: a scan archetype, audible glide."""
    frames = []
    for k in range(32):
        beta = k / 31.0
        p = 2 + int(beta * 14)
        partials = [float(beta ** 1.2 * (1.0 / q ** 0.9)
                          * (1.0 if q % 2 else 0.35)) for q in range(1, p + 1)]
        frames.append({"partials": partials, "inharm": 0.0})
    return WtGenome(
        f0=110.0,
        pitch={"type": "glide", "rate": 0.5, "depth": 0.0, "end": 0.5},
        frames=frames,
        morph={"type": "lfo", "rate": 0.3, "depth": 1.0, "shape": "sin",
               "duty": 0.0, "bps": [0.1, 0.5, 0.9]},
        voices=[{"detune": 0.0, "level": 1.0, "pan": 0.0,
                 "morph_offset": 0.0, "attack": 0.05, "decay_tau": 8.0}],
        vector={"type": "fixed", "x0": 0.0, "y0": 0.0, "rate_x": 0.1,
                "rate_y": 0.13, "depth": 0.0, "shape_x": "sin",
                "shape_y": "tri"},
        seed=22,
    )


def _sequencer() -> WtGenome:
    """The Wavestation: eight unrelated frames stepped at 2 Hz (duty
    crossfades the edges), the pitch jumping fifths on its own step
    clock, two voices detuned 7c apart. Rhythmic timbre from table
    ORDER alone — the sequence archetype's known answer."""
    frames = [
        {"partials": [1.0, 0.05], "inharm": 0.0},                     # sine
        {"partials": [1.0 / k for k in range(1, 11)], "inharm": 0.0},  # organ
        {"partials": [0.02, 0.02, 0.02, 0.02, 0.4, 1.0, 1.0, 0.5],
         "inharm": 0.0},                                             # formant
        {"partials": [1.0 / k ** 0.7 for k in range(1, 15)],
         "inharm": 0.0},                                             # bright
        {"partials": [1.0, 0.8, 0.6, 0.5, 0.4], "inharm": 0.25},     # metal
        {"partials": [(1.0 / k if k % 2 else 0.05) for k in range(1, 13)],
         "inharm": 0.0},                                             # hollow
        {"partials": [float(0.9 ** k) for k in range(1, 17)],
         "inharm": 0.0},                                             # dense
        {"partials": [1.0, 0.3], "inharm": 0.0},                     # sine2
    ]
    voices = []
    for det, pan in ((7.0, 0.3), (-7.0, -0.3)):
        voices.append({"detune": det, "level": 0.8, "pan": pan,
                       "morph_offset": 0.0, "attack": 0.005,
                       "decay_tau": 4.0})
    return WtGenome(
        f0=220.0,
        pitch={"type": "step", "rate": 2.0, "depth": 700.0, "end": 1.0},
        frames=frames,
        morph={"type": "step", "rate": 2.0, "depth": 1.0, "shape": "saw",
               "duty": 0.15, "bps": [0.1, 0.5, 0.9]},
        voices=voices,
        vector={"type": "lfo2d", "x0": 0.0, "y0": 0.0, "rate_x": 0.3,
                "rate_y": 0.21, "depth": 0.25, "shape_x": "sin",
                "shape_y": "tri"},
        seed=33,
    )


_BUILDERS = {"organpipe": _organpipe, "surfer": _surfer,
             "sequencer": _sequencer}

CHAMPIONS = tuple(_BUILDERS)

_BLURBS = {
    "organpipe":
        "the additive reference — 12 harmonics held still; the medium at rest",
    "surfer":
        "the PPG sweep — sine -> reeded stack while the pitch glides down",
    "sequencer":
        "the Wavestation — eight unrelated frames stepped into rhythm",
}


def champion(name: str) -> WtGenome:
    if name not in _BUILDERS:
        raise ValueError(f"unknown champion '{name}' "
                         f"(have: {', '.join(CHAMPIONS)})")
    return _BUILDERS[name]()


def blurb(name: str) -> str:
    return _BLURBS[name]


def stats(audio: np.ndarray, sr: int = 44100) -> dict:
    """The listening measurements prism's champion table carries."""
    x = np.asarray(audio, np.float64)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    rms = float(np.sqrt((mono ** 2).mean()))
    peak = float(np.abs(mono).max())
    spec = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sr)
    cen = float((spec * freqs).sum() / (spec.sum() + 1e-12))
    crest = peak / (rms + 1e-12)
    return {"rms": round(rms, 3), "centroid_hz": round(cen),
            "crest": round(crest, 1)}