"""champions: three hand-built genomes, one per traversal.

The point of a champion is to prove the physics before any evolution
happens: a genome you can point at, a render you can hold to the ear,
and a picture that shows exactly why the sound looks the way it does.
Every one of these is built with the same public API an evolved
specimen will be built with — no tricks, no shortcuts.

    bellfield   spectra    — ridges of sound bending over log-frequency
    glassvoice  wavetable  — a table that walks sine → reed and back
    tide        field      — nested slow-drifting FM at audio rate

Their pictures are their scores: bellfield seen from above is the
landscape its sound walks through.
"""

from __future__ import annotations

import math

import numpy as np

from .genome import Genome, INPUT_IDS

TAU = 2.0 * math.pi


def _hid(g: Genome, fn: str, bias: float = 0.0) -> int:
    return g.add_hidden(fn, bias)


def _c(g: Genome, src_name_or_id, dst: int, w: float) -> None:
    src = INPUT_IDS[src_name_or_id] if isinstance(src_name_or_id, str) \
        else src_name_or_id
    g.conn(src, dst, w)


# ------------------------------------------------------------------ bellfield


def _bellfield() -> Genome:
    """spectra traversal. x = seconds, y = log-frequency [-1, 1].
    Five ridges bend at different rates; one breathes through abs(sin).
    In sound: a slow glass chord whose partials glide and cross."""
    g = Genome(seed=1)

    drifts = []                          # slow phases, one per ridge
    for rate, bias in ((0.23, 0.7), (0.41, 2.1), (0.17, 0.0),
                       (0.31, 4.0)):
        n = _hid(g, "sin", bias)
        _c(g, "x", n, TAU * rate)
        drifts.append(n)
    breathe = _hid(g, "sin", 1.0)        # the breathing ridge's driver
    _c(g, "x", breathe, TAU * 0.13)
    breathe_abs = _hid(g, "abs")
    _c(g, breathe, breathe_abs, 1.0)

    shapes = ((drifts[0], 0.18, 18.0, 0.95),   # (driver, bend, width⁻¹, amp)
              (drifts[1], 0.10, 11.0, 0.65),
              (drifts[2], 0.25, 13.0, 0.85),
              (drifts[3], 0.15, 8.0, 0.50),
              (breathe_abs, 0.22, 10.0, 0.40))
    amp = g.add_output("amp")
    for driver, bend, sharp, w in shapes:
        ridge = _hid(g, "lin")           # y - bend · drift
        _c(g, "y", ridge, 1.0)
        _c(g, driver, ridge, -bend)
        gauss = _hid(g, "gauss")
        _c(g, ridge, gauss, sharp)
        _c(g, gauss, amp, w)

    pan = g.add_output("pan", fn="sin")  # slow wander, stereo = the room
    _c(g, "t", pan, TAU * 0.05)
    return g


# ------------------------------------------------------------------ glassvoice


def _glassvoice() -> Genome:
    """wavetable traversal. x = phase [-1, 1], y = morph [-1, 1].
    A morphing FM voice: the morph axis walks the inner phase, so the
    spectrum slides from near-pure to reedy and back. Band-limited at
    every table — the reed never aliases."""
    g = Genome(seed=2)

    inner = _hid(g, "sin")               # sin(1.3πx + 0.9πy)
    _c(g, "x", inner, 1.3 * math.pi)
    _c(g, "y", inner, 0.9 * math.pi)

    outer = _hid(g, "sin")               # sin(πx + 2.4·inner − 0.6y)
    _c(g, "x", outer, math.pi)
    _c(g, inner, outer, 2.4)
    _c(g, "y", outer, -0.6)

    reed = _hid(g, "sq")                 # a signed-square shadow
    _c(g, outer, reed, 0.35)

    wave = g.add_output("wave")
    _c(g, outer, wave, 1.0)
    _c(g, reed, wave, 0.45)

    pan = g.add_output("pan", fn="sin")
    _c(g, "t", pan, TAU * 0.05)
    return g


# ------------------------------------------------------------------ tide


def _tide() -> Genome:
    """field traversal. x = seconds — weights are frequencies.
    Nested drift-FM: a carrier and a detuned partner ride the same
    slow index; a 0.017 Hz swell pushes both into a tanh waveshaper,
    so the weather is saturation depth, not an envelope."""
    g = Genome(seed=3)

    slow = _hid(g, "sin")                # 0.055 Hz wander
    _c(g, "x", slow, TAU * 0.055)
    idx = _hid(g, "sin")                 # the drifting FM index
    _c(g, slow, idx, 3.2)

    carrier = _hid(g, "sin")             # sin(2π·55t + 2.4·idx)
    _c(g, "x", carrier, TAU * 55.0)
    _c(g, idx, carrier, 2.4)
    partner = _hid(g, "sin")             # detuned, negative index
    _c(g, "x", partner, TAU * 55.35)
    _c(g, idx, partner, -1.7)

    swell = _hid(g, "sin", 0.6)          # 0.017 Hz saturation weather
    _c(g, "x", swell, TAU * 0.017)

    field = g.add_output("field", fn="tanh")
    _c(g, carrier, field, 1.4)
    _c(g, partner, field, 1.4)
    _c(g, swell, field, 0.4)

    pan = g.add_output("pan", fn="sin")
    _c(g, "x", pan, TAU * 0.023)
    return g


# ------------------------------------------------------------------ registry

CHAMPIONS: dict[str, dict] = {
    "bellfield": {
        "mode": "spectra",
        "blurb": ("five log-frequency ridges that bend at their own "
                  "rates — a slow glass chord; the picture is the "
                  "landscape the sound walks through"),
        "build": _bellfield,
    },
    "glassvoice": {
        "mode": "wavetable",
        "blurb": ("a single-cycle table that walks from near-pure sine "
                  "to reeded shadow and back, band-limited every step"),
        "build": _glassvoice,
    },
    "tide": {
        "mode": "field",
        "blurb": ("direct field: nested slow-drifting FM at audio rate, "
                  "oversampled and lowpassed — inharmonic but alive, "
                  "its weather is saturation depth"),
        "build": _tide,
    },
}


def champion(name: str) -> Genome:
    """A fresh instance of a champion genome."""
    return CHAMPIONS[name]["build"]()


# ------------------------------------------------------------------ labels


def anatomy(g: Genome) -> dict:
    """Ground truth off the genome — the maze discipline: describe
    what a specimen is without listening to it."""
    fns: dict[str, int] = {}
    for n in g.nodes.values():
        if n.kind != "in":
            fns[n.fn] = fns.get(n.fn, 0) + 1
    used = sorted({c.src for c in g.conns if c.on})
    senses = [name for name, nid in INPUT_IDS.items() if nid in used]
    return {
        "nodes": sum(1 for n in g.nodes.values() if n.kind != "in"),
        "conns": sum(1 for c in g.conns if c.on),
        "outputs": list(g.outputs),
        "functions": fns,
        "senses": senses,
        "seed": g.seed,
    }