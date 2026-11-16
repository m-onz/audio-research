"""champions: hand-built score organisms — known answers first.

The cryptid precedent: before any evolution, a few champions prove
the physics. Each is a minimal CPPN whose behavior can be asserted
from first principles, because each wires a scaffold sense straight
to an output through one function:

    name      wiring               known answer
    ------    ------------------   ----------------------------------------
    follower  step(s0 - 0.35)     strikes exactly at spine onsets —
                                  the attack-lock; every run starts on
                                  the master train's attacks
    answerer  gauss(6*s0 - 3.0)    fires MID-DECAY, between onsets —
                                  the temporal coordinate frame made
                                  audible: the network knows where it
                                  is inside each event (the paper's
                                  subtle trick, proven)
    splitter  step(sk - 0.30)     register-locked arrangement: low /
                                  mid / high voices follow their own
                                  bands (s1, s2, s3) — multi-channel
                                  scaffolding, Johnny Cope style
    swell     tanh(3*s0 - 0.6)    graded strikes: velocity FOLLOWS the
                                  decay phase — the paper's volume
                                  nuance, a curve not a gate
    kit       six voices, three    the ensemble: kick on the bass
             clocks, subdivision  clock, rim at the mid band's 40%
             multipliers, one    phase, percA on the high clock,
             weather-gated voice percB subdividing the mid clock x2,
                                  hat subdividing the master clock x4,
                                  openhat opening only in LOUD
                                  weather — polymeter and evolution,
                                  all pure functions of the spine

All three are PURE: no time, no position input — every event they
produce is a function of the spine's state alone. If evolution later
finds something these cannot, good — that is what the mutation zoo
is for. Champions are the ruler, not the ceiling.

random_score_genome is the founder for breeding: the same minimal
NEAT-style start (loose hidden chain, one output), but wired onto
the SCAFFOLD senses instead of free geometry — from the first
generation, structure comes from the spine.
"""

from __future__ import annotations

import numpy as np

from .genome import FUNCTIONS, Genome, INPUT_IDS

__all__ = ["CHAMPIONS", "build_champion", "follower", "answerer",
           "splitter", "swell", "kit", "random_score_genome"]


def follower() -> Genome:
    """Strike at the spine's attacks: audible while the master spike
    is fresh (level 1 while s0 > 0.35, i.e. the first ~65% of each
    inter-onset interval)."""
    g = Genome(seed=1908)                       # the paper's year
    out = g.add_output("pulse", fn="step", bias=-0.35)
    g.conn(INPUT_IDS["s0"], out, 1.0)
    return g


def answerer() -> Genome:
    """Fire between the attacks: peak when the master spike has
    decayed to half (gauss peaks where 6*s0 - 3 = 0), audible for the
    middle half of each decay. Only the coordinate frame makes this
    possible — a flat rms input has no 'mid-note' to find."""
    g = Genome(seed=8032)
    out = g.add_output("answer", fn="gauss", bias=-3.0)
    g.conn(INPUT_IDS["s0"], out, 6.0)
    return g


def splitter() -> Genome:
    """One voice per register, each driven by its own band train:
    the low voice answers the bass, the high voice the air. The
    arrangement is band-locked by construction."""
    g = Genome(seed=2610)
    for name, sense in (("low", "s1"), ("mid", "s2"), ("high", "s3")):
        out = g.add_output(name, fn="step", bias=-0.30)
        g.conn(INPUT_IDS[sense], out, 1.0)
    return g


def swell() -> Genome:
    """Graded strikes: velocity follows the decay phase itself —
    tanh(3*s0 - 0.6) is ~1 at the attack and tails with the spike,
    inaudible by the last quarter. The paper's volume nuance, proven:
    level is a curve, not a gate."""
    g = Genome(seed=2077)
    out = g.add_output("swell", fn="tanh", bias=-0.6)
    g.conn(INPUT_IDS["s0"], out, 3.0)
    return g


def kit() -> Genome:
    """The ensemble: six voices, three clocks, one weather gate —
    polymeter and evolution as pure functions of the spine.

    POLYMETER the spine's bands already run at different meters
    (bass onsets x full flux x highs); each voice locks its own
    clock and some SUBDIVIDE it — sin(4pi*s2) fires twice per
    mid-band IOI, sin(8pi*s0) four times per master IOI — so the
    layers phase and cross as the spine's own meters drift.

      kick     step(s1-0.35)      the bass clock's attacks — the floor
      rim      gauss(6*s2-2.4)    one hit at 40% of the MID clock's IOI:
                                 the offbeat cross
      percA    tanh(2.5*s3-0.55)  graded strikes on the HIGH clock
      percB    tanh(3*sin(4pi s2)-0.5)  the mid clock, subdivided x2
      hat      tanh(2.2*sin(8pi s0)-0.45) the master clock, x4, soft
      openhat  gauss(9*s0 + 3*weather - 8.1)  the OPEN that lets the
                                 next hit land inside it — quiet weather
                                 pushes the peak off the end of the IOI
                                 (silent); loud passages slide it in,
                                 and the louder the weather the earlier
                                 it opens. Density AND phase evolve
                                 with the spine's loudness.

    Under a form's tempo map the tick rate changes per section, so
    the subdivision windows catch more or fewer ticks — the kit
    rolls in the fast sections and breathes in the slow ones."""
    import numpy as _np
    g = Genome(seed=4242)
    # clock hiddens
    ph4 = g.add_hidden("sin")                 # sin(4pi*s2): mid x2
    g.conn(INPUT_IDS["s2"], ph4, 4.0 * _np.pi)
    ph8 = g.add_hidden("sin")                 # sin(8pi*s0): master x4
    g.conn(INPUT_IDS["s0"], ph8, 8.0 * _np.pi)
    ohw = g.add_hidden("tanh", bias=-2.6)      # the weather gate
    g.conn(INPUT_IDS["live_rms"], ohw, 4.0)
    # voices
    kick = g.add_output("kick", fn="step", bias=-0.35)
    g.conn(INPUT_IDS["s1"], kick, 1.0)
    rim = g.add_output("rim", fn="gauss", bias=-2.4)
    g.conn(INPUT_IDS["s2"], rim, 6.0)
    perc_a = g.add_output("percA", fn="tanh", bias=-0.55)
    g.conn(INPUT_IDS["s3"], perc_a, 2.5)
    perc_b = g.add_output("percB", fn="tanh", bias=-0.5)
    g.conn(ph4, perc_b, 3.0)
    hat = g.add_output("hat", fn="tanh", bias=-0.45)
    g.conn(ph8, hat, 2.2)
    openhat = g.add_output("openhat", fn="gauss", bias=-8.1)
    g.conn(INPUT_IDS["s0"], openhat, 9.0)
    g.conn(ohw, openhat, 3.0)
    return g


CHAMPIONS = {
    "follower": (follower, "step(s0-0.35): attack-locked strikes"),
    "answerer": (answerer, "gauss(6*s0-3): fires mid-decay, between onsets"),
    "splitter": (splitter, "step per band: low/mid/high register lock"),
    "swell": (swell, "tanh(3*s0-0.6): velocity follows the decay curve"),
    "kit": (kit, "six-voice ensemble: three clocks, subdivision "
                "multipliers, weather-gated openhat"),
}


def build_champion(name: str) -> Genome:
    if name not in CHAMPIONS:
        raise ValueError(f"unknown champion {name!r}; "
                         f"valid: {sorted(CHAMPIONS)}")
    return CHAMPIONS[name][0]()


def random_score_genome(rng, n_hidden: int = 3) -> Genome:
    """The founder for score evolution: prism's random founder, but
    the hidden chain hears the SPIKE STREAMS (s0..s4) and weather,
    not free geometry. Two outputs from the start — a struck voice
    and a moving voice — NEAT-minimal, scaffold-attached. Everything
    after this is mutation."""
    rng = rng if isinstance(rng, np.random.Generator) \
        else np.random.default_rng(rng)
    g = Genome(seed=int(rng.integers(2**31)))
    fns = list(FUNCTIONS)
    senses = ("s0", "s1", "s2", "s3", "s4",
              "live_rms", "live_centroid", "live_flux")
    prev = None
    for _ in range(n_hidden):
        nid = g.add_hidden(fns[int(rng.integers(len(fns)))])
        for name in senses:
            if rng.random() < 0.6:
                g.conn(INPUT_IDS[name], nid, 0.8 * rng.standard_normal())
        if prev is not None and rng.random() < 0.7:
            g.conn(prev, nid, 0.8 * rng.standard_normal())
        prev = nid
    struck = g.add_output("pulse", fn="tanh")
    g.conn(prev if prev is not None else INPUT_IDS["s0"], struck,
           float(0.8 * rng.standard_normal() or 0.5))
    moving = g.add_output("flow", fn="tanh")
    g.conn(INPUT_IDS["live_flux"], moving, 0.8 * rng.standard_normal())
    if rng.random() < 0.5 and prev is not None:
        g.conn(prev, moving, 0.5 * rng.standard_normal())
    return g