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

All three are PURE: no time, no position input — every event they
produce is a function of the spine's state alone. If evolution later
finds something these cannot, good — that is what the mutation zoo
is for. Champions are the ruler, not the ceiling.
"""

from __future__ import annotations

from .genome import Genome, INPUT_IDS

__all__ = ["CHAMPIONS", "build_champion", "follower", "answerer", "splitter"]


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


CHAMPIONS = {
    "follower": (follower, "step(s0-0.35): attack-locked strikes"),
    "answerer": (answerer, "gauss(6*s0-3): fires mid-decay, between onsets"),
    "splitter": (splitter, "step per band: low/mid/high register lock"),
}


def build_champion(name: str) -> Genome:
    if name not in CHAMPIONS:
        raise ValueError(f"unknown champion {name!r}; "
                         f"valid: {sorted(CHAMPIONS)}")
    return CHAMPIONS[name][0]()