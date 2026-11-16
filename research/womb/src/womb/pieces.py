"""presets: named organ pieces — starting points for original A/V work.

Every preset is an organ timeline; pass `corpus=` (loom output, the EP,
anything) to give the organs your own grain material. Same open-registry
philosophy as loom/doom: copy, tweak, add your own.
"""

PIECES: dict[str, list[dict]] = {
    # The physics flagship: struck plates whose visible modes are their
    # audible tones, dissolving into a synchronized oscillator field.
    "plate": [
        dict(organ="waveplate", palette="ice"),
        dict(organ="waveplate", palette="inferno", strike_every=6),
        dict(organ="kuramoto", palette="vhs_night"),
        dict(organ="waveplate", palette="blood", damp=0.997),
    ],
    # Biology: growing automata firing your corpus, then its inhabitants.
    "colony": [
        dict(organ="automata", palette="toxic"),
        dict(organ="swarm", palette="vhs_night"),
        dict(organ="automata", palette="inferno"),
        dict(organ="swarm", palette="toxic"),
    ],
    # The corpus as territory: a walker tours your own material's map.
    "territory": [
        dict(organ="territory", palette="black_bronze"),
        dict(organ="automata", palette="black_bronze"),
        dict(organ="territory", palette="black_bronze"),
        dict(organ="kuramoto", palette="verdigris"),
    ],
    # All five organs, one each — the breadth tour.
    "cabinet": [
        dict(organ="waveplate", palette="ice"),
        dict(organ="automata", palette="toxic"),
        dict(organ="swarm", palette="blood"),
        dict(organ="kuramoto", palette="ps2_dusk"),
        dict(organ="territory", palette="amber"),
    ],
}


def list_pieces() -> list[str]:
    return sorted(PIECES)