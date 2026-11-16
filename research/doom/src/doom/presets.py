"""presets: named looks.

Each preset is a look + a scene schedule. The plan entries follow
timeline.py; `fx` is the default chain for segments that don't set one.
Copy, tweak, or add your own — same philosophy as loom's presets.
"""

PRESETS: dict[str, dict] = {
    # Full demoscene throwback: classic effects, clean-ish signal, palette heat.
    "demo": dict(
        plan=[
            dict(scene="starfield", start=0, end=None, transition="cut"),
            dict(scene="plasma", transition="cut"),
            dict(scene="tunnel", transition="cut"),
            dict(scene="metaballs", transition="glitch"),
            dict(scene="fire", transition="cut"),
            dict(scene="starfield", transition="glitch"),
        ],
        fx=["glow", "scanlines"],
    ),
    # PS2 memory: low-poly flyover + dithered 15-bit everything.
    "ps2": dict(
        plan=[
            dict(scene="lowpoly", transition="cut", palette="ps2_dusk"),
            dict(scene="plasma", transition="glitch", palette="ps2_dusk"),
            dict(scene="lowpoly", transition="dissolve"),
            dict(scene="starfield", transition="glitch", palette="ice"),
        ],
        fx=["psx(levels=24)", "scanlines", "glow"],
    ),
    # Taped evidence: everything through a dying VCR.
    "vhs": dict(
        plan=[
            dict(scene="scope", transition="glitch"),
            dict(scene="boids", transition="glitch"),
            dict(scene="tunnel", transition="cut"),
            dict(scene="spectro", transition="dissolve"),
            dict(scene="feedback", transition="glitch"),
        ],
        fx=["vhs", "scanlines", "rgb_split"],
    ),
    # Onset war: corruption lands on every transient.
    "glitchcore": dict(
        plan=[
            dict(scene="plasma", transition="glitch"),
            dict(scene="flow", transition="glitch"),
            dict(scene="starfield", transition="glitch"),
            dict(scene="spectro", transition="glitch"),
            dict(scene="feedback", transition="glitch"),
            dict(scene="plasma", transition="glitch"),
        ],
        fx=["corrupt(amount=0.0008)", "slices", "block_glitch", "rgb_split", "strobe", "grain"],
    ),
    # Laboratory: the honest visualizations, lightly dressed.
    "scope_laboratory": dict(
        plan=[
            dict(scene="scope", transition="dissolve", palette="green_terminal"),
            dict(scene="spectro", transition="dissolve", palette="inferno"),
            dict(scene="mandel", transition="dissolve", palette="ice"),
            dict(scene="rd", transition="dissolve", palette="ice"),
        ],
        fx=["scanlines", "glow"],
    ),
    # Nature of Code: the simulation chapters, straight.
    "noc": dict(
        plan=[
            dict(scene="boids", transition="dissolve"),
            dict(scene="flow", transition="dissolve"),
            dict(scene="lenia", transition="dissolve"),
            dict(scene="rd", transition="dissolve"),
            dict(scene="fire", transition="glitch"),
        ],
        fx=["grain"],
    ),
    # Psychedelic drift: self-organizing systems + hue rotation.
    " organism": dict(
        plan=[
            dict(scene="lenia", transition="dissolve"),
            dict(scene="rd", transition="dissolve"),
            dict(scene="metaballs", transition="dissolve"),
            dict(scene="worms", transition="dissolve"),
        ],
        fx=["hue_cycle", "glow", "grain"],
    ),
}

# (typo-proof alias)
PRESETS["organism"] = PRESETS.pop(" organism")


def list_presets() -> list[str]:
    return sorted(PRESETS)