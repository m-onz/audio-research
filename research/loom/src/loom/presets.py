"""Presets: named starting points for `loom.compose(preset=...)`.

A preset is just a dict of compose() defaults — copy one, tweak it, or add
your own to PRESETS. Explicit compose() kwargs always win over the preset.

Each preset is tuned for a different *kind of listening*: drift is ambient
granularity, storm is maximal-contrast collage, pulse is rhythm-forward,
ghost re-processes the corpus into its own shadow, strata builds
feature-separated layers.
"""

from __future__ import annotations

# Base defaults shared by every compose() call (presets override these).
BASE_DEFAULTS: dict = dict(
    length=120.0,
    seed=0,
    sr=None,
    chunker="fixed",
    chunker_params=None,
    select="all",
    passes=None,
    arranger="sequence",
    arranger_params=None,
    processors=None,
    normalize=True,
    bits=16,
    soft_clip=False,
    store="memory",
)

PRESETS: dict[str, dict] = {
    # Ambient granular drift: onset chunks, darkening lowpass, thinning cloud.
    "drift": dict(
        chunker="onset",
        chunker_params={"min_seconds": 0.3, "max_seconds": 6.0},
        arranger="cloud",
        arranger_params=dict(
            density=[18, 110, 24],
            speed=[1.0, 0.55],
            gain=[0.6, 1.0, 0.5],
            pan_spread=0.8,
            fade=0.05,
            order="random",
        ),
        processors=[["lowpass", {"cut_hz": [3000, 500], "width_oct": 0.4}], "fades"],
        select="dur>0.1",
    ),
    # Maximal-contrast collage: farthest-point ordering, tape-slow arc.
    "storm": dict(
        chunker="onset",
        chunker_params={"min_seconds": 0.2, "max_seconds": 10.0},
        arranger="sequence",
        arranger_params=dict(
            order="farthest",
            gap=-0.35,
            speed=[1.0, 0.72, 0.9],
            gain=1.0,
            fade=0.006,
        ),
        processors=["normalize", "fades"],
    ),
    # Rhythm-forward: bass hits on a Euclid pulse, texture raining above.
    "pulse": dict(
        chunker="onset",
        chunker_params={"min_seconds": 0.1, "max_seconds": 2.0},
        arranger="layers",
        arranger_params=dict(
            voices=[
                dict(select="bass>0.6 and rms>0.4", arranger="euclid",
                     params=dict(bpm=138, steps=16, pulses=9, rotate=3, gate=0.5, accent=0.6), gain=1.0),
                dict(select="air>0.55", arranger="cloud",
                     params=dict(density=[30, 160, 40], speed=[0.8, 1.6], pan_spread=0.9, fade=0.04), gain=0.5),
            ]
        ),
        processors=["fades"],
    ),
    # Re-process the corpus into its own shadow, then let it speak in
    # the corpus's own grammar (markov over clusters).
    "ghost": dict(
        chunker="onset",
        chunker_params={"min_seconds": 0.25, "max_seconds": 5.0},
        passes=[
            dict(select="rms>0.5", processors=[["lowpass", {"cut_hz": 900}], "reverse"],
                 tag="dark"),
            dict(select="tag=dark and rms>0.6", processors=["echo"], tag="dark_echo"),
        ],
        select="tag=dark or tag=dark_echo or rms<0.5",
        arranger="sequence",
        arranger_params=dict(order="markov", gap=0.0, speed=[1.0, 0.85], fade=0.01),
        processors=["fades", "normalize"],
    ),
    # Feature-separated strata: lows walk, mids pulse, highs drift.
    "strata": dict(
        chunker="onset",
        chunker_params={"min_seconds": 0.15, "max_seconds": 4.0},
        arranger="layers",
        arranger_params=dict(
            voices=[
                dict(select="bass>0.75", arranger="sequence",
                     params=dict(order="by:rms", gap=-0.1, speed=0.85), gain=1.0),
                dict(select="mid>0.5 and mid<0.8", arranger="euclid",
                     params=dict(bpm=100, steps=12, pulses=5, gate=0.3, accent=0.55), gain=0.7),
                dict(select="air>0.7", arranger="cloud",
                     params=dict(density=[50, 10, 50], pan_spread=1.0, fade=0.08), gain=0.4),
            ]
        ),
        processors=["fades"],
    ),
    # Chance collage: random windows chased by similarity, drifting speed.
    "collage": dict(
        chunker="random",
        chunker_params={"n": 120, "min_seconds": 0.4, "max_seconds": 3.0},
        arranger="sequence",
        arranger_params=dict(order="chase", gap=-0.2, speed=[0.9, 1.3, 0.7], fade=0.008),
        processors=[["lowpass", {"cut_hz": [6000, 1200]}], "fades"],
    ),
}


def apply_preset(name: str, explicit: dict) -> dict:
    """Base defaults <- preset <- explicit kwargs (explicit always wins).

    `explicit` contains only kwargs actually passed by the caller (the
    compose() _UNSET sentinel has already filtered them), so a preset can
    safely provide its own chunker/arranger/etc.
    Nested dicts (chunker_params, arranger_params) merge key-wise so a
    preset can be nudged without re-declaring it.
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    out = _deep_copy_config(BASE_DEFAULTS)
    out.update(_deep_copy_config(PRESETS[name]))
    for k, v in explicit.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k in ("chunker_params", "arranger_params"):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def _deep_copy_config(cfg: dict) -> dict:
    import copy

    return copy.deepcopy(cfg)


def list_presets() -> list[str]:
    return sorted(PRESETS)