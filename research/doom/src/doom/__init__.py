"""doom — perfectly audio-synchronized generative video.

    import doom
    doom.render(audio="piece.wav", out="video.mp4", preset="glitchcore")

Point it at any audio file; it analyzes per-frame features (the sync
clock), schedules scenes, renders numpy frames, and pipes them through
ffmpeg into an H.264 .mp4 with the audio muxed in the same pass.
"""

__version__ = "0.1.0"

from .audio import AudioFeatures, analyze_audio, probe_duration
from .compose import render
from .fx import FX, build_fx_chain, list_fx, register_fx, run_chain
from .palettes import PALETTES, apply_palette, get_palette, psx_quantize, rgb555
from .presets import PRESETS, list_presets
from .scenes import SCENES, list_scenes, make_scene, register_scene
from .timeline import normalize_plan, plan_from_audio
from .writer import FFmpegWriter, verify_mp4

__all__ = [
    "__version__",
    "render",
    "analyze_audio",
    "AudioFeatures",
    "probe_duration",
    "SCENES", "register_scene", "make_scene", "list_scenes",
    "FX", "register_fx", "build_fx_chain", "run_chain", "list_fx",
    "PALETTES", "get_palette", "apply_palette", "rgb555", "psx_quantize",
    "PRESETS", "list_presets",
    "plan_from_audio", "normalize_plan",
    "FFmpegWriter", "verify_mp4",
]