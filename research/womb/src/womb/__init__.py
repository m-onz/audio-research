"""womb — where loom and doom meet.

Joint audiovisual generation, fully offline: sonify video back into sound,
feed the drift back through loom/doom-style rendering, and let generations
degrade on purpose. The artifact is always a finished video.

    import womb
    reel = womb.metabolize("piece.wav", out="my_reel", passes=3,
                           preset="metabolism", seed=7)
    print(reel.drift_report())
"""

__version__ = "0.1.0"

from .degrade import AUDIO_DEGRADES, VIDEO_DEGRADES, degrade_audio, degrade_video
from .metabolize import PRESETS, metabolize
from .lowpoly import LOWPOLY_DEFAULTS, lowpoly_frame, reprocess
from .organs import ORGANS, CorpusBank, list_organs, load_corpus_bank, make_organ, register_organ
from .search import mutate_genome, random_genome, search, walk
from .piece import generate, normalize_plan
from .pieces import PIECES, list_pieces
from .reel import Reel
from .sonify import SONIFY_MODES, sonify, write_wav
from .video import FrameStats, decode_frames, decode_luma, probe_video, video_stats
from .cryptid import (CHAMPIONS, build_champion, describe_genome, load_specimen,
                      render as cryptid_render, save_specimen, viability)
from . import breed

__all__ = [
    "__version__",
    "generate", "normalize_plan",
    "ORGANS", "register_organ", "make_organ", "list_organs",
    "CorpusBank", "load_corpus_bank",
    "search", "walk", "random_genome", "mutate_genome",
    "lowpoly_frame", "reprocess", "LOWPOLY_DEFAULTS",
    "PIECES", "list_pieces",
    "metabolize", "PRESETS",
    "Reel",
    "sonify", "SONIFY_MODES", "write_wav",
    "degrade_audio", "degrade_video", "AUDIO_DEGRADES", "VIDEO_DEGRADES",
    "probe_video", "decode_frames", "decode_luma", "video_stats", "FrameStats",
    "cryptid_render", "CHAMPIONS", "build_champion", "viability",
    "save_specimen", "load_specimen", "describe_genome", "breed",
]