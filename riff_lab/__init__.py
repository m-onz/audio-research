"""riff_lab — genetic riff generator for DnB / dubstep performance patches.

Each riff is a small genetic individual rendered as a FOLDER you can drop
into an alien-av-toolkit patch:

    riffs/gen_001/riff_001/
        _preview.wav        full audio render — audition it
        _genome.json        the genetic material (breed from it)
        _info.txt           how to use it in the patch
        patterns/           ready-to-paste alien (seq ...) lines:
            bass.txt  noise.txt  fill.txt  sub.txt  all.txt
        sounds/
            bass/           numbered oneshots (pattern index = file)
            noise/          the interjected noisy sounds
            fill/           end-of-phrase higher-pitched noises + fills

Source sounds are picked from your corpus under ./sounds (see sounds/README),
optionally pushed through oneshot_lab transforms. Sub bass patterns are
exported as MIDI note numbers for m-sub2~-style voices ("- - - - 23 - - - 45").
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# echo (the perceptual encoder / novelty "memory organ") ships alongside this
# package in ../echo/src — make it importable regardless of the CWD.
import sys
_ECHO_SRC = _ROOT / "echo" / "src"
if _ECHO_SRC.is_dir() and str(_ECHO_SRC) not in sys.path:
    sys.path.insert(0, str(_ECHO_SRC))

from oneshot_lab import core           # noqa: E402  (sibling top-level package)

__all__ = ["core"]