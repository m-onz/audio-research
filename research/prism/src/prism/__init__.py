"""prism: the pattern organism — CPPNs bred for the ear and the eye.

One genome, many phenotypes: the same compositional graph is a
spectrogram, a wavetable bank, a direct waveform, and a picture.
Phase A proved the physics (three champions, determinism, speed);
Phase B is the breeding loop — echo admission, unbounded alphabet,
rotating-lens selection, and the arms study (engine.py).
"""

from .champions import CHAMPIONS, anatomy, champion
from .genome import (FUNCTIONS, INPUTS, INPUT_IDS, Conn, Genome, Node,
                     random_genome)
from .render import MODES, SR, picture, render, write_png, write_wav

__version__ = "0.2.0"
__all__ = [
    "Genome", "Node", "Conn", "INPUTS", "INPUT_IDS", "FUNCTIONS",
    "random_genome", "render", "picture", "write_wav", "write_png",
    "MODES", "SR", "CHAMPIONS", "champion", "anatomy", "__version__",
]