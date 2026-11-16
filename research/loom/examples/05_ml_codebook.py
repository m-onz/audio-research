"""ML extension: a learned processor running inline during offline render.

CodebookProcessor fits a k-means codebook over corpus STFT magnitudes,
then rewrites each chunk's spectra to its nearest codebook entry — input
material 'speaks in the corpus's timbres'. The fitted instance is passed
directly into the processor chain. This is the pattern for any learned
model: fit -> instance -> chain.

Run:  python3 examples/05_ml_codebook.py path/to/corpus
"""

import sys
from pathlib import Path

import loom
from loom.ml import CodebookProcessor

sources = sys.argv[1:] or [Path(__file__).parents[2] / "audio_lab" / "clubbed"]
corpus = Path(sources[0]) if len(sys.argv) == 1 else sources

# 1. fit the model on the corpus (folder of wavs or a single file)
codebook = CodebookProcessor(n_fft=2048, blend=0.85).fit_folder(corpus, k=96)

# 2. use it inline: the *instance* goes into the chain spec
result = loom.compose(
    sources=corpus,
    length=60.0,
    seed=13,
    chunker="onset",
    chunker_params={"min_seconds": 0.4, "max_seconds": 4.0},
    arranger="sequence",
    arranger_params={"order": "farthest", "gap": -0.25},
    processors=[codebook, "fades"],  # fitted instance + builtin
    out="loom_out/codebook.wav",
)
print(result.summary())

# 3. or register the fitted instance under a name for string specs
loom.register_processor_instance("my_codebook", codebook)
r2 = loom.compose(
    sources=corpus,
    length=30.0,
    seed=14,
    chunker="random",
    chunker_params={"n": 40, "min_seconds": 1.0, "max_seconds": 3.0},
    arranger="cloud",
    arranger_params={"density": 90},
    processors="my_codebook(dream=0.15)",  # 15% of frames jump to random entries
    out="loom_out/codebook_dream.wav",
)
print(r2.summary())