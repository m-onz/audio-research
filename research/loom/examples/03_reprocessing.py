"""Re-processing: passes that transform, tag, and re-combine chunks.

A pass selects chunks by feature quantile or tag, runs a processor chain,
and derives new chunks (keeping lineage). Later passes — and the final
selection — can query those tags. This is how a corpus becomes its own
shadow: the piece below is built from processed copies of its loudest parts.

Run:  python3 examples/03_reprocessing.py path/to/corpus
"""

import sys
from pathlib import Path

import loom

sources = sys.argv[1:] or [Path(__file__).parents[2] / "audio_lab" / "clubbed"]

result = loom.compose(
    sources=sources,
    length=90.0,
    seed=21,
    chunker="onset",
    chunker_params={"min_seconds": 0.3, "max_seconds": 6.0},
    passes=[
        # loud chunks -> lowpassed + reversed, tagged 'dark'
        dict(select="rms>0.5", processors=[["lowpass", {"cut_hz": 800}], "reverse"], tag="dark"),
        # half the dark chunks -> crushed + echoed, tagged 'crushed'
        dict(select="tag=dark and rms>0.6", processors=["crush(bits=5)", "echo"], tag="crushed"),
        # in-place pass: everything left gets a gentle lowpass
        dict(select="all", processors=[["lowpass", {"cut_hz": 12000}]], replace=True),
    ],
    # weave the three families together in corpus-grammar order
    select="tag=dark or tag=crushed",
    arranger="sequence",
    arranger_params={"order": "markov", "gap": -0.15, "speed": [1.0, 0.8, 0.95]},
    processors=["fades", "normalize"],
    store="disk",  # derived chunks spill to disk
    out="loom_out/reprocessed.wav",
)
print(result.summary())

for c in result.catalog.query("tag=crushed")[:3]:
    print("  " + result.catalog.chunks[c].describe())  # lineage + features