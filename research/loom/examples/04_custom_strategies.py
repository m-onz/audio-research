"""Custom strategies: the library is a toolkit, not a fixed pipeline.

Chunkers, processors, and arrangers are all open registries. Anything with
the right signature can be registered by name and used from compose() /
chain specs. This example adds:

- a 'loudest_peaks' chunker that hunts the highest-RMS windows
- a 'shimmer' processor (slow AM / tremolo)
- a 'pingpong' arranger that bounces chunks left-right

Run:  python3 examples/04_custom_strategies.py path/to/corpus
"""

import sys
from pathlib import Path

import numpy as np

import loom
from loom.chunks import ChunkRef
from loom.processors import Processor, register_processor
from loom.arrangers import register_arranger
from loom.score import Event


# ---- a custom chunker: N highest-energy windows per source ------------------

def loudest_peaks(sources, rng, n=8, seconds=2.0, probe=0.5):
    """Scan coarse probes, keep the windows around the loudest energy peaks."""
    refs = []
    for si in range(len(sources)):
        src = sources[si]
        dur = src.duration
        starts = np.arange(0.0, max(0.0, dur - seconds), probe)
        energies = [float(np.abs(src.read(t, min(seconds, dur - t))).mean()) for t in starts]
        best = np.argsort(energies)[::-1][: int(n)]
        for i in sorted(best):
            t = float(starts[i])
            refs.append(ChunkRef(si, t, min(seconds, dur - t)))
    return refs


# ---- a custom processor: tremolo -------------------------------------------

@register_processor
class Shimmer(Processor):
    name = "shimmer"

    def process(self, audio, sr, t, rng, rate_hz=6.0, depth=0.5, **_):
        n = audio.shape[0]
        lfo = 0.5 * depth * (1.0 - np.cos(2 * np.pi * np.arange(n) * float(rate_hz) / sr))
        return audio * (1.0 - depth + lfo)[:, None]


# ---- a custom arranger: ping-pong bounce ------------------------------------

@register_arranger
def pingpong(catalog, ids, length, sr, rng, params, chain):
    """Alternate chunks hard-left / hard-right, sequence style."""
    p = dict(params or {})
    gap = loom.as_curve(p.get("gap"), 0.0)
    ordered = loom.order_ids(catalog, ids, kind=p.get("order", "random"), rng=rng,
                             n=max(1, 4 * len(ids)))
    events, t, k = [], 0.0, 0
    while t < length:
        u = t / length
        dur = catalog.chunks[ordered[k % len(ordered)]].dur
        events.append(Event(chunk=ordered[k % len(ordered)], t=t, dur=dur,
                            pan=(-1.0 if k % 2 == 0 else 1.0),
                            chain=chain, note="pingpong"))
        t += dur + gap.at(u)
        k += 1
    return events


# ---- use them all ------------------------------------------------------------

sources = sys.argv[1:] or [Path(__file__).parents[2] / "audio_lab" / "clubbed"]

result = loom.compose(
    sources=sources,
    length=60.0,
    seed=4,
    chunker=loudest_peaks,          # callable chunker, or register one:
    chunker_params={"n": 10, "seconds": 3.0},
    arranger="pingpong",            # registered above
    arranger_params={"gap": -0.2},
    processors=["shimmer", "fades"],
    out="loom_out/custom.wav",
)
print(result.summary())