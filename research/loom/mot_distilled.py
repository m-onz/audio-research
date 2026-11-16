"""MOT, Distilled — a 5-minute condensation of the 55-minute live set.

Concept: the set remixes itself. Every voice is the live set playing a role:

  statement   the loudest moment of the set, at the open and (reversed) close
  sub pulse   its bass-heavy tonal chunks, truncated to kicks, pitch-ordered
              so the bassline slowly rises through the piece
  snap        its percussive transients on an offset Euclid grid
  shrapnel    flashes of the set itself flying over the groove
  ghost       its darkest passages, reversed + echoed — the breakdown wash
  sky         its air, highpassed — shimmer on top
  riser       its noisy sustains, driven + echoed — transition swells

Form (300 s, curves are 13 evenly-spaced points over [0, 1]):
  0:00-0:50  ghost/sky wash opens; the set's loudest word states itself
  0:50-2:30  pulse enters and builds to full energy
  2:30-3:20  breakdown: pulse drops out, ghosts swell, risers cry
  3:20-4:50  second build and climax
  4:50-5:00  dissolve; the set un-says itself (reversed statement)

Run:  python3 mot_distilled.py [seed]
"""

import sys

import loom
from loom.processors import build_chain

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1369
LENGTH = 300.0

# 13-point arcs over the piece (each point = 25 s)
SUB_GAIN = [0, 0, 0.25, 1.0, 1.1, 1.1, 0.1, 0, 0.95, 1.1, 1.1, 0.55, 0]
SNAP_GAIN = [0, 0, 0.15, 0.75, 0.95, 0.95, 0.05, 0, 0.65, 0.95, 0.95, 0.45, 0]
SHRAPNEL_DENSITY = [0, 5, 30, 55, 65, 65, 8, 0, 45, 70, 75, 30, 0]
GHOST_DENSITY = [35, 45, 30, 18, 14, 16, 100, 85, 30, 20, 18, 30, 20]
SKY_DENSITY = [6, 8, 12, 14, 18, 20, 14, 16, 16, 22, 18, 8, 4]
RISER_DENSITY = [0, 0, 0, 0, 4, 14, 0, 10, 0, 16, 0, 6, 0]
DRONE_DENSITY = [0, 0, 20, 30, 35, 35, 25, 30, 25, 35, 35, 20, 10]

result = loom.compose(
    sources="../corpus2/MOT-oct26-everything.wav",
    length=LENGTH,
    seed=SEED,
    sr=48000,
    bits=24,
    store="disk",
    chunker="onset",
    chunker_params={"min_seconds": 0.25, "max_seconds": 8.0},
    # ---- re-processing passes: build the voice families -----------------
    passes=[
        dict(select="bass>0.55 and rms>0.45 and pitch_conf>0.35",
             processors=[["lowpass", {"cut_hz": 340}], "normalize"],
             tag="sub"),
        dict(select="crest>0.72 and air>0.35",
             processors=[["bandpass", {"lo_hz": 900, "hi_hz": 9000}], "normalize"],
             tag="snap"),
        dict(select="centroid<0.35 and dur>0.75",
             processors=["reverse", ["echo", {"delay_s": 0.45, "feedback": 0.5, "mix": 0.5}]],
             tag="ghost"),
        dict(select="air>0.65",
             processors=[["highpass", {"cut_hz": 400}], "normalize"],
             tag="sky"),
        dict(select="flatness>0.5 and dur>0.6",
             processors=[["highpass", {"cut_hz": 250}], ["drive", {"amount": 2.0}],
                         ["echo", {"delay_s": 0.33, "feedback": 0.6, "mix": 0.5}]],
             tag="riser"),
        dict(select="dur>0.8 and rms>0.5",
             processors=[["lowpass", {"cut_hz": 700}], ["drive", {"amount": 1.5}]],
             tag="drone"),
    ],
    # ---- arrangement: six voices over the form ---------------------------
    arranger="layers",
    arranger_params={"voices": [
        dict(select="tag=sub", arranger="euclid", gain=1.4,
             params=dict(bpm=136, steps=16, pulses=4, rotate=0, gate=0.9,
                         accent=1.0, order="by:pitch", gain=SUB_GAIN)),
        dict(select="tag=snap", arranger="euclid", gain=1.0,
             params=dict(bpm=136, steps=16, pulses=7, rotate=2, gate=0.45,
                         accent=0.6, order="random", gain=SNAP_GAIN)),
        dict(select="rms>0.6 and dur>0.3 and dur<0.8", arranger="cloud", gain=0.6,
             params=dict(density=SHRAPNEL_DENSITY, pan_spread=0.55,
                         fade=0.02, order="random")),
        dict(select="tag=ghost", arranger="cloud", gain=0.8,
             params=dict(density=GHOST_DENSITY, pan_spread=0.85, fade=0.06,
                         speed=[1.0, 0.88, 1.0], order="random")),
        dict(select="tag=sky", arranger="cloud", gain=0.55,
             params=dict(density=SKY_DENSITY, pan_spread=1.0, fade=0.05,
                         order="random")),
        dict(select="tag=riser", arranger="cloud", gain=0.8,
             params=dict(density=RISER_DENSITY, pan_spread=0.7, fade=0.04,
                         order="random")),
        dict(select="tag=drone", arranger="cloud", gain=0.35,
             params=dict(density=DRONE_DENSITY, pan_spread=0.3, fade=0.1,
                         speed=0.5, max_dur=6.0, order="random")),
    ]},
    processors=["fades"],
    dry_run=True,   # arrange first; statement events are added by hand below
)

score = result.score
catalog = result.catalog

# ---- bookend statements: the set's own loudest word -----------------------
ranks = catalog.ranks()
loudest = int(ranks[:, 0].argmax())
loud_chunk = catalog.chunks[loudest]
statement_dur = min(loud_chunk.dur, 3.0)
from loom.score import Event

score.events.insert(0, Event(
    chunk=loudest, t=0.0, dur=statement_dur, speed=1.0, gain=0.95,
    pan=0.0, fade_in=0.04, fade_out=0.9, truncate=True,
    chain=build_chain(["fades"]), note="statement-open",
))
reversed_stmt = Event(
    chunk=loudest, t=LENGTH - statement_dur - 1.0, dur=statement_dur,
    speed=1.0, gain=0.8, pan=0.0, fade_in=0.8, fade_out=1.5, truncate=True,
    chain=build_chain(["reverse", "fades"]), note="statement-close",
)
score.events.append(reversed_stmt)
score.events.sort(key=lambda e: e.t)

out = f"loom_out/MOT_distilled_seed{SEED}.wav"
stats = loom.render(score, catalog, out, bits=24, normalize=True, headroom_db=0.3,
                    soft_clip=True)
print(f"{out}  peak={stats['peak']:.3f}  events={len(score.events)}")

# ---- provenance: score + manifest next to the master ----------------------
import hashlib
import json
from pathlib import Path

out_path = Path(stats["path"])
score_path = out_path.with_suffix(".score.json")
score.write(score_path)
manifest = {
    "title": "everysongiown, distilled",
    "source": "corpus2/MOT-oct26-everything.wav (55 min live set @ MOT, oct 26)",
    "concept": ("the live set remixes itself: its bass becomes the pulse, its "
                "transients the snaps, its dark passages the breakdown ghosts, "
                "its air the sky; loudest word of the set opens and (reversed) "
                "closes the piece"),
    "form": ["0:00 ghost/sky wash + statement", "0:50 pulse enters",
             "1:30 full energy", "2:30 breakdown (pulse out, ghosts swell)",
             "3:20 rebuild", "4:10 second climax", "4:50 dissolve + reversed statement"],
    "seed": SEED, "length": LENGTH, "sr": 48000, "bits": 24,
    "bpm": 136, "events": len(score.events),
    "voices": {str(n): c for n, c in sorted(
        ((e.note, 1) for e in score.events), key=lambda kv: kv[0])},
    "render": {k: v for k, v in stats.items() if k != "path"},
    "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    "score": score_path.name,
}
manifest_path = out_path.with_suffix(".manifest.json")
manifest_path.write_text(json.dumps(manifest, indent=1))
print(f"manifest: {manifest_path}")
print(f"score:    {score_path}")
print("by voice:")
from collections import Counter
for note, n in sorted(Counter(e.note for e in score.events).items(), key=lambda kv: -kv[1]):
    print(f"  {str(note):<16} {n}")