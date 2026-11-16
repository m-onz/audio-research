"""everysongiown, bit rot — the live set taken apart into bits.

A bitty, glitchy, harmonic companion to "everysongiown, distilled".
Where the last piece was dense and driving, this one is sparse and
holed — silence is a structural voice at three scales:

  macro    two near-silence windows (~1:50 and ~3:30) where only a
           single stretched sound or ghost ping survives
  meso     Euclid pulses with 68% empty steps; density curves with
           built-in holes every phrase
  micro    the `chop` processor gates patterns INSIDE hits

The set's own material, again in every role:

  bit-pulse   its percussive hits on a 5-in-16 Euclid grid — every hit
              runs through `morph`, so each one gets a different
              treatment (chop / crush / ring / bandpass / repitch)
  sub-drone   the set's ~50 Hz bass family as a near-static foundation
  harm-mid    its tonal chunks walked by CONSONANT intervals (fifths,
              octaves, fourths) through the 50-750 Hz register
  harm-high   the same walk in the shimmer register — layered octaves  morph-text  its mid-energy chunks, one pooled transformation per
              event: the "different effect every couple of seconds" voice
  stutter    its loudest transients as stutter-fills: one slice
              re-triggered, each repeat pitched down or up progressively
  stretch    its dark sustained chunks time-stretched (pitch preserved)
              — the sound of the set being dragged through itself
  ghost      its darkest passages, reversed + softly echoed — memory pings

Run:  python3 mot_bitrot.py [seed]
"""

import sys

import loom

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LENGTH = 240.0  # 4:00

# 13-point arcs (each point = 20 s). Zeroes are deliberate: silence.
PULSE_GAIN = [0, 0, 0.4, 0.8, 1.0, 0.3, 0, 0.6, 1.0, 1.0, 0.1, 0, 0]
HARMLOW_DENSITY = [4, 14, 22, 26, 30, 10, 4, 20, 30, 30, 12, 2, 0]
HARMMID_DENSITY = [0, 8, 14, 18, 22, 8, 2, 14, 22, 22, 10, 0, 0]
HARMHIGH_DENSITY = [0, 0, 0, 10, 14, 6, 0, 10, 18, 18, 8, 0, 0]
MORPH_DENSITY = [2, 6, 10, 14, 20, 8, 2, 14, 30, 30, 15, 0, 0]
STUTTER_DENSITY = [0, 0, 0, 0, 0, 12, 0, 4, 8, 4, 10, 0, 0]
STRETCH_DENSITY = [4, 2, 0, 0, 0, 0, 5, 0, 2, 0, 0, 0, 4]
GHOST_DENSITY = [2, 3, 3, 4, 4, 3, 3, 4, 4, 4, 6, 2, 0]

# ---- pools for the morph voices (the diversity engine) ----------------------
# pulse hits: short-form treatments (each hit is ~60 ms)
PULSE_MORPH = [
    ["chop", {"steps": 4, "pulses": 2}],
    ["chop", {"steps": 8, "pulses": 3, "rotate": 2}],
    ["crush", {"bits": 4}],
    ["ring", {"freq_hz": 330.0, "mix": 0.4}],
    ["bandpass", {"lo_hz": 200, "hi_hz": 900}],
    ["speed", {"x": 0.6}],
    ["speed", {"x": 1.5}],
    ["gate", {"threshold_db": -30.0}],
    "normalize",
]
# texture events (0.4-3 s): long-form treatments, stretch + varispeed included
TEXTURE_MORPH = [
    ["chop", {"steps": 4, "pulses": 2}],
    ["chop", {"steps": 8, "pulses": 3, "rotate": 2}],
    ["crush", {"bits": 4}],
    ["crush", {"bits": 6}],
    ["ring", {"freq_hz": 440.0, "mix": 0.4}],
    ["ring", {"freq_hz": 587.0, "mix": 0.35}],
    ["drive", {"amount": 2.5}],
    ["bandpass", {"lo_hz": 400, "hi_hz": 1600}],
    ["stretch", {"factor": 1.6}],
    ["stretch", {"factor": 0.6}],
    "reverse",
    ["speed", {"x": 0.53}],
]

def build(seed=5):
    """Arrange the piece (dry run) -> ComposeResult with score + catalog."""
    result = loom.compose(
        sources="../corpus2/MOT-oct26-everything.wav",
        length=LENGTH,
        seed=seed,
        sr=48000,
        bits=24,
        store="disk",
        chunker="onset",
        chunker_params={"min_seconds": 0.25, "max_seconds": 8.0},
        # one pass: the ghost family (memory pings)
        passes=[
            dict(select="centroid<0.3 and dur>0.7",
                 processors=["reverse", ["echo", {"delay_s": 0.5, "feedback": 0.45, "mix": 0.35}]],
                 tag="ghost"),
        ],
        arranger="layers",
        arranger_params={"voices": [
            dict(select="crest>0.6 and rms>0.45 and flatness<0.6",
                 arranger="euclid", gain=1.1,
                 chain=[["morph", {"processors": PULSE_MORPH}], "fades"],
                 params=dict(bpm=120, steps=16, pulses=5, rotate=2, gate=0.5,
                             accent=0.5, order="random", gain=PULSE_GAIN)),
            dict(select="pitch_conf>0.5 and pitch>0.3 and pitch<0.62 and flatness<0.55",
                 arranger="cloud", gain=0.6,
                 chain=[["highpass", {"cut_hz": 35}], "fades"],
                 params=dict(density=HARMLOW_DENSITY, pan_spread=0.3, fade=0.3,
                             max_dur=3.5, order="random")),
            dict(select="pitch_conf>0.5 and pitch>0.62 and pitch<0.85 and flatness<0.6",
                 arranger="cloud", gain=0.55,
                 chain=[["highpass", {"cut_hz": 150}], "fades"],
                 params=dict(density=HARMMID_DENSITY, pan_spread=0.5, fade=0.2,
                             order="harmonic")),
            dict(select="pitch_conf>0.5 and pitch>0.85 and flatness<0.6",
                 arranger="cloud", gain=0.42,
                 chain=[["highpass", {"cut_hz": 400}], "fades"],
                 params=dict(density=HARMHIGH_DENSITY, pan_spread=0.7, fade=0.25,
                         order="harmonic")),
            dict(select="rms>0.5 and crest>0.5 and flatness<0.6 and dur>0.4",
                 arranger="cloud", gain=0.55,
                 chain=[["morph", {"processors": TEXTURE_MORPH}], "fades"],
                 params=dict(density=MORPH_DENSITY, pan_spread=0.8, fade=0.03,
                             order="random")),
            dict(select="rms>0.7 and crest>0.6 and flatness<0.7",
                 arranger="cloud", gain=0.8,
                 chain=[["stutter", {"count": 5, "step_semi": 1.5, "direction": "random",
                                    "slice_s": 0.11, "decay": 0.8}], "fades"],
                 params=dict(density=STUTTER_DENSITY, fade=0.02, order="random")),
            dict(select="centroid<0.45 and dur>0.7 and flatness<0.6",
                 arranger="cloud", gain=0.6,
                 chain=[["stretch", {"factor": 3.0, "grain_s": 0.06}],
                        ["lowpass", {"cut_hz": 2500}], "fades"],
                 params=dict(density=STRETCH_DENSITY, fade=0.3, order="random")),
            dict(select="tag=ghost",
                 arranger="cloud", gain=0.45,
                 chain=["fades"],
                 params=dict(density=GHOST_DENSITY, fade=0.1, order="random")),
        ]},
        processors=["fades"],
        dry_run=True,
    )
    return result


if __name__ == "__main__":
    result = build(SEED)
    score = result.score
    catalog = result.catalog
    out = f"loom_out/MOT_bitrot_seed{SEED}.wav"
    stats = loom.render(score, catalog, out, bits=24, normalize=True, headroom_db=0.3)

    # provenance: score + manifest next to the master
    import hashlib
    import json
    from pathlib import Path

    out_path = Path(stats["path"])
    score_path = out_path.with_suffix(".score.json")
    score.write(score_path)
    manifest = {
        "title": "everysongiown, bit rot",
        "source": "corpus2/MOT-oct26-everything.wav (55 min live set @ MOT, oct 26)",
        "concept": ("the set taken apart into bits: holey Euclid pulses with a "
                     "different morph treatment per hit, consonant-walked harmonic "
                     "beds, stutter fills with progressive pitch drops, "
                     "time-stretch moments, and silence as structure at three "
                     "scales"),
        "form": ["0:00 stretched open, sparse", "0:40 pulse enters",
                 "1:15 full bitty section", "2:00 near-silence window",
                 "2:35 rebuild", "3:10 climax", "3:40 dissolve + final stretch"],
        "seed": SEED, "length": LENGTH, "sr": 48000, "bits": 24,
        "bpm": 120, "events": len(score.events),
        "render": {k: v for k, v in stats.items() if k != "path"},
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "score": score_path.name,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1))

    print(f"{out}  peak={stats['peak']:.3f}")
    from collections import Counter
    for note, n in sorted(Counter(e.note for e in score.events).items(), key=lambda kv: -kv[1]):
        print(f"  {str(note):<16} {n}")

    # silence report: quietest 3-second windows
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(out, always_2d=True)
    m = np.abs(x.mean(axis=1))
    w3 = 3 * sr
    peaks3 = [m[i:i + w3].max() for i in range(0, len(m) - w3, w3 // 2)]
    quiet = [round(1.5 + i * 1.5, 1) for i, p in enumerate(peaks3) if p < 0.02]
    print(f"near-silent 3s windows at: {quiet}")