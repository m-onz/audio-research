"""Score surgery: compose, inspect, edit, and re-render.

A Score is plain data — JSON-serializable. Arrange once, then edit the
event list by hand (or by script) and render the edited score. This
separation is what makes loom a good reference for offline rendering:
composition and rendering never tangle.

Run:  python3 examples/06_score_surgery.py path/to/corpus
"""

import json
import sys
from pathlib import Path

import loom

sources = sys.argv[1:] or [Path(__file__).parents[2] / "audio_lab" / "clubbed"]

# 1. arrange only (dry run)
result = loom.compose(
    sources=sources,
    length=45.0,
    seed=8,
    chunker="onset",
    chunker_params={"min_seconds": 0.2, "max_seconds": 4.0},
    arranger="cloud",
    arranger_params={"density": 100},
    dry_run=True,
)
score = result.score
print("original:", json.dumps(score.summary(), indent=1))

# 2. edit: mute the middle third, boost the finale
for e in score.events:
    if score.length / 3 <= e.t < 2 * score.length / 3:
        e.gain = 0.0
    elif e.t >= 2 * score.length / 3:
        e.gain *= 1.5

# 3. persist the score as JSON (inspectable, diffable, archivable)
score.write("loom_out/score.json")

# 4. render the edited score
stats = loom.render(score, result.catalog, "loom_out/surgery.wav", bits=24)
print("rendered:", json.dumps(stats, indent=1))

# 5. reload a score from disk and render it again — full reproducibility
score2 = loom.Score.read("loom_out/score.json")
stats2 = loom.render(score2, result.catalog, "loom_out/surgery_again.wav", bits=24)
print("re-rendered:", stats2["path"])
assert stats["peak"] == stats2["peak"], "scores must render identically"