"""reel: the artifact — one directory, full lineage.

A Reel is womb's unit of work: a folder holding the seed audio, every
pass's stem and video, and a manifest that records how each generation was
made and what it drifted. The *artifact* is always `final.mp4` (the last
video), but the whole lineage stays inspectable and reproducible.

```
my_reel/
  manifest.json          # lineage + telemetry (the drift ledger)
  pass_0/stem.wav        # the seed audio
  pass_0/video.mp4       # doom render of the seed
  pass_1/stem.wav        # sonified + degraded from pass_0/video.mp4
  pass_1/video.mp4
  final.mp4              # copy of the last video = the artifact
  final.wav               # copy of the last stem
```
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

__all__ = ["Reel"]


class Reel:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            self.manifest = {"passes": [], "name": self.root.name}

    # ---------------------------------------------------------------- state
    def save(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=1))

    @property
    def n_passes(self) -> int:
        return len(self.manifest["passes"])

    def pass_dir(self, i: int) -> Path:
        d = self.root / f"pass_{i}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------------------------------------------------------- building
    @staticmethod
    def _place(src, dst: Path) -> None:
        """Copy src to dst unless it is already there (idempotent)."""
        if Path(src).resolve() != dst.resolve():
            shutil.copy(str(src), dst)

    def add_seed(self, audio_path) -> Path:
        """pass_0 stem: the seed audio, copied in."""
        dst = self.pass_dir(0) / "stem.wav"
        self._place(audio_path, dst)
        return dst

    def add_pass(self, kind: str, stem: Path | None = None,
                 video: Path | None = None, record: dict | None = None) -> int:
        """Append a generation pass; returns its index."""
        i = self.n_passes
        d = self.pass_dir(i)
        if stem is not None:
            self._place(stem, d / "stem.wav")
        if video is not None:
            self._place(video, d / "video.mp4")
        entry = {"index": i, "kind": kind}
        if record:
            entry.update(record)
        self.manifest["passes"].append(entry)
        self.save()
        return i

    def stem(self, i: int) -> Path:
        return self.pass_dir(i) / "stem.wav"

    def video(self, i: int) -> Path:
        return self.pass_dir(i) / "video.mp4"

    def finalize(self) -> Path:
        """The artifact: the last pass's video (+ its stem) copied to top level."""
        last = self.n_passes - 1
        if last < 0:
            raise RuntimeError("empty reel — nothing to finalize")
        shutil.copy(str(self.video(last)), self.root / "final.mp4")
        if self.stem(last).exists():
            shutil.copy(str(self.stem(last)), self.root / "final.wav")
        self.manifest["final"] = "final.mp4"
        self.save()
        return self.root / "final.mp4"

    # ---------------------------------------------------------------- drift
    def drift_report(self) -> str:
        """The drift ledger as text: per-pass telemetry and deltas."""
        rows = []
        header = (f"{'pass':>4} {'kind':<10} {'rms':>6} {'centHz':>7} "
                   f"{'luma':>6} {'motion':>7} {'aDrift':>7} {'vDrift':>7}")
        rows.append(header)
        for p in self.manifest["passes"]:
            rows.append(
                f"{p['index']:>4} {p['kind']:<10} "
                f"{p.get('rms', 0):>6.3f} {p.get('centroid', 0):>7.0f} "
                f"{p.get('luma', 0):>6.3f} {p.get('motion', 0):>7.4f} "
                f"{p.get('audio_drift', 0):>7.3f} {p.get('video_drift', 0):>7.3f}"
            )
        rows.append("")
        return "\n".join(rows)

    # ---------------------------------------------------------------- summary
    def summary(self) -> str:
        m = self.manifest
        lines = [f"reel: {m['name']}  ({self.n_passes} passes)"]
        for p in m["passes"]:
            lines.append(f"  pass {p['index']}: {p['kind']}  "
                         + (f"sonify={p.get('sonify_mode')} " if p.get("sonify_mode") else "")
                         + (f"degrade={p.get('audio_degrades')}" if p.get("audio_degrades") else ""))
        if (self.root / "final.mp4").exists():
            lines.append(f"  artifact: {self.root / 'final.mp4'}")
        return "\n".join(lines)