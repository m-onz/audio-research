"""form: the tempo map — sections, slow bits, and incredible fast BPM.

NEAT Drummer samples its CPPN on a uniform tick grid because the MIDI
song already had one tempo. A maestro composition does not have to:
the FORM is an explicit compositional decision (yours, not the
genome's) — a sequence of sections, each with its own BPM and bar
count, that maps tick INDEX to tick TIME. The organism is still pure:
it sees the spine's state at each tick; the form decides how fast
those ticks flow. The same genome pattern becomes a grind at 92 bpm
and a drum roll at 320 — the tempo changes are structural, not
encoded.

    sections: [(bpm 92, 8 bars), (bpm 196, 6), (bpm 78, 4), (bpm 320, 8), ...]
         -> tick_times: strictly increasing seconds, ticks_per_beat per beat

The post layer reads the form too: stutters belong in the fast
sections, stretches in the slow bits — damage placed where it lives
best. Named forms are starting points; --form-json is the open dial.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Section", "Form", "FORMS", "build_form", "describe_form"]

BEATS_PER_BAR = 4


@dataclass
class Section:
    bpm: float
    bars: int
    label: str = ""

    @property
    def seconds(self) -> float:
        return self.bars * BEATS_PER_BAR * 60.0 / self.bpm

    def to_dict(self) -> dict:
        return {"bpm": self.bpm, "bars": self.bars, "label": self.label,
                "seconds": round(self.seconds, 2)}


class Form:
    """A tempo map: sections in order, one tick rate each."""

    def __init__(self, sections: list, ticks_per_beat: int = 4):
        secs: list[Section] = []
        for s in sections:
            if isinstance(s, Section):
                secs.append(s)
            else:
                label = str(s[2]) if len(s) > 2 else ""
                secs.append(Section(float(s[0]), int(s[1]), label))
        self.sections = secs
        self.ticks_per_beat = int(ticks_per_beat)

    # ------------------------------------------------------------ the grid

    def tick_times(self, length_s: float | None = None) -> np.ndarray:
        """Tick index -> time in seconds. Sections run in order; with
        `length_s`, the last section's tempo extends until the piece
        reaches that length (a piece never ends mid-air)."""
        times: list[float] = [0.0]
        for sec in self.sections:
            n = sec.bars * BEATS_PER_BAR * self.ticks_per_beat
            period = 60.0 / (sec.bpm * self.ticks_per_beat)
            for _ in range(n):
                times.append(times[-1] + period)
        if length_s is not None:
            period = 60.0 / (self.sections[-1].bpm * self.ticks_per_beat)
            while times[-1] < length_s:
                times.append(times[-1] + period)
        return np.asarray(times, np.float64)

    def section_bounds(self, length_s: float | None = None
                       ) -> list[dict]:
        """Where each section starts and ends in seconds (post's map of
        the piece: fast sections stutter, slow sections stretch)."""
        tt = self.tick_times(length_s)
        bounds, t0, i = [], 0.0, 0
        for sec in self.sections:
            n = sec.bars * BEATS_PER_BAR * self.ticks_per_beat
            t1 = tt[min(i + n, len(tt) - 1)]
            bounds.append({"t0": round(t0, 3), "t1": round(t1, 3),
                           "bpm": sec.bpm, "label": sec.label or f"{sec.bpm:g}"})
            t0, i = t1, i + n
        if length_s is not None and bounds:
            bounds[-1]["t1"] = round(max(tt[-1], length_s), 3)
        return bounds

    def to_dict(self) -> dict:
        return {"ticks_per_beat": self.ticks_per_beat,
                "sections": [s.to_dict() for s in self.sections],
                "total_seconds": round(sum(s.seconds for s in self.sections), 1)}


# ------------------------------------------------------------------ the zoo

FORMS: dict[str, list] = {
    # slow grind -> build -> fast charge -> break -> VERY fast -> insane -> outro
    "tectonic": [
        (92, 16, "grind"), (138, 16, "build"), (196, 12, "charge"),
        (78, 8, "break"), (260, 12, "rush"), (138, 8, "reset"),
        (320, 16, "insane"), (104, 12, "outro"),
    ],
    # a monotonic ladder from 70 to 330: the accelerator
    "accel": [
        (70, 8, "deep"), (88, 12, "rise"), (104, 12, "walk"),
        (122, 12, "run"), (142, 12, "drive"), (168, 12, "charge"),
        (196, 12, "sprint"), (232, 12, "redline"), (280, 12, "escape"),
        (330, 16, "liftoff"),
    ],
    # hard alternation: fast slams against slow bits, tempo as dynamics
    "flip": [
        (150, 16, "pulse"), (75, 8, "sink"), (300, 8, "slam"),
        (85, 8, "swim"), (280, 8, "slam"), (70, 8, "bottom"),
        (240, 16, "roll"), (95, 12, "settle"),
    ],
    # the dance floor: long jackin' stretches, one break, one rush —
    # the club pipeline drives this one (four-on-the-floor aware)
    "club": [
        (122, 20, "floor"), (126, 16, "build"), (132, 12, "lift"),
        (74, 8, "break"), (128, 20, "drop"), (150, 12, "rush"),
        (122, 8, "outro"),
    ],
    # one tempo, the control (and the spine's own reading)
    "flat": [(120, 128, "flat")],
}


def build_form(spec: str | list | dict) -> Form:
    """A named form, a json path, a json list/dict of sections, or
    'name@tpb' for a custom tick density."""
    tpb = 4
    if isinstance(spec, str) and "@" in spec:
        spec, tpb_s = spec.rsplit("@", 1)
        tpb = int(tpb_s)
    if isinstance(spec, str):
        p = Path(spec)
        if spec in FORMS:
            return Form(FORMS[spec], ticks_per_beat=tpb)
        if p.exists():
            spec = json.loads(p.read_text())
        else:
            raise ValueError(f"unknown form {spec!r}; valid: {sorted(FORMS)}"
                             " or a json path/list")
    if isinstance(spec, dict):
        tpb = int(spec.get("ticks_per_beat", tpb))
        spec = spec.get("sections", [])
    return Form(spec, ticks_per_beat=tpb)


def describe_form(form: Form) -> str:
    L = ["  ".join(f"[{s.label}] {s.bpm:g}bpm x{s.bars}b"
                   for s in form.sections)]
    L.append(f"  total ~{sum(s.seconds for s in form.sections):.0f}s"
             f" at {form.ticks_per_beat} ticks/beat")
    return "\n".join(L)