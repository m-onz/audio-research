"""Score: the arrangement as data.

An arranger's whole job is to produce a `Score` — a flat list of `Event`s
(chunk id, time, duration, speed, gain, pan, fades, processor chain). A score
is JSON-serializable, inspectable, editable by hand, and renderable
deterministically. Composition and rendering never touch each other's
concerns; this separation is what makes the library a good reference:
arrangers reason about *placement*, the renderer reasons about *audio*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .processors import Chain, build_chain

__all__ = ["Event", "Score"]


@dataclass
class Event:
    """One chunk placement on the timeline."""

    chunk: int
    t: float            # start time in seconds
    dur: float          # sounded duration in seconds (after speed)
    speed: float = 1.0  # playback rate (varispeed)
    gain: float = 1.0
    pan: float = 0.0    # -1 hard left .. +1 hard right
    fade_in: float = 0.003
    fade_out: float = 0.003
    truncate: bool = False  # cut audio that outlasts `dur` (rhythm mode)
    chain: Chain | None = None
    note: str | None = None

    def to_json(self) -> dict:
        # NOTE: no rounding — JSON round-trips must be bit-exact for
        # reproducible re-rendering.
        d = {
            "chunk": self.chunk,
            "t": self.t,
            "dur": self.dur,
            "speed": self.speed,
            "gain": self.gain,
            "pan": self.pan,
            "fades": [self.fade_in, self.fade_out],
            "truncate": self.truncate,
        }
        if self.chain is not None:
            d["chain"] = self.chain.describe()
        if self.note:
            d["note"] = self.note
        return d

    @staticmethod
    def from_json(d: dict, chain: Chain | None = None) -> "Event":
        fades = d.get("fades", [0.003, 0.003])
        ev = Event(
            chunk=int(d["chunk"]),
            t=float(d["t"]),
            dur=float(d["dur"]),
            speed=float(d.get("speed", 1.0)),
            gain=float(d.get("gain", 1.0)),
            pan=float(d.get("pan", 0.0)),
            fade_in=float(fades[0]),
            fade_out=float(fades[1]),
            truncate=bool(d.get("truncate", False)),
            chain=chain or build_chain(d.get("chain")),
        )
        ev.note = d.get("note")
        return ev


@dataclass
class Score:
    """A finished arrangement ready for rendering."""

    sr: int
    length: float
    seed: int
    events: list[Event] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(round(self.length * self.sr))

    def to_json(self) -> dict:
        shared_chain = self.events[0].chain.describe() if self.events and self.events[0].chain else None
        return {
            "sr": self.sr,
            "length": self.length,
            "seed": self.seed,
            "n_events": len(self.events),
            "n_chunks_used": len({e.chunk for e in self.events}),
            "chain": shared_chain,
            "events": [e.to_json() for e in self.events],
        }

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))
        return path

    @classmethod
    def from_json(cls, d: dict) -> "Score":
        chain = build_chain(d.get("chain"))
        events = [Event.from_json(e, chain) for e in d.get("events", [])]
        return cls(sr=int(d["sr"]), length=float(d["length"]), seed=int(d.get("seed", 0)), events=events)

    @classmethod
    def read(cls, path) -> "Score":
        return cls.from_json(json.loads(Path(path).read_text()))

    def summary(self) -> dict:
        durs = [e.dur for e in self.events] or [0.0]
        return {
            "sr": self.sr,
            "length": round(self.length, 2),
            "events": len(self.events),
            "chunks_used": len({e.chunk for e in self.events}),
            "event_dur_median": round(sorted(durs)[len(durs) // 2], 3),
            "coverage": round(sum(durs) / max(1e-9, self.length), 3),
        }