"""palette: voices -> corpus sounds, chosen from YOUR material.

A voice name is a role; the palette resolves it to a real file from
the corpus (families read from the corpus manifest; keyword pools
carry the audio_lab's proven wisdom — kicks for the floor, hats for
the air, harmonic textures for flow, evolved specimens for spice).
Deterministic from (seed, voice): the same piece re-renders exactly;
a different seed is a different outfit for the same organism.

Override anything with explicit voice->file pairs; the auto pick is
the floor, not the ceiling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["auto_voices", "POOLS", "corpus_files"]

import re


def _name_has(name: str, key: str) -> bool:
    """Word-ish boundary match: 'rim' matches RIM and RIMSHOT_2 but
    NOT 'experiments' (the substring trap that cast a cloud texture
    as a rimshot). Underscores/digits are separators, like a drummer
    would read the name."""
    return re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])",
                     name.lower()) is not None


# percussive roles are chosen by FILE keywords — (include, exclude):
# the bank families mix kicks, hats and stabs in one folder, and the
# names carry the roles. "open" claims the open hats BEFORE "hat"
# gets them (order matters: openhat is checked first in _role_of).
ROLE_KEYS: dict[str, tuple[tuple, tuple]] = {
    "kick":    (("kick", "808"), ()),          # the floor: real kicks
    "low":     (("kick", "bass", "sub", "808", "mbass"), ()),
    "rim":     (("rim", "rimshot"), ()),
    "openhat": (("open",), ()),
    "hat":     (("hat", "hh", "ride", "crash", "shaker", "cymbal"),
                ("open",)),
    "high":    (("hat", "hh", "ride", "crash", "shaker", "cymbal", "air"),
                ("open",)),
    "perc":    (("perc", "tom", "tabla", "guiro", "wood", "shaker",
                 "clap", "conga"), ()),
    "mid":     (("stab", "snare", "tom", "perc", "mhit", "clap", "wood",
                 "chop", "tabla"), ()),
    "impact":  (("crash", "impact", "riser", "fx"), ()),
}

# the kit image: default gain and pan per role — hats off to the
# left, rims to the right, the floor in the middle
ROLE_MIX: dict[str, tuple[float, float]] = {
    "kick": (1.00, 0.00), "low": (1.00, 0.00), "pulse": (1.00, 0.00),
    "rim": (0.85, 0.45), "mid": (0.85, 0.10),
    "openhat": (0.72, 0.30), "hat": (0.62, -0.35), "high": (0.70, -0.25),
    "perc": (0.78, -0.15), "impact": (0.85, 0.00),
    "flow": (0.90, 0.00), "answer": (0.80, 0.20),
    "swell": (0.80, 0.00), "texture": (0.90, 0.00),
}

# role pools, by corpus manifest family (the keyword miss fallback)
POOLS: dict[str, list[str]] = {
    "kick":   ["audio_lab/sounds/kick", "audio_lab/sounds/mbass"],
    "low":    ["audio_lab/sounds/kick", "audio_lab/sounds/mbass"],
    "rim":    ["audio_lab/sounds/rimshot"],
    "openhat": ["audio_lab/sounds/openhh", "audio_lab/sounds/openhihats"],
    "hat":    ["audio_lab/sounds/hihat", "audio_lab/sounds/rides"],
    "high":   ["audio_lab/sounds/hihat", "audio_lab/sounds/rides",
               "audio_lab/sounds/clap", "audio_lab/rhythms/bank"],
    "perc":   ["audio_lab/sounds/perc", "audio_lab/sounds/shaker",
               "audio_lab/sounds/woodblock", "audio_lab/sounds/toms"],
    "impact": ["audio_lab/sounds/crash", "audio_lab/sounds/mfx",
               "audio_lab/storm_pieces"],
    "mid":    ["audio_lab/sounds/mhit", "audio_lab/sounds/perc",
               "audio_lab/sounds/snare", "audio_lab/rhythms/bank"],
    "pulse":  ["audio_lab/sounds/kick", "audio_lab/sounds/mbass"],
    "flow":   ["audio_lab/harmonic_corpus", "audio_lab/fusion_harmonic_corpus",
               "audio_lab/fusion_pieces", "bomb/corpus"],
    "answer": ["audio_lab/alien_rhythms", "audio_lab/sounds/perc",
               "audio_lab/sounds/rimshot", "audio_lab/dnb_chops"],
    "swell":  ["audio_lab/sounds/crash", "audio_lab/sounds/openhh",
               "audio_lab/storm_pieces", "audio_lab/sounds/mfx"],
    "texture": ["audio_lab/harmonic_corpus", "audio_lab/curated_corpus",
                 "cryptid/runs/gen_0030", "womb/womb_out", "final_final_final"],
}
_ALL = sorted({f for pool in POOLS.values() for f in pool})

_BAD = ("site-packages",)          # never reach into venvs


def corpus_files(corpus_dir) -> dict[str, list[str]]:
    """family -> [file path] from the corpus manifest; a missing
    manifest falls back to globbing (family '<all>')."""
    corpus_dir = Path(corpus_dir)
    families: dict[str, list[str]] = {}
    manifest = corpus_dir / "manifest.json"
    if manifest.exists():
        for entry in json.loads(manifest.read_text()):
            fam = entry.get("family", "")
            if any(b in fam for b in _BAD):
                continue
            p = corpus_dir / entry["file"]
            if p.exists():
                families.setdefault(fam, []).append(str(p))
    if not families:
        families["<all>"] = [str(p) for p in sorted(corpus_dir.glob("*.wav"))]
    return families


def _role_of(voice: str) -> str:
    """Voice name -> role. Order matters: openhat before hat (the
    name contains 'hat'), rim and kick before the generic pools."""
    v = voice.lower()
    if v in ROLE_KEYS or v in POOLS:
        return v
    if any(w in v for w in ("openhat", "open_hat", "openhh")):
        return "openhat"
    if "impact" in v or "crash" in v:
        return "impact"
    if "rim" in v:
        return "rim"
    if any(w in v for w in ("kick", "bass", "sub")):
        return "kick"
    if any(w in v for w in ("low",)):
        return "low"
    if any(w in v for w in ("hat", "air", "top")):
        return "hat"
    if any(w in v for w in ("high",)):
        return "high"
    if any(w in v for w in ("mid", "stab", "snare")):
        return "mid"
    if "perc" in v:
        return "perc"
    if any(w in v for w in ("flow", "pad", "drone", "tex")):
        return "flow"
    if any(w in v for w in ("answer", "echo", "ghost")):
        return "answer"
    if any(w in v for w in ("swell", "crash", "riser")):
        return "swell"
    return "texture"


def _candidates_for(role: str, by_family: dict[str, list[str]]) -> list[str]:
    """Percussive roles: keyword-filter the WHOLE corpus by file name
    (include AND exclude). Texture roles: family pools. Fallback:
    the role's family pool, then everything."""
    if role in ROLE_KEYS:
        inc, exc = ROLE_KEYS[role]
        names = {f for fs in by_family.values() for f in fs}
        hits = sorted({f for f in names
                       if any(_name_has(Path(f).name, k) for k in inc)
                       and not any(_name_has(Path(f).name, k) for k in exc)})
        if hits:
            return hits
    pools = POOLS.get(role, _ALL)
    fam_hits = sorted({f for p in pools for f in by_family.get(p, [])})
    if fam_hits:
        return fam_hits
    return sorted({f for fs in by_family.values() for f in fs})


def _stable(seed: int, voice: str) -> int:
    h = hashlib.sha256(f"{seed}|{voice}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def auto_voices(voice_names, corpus_dir="corpus", seed: int = 0,
                overrides: dict | None = None) -> tuple[dict, dict]:
    """Resolve each score voice to a corpus file. Returns
    (voice_map for the mapper, provenance {voice: family/file}).
    overrides: {voice: path} win over the auto pick."""
    fams = corpus_files(corpus_dir)
    by_family: dict[str, list[str]] = {}
    for fam, files in fams.items():
        by_family.setdefault(fam, []).extend(files)
    overrides = overrides or {}
    voice_map, provenance = {}, {}
    for v in voice_names:
        if v in overrides:
            spec = {"file": overrides[v], "gain": 1.0, "pan": 0.0,
                    "speed": 1.0, "truncate": False}   # full shape —
            voice_map[v] = spec                    # club's manifest and
            provenance[v] = {"file": str(overrides[v]), "family": "<override>"}
            continue
        pick_pool = _candidates_for(_role_of(v), by_family)
        candidates = pick_pool
        if not candidates:
            raise SystemExit(f"no corpus files found under {corpus_dir}")
        pick = candidates[_stable(seed, v) % len(candidates)]
        fam = next((f for f, fs in by_family.items() if pick in fs), "<all>")
        gain, pan = ROLE_MIX.get(_role_of(v), (0.9, 0.0))
        voice_map[v] = {"file": pick, "gain": gain, "pan": pan}
        provenance[v] = {"file": pick, "family": fam}
    return voice_map, provenance