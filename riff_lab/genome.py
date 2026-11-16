"""genome: the riff's genetic material — now expression-based.

A genome fully describes a riff: tempo / bars / scale, one ALIEN PATTERN
EXPRESSION per voice (the generative program, evaluated every bar so the
line breathes), the source policy per voice (chopped corpus events /
tonal one-shots / synthesis), the transform genetics (chain depth,
crossbreed probability, bass resynth), and the per-hit processing genes
(so repeats are never identical). All JSON-serializable; mutation is
expression surgery + gene drift; crossover grafts expressions and
mixes genes.
"""
import copy

import numpy as np

from . import patterns as P
from . import drums as D

SCALES = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pent_minor": [0, 3, 5, 7, 10],
    "hirajoshi": [0, 2, 3, 7, 8],
}
ROOTS = {"D1": 26, "E1": 28, "F1": 29, "G1": 31, "A1": 33, "Bb1": 34}
GENRES = ("dnb", "techstep", "halftime")

SOURCE_MODES = {
    "bass": (["oneshot", "oneshot", "chop", "synth"]),   # tonal-leaning
    "noise": (["chop", "chop", "chop", "oneshot"]),       # crisp events
    "fill": (["chop", "chop", "oneshot"]),
}


def mtof(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def scale_notes(root, scale, lo=22, hi=48):
    deg = SCALES[scale]
    out = []
    for octv in range(-1, 3):
        for d in deg:
            m = root + d + 12 * octv
            if lo <= m <= hi and m not in out:
                out.append(m)
    return sorted(out)


def json_copy(g):
    return copy.deepcopy(g)


def assemble_drums(g):
    """Build each drum voice's full phrase from the archetype backbone +
    the evolvable fill (fill lands on the last bar of the snare/perc)."""
    arch = P.ARCHETYPES[g.get("drum_style", "classic")]
    fill = g.get("drum_fill")
    bars = g.get("bars", 4)
    out = {}
    for v in D.DRUM_VOICES:
        bb = arch.get(v)
        if bb is None:
            continue
        last = fill if (v in P.FILL_VOICES and fill is not None) else bb
        out[v] = P.drum_full(bb, last, bars)
    return out


# ----------------------------------------------------------------- genesis

def random_genome(seed, bars=None, genre=None, bpm=None):
    r = np.random.default_rng(seed)
    g = {"seed": int(seed), "version": 2}
    g["genre"] = genre or str(r.choice(["dnb", "dnb", "techstep", "halftime"]))
    if bpm is None:
        bpm = float(r.uniform(170, 178)) if g["genre"] in ("dnb", "techstep") \
            else float(r.uniform(136, 145))
    g["bpm"] = round(bpm, 1)
    g["bars"] = int(bars or r.choice([4, 4, 8, 8, 16]))
    g["steps_per_bar"] = 16
    g["swing"] = round(float(r.uniform(0.0, 0.18)), 3)
    g["scale"] = str(r.choice(sorted(SCALES)))
    g["root"] = str(r.choice(sorted(ROOTS)))

    g["n_variants"] = int(r.integers(2, 6))        # bass sounds in the folder
    g["n_noise_sources"] = int(r.integers(2, 6))
    g["n_fill_sources"] = int(r.integers(2, 6))
    g["notes"] = scale_notes(ROOTS[g["root"]], g["scale"])[: max(4, g["n_variants"] + 1)]

    # alien pattern expressions (the motifs)
    g["bass_expr"] = P.gen_voice(r, "bass", g["n_variants"])
    g["noise_expr"] = P.gen_voice(r, "noise", g["n_noise_sources"])
    g["fill_expr"] = P.gen_voice(r, "fill", g["n_fill_sources"])
    # drums: a CONSTRAINED classic break (archetype) + an EVOLVABLE fill.
    # the backbone stays recognizable DnB/jungle/techstep; the fill is where
    # complexification happens (end-of-phrase rolls/edits grow over the run).
    if g["genre"] == "halftime":
        g["drum_style"] = "halftime" if r.random() < 0.7 else str(r.choice(P.STYLES))
    elif g["genre"] == "techstep":
        g["drum_style"] = "techstep" if r.random() < 0.6 else str(r.choice(["classic", "jungle", "techstep"]))
    else:
        g["drum_style"] = str(r.choice(["classic", "classic", "jungle", "techstep"]))
    g["drum_fill"] = P.gen_fill(r, D.spans().get("snare", 1))
    g["drum_exprs"] = assemble_drums(g)

    # source policy per voice
    g["source_mode"] = {v: str(r.choice(SOURCE_MODES[v])) for v in
                        ("bass", "noise", "fill")}

    # transform genetics (open-ended: chains, crossbreed, resynth)
    g["transform_p"] = round(float(r.uniform(0.3, 0.9)), 2)
    g["chain_len"] = int(r.integers(1, 4))          # stages per transform
    g["crossbreed_p"] = round(float(r.uniform(0.0, 0.5)), 2)
    g["resynth_p"] = round(float(r.uniform(0.15, 0.6)), 2)
    g["drive_max"] = round(float(r.uniform(0.0, 0.6)), 2)

    # per-hit processing genes (so no two triggers are identical)
    g["hit_pitch_p"] = round(float(r.uniform(0.0, 0.4)), 2)
    g["hit_rev_p"] = round(float(r.uniform(0.0, 0.15)), 2)
    g["hit_drive_p"] = round(float(r.uniform(0.0, 0.3)), 2)
    g["hit_crush_p"] = round(float(r.uniform(0.0, 0.2)), 2)
    g["hit_jit_ms"] = round(float(r.uniform(1.0, 7.0)), 2)

    # character dial — 0 = pure novelty (defects-only). Evolvable.
    g["strictness"] = 0.0

    g["gains"] = {"bass": round(float(r.uniform(0.8, 0.95)), 2),
                  "noise": round(float(r.uniform(0.3, 0.6)), 2),
                  "fill": round(float(r.uniform(0.35, 0.65)), 2),
                  "sub": round(float(r.uniform(0.3, 0.5)), 2)}
    g["pans"] = {"noise": round(float(r.uniform(-0.4, 0.4)), 2),
                 "fill": round(float(r.uniform(-0.3, 0.3)), 2)}
    g["voice_names"] = {"bass": "bass", "noise": "noise",
                        "fill": "fill", "sub": "sub"}
    return g


# ----------------------------------------------------------------- GA ops

def _sync_genre(g):
    g["genre"] = "dnb" if g["bpm"] >= 160 else "halftime"
    return g


def _span(g, voice):
    return {"bass": g["n_variants"], "noise": g["n_noise_sources"],
            "fill": g["n_fill_sources"]}[voice]


def mutate(g, rng, rate=0.3):
    m = json_copy(g)
    m["seed"] = int(rng.integers(2 ** 31))
    for voice, key in (("bass", "bass_expr"), ("noise", "noise_expr"),
                       ("fill", "fill_expr")):
        if rng.random() < max(0.4, rate):
            m[key] = P.mutate_expr(m[key], rng, voice, _span(m, voice))
    # drums: backbone stays constrained (rarely switch archetype);
    # COMPLEXIFY the fill (this is where drum intricacy evolves)
    if rng.random() < 0.15:
        m["drum_style"] = str(rng.choice(P.STYLES))
    if rng.random() < max(0.5, rate):
        base = m.get("drum_fill") or P.gen_fill(rng, D.spans().get("snare", 1))
        m["drum_fill"] = P.mutate_expr(base, rng, "fill", D.spans().get("snare", 1))
    m["drum_exprs"] = assemble_drums(m)
    if rng.random() < 0.35:
        m["bpm"] = round(float(np.clip(m["bpm"] + rng.uniform(-3, 3), 128, 182)), 1)
    if rng.random() < 0.25:
        m["swing"] = round(float(np.clip(m["swing"] + rng.uniform(-0.04, 0.04), 0, 0.2)), 3)
    for gene, lo, hi, d in (("transform_p", 0.0, 1.0, 0.2),
                            ("crossbreed_p", 0.0, 0.8, 0.2),
                            ("resynth_p", 0.0, 0.9, 0.2),
                            ("drive_max", 0.0, 1.0, 0.2),
                            ("hit_pitch_p", 0.0, 0.6, 0.2),
                            ("hit_drive_p", 0.0, 0.5, 0.15),
                            ("hit_crush_p", 0.0, 0.4, 0.15),
                            ("strictness", 0.0, 1.0, 0.2)):
        if rng.random() < 0.3:
            m[gene] = round(float(np.clip(m.get(gene, 0.0) + rng.uniform(-d, d), lo, hi)), 2)
    if rng.random() < 0.3:
        m["chain_len"] = int(np.clip(m.get("chain_len", 2) + rng.integers(-1, 2), 1, 4))
    for voice in ("bass", "noise", "fill"):
        if rng.random() < 0.2:
            m["source_mode"][voice] = str(rng.choice(SOURCE_MODES[voice]))
    if rng.random() < 0.25:                      # re-pitch / re-scale
        m["root"] = str(rng.choice(sorted(ROOTS)))
        m["scale"] = str(rng.choice(sorted(SCALES)))
        m["notes"] = scale_notes(ROOTS[m["root"]], m["scale"])[: len(m["notes"])]
    if rng.random() < 0.2:                        # folder sizes drift
        m["n_noise_sources"] = int(np.clip(m["n_noise_sources"] + rng.integers(-1, 2), 2, 7))
        m["n_fill_sources"] = int(np.clip(m["n_fill_sources"] + rng.integers(-1, 2), 2, 7))
    return _sync_genre(m)


def crossover(a, b, rng):
    c = json_copy(a)
    c["seed"] = int(rng.integers(2 ** 31))
    for key in ("bass_expr", "noise_expr", "fill_expr"):
        c[key] = P.crossover_expr(a[key], b[key], rng)
    c["drum_style"] = a.get("drum_style") if rng.random() < 0.5 else b.get("drum_style")
    fa, fb = a.get("drum_fill"), b.get("drum_fill")
    if fa and fb:
        c["drum_fill"] = P.crossover_expr(fa, fb, rng)
    else:
        c["drum_fill"] = copy.deepcopy(fa or fb)
    c["drum_exprs"] = assemble_drums(c)
    for k in ("bpm", "swing", "scale", "root", "genre",
              "transform_p", "chain_len", "crossbreed_p", "resynth_p",
              "drive_max", "hit_pitch_p", "hit_rev_p", "hit_drive_p",
              "hit_crush_p", "hit_jit_ms", "strictness",
              "n_variants", "n_noise_sources", "n_fill_sources",
              "gains", "pans"):
        src = a if rng.random() < 0.5 else b
        if k in src:
            c[k] = json_copy(src[k])
    c["source_mode"] = {v: (a["source_mode"][v] if rng.random() < 0.5
                            else b["source_mode"][v])
                        for v in ("bass", "noise", "fill")}
    c["notes"] = scale_notes(ROOTS[c["root"]], c["scale"])[: max(4, c["n_variants"] + 1)]
    return _sync_genre(c)


# ----------------------------------------------------------------- export

def materialize(expr, rng, bars, steps=16, as_index=True):
    """Evaluate the expression to bars*steps cells for export / sub
    derivation (the program phases as polymeter across the whole run)."""
    stream = P.eval_stream(expr, rng, bars * steps)
    out = []
    for v in stream:
        if not P.is_hit(v):
            out.append(0)
        elif as_index:
            out.append(max(1, int(round(float(v)))))
        else:
            out.append(1)
    return out


def pattern_to_alien(voice, expr):
    """Export the GENERATIVE expression — paste straight into [alien]; it
    breathes in the patch exactly as it did in evolution."""
    return f"{voice} {P.to_src(expr)}"


def sub_line_src(g, materialized_bass):
    """A materialized sub note stream from the bass index stream (note of
    the wrapped index at strong steps)."""
    notes = g["notes"]
    out = []
    for v in materialized_bass:               # the sub follows every bass hit
        out.append(str(notes[(v - 1) % len(notes)]) if v else "-")
    return "sub (seq " + " ".join(out) + ")"
