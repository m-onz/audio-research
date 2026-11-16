"""build: resolve a genome into a riff FOLDER.

Sources are drawn per the genome's source policy — CHOPPED corpus events
(short, crisp, event-aligned), tonal one-shots, or synthesis — then
pushed through the open arsenal (transform chains + crossbreeding +
bass tuning). The ONLY gate is ears.defects() (strictness 0 = pure
novelty); a defective candidate is substituted. Patterns export as the
live alien expressions. Output contract is unchanged:

    sounds/{bass,noise,fill}/  numbered oneshots
    patterns/{bass,noise,fill,sub,all}.txt   alien lines (paste into [alien])
    _preview.wav  _genome.json  _info.txt
"""
import json
from pathlib import Path

import numpy as np

from . import arsenal, chop, core, drums, ears, genome, pdpatch, pool, render

VOICE_FILTERS = {"bass": (28.0, 6000.0), "noise": (300.0, 12000.0),
                 "fill": (300.0, 15000.0)}
MAX_SHOT_S = {"bass": 0.8, "noise": 0.6, "fill": 0.9}

# the recently-generated bass libraries — the bass voice is drawn from here
GEN_BASS_DIRS = ["m-onz_bass", "m-onz_bass2", "m-onz_bass3", "m-onz_dub",
                 "m-onz_squelch", "m-onz_neuro", "m-onz_neuro2", "m-onz_neuro3"]

_CHOP = None
_ROLES = None
_XW = None
_GEN_BASS = None


def _gen_bass():
    """All generated bass one-shots (cached). The bass voice uses these."""
    global _GEN_BASS
    if _GEN_BASS is None:
        files = []
        for d in GEN_BASS_DIRS:
            p = Path(d)
            if p.is_dir():
                files += [str(q) for q in sorted(p.glob("*.wav"))
                          if not q.name.startswith("_")]
        _GEN_BASS = files
    return _GEN_BASS


def _chop_lib():
    global _CHOP, _ROLES, _XW
    if _CHOP is None:
        _CHOP = chop.build()
        _ROLES = chop.by_role(_CHOP)
        _XW = chop._whiten(_CHOP)
    return _CHOP, _ROLES, _XW


# ----------------------------------------------------------------- dsp helpers

def _trim_onset(x, sr, floor_db=-38.0):
    m = np.abs(np.asarray(x, np.float32))
    m = m.mean(axis=1) if m.ndim > 1 else m
    if len(m) < 64:
        return x
    pk = float(m.max())
    if pk < 1e-6:
        return x
    above = np.flatnonzero(m > pk * 10 ** (floor_db / 20.0))
    if not len(above) or above[0] == 0:
        return x
    return np.ascontiguousarray(x[max(0, int(above[0]) - int(0.001 * sr)):])


def _trim_tail(x, sr, floor_db=-40.0, min_s=0.04):
    m = np.abs(np.asarray(x, np.float32))
    m = m.mean(axis=1) if m.ndim > 1 else m
    if len(m) < 64:
        return x
    thr = 10 ** (floor_db / 20.0) * max(float(np.max(m)), 1e-6)
    above = np.flatnonzero(m > thr)
    if not len(above):
        return x
    end = max(int(above[-1]) + int(0.02 * sr), int(min_s * sr))
    y = np.ascontiguousarray(x[: min(len(m), end)])
    f = min(len(y) // 4, int(0.02 * sr))
    if f > 2:
        w = np.linspace(1, 0, f, dtype=np.float32)
        y[-f:] *= w[:, None] if y.ndim > 1 else w
    return y


def _butter(x, sr, hp, lp, order=2):
    from scipy.signal import butter, lfilter, lfilter_zi
    y = np.asarray(x, np.float32).copy()
    if y.ndim == 1:
        y = y[:, None]
    for cut, bt in ((hp, "high"), (lp, "low")):
        if not cut:
            continue
        b, a = butter(order, cut / (sr / 2), btype=bt)
        zi0 = lfilter_zi(b, a)
        for ch in range(y.shape[1]):
            y[:, ch], _ = lfilter(b, a, y[:, ch], zi=zi0 * y[0, ch])
    return y


def _write_shot(path, x, sr, voice):
    import soundfile as sf
    y = core.safe(np.asarray(x, np.float32))
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    m = float(np.mean(y))
    if abs(m) > 0.005:
        y = y - np.float32(m)
    pk = float(np.abs(y).max())
    if pk > 1e-4:
        y = y * (0.9 / pk)
    fi = {"bass": 16, "noise": 3, "fill": 6}[voice]
    fo = {"bass": 24, "noise": 8, "fill": 12}[voice]
    fin = min(len(y) // 8, int(fi / 1000 * sr))
    fout = min(len(y) // 8, int(fo / 1000 * sr))

    def hc(nn):
        return 0.5 - 0.5 * np.cos(np.pi * np.arange(nn) / max(1, nn - 1))
    if fin > 2:
        y[:fin] *= hc(fin)[:, None]
    if fout > 2:
        y[-fout:] *= hc(fout)[::-1][:, None]
    sf.write(str(path), y, sr, subtype="PCM_16")


# ----------------------------------------------------------------- source draw

def _raw_sources(g, voice, pl, rng, want, usage):
    """Raw (untransformed) stereo sources for a voice, by source policy.
    Returns list of (x, sr, provenance_str)."""
    mode = g["source_mode"][voice]
    lib, roles, Xw = _chop_lib()
    out = []
    if voice == "bass":                           # bass always from the generated library
        files = _gen_bass()
        if files:
            idx = rng.permutation(len(files))[: want + 3]
            for j in idx:
                try:
                    x, sr = core.read_audio(files[int(j)], max_in_s=2.5)
                except Exception:
                    continue
                if len(x) >= 256:
                    out.append((x, sr, f"gen:{Path(files[int(j)]).stem[:20]}"))
            if out:
                return out
    if mode == "synth" and voice == "bass":
        return out                                # handled per-slot as founder
    if mode == "chop" or (mode == "synth" and voice != "bass"):
        picks = chop.pick_fresh(lib, roles.get(voice, []), want + 3, rng,
                                usage, Xw)
        for i in picks:
            x, sr = chop.read_slice(lib[i])
            if x is not None and len(x) >= 256:
                out.append((x, sr, f"chop:{Path(lib[i]['path']).stem[:18]}"))
    else:                                          # oneshot pool
        recs = pl.get(voice, [])
        if recs:
            idx = rng.permutation(len(recs))[: want + 3]
            for j in idx:
                try:
                    x, sr = core.read_audio(recs[int(j)]["path"], max_in_s=1.6)
                except Exception:
                    continue
                if len(x) >= 256:
                    out.append((x, sr, f"pool:{Path(recs[int(j)]['path']).stem[:18]}"))
    return out


# ----------------------------------------------------------------- processing

def _process(g, voice, x, sr, note, rng, partners):
    """Tune (bass) + transform chain + crossbreed. Returns (y, tags)."""
    tags = []
    if sr != 44100:
        x = core.resample(x, sr / 44100.0); sr = 44100
    x = _trim_onset(x, sr)
    if voice == "bass" and note is not None:
        f0 = core.dominant_f0(x, sr, default=0.0)
        if f0 <= 0 or rng.random() < g.get("resynth_p", 0.3):
            x = arsenal.partial_resynth(x, sr, rng, note,
                                        drive=float(rng.uniform(0, g.get("drive_max", 0.3))))
            tags.append("resynth")
        else:
            semis = int(round(12.0 * np.log2(genome.mtof(note) / f0)))
            if abs(semis) >= 1 and abs(semis) <= 24:
                from oneshot_lab import pitchfx
                x = pitchfx.pitch_shift_tape(x, sr, rng, semis=semis)
                tags.append(f"pitch{semis:+d}")
    if rng.random() < g.get("crossbreed_p", 0.0) and partners:
        p = partners[int(rng.integers(len(partners)))]
        x, nm = arsenal.crossbreed(x, p, sr, rng, MAX_SHOT_S[voice])
        if nm:
            tags.append(nm)
    if rng.random() < g.get("transform_p", 0.5):
        x, used = arsenal.transform_chain(x, sr, rng, voice,
                                          g.get("chain_len", 2),
                                          MAX_SHOT_S[voice])
        tags += used
    x = _trim_onset(x, sr)
    x = _trim_tail(x, sr)
    x = core.clamp_dur(core.safe(x), sr, MAX_SHOT_S[voice])
    x = _butter(x, sr, *VOICE_FILTERS[voice])
    return x, tags, sr


def _resolve_voice(g, voice, pl, rng, strictness, usage):
    want = {"bass": g["n_variants"], "noise": g["n_noise_sources"],
            "fill": g["n_fill_sources"]}[voice]
    raws = _raw_sources(g, voice, pl, rng, want, usage)
    partners = [x for x, _, _ in raws]
    notes = g["notes"]
    sounds, prov = [], []
    taste = ears.load_taste()
    for j in range(want):
        note = notes[j % len(notes)] if voice == "bass" else None
        made = None
        for attempt in range(min(4, max(1, len(raws)))):
            if raws:
                x, sr, src = raws[(j + attempt) % len(raws)]
            elif voice == "bass":
                x, sr, src = arsenal.founder_synth(rng, note or notes[0]), 44100, "synth"
            else:
                break
            try:
                y, tags, sr = _process(g, voice, x, sr, note, rng, partners)
            except Exception:
                continue
            f0h = genome.mtof(note) if (voice == "bass" and note) else 0.0
            if ears.passes(y, sr, voice, strictness, f0_hint=f0h, taste=taste):
                made = (y, sr, src, tags); break
            if not ears.defects(y, sr):
                made = made or (y, sr, src, tags)
        if made is None and voice == "bass":
            made = (arsenal.founder_synth(rng, note or notes[0]), 44100, "synth", ["founder"])
        if made is None:
            continue
        y, sr, src, tags = made
        sounds.append(y)
        prov.append(dict(file=f"{j + 1:02d}.wav", source=src,
                         note=note, transform="+".join(tags) or None))
    return sounds, prov


# ----------------------------------------------------------------- build

def prepare_riff(g, pl, strictness=None, source_use=None):
    rng = np.random.default_rng(g["seed"])
    strictness = g.get("strictness", 0.0) if strictness is None else strictness
    usage = source_use if source_use is not None else {}
    sounds, prov = {}, {}
    for voice in ("bass", "noise", "fill"):
        sounds[voice], prov[voice] = _resolve_voice(g, voice, pl, rng,
                                                    strictness, usage)
    if not sounds["bass"]:
        sounds["bass"] = [arsenal.founder_synth(rng, g["notes"][0])]
        prov["bass"] = [dict(file="01.wav", source="synth", note=g["notes"][0],
                             transform="founder")]
    drng = np.random.default_rng((int(g["seed"]) + 4242) & 0x7fffffff)
    drum_sounds, drum_prov = drums.make_kit(drng, variants=2)
    events, bass_idx, n_steps = render.collect_events(g, sounds, drum_sounds)
    preview = render.render_preview(g, events, n_steps)
    v = g["voice_names"]
    mat_bass = genome.materialize(g["bass_expr"],
                                  np.random.default_rng(g["seed"] + 1), g["bars"])
    pats = {"bass": genome.pattern_to_alien(v["bass"], g["bass_expr"]),
            "noise": genome.pattern_to_alien(v["noise"], g["noise_expr"]),
            "fill": genome.pattern_to_alien(v["fill"], g["fill_expr"]),
            "sub": genome.sub_line_src(g, mat_bass)}
    for dv, dexpr in (g.get("drum_exprs") or {}).items():
        pats[dv] = genome.pattern_to_alien(dv, dexpr)
    return dict(genome=g, provenance=prov, sounds=sounds, pats=pats,
                preview=preview, sel=None, drum_prov=drum_prov,
                drum_sounds=drum_sounds)


def write_riff(prep, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g, prov, pats = prep["genome"], prep["provenance"], prep["pats"]
    for voice in ("bass", "noise", "fill"):
        vdir = out / "sounds" / voice
        vdir.mkdir(parents=True, exist_ok=True)
        for i, y in enumerate(prep["sounds"][voice]):
            _write_shot(vdir / f"{i + 1:02d}.wav", y, 44100, voice)
    # the custom per-generation kit (picked + transformed) — saved per idea
    import soundfile as _sf
    for dvoice, bufs in (prep.get("drum_sounds") or {}).items():
        ddir = out / "sounds" / "drums" / dvoice
        ddir.mkdir(parents=True, exist_ok=True)
        for i, y in enumerate(bufs):
            yy = core.safe(np.asarray(y, np.float32))
            if yy.ndim == 1:
                yy = np.stack([yy, yy], axis=1)
            pk = float(np.abs(yy).max())
            if pk > 1e-4:
                yy = yy * (0.9 / pk)
            _sf.write(str(ddir / f"{i + 1:02d}.wav"), yy, 44100, subtype="PCM_16")
    pat_dir = out / "patterns"
    pat_dir.mkdir(exist_ok=True)
    for name, line in pats.items():
        (pat_dir / f"{name}.txt").write_text("; " + line + ";\n")
    (pat_dir / "all.txt").write_text("; " + ";\n; ".join(pats.values()) + ";\n")
    core.write_audio(out / "_preview.wav", prep["preview"], 44100)
    # the importable Pd patch: every pattern as a global message
    title = f"riff_lab {out.parent.name}/{out.name}  ({g['genre']} {g['bpm']}bpm)"
    (out / "_patterns.pd").write_text(pdpatch.build_pd(title, list(pats.values())))
    with open(out / "_genome.json", "w") as fh:
        json.dump(dict(genome=g, provenance=prov), fh, indent=1)
    (out / "_info.txt").write_text(_info_text(g, prov, pats, prep.get("drum_prov")))
    return out


def build_riff(g, pl, out_dir):
    return write_riff(prepare_riff(g, pl), out_dir)


def _info_text(g, prov, pats, drum_prov=None):
    L = [f"RIFF — {g['genre']} / {g['bars']} bars / {g['scale']} on {g['root']}"
         f" / swing {g['swing']}  (clock: 80 ms/step)",
         f"DRUMS — {g.get('drum_style', 'classic')} break + evolved fill", "",
         "CUSTOM KIT (unique to this idea — picked + transformed per voice):"]
    if drum_prov:
        for v, items in drum_prov.items():
            L.append(f"  [{v}] " + ", ".join(f"{nm}({tg})" for nm, tg in items))
    L += ["  BASS voice is drawn from the generated m-onz bass library.", "",
         "FASTEST IMPORT: open _patterns.pd and click SEND-ALL (or just open",
         "it) — every pattern broadcasts to the running dnb-1 patch as global",
         "messages to each alien voice (kick/snare/hihat/openhh/perc + bass/",
         "noise/fill/sub). Drums play this idea's custom kit "
         "(saved in sounds/drums/).", "",
         "TO USE THE EVOLVED SOUNDS TOO:",
         "  1. copy sounds/bass sounds/noise sounds/fill into the patch",
         "  2. one _ voice per folder (alien -> else/seq -> playdir~)",
         "  3. values wrap modulo folder size; sub = MIDI notes -> m-sub2~", "",
         f"  source policy: {g['source_mode']}", "",
         "SOUNDS (riff voices — drums come from the patch kit):"]
    for voice in ("bass", "noise", "fill"):
        L.append(f"  [{voice}]")
        for p in prov[voice]:
            t = f"  ({p['transform']})" if p.get("transform") else ""
            nt = f" note {p['note']}" if p.get("note") else ""
            L.append(f"    {p['file']} <- {p['source']}{nt}{t}")
    L += ["", "  (the custom kit samples are saved in sounds/drums/<voice>/)",
          "", "PATTERNS (alien):"]
    for k, line in pats.items():
        L.append(f"  ; {line};")
    return "\n".join(L) + "\n"
