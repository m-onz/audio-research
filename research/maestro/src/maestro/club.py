"""club: the night-club pipeline — a driving floor, an evolving top.

The audio_lab/loom review distilled: every club piece in this
ecosystem is TWO things stacked — a DRIVING rhythm section (the
fusion recipe's spine+duck+parts, club_lab's GA beat slots, edm_lab's
four-on-the-floor drop) and a GENERATED layer riding it (fusion's
stutter_leads and map_dialogs, loom's layered voices). maestro's
version keeps both halves honest about what they are:

    DRIVER  the hand-designed dancefloor skeleton — the paper's
            "scaffold produced by competent humans", made explicit:
            four-on-the-floor kick on the FORM's beats (accented
            downbeats, double-time in the rush, none in the break),
            rimshot on the offbeat, closed-hat 16ths, crash impacts
            at section boundaries, and ONE spine-locked polymetric
            perc crossing the grid (the spine's own meter against
            the form's).
    TOP     a maestro pattern, EVOLVED: breed organisms on this
            spine (`maestro run`), take the freshest specimen, and
            its score rides the driver — stabs, textures, ghost
            kicks at half gain, the part that never repeats.

club_lab's duck (depth 0.35, tau 0.14) glues the floor; post still
places its damage where the form says it lives. Everything
deterministic from the seed; the manifest names both organisms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .champions import random_score_genome
from .form import BEATS_PER_BAR, Form, build_form
from .genome import INPUT_IDS, Genome
from .mapper import mapped_audio
from .palette import auto_voices
from .post import duck, post_process
from .scaffold import load_spine
from .score import score_from, score_summary, write_score
from echo.probes import SR, write_wav

__all__ = ["driver_score", "club_compose", "best_specimen",
           "resolve_top_genome", "quantize_top"]


# ------------------------------------------------------------------ driver


def _crossperc() -> Genome:
    """The driver's one spine-locked voice: graded strikes on the high
    band's clock — the spine's own meter crossing the form's grid."""
    g = Genome(seed=909)
    out = g.add_output("percB", fn="tanh", bias=-0.55)
    g.conn(INPUT_IDS["s3"], out, 2.5)
    return g


def driver_score(scaf, form: Form, length: float, impacts: bool = True,
                 rush_double: bool = True) -> dict:
    """The dancefloor skeleton, read off the form's tick grid. Beats
    are ticks 0, tpb, 2*tpb... per section; dance sections (bpm >=
    100) get the kick/rim/hats, the break keeps a rim pulse, rush
    sections (bpm >= 140) double the kick. A maestro-shaped score
    dict, mappable like any other."""
    tt = form.tick_times(length)
    tpb = form.ticks_per_beat
    n = len(tt)
    events: list[dict] = []

    def ev(voice, j, vel):
        if j >= n:
            return
        d = float(tt[j + 1] - tt[j]) if j + 1 < n else float(tt[-1] - tt[-2])
        events.append({"voice": voice, "tick": int(j),
                       "t": float(tt[j]), "dur": d,
                       "velocity": float(np.clip(vel, 0.0, 1.0))})

    i = 0
    for si, sec in enumerate(form.sections):
        n_sec = sec.bars * BEATS_PER_BAR * tpb
        if i >= n:
            break
        sec_end = min(i + n_sec, n)
        dance = sec.bpm >= 100.0
        rush = sec.bpm >= 140.0
        half = max(1, tpb // 2)
        if dance:
            for j in range(i, sec_end, tpb):
                beat = (j - i) // tpb
                ev("kick", j, 1.0 if beat % 4 == 0 else 0.86)   # accent
                if rush_double and rush and j + half < sec_end:
                    ev("kick", j + half, 0.68)                  # double-time
                if j + half < sec_end:
                    ev("rim", j + half, 0.80)                   # the offbeat
            step = max(1, tpb // 4)
            for j in range(i, sec_end, step):
                ev("hat16", j, 0.50)                            # 16ths
        else:
            for j in range(i, sec_end, tpb):
                ev("rim", j, 0.60)                              # break pulse
        if impacts and si > 0:
            ev("impact", i, 1.0)                                # the drop
        i = sec_end

    # the polymetric perc: a tiny pure organism over the scaffold
    cross = score_from(_crossperc(), scaf, mode="strike")
    for e in cross["events"]:
        if e["voice"] == "percB" and e["t"] <= tt[-1]:
            events.append(e)
    events.sort(key=lambda e: (e["t"], e["voice"]))

    return {
        "maestro": "score", "version": 1,
        "grid": {"bpm": None, "ticks_per_beat": tpb,
                 "tick_hz": float(n / max(tt[-1], 1e-9)), "n_ticks": n,
                 "length_s": scaf.length_s, "uniform": False,
                 "form": form.to_dict(), "driver": True},
        "spine": {"path": scaf.path, "sha256": scaf.sha256,
                  "spine_length_s": scaf.spine_length_s},
        "senses": ["grid"], "threshold": 0.0, "mode": "strike",
        "note": "hand-designed dancefloor skeleton (the driver)",
        "genome": {"driver": "grid+crossperc"},
        "n_events": len(events), "events": events, "matrix": {},
    }


# ------------------------------------------------------------------ the top


def resolve_top_genome(spec: str) -> Genome:
    """The evolving layer's organism: a run specimen json
    (`maestro_out/run/novelty_s0/spec_00.json`), a genome json, or
    `random:N` for a founder."""
    if spec.startswith("random:"):
        return random_score_genome(np.random.default_rng(int(spec[7:])))
    d = json.loads(Path(spec).read_text())
    return Genome.from_dict(d["genome"] if "genome" in d else d)


def best_specimen(run_dir) -> Path:
    """The freshest voice of a breeding run: the specimen with the
    highest novelty at birth. Feed it to club --top."""
    run_dir = Path(run_dir)
    best, best_nov = None, -1.0
    for p in sorted(run_dir.glob("spec_*.json")):
        d = json.loads(p.read_text())
        nov = d.get("nov_at_birth")
        if nov is not None and float(nov) > best_nov:
            best, best_nov = p, float(nov)
    if best is None:
        raise SystemExit(f"no specimens with novelties in {run_dir} — "
                        "is this a maestro run directory?")
    return best


# ------------------------------------------------------------------ compose


def quantize_top(score: dict, form: Form, length: float,
                 window: str = "half", max_per: int = 4) -> dict:
    """Snap the evolving layer onto the dancefloor's grid.

    The bred organism fires wherever its spike phases say — dense,
    continuous, gloriously un-danceable. This is club_lab's slot
    discipline, pointed at maestro's phenotype: half-beat windows of
    the FORM's grid; per window, each voice keeps ONE event (its
    peak, velocity intact) SNAPPED to the window tick; at most
    `max_per` voices sound per window (the loudest). The evolution
    survives — WHICH voices, HOW loud, WHEN they crest is all the
    organism's — but it lands on the 8th-note grid, over the floor.
    """
    tpb = form.ticks_per_beat
    wt = {"beat": tpb, "half": max(1, tpb // 2),
          "quarter": max(1, tpb // 4)}.get(window, max(1, tpb // 2))
    tt = form.tick_times(length)
    n = min(score["grid"]["n_ticks"], len(tt))

    best: dict[tuple[int, str], dict] = {}
    for e in score["events"]:
        if e["tick"] >= n:
            continue
        w = e["tick"] // wt
        k = (w, e["voice"])
        if k not in best or e["velocity"] > best[k]["velocity"]:
            best[k] = e
    by_w: dict[int, list] = {}
    for (w, _v), e in best.items():
        by_w.setdefault(w, []).append(e)
    out_events = []
    for w, evs in sorted(by_w.items()):
        evs.sort(key=lambda e: -e["velocity"])
        for e in (evs[: max_per] if max_per else evs):
            j = w * wt
            d = float(tt[j + 1] - tt[j]) if j + 1 < len(tt) \
                else float(tt[-1] - tt[-2])
            out_events.append({"voice": e["voice"], "tick": int(j),
                               "t": float(tt[j]), "dur": d,
                               "velocity": float(e["velocity"])})
    out_events.sort(key=lambda e: (e["t"], e["voice"]))
    out = dict(score)
    out["events"] = out_events
    out["n_events"] = len(out_events)
    out["note"] = (f"quantized to {window}-beat grid, max {max_per} "
                   "voices/window (the evolving pattern, danceable)")
    return out


def club_compose(spine: str, form: str = "club", top: str = "random:7",
                 seed: int = 0, length: float = 180.0,
                 corpus: str = "corpus", driver_gain: float = 1.0,
                 top_gain: float = 0.5, spine_gain: float = 0.40,
                 duck_depth: float = 0.35, post_intensity: float = 1.0,
                 loop: bool = True, mode: str = "strike",
                 threshold: float = 0.12, senses=("spikes", "weather"),
                 voice_overrides: dict | None = None, impacts: bool = True,
                 rush_double: bool = True, top_grid: bool = True,
                 top_window: str = "half", top_voices_per: int = 4,
                 out: str = "maestro_out/club", verbose: bool = True) -> dict:
    """Driver + evolved top -> the night-club piece. The driver's
    voices cast at full gain; the top's at `top_gain` (quantized onto
    the form's grid when top_grid); club_lab's duck on every driver
    kick; post where the form says it lives."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = build_form(form)
    scaf0 = load_spine(spine, bpm=120.0, ticks_per_beat=f.ticks_per_beat)
    tt = f.tick_times(length)
    scaf = scaf0.sample(tt, loop=loop)

    driver = driver_score(scaf, f, length, impacts=impacts,
                          rush_double=rush_double)
    top_g = resolve_top_genome(top)
    top_sc = score_from(top_g, scaf, senses=senses, threshold=threshold,
                        mode=mode, form=f.to_dict())
    if top_grid:
        top_sc = quantize_top(top_sc, f, length, window=top_window,
                              max_per=top_voices_per)
    write_score(driver, out / "driver.score.json")
    write_score(top_sc, out / "top.score.json")

    d_voices = sorted({e["voice"] for e in driver["events"]})
    t_voices = sorted(top_sc["matrix"].keys())
    d_map, d_prov = auto_voices(d_voices, corpus_dir=corpus, seed=seed,
                                overrides=voice_overrides)
    # overrides reach the top too: a bred specimen's out2..outN voices
    # would otherwise fall to the texture pools — with --voice the
    # evolved percussion can be cast as real percs/hats/rimshots.
    t_map, t_prov = auto_voices(t_voices, corpus_dir=corpus,
                                 seed=seed + 555,              # auto picks
                                 overrides=voice_overrides)   # stay cast apart
    for v in d_map:
        d_map[v]["gain"] *= driver_gain
    for v in t_map:
        t_map[v]["gain"] *= top_gain
    d_map["spine_gain"] = spine_gain
    t_map["spine_gain"] = 0.0           # the spine rides under the driver only

    mix = mapped_audio(driver, d_map, tail_s=0.5, spine_loop=loop) \
        + mapped_audio(top_sc, t_map, tail_s=0.5)

    kick_times = [e["t"] for e in driver["events"] if e["voice"] == "kick"]
    if duck_depth > 0 and kick_times:
        mix = duck(mix, SR, kick_times, depth=duck_depth, tau=0.14)

    bounds = f.section_bounds(length)
    piece, post_ledger = post_process(mix, SR, bounds, seed=seed,
                                      intensity=post_intensity)
    wav_path = out / "piece.wav"
    write_wav(wav_path, piece.astype(np.float32), SR)

    manifest = {
        "maestro": "club", "version": 1, "seed": seed,
        "spine": {"path": scaf.path, "sha256": scaf.sha256,
                  "spine_length_s": scaf.spine_length_s, "loop": loop},
        "form": f.to_dict(),
        "driver": {"kind": "grid skeleton + spine-locked percB",
                   "voices": {v: {"file": d_map[v]["file"],
                                  "gain": d_map[v]["gain"],
                                  "pan": d_map[v]["pan"],
                                  "family": d_prov[v]["family"]}
                              for v in d_voices},
                   "n_events": driver["n_events"]},
        "top": {"genome_source": str(top),
                "genome": top_g.to_dict(),
                "grid_quantized": top_grid,
                "window": top_window if top_grid else None,
                "voices": {v: {"file": t_map[v]["file"],
                               "gain": t_map[v]["gain"],
                               "pan": t_map[v]["pan"],
                               "family": t_prov[v]["family"]}
                           for v in t_voices},
                "n_events": top_sc["n_events"]},
        "spine_gain": spine_gain, "top_gain": top_gain,
        "duck_depth": duck_depth, "post": post_ledger,
        "driver_summary": score_summary(driver),
        "top_summary": score_summary(top_sc),
        "piece": {"path": str(wav_path), "sr": SR,
                  "length_s": round(float(piece.shape[0] / SR), 2)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    if verbose:
        print(f"[maestro] form   : {form}")
        print(f"[maestro] spine  : {spine} (loop {loop})")
        ds = score_summary(driver)
        print(f"[maestro] driver : {ds['n_events']} events")
        for v, d in sorted(ds["voices"].items()):
            print(f"  {v:8s} {d['events']:5d} events  {d['density']:5.1f}/s"
                  f"  <- {Path(d_prov[v]['file']).name}")
        ts = score_summary(top_sc)
        print(f"[maestro] top    : {ts['n_events']} events"
              f"  (genome: {top})")
        for v, d in sorted(ts["voices"].items()):
            print(f"  {v:8s} {d['events']:5d} events  {d['density']:5.1f}/s"
                  f"  <- {Path(t_prov[v]['file']).name}")
        kinds = {}
        for e in post_ledger:
            kinds[e["type"]] = kinds.get(e["type"], 0) + 1
        print(f"[maestro] post   : " + ", ".join(f"{k}x{v}"
                                                 for k, v in sorted(kinds.items())))
        print(f"[maestro] piece  : {wav_path}"
              f" ({piece.shape[0] / SR:.1f}s)")
    return manifest