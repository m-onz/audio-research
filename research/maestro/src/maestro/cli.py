"""cli: maestro score | map | run | report | preview | info | champions.

    maestro score --spine club.wav --champion follower -o score.json
    maestro score --spine spine.wav --random --seed 7 --mode gate --bpm auto
    maestro map score.json --voice pulse=kick.wav --voice flow=air.wav \
              --spine-gain 0.5 -o piece.wav
    maestro run --spine spine.wav --arms novelty random --gens 240
    maestro report maestro_out/run1
    maestro preview score.json -o preview.wav --spine-gain 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .champions import CHAMPIONS, build_champion, random_score_genome
from .club import club_compose
from .compose import compose
from .form import FORMS, Form, describe_form
from .genome import Genome
from .mapper import render_mapped
from .preview import render_preview
from .scaffold import SENSES, estimate_bpm, load_spine
from .score import load_score, score_from, score_summary, write_score

__all__ = ["main"]


def _parse_senses(s: str) -> tuple:
    senses = tuple(x.strip() for x in s.split(",") if x.strip())
    bad = [x for x in senses if x not in SENSES]
    if bad:
        raise SystemExit(f"unknown senses {bad}; valid: {SENSES}")
    return senses


def _parse_bpm(v) -> float:
    """--bpm is a number or 'auto' (estimated from the spine's own
    onsets — the grid need not be guessed)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        raise SystemExit(f"--bpm must be a number or 'auto', got {v!r}")


def _load_spine_args(args, spine):
    """Shared spine loader; resolves --bpm auto by measuring the spine
    once at a provisional grid (onsets live in the frame domain, they
    do not depend on the tick grid) and rebuilding at the estimate."""
    tpb = getattr(args, "tpb", 4)
    tick_hz = getattr(args, "tick_hz", None)
    length = getattr(args, "length", None)
    sr = getattr(args, "sr", 44100)
    bpm = getattr(args, "bpm", 120.0)
    if str(bpm) == "auto" and tick_hz is None:
        scaf0 = load_spine(spine, bpm=120.0, ticks_per_beat=tpb,
                           tick_hz=None, length_s=length, sr=sr)
        est = estimate_bpm(scaf0.onsets["s0"])
        if est is None:
            print("[maestro] bpm auto: too few onsets to read — using 120")
            est = 120.0
        print(f"[maestro] bpm auto: estimated {est:g} from "
              f"{len(scaf0.onsets['s0'])} onsets")
        bpm = est
    else:
        bpm = _parse_bpm(bpm)
    return load_spine(spine, bpm=bpm, ticks_per_beat=tpb,
                      tick_hz=tick_hz, length_s=length, sr=sr)


def _genome(args) -> Genome:
    picks = sum(bool(x) for x in (args.champion, args.genome, args.random))
    if picks != 1:
        raise SystemExit("give exactly one of --champion, --genome, --random")
    if args.champion:
        return build_champion(args.champion)
    if args.genome:
        return Genome.from_dict(json.loads(Path(args.genome).read_text()))
    return random_score_genome(np.random.default_rng(args.random_seed))


def _cmd_club(args):
    overrides = {}
    for spec in args.voice or []:
        if "=" not in spec:
            raise SystemExit(f"--voice wants name=file.wav — got {spec!r}")
        name, path = spec.split("=", 1)
        overrides[name.strip()] = path.strip()
    club_compose(spine=args.spine, form=args.form, top=args.top,
                 seed=args.seed, length=args.length, corpus=args.corpus,
                 driver_gain=args.driver_gain, top_gain=args.top_gain,
                 spine_gain=args.spine_gain, duck_depth=args.duck,
                 post_intensity=args.post, loop=not args.no_loop,
                 mode=args.mode, threshold=args.threshold,
                 senses=_parse_senses(args.senses),
                 voice_overrides=overrides, impacts=not args.no_impact,
                 rush_double=not args.no_rush_double,
                 top_grid=not args.no_top_grid,
                 top_window=args.top_window,
                 top_voices_per=args.top_voices_per,
                 out=args.out, verbose=True)
    return 0


def _cmd_compose(args):
    overrides = {}
    for spec in args.voice or []:
        if "=" not in spec:
            raise SystemExit(f"--voice wants name=file.wav — got {spec!r}")
        name, path = spec.split("=", 1)
        overrides[name.strip()] = path.strip()
    compose(spine=args.spine, form=args.form, length=args.length,
            seed=args.seed, champion=args.champion, genome_path=args.genome,
            random_seed=args.random_seed, mode=args.mode,
            threshold=args.threshold, senses=_parse_senses(args.senses),
            corpus=args.corpus, voice_overrides=overrides,
            spine_gain=args.spine_gain, duck_depth=args.duck,
            post_intensity=args.post, loop=not args.no_loop, out=args.out,
            verbose=True)
    return 0


def _cmd_forms(args):
    for name, sections in FORMS.items():
        f = Form(sections)
        print(f"{name:10s} " + " ".join(
            f"[{s.label}]{s.bpm:g}" for s in f.sections)
            + f"  (~{sum(s.seconds for s in f.sections):.0f}s)")
    return 0


def _cmd_score(args):
    genome = _genome(args)
    scaf = _load_spine_args(args, args.spine)
    score = score_from(genome, scaf, senses=_parse_senses(args.senses),
                       threshold=args.threshold, mode=args.mode)
    out = Path(args.out)
    write_score(score, out)
    print(scaf.info())
    s = score_summary(score)
    print(f"score   : {out}  ({s['n_events']} events, "
          f"{len(s['voices'])} voices, mode {s['mode']})")
    for voice, d in s["voices"].items():
        print(f"  {voice:12s} {d['events']:5d} events  "
              f"{d['density']:6.1f}/s  vel mean {d['velocity_mean']:.2f} "
              f"max {d['velocity_max']:.2f}")
    pv = out.with_name(out.stem + "_preview.wav")
    render_preview(score, pv, spine_gain=args.preview_spine_gain)
    print(f"preview : {pv}  (listen: the score over "
          f"{args.preview_spine_gain:g}x spine)")
    print(f"map it  : maestro map {out} --voice <name>=<sound.wav> -o piece.wav")
    return 0


def _cmd_map(args):
    score = load_score(args.score)
    voice_map: dict = {"voices": {}, "spine_gain": args.spine_gain}
    for spec in args.voice or []:
        if "=" not in spec:
            raise SystemExit(f"--voice wants name=file.wav[,gain=..,pan=..] — got {spec!r}")
        name, rest = spec.split("=", 1)
        parts = {p.split("=")[0]: p.split("=")[1] for p in rest.split(",")
                 if "=" in p}
        if "file" not in parts:
            parts = {"file": rest}
        voice_map["voices"][name.strip()] = parts
    if args.map_file:
        loaded = json.loads(Path(args.map_file).read_text())
        voice_map["voices"].update(loaded.get("voices", {}))
        voice_map["spine_gain"] = loaded.get(
            "spine_gain", voice_map["spine_gain"])
    out = render_mapped(score, voice_map, args.out)
    from echo.probes import decode, SR
    x = decode(out, sr=SR)
    print(f"piece   : {out}  ({len(x)/SR:.1f}s, "
          f"{len(voice_map['voices'])} voices mapped, "
          f"spine_gain {voice_map['spine_gain']:g})")
    return 0


def _cmd_run(args):
    from .engine import run_experiment
    bpm = args.bpm
    if str(bpm) == "auto":
        scaf0 = load_spine(args.spine, bpm=120.0,
                           ticks_per_beat=args.tpb,
                           length_s=args.eval_len)
        est = estimate_bpm(scaf0.onsets["s0"])
        if est is None:
            print("[maestro] bpm auto: too few onsets to read — using 120")
            est = 120.0
        print(f"[maestro] bpm auto: estimated {est:g} from "
              f"{len(scaf0.onsets['s0'])} onsets")
        bpm = est
    else:
        bpm = _parse_bpm(bpm)
    run_experiment(out=args.out, seeds=tuple(args.seeds), gens=args.gens,
                   pop=args.pop, births=args.births,
                   arms=tuple(args.arms), spine=args.spine, mode=args.mode,
                   strictness=args.strictness, eval_len=args.eval_len,
                   threshold=args.threshold, bpm=bpm,
                   ticks_per_beat=args.tpb, specimen_len=args.specimen_len,
                   verbose=True)
    return 0


def _cmd_report(args):
    from .engine import report_run
    print(report_run(args.run))
    return 0


def _cmd_preview(args):
    score = load_score(args.score)
    out = render_preview(score, args.out, spine_gain=args.spine_gain)
    print(f"preview : {out}  ({score['n_events']} events over "
          f"spine_gain {args.spine_gain:g})")
    return 0


def _cmd_info(args):
    score = load_score(args.score)
    g, sp = score["grid"], score["spine"]
    print(f"spine   : {sp['path'] or '<array>'}  "
          f"sha {sp['sha256'][:12] if sp['sha256'] else '-'}")
    print(f"grid    : {g['n_ticks']} ticks @ {g['tick_hz']:g} Hz "
          f"({g['length_s']:.1f}s"
          + (f", bpm {g['bpm']:g} x {g['ticks_per_beat']}" if g["bpm"]
             else "") + ")")
    print(f"genome  : {len(score['genome']['nodes'])} nodes, "
          f"{len(score['genome']['conns'])} conns, "
          f"outputs {list(score['genome']['outputs'])}")
    print(f"senses  : {', '.join(score['senses'])}  "
          f"mode {score['mode']}  threshold {score['threshold']}")
    s = score_summary(score)
    for voice, d in s["voices"].items():
        print(f"  {voice:12s} {d['events']:5d} events  "
              f"{d['density']:6.1f}/s  vel mean {d['velocity_mean']:.2f} "
              f"max {d['velocity_max']:.2f}")
    print("first events:")
    for e in score["events"][:12]:
        print(f"  t {e['t']:7.3f}  {e['voice']:12s} vel {e['velocity']:.2f} "
              f"dur {e['dur']:.3f}")
    return 0


def _cmd_champions(args):
    for name, (_, desc) in CHAMPIONS.items():
        print(f"{name:10s} {desc}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="maestro",
        description="maestro: the orchestration organ — a CPPN whose inputs "
                    "are a spike-encoded spine, whose outputs are voices, "
                    "whose phenotype is a score. (NEAT Drummer's "
                    "scaffolding, open-ended edition.)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # -- score
    sc = sub.add_parser("score", help="genome + spine -> score JSON + preview")
    sc.add_argument("--spine", required=True,
                    help="the scaffold: any audio file")
    src = sc.add_mutually_exclusive_group()
    src.add_argument("--champion", "-c",
                     help=f"hand-built: {sorted(CHAMPIONS)}")
    src.add_argument("--genome", "-g", help="genome JSON (from a run, by hand)")
    src.add_argument("--random", action="store_true",
                     help="a random score founder (minimal topology, "
                          "wired onto the scaffold senses)")
    sc.add_argument("--random-seed", type=int, default=0)
    sc.add_argument("--bpm", default="120",
                    help="tick grid tempo, or 'auto' (estimated from the "
                         "spine's onsets)")
    sc.add_argument("--tpb", "--ticks-per-beat", type=int, default=4)
    sc.add_argument("--tick-hz", type=float, default=None,
                    help="direct tick rate (overrides bpm/tpb)")
    sc.add_argument("--length", type=float, default=None,
                    help="piece length in s (default: spine length)")
    sc.add_argument("--sr", type=int, default=44100)
    sc.add_argument("--mode", choices=("strike", "gate"), default="strike",
                    help="strike: every audible tick = an event (the "
                         "paper's drums). gate: rising edge holds while "
                         "audible (sustained material)")
    sc.add_argument("--threshold", type=float, default=0.12,
                    help="audibility floor (velocity below = silence)")
    sc.add_argument("--senses", default="spikes,weather",
                    help="comma list from: spikes,weather,position,time "
                         "(position/time are impure: structure not "
                         "inherited from the spine)")
    sc.add_argument("--preview-spine-gain", type=float, default=0.6)
    sc.add_argument("-o", "--out", default="maestro_out/score.json")
    sc.set_defaults(func=_cmd_score)

    # -- club (driver + evolving top)
    cl = sub.add_parser("club",
                        help="night-club piece: driving floor + evolved top")
    cl.add_argument("--spine", required=True)
    cl.add_argument("--top", default="random:7",
                    help="the evolving layer's organism: a run specimen "
                    "json (see best_specimen / journey files), a genome "
                    "json, or random:N for a founder")
    cl.add_argument("--form", default="club",
                    help=f"tempo map (club is the danceable one): {sorted(FORMS)}")
    cl.add_argument("--length", type=float, default=180.0)
    cl.add_argument("--seed", type=int, default=0)
    cl.add_argument("--corpus", default="corpus")
    cl.add_argument("--voice", action="append", default=None,
                    metavar="NAME=FILE",
                    help="override a voice's sound (driver or top)")
    cl.add_argument("--driver-gain", type=float, default=1.0)
    cl.add_argument("--top-gain", type=float, default=0.5)
    cl.add_argument("--spine-gain", type=float, default=0.40)
    cl.add_argument("--duck", type=float, default=0.35,
                    help="club_lab's sidechain depth (0 = off)")
    cl.add_argument("--post", type=float, default=1.0)
    cl.add_argument("--mode", choices=("strike", "gate"), default="strike")
    cl.add_argument("--threshold", type=float, default=0.12)
    cl.add_argument("--senses", default="spikes,weather")
    cl.add_argument("--no-loop", action="store_true")
    cl.add_argument("--no-impact", action="store_true",
                    help="no crash at section boundaries")
    cl.add_argument("--no-rush-double", action="store_true",
                    help="no double-time kicks in rush sections")
    cl.add_argument("--no-top-grid", action="store_true",
                    help="let the top fire free (not quantized to the grid)")
    cl.add_argument("--top-window", default="half",
                    choices=("beat", "half", "quarter"),
                    help="the quantize window for the evolving top")
    cl.add_argument("--top-voices-per", type=int, default=4,
                    help="max top voices sounding per window")
    cl.add_argument("-o", "--out", default="maestro_out/club")
    cl.set_defaults(func=_cmd_club)

    # -- compose (the whole piece)
    cp = sub.add_parser("compose",
                        help="spine + form + organism + corpus + post -> the piece")
    cp.add_argument("--spine", required=True, help="the scaffold: any audio file")
    src = cp.add_mutually_exclusive_group()
    src.add_argument("--champion", "-c",
                     help=f"hand-built organism: {sorted(CHAMPIONS)}")
    src.add_argument("--genome", "-g", help="genome json (e.g. a run specimen)")
    src.add_argument("--random-seed", type=int, default=None,
                     help="a random score founder at this seed")
    cp.add_argument("--form", default="tectonic",
                    help=f"tempo map: {sorted(FORMS)}, a json list, or a json file")
    cp.add_argument("--length", type=float, default=180.0,
                    help="target piece length in s (form extends to reach it)")
    cp.add_argument("--seed", type=int, default=0,
                    help="the composition seed: voices, post, everything")
    cp.add_argument("--corpus", default="corpus",
                    help="corpus dir (manifest.json read for families)")
    cp.add_argument("--voice", action="append", default=None,
                    metavar="NAME=FILE", help="override a voice's sound")
    cp.add_argument("--spine-gain", type=float, default=0.45,
                    help="spine under the piece (0 = score alone)")
    cp.add_argument("--duck", type=float, default=0.28,
                    help="sidechain depth from the low voice (0 = off)")
    cp.add_argument("--post", type=float, default=1.0,
                    help="post-processing intensity dial (0 = none)")
    cp.add_argument("--no-loop", action="store_true",
                    help="don't tile a short spine across the piece")
    cp.add_argument("--mode", choices=("strike", "gate"), default="strike")
    cp.add_argument("--threshold", type=float, default=0.12)
    cp.add_argument("--senses", default="spikes,weather")
    cp.add_argument("-o", "--out", default="maestro_out/piece")
    cp.set_defaults(func=_cmd_compose)

    fs = sub.add_parser("forms", help="list the named tempo maps")
    fs.set_defaults(func=_cmd_forms)

    # -- map
    mp = sub.add_parser("map", help="score + voice->sound map -> the piece")
    mp.add_argument("score")
    mp.add_argument("--voice", action="append", default=None,
                    metavar="NAME=FILE[,gain=G][,pan=P][,speed=S][,truncate]",
                    help="map a score voice onto a sound (repeatable)")
    mp.add_argument("--map-file", default=None,
                    help="a voice-map json: {\"voices\": {...}, \"spine_gain\": ..}")
    mp.add_argument("--spine-gain", type=float, default=0.0,
                    help="mix the spine underneath (0 = score alone)")
    mp.add_argument("-o", "--out", default=None)
    mp.set_defaults(func=_cmd_map)

    # -- run (the breeding loop)
    rn = sub.add_parser("run", help="breed score organisms over a spine")
    rn.add_argument("--spine", required=True)
    rn.add_argument("-o", "--out", default="maestro_out/run1")
    rn.add_argument("--seeds", type=int, nargs="*", default=[0])
    rn.add_argument("--gens", type=int, default=240)
    rn.add_argument("--pop", type=int, default=24)
    rn.add_argument("--births", type=int, default=12)
    rn.add_argument("--arms", nargs="*", default=["novelty", "random", "greedy"])
    rn.add_argument("--mode", choices=("strike", "gate"), default="strike")
    rn.add_argument("--strictness", type=float, default=0.5,
                    help="MC band dial: 0.1 narrow, 1.0 wide, 0.5 calibrated")
    rn.add_argument("--eval-len", type=float, default=8.0,
                    help="seconds of spine per child evaluation")
    rn.add_argument("--specimen-len", type=float, default=24.0,
                    help="seconds of spine per final artifact")
    rn.add_argument("--threshold", type=float, default=0.12)
    rn.add_argument("--bpm", default="120", help="number or 'auto'")
    rn.add_argument("--tpb", "--ticks-per-beat", type=int, default=4)
    rn.set_defaults(func=_cmd_run)

    rp = sub.add_parser("report", help="print a run's report")
    rp.add_argument("run")
    rp.set_defaults(func=_cmd_report)

    # -- preview / info / champions
    pv = sub.add_parser("preview", help="score JSON -> listenable wav")
    pv.add_argument("score")
    pv.add_argument("--spine-gain", type=float, default=0.6,
                    help="0 = score alone, >0 mixes the spine underneath")
    pv.add_argument("-o", "--out", default=None)
    pv.set_defaults(func=_cmd_preview)

    inf = sub.add_parser("info", help="inspect a score")
    inf.add_argument("score")
    inf.set_defaults(func=_cmd_info)

    ch = sub.add_parser("champions", help="list the hand-built organisms")
    ch.set_defaults(func=_cmd_champions)

    args = p.parse_args(argv)
    if args.cmd == "preview":
        args.out = args.out or str(
            Path(args.score).with_name(Path(args.score).stem + "_preview.wav"))
    if args.cmd == "map":
        args.out = args.out or str(
            Path(args.score).with_name(Path(args.score).stem + "_piece.wav"))
    return args.func(args)


if __name__ == "__main__":
    main()