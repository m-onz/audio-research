"""cli: prism score | preview | info | champions.

    prism score --spine club.wav --champion follower -o score.json
    prism score --spine spine.wav --random --seed 7 --mode gate
    prism preview score.json -o preview.wav --spine-gain 0.6
    prism info score.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .champions import CHAMPIONS, build_champion
from .genome import Genome, random_genome
from .preview import render_preview
from .scaffold import SENSES, load_spine
from .score import load_score, score_from, score_summary, write_score

__all__ = ["main"]


def _parse_senses(s: str) -> tuple:
    senses = tuple(x.strip() for x in s.split(",") if x.strip())
    bad = [x for x in senses if x not in SENSES]
    if bad:
        raise SystemExit(f"unknown senses {bad}; valid: {SENSES}")
    return senses


def _genome(args) -> Genome:
    picks = sum(bool(x) for x in (args.champion, args.genome, args.random))
    if picks != 1:
        raise SystemExit("give exactly one of --champion, --genome, --random")
    if args.champion:
        return build_champion(args.champion)
    if args.genome:
        return Genome.from_dict(json.loads(Path(args.genome).read_text()))
    return random_genome(np.random.default_rng(args.random_seed))


def _cmd_score(args):
    genome = _genome(args)
    scaf = load_spine(args.spine, sr=args.sr,
                     bpm=args.bpm, ticks_per_beat=args.tpb,
                     tick_hz=args.tick_hz, length_s=args.length)
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
    pv = Path(out).with_name(Path(out).stem + "_preview.wav")
    render_preview(score, pv, spine_gain=args.preview_spine_gain)
    print(f"preview : {pv}  (listen: the score over "
          f"{args.preview_spine_gain:g}x spine)")


def _cmd_preview(args):
    score = load_score(args.score)
    out = render_preview(score, args.out, spine_gain=args.spine_gain)
    print(f"preview : {out}  ({score['n_events']} events over "
          f"spine_gain {args.spine_gain:g})")


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


def _cmd_champions(args):
    for name, (_, desc) in CHAMPIONS.items():
        print(f"{name:10s} {desc}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="prism",
        description="prism: the score organism — a CPPN whose inputs are "
                    "a spike-encoded spine, whose outputs are voices, whose "
                    "phenotype is a score. (NEAT Drummer's scaffolding, "
                    "open-ended edition.)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="genome + spine -> score JSON + preview")
    sc.add_argument("--spine", required=True, help="the scaffold: any audio file")
    src = sc.add_mutually_exclusive_group()
    src.add_argument("--champion", "-c", help=f"hand-built: {sorted(CHAMPIONS)}")
    src.add_argument("--genome", "-g", help="genome JSON (from a run, by hand)")
    src.add_argument("--random", action="store_true",
                     help="a random small founder (NEAT-style: minimal topology)")
    sc.add_argument("--random-seed", type=int, default=0)
    sc.add_argument("--bpm", type=float, default=120.0)
    sc.add_argument("--tpb", "--ticks-per-beat", type=int, default=4)
    sc.add_argument("--tick-hz", type=float, default=None,
                    help="direct tick rate (overrides bpm/tpb)")
    sc.add_argument("--length", type=float, default=None,
                    help="piece length in s (default: spine length)")
    sc.add_argument("--sr", type=int, default=44100)
    sc.add_argument("--mode", choices=("strike", "gate"), default="strike",
                    help="strike: every audible tick = an event (the paper's "
                         "drums). gate: rising edge holds while audible "
                         "(sustained material)")
    sc.add_argument("--threshold", type=float, default=0.12,
                    help="audibility floor (velocity below = silence)")
    sc.add_argument("--senses", default="spikes,weather",
                    help="comma list from: spikes,weather,position,time "
                         "(position/time are impure: structure not "
                         "inherited from the spine)")
    sc.add_argument("--preview-spine-gain", type=float, default=0.6)
    sc.add_argument("-o", "--out", default="prism_out/score.json")
    sc.set_defaults(func=_cmd_score)

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
    args.func(args)


if __name__ == "__main__":
    main()