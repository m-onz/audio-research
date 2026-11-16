"""cli: scan champions | render | random | info | audition."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import (WtGenome, render, random_genome, mutate, viable, tonality,
               write_wav, to_dict, from_dict, champion, blurb, stats,
               CHAMPIONS, SECONDS, SR)


def _load_target(name: str):
    """A champion name (wt or sub) or a path to a genome json."""
    from .champions import CHAMPIONS
    from .substrate import SUB_CHAMPIONS, champion as sub_champion
    if name in CHAMPIONS:
        return champion(name)
    if name in SUB_CHAMPIONS:
        return sub_champion(name)
    p = Path(name)
    if not p.exists():
        raise SystemExit(f"no champion '{name}' and no file '{name}'")
    with open(p) as f:
        d = json.load(f)
    if d.get("scan") == "subgenome":
        from .substrate import from_dict as sub_from
        return sub_from(d)
    return from_dict(d)


def cmd_champions(args):
    for name in CHAMPIONS:
        g = champion(name)
        a = g.anatomy()
        print(f"{name:12s} {blurb(name)}")
        print(f"             frames={a['frames']} partials={a['partials']} "
              f"morph={a['morph']} pitch={a['pitch']} f0={a['f0']}Hz "
              f"voices={a['voices']} -> {a['archetype']}")
    print("\nlisten:  scan render surfer -l 8 -o scan_out/surfer.wav")


def cmd_render(args):
    g = _load_target(args.target)
    if args.device:
        import scan.substrate as sub
        sub.set_device(args.device)
    from .substrate import render as sub_render
    from .organism import render as wt_render
    render_fn = sub_render if hasattr(g, "cppn") else wt_render
    t0 = time.time()
    audio = render_fn(g, seconds=args.length, hi=args.hi)
    dt = (time.time() - t0) * 1000.0
    out = write_wav(args.out, audio)
    st = stats(audio)
    print(f"{out}  {audio.shape[0] / SR:.1f}s  render {dt:.0f} ms  "
          f"rms {st['rms']} centroid {st['centroid_hz']}Hz "
          f"crest {st['crest']}  archetype {g.archetype()}")


def cmd_random(args):
    rng = np.random.default_rng(args.seed)
    g = random_genome(rng, seed=int(rng.integers(1 << 30)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(to_dict(g), f, indent=1)
        print(f"{args.out}  {g.archetype()}")
    else:
        print(json.dumps(to_dict(g), indent=1))


def cmd_audition(args):
    """Render n random founders into a directory — You are the judge."""
    rng = np.random.default_rng(args.seed)
    made, tries = 0, 0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    while made < args.n and tries < args.n * 30:
        tries += 1
        g = random_genome(rng, seed=int(rng.integers(1 << 30)))
        audio = render(g, seconds=SECONDS)
        if not viable(audio):
            continue
        made += 1
        name = f"spec_{made:02d}_{g.archetype()}.wav"
        write_wav(Path(args.out) / name, audio)
        with open(Path(args.out) / f"{name[:-4]}.json", "w") as f:
            json.dump(to_dict(g), f, indent=1)
        print(f"{name}  tonality {tonality(g):.1f}")
    print(f"{made} viable specimens from {tries} tries -> {args.out}/")


def cmd_info(args):
    g = _load_target(args.target)
    a = g.anatomy()
    print(f"archetype   {a['archetype']}")
    print(f"tonality    {tonality(g):.2f}  (the greedy trap: higher = purer)")
    print(f"frames      {a['frames']}  partials {a['partials']}  "
          f"inharm {a['inharm_mean']}")
    print(f"pitch       {a['pitch']}  f0={a['f0']}Hz")
    print(f"morph       {a['morph']}")
    print(f"voices      {a['voices']}  detunes {a['detunes']}")
    print(f"vector      {a['vector']}")
    print(f"seed        {g.seed}")


def cmd_run(args):
    from .engine import run_experiment
    if args.device:
        import scan.substrate as sub
        sub.set_device(args.device)
    run_experiment(out=args.out, seeds=tuple(args.seeds), gens=args.gens,
                   pop=args.pop, births=args.births,
                   arms=tuple(args.arms), verbose=True,
                   jobs=args.jobs, organism_name=args.organism)


def cmd_report(args):
    from .engine import report_run
    summary = json.loads((Path(args.run) / "experiment.json").read_text())
    print(report_run(summary))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scan",
                                 description="the wavetable arena (phase A)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("champions", help="the three known answers")
    p.set_defaults(fn=cmd_champions)

    p = sub.add_parser("render", help="genome/champion -> wav")
    p.add_argument("target", help="champion name or genome json path")
    p.add_argument("-l", "--length", type=float, default=SECONDS)
    p.add_argument("-o", "--out", default="scan_out/render.wav")
    p.add_argument("--hi", action="store_true",
                   help="render at full rate (the aliasing check path)")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("random", help="a Phase B founder")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(fn=cmd_random)

    p = sub.add_parser("audition", help="render n random founders, listen")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("-n", type=int, default=8)
    p.add_argument("-o", "--out", default="scan_out/audition")
    p.set_defaults(fn=cmd_audition)

    p = sub.add_parser("info", help="a genome's anatomy")
    p.add_argument("target", help="champion name or genome json path")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("run", help="breed the four arms (phase B)")
    p.add_argument("-o", "--out", default="scan_out/run1")
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--births", type=int, default=8)
    p.add_argument("--arms", nargs="*",
                   default=["greedy", "random", "cell", "novelty"])
    p.add_argument("--organism", default="wt", choices=["wt", "sub"],
                   help="wt = direct frames | sub = DeepHyperNEAT "
                        "substrate frames (torch, GPU-capable)")
    p.add_argument("--device", default=None,
                   choices=["auto", "cpu", "mps"],
                   help="substrate evaluation device (sub organism)")
    p.add_argument("--jobs", type=int, default=None,
                   help="parallel (arm, seed) processes; default = one "
                        "per core; 1 = serial. Results are identical.")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="print a run's report")
    p.add_argument("run")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()