"""cli: prism render | picture | info | champions | modes | random
            | seed | report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .champions import CHAMPIONS, anatomy, champion
from .genome import Genome, random_genome
from .render import MODES, SR, picture, render, write_png, write_wav


def _load(target: str):
    """A champion by name, or a genome json from disk."""
    if target in CHAMPIONS:
        g = champion(target)
        return g, CHAMPIONS[target]["mode"]
    p = Path(target)
    if not p.exists():
        raise SystemExit(f"no champion named '{target}' and no file at '{p}'")
    d = json.loads(p.read_text())
    gd = d["genome"] if "genome" in d else d
    return Genome.from_dict(gd), d.get("mode", "spectra")


def cmd_render(a):
    g, mode = _load(a.target)
    mode = a.mode if a.mode != "auto" else mode
    audio = render(g, length=a.length, sr=a.sr, mode=mode, seed=a.seed,
                   f0=a.f0, morph_hz=a.morph_hz)
    out = a.out or f"{Path(a.target).stem}_{mode}_{int(a.length)}s.wav"
    write_wav(out, audio, a.sr)
    rms = float((audio ** 2).mean() ** 0.5)
    print(f"[prism] {Path(a.target).stem} · {mode} · {len(audio)/a.sr:.1f}s "
          f"· rms {rms:.3f} · {out}")
    return 0


def cmd_picture(a):
    g, _ = _load(a.target)
    w, h = (int(v) for v in a.size.split("x"))
    img = picture(g, width=w, height=h, t=a.t)
    out = a.out or f"{Path(a.target).stem}_{w}x{h}.png"
    write_png(out, img)
    print(f"[prism] {Path(a.target).stem} · picture {w}x{h} · {out}")
    return 0


def cmd_info(a):
    g, mode = _load(a.target)
    info = anatomy(g)
    print(f"genome seed {info['seed']} · preferred traversal {mode}")
    print(f"  senses wired : {', '.join(info['senses'])}")
    print(f"  nodes {info['nodes']} · connections {info['conns']}")
    print(f"  functions     : "
          + " ".join(f"{k}×{v}" for k, v in sorted(info["functions"].items())))
    print(f"  outputs      : {', '.join(info['outputs'])}  (the alphabet)")
    return 0


def cmd_champions(_a):
    for name, spec in CHAMPIONS.items():
        print(f"  {name:12s} {spec['mode']:9s} {spec['blurb']}")
    return 0


def cmd_modes(_a):
    for m in MODES:
        print(f"  {m}")
    return 0


def cmd_random(a):
    import numpy as np
    g = random_genome(np.random.default_rng(a.seed))
    d = {"genome": g.to_dict(), "mode": a.mode,
         "born": "random_genome", "seed": a.seed}
    out = a.out or f"prism_random_{a.seed}.json"
    Path(out).write_text(json.dumps(d, indent=1))
    print(f"[prism] founder written · {out}")
    return 0


def cmd_seed(a):
    from .engine import ARMS, run_experiment
    run_experiment(out=a.out, seeds=tuple(a.seeds), gens=a.gens,
                   pop=a.pop, births=a.births, arms=tuple(a.arms),
                   organism=a.organism, strictness=a.strictness,
                   eval_len=a.eval_len, verbose=True)
    return 0


def cmd_report(a):
    from .engine import report_run
    print(report_run(a.run))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="prism",
        description="the pattern organism — one CPPN, many phenotypes: "
                    "the same genome draws the picture and plays the sound.")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("render", help="a genome → audio")
    c.add_argument("target", help="champion name or genome json path")
    c.add_argument("-l", "--length", type=float, default=30.0)
    c.add_argument("--mode", default="auto", choices=["auto", *MODES])
    c.add_argument("--sr", type=int, default=SR)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--f0", type=float, default=110.0,
                   help="wavetable fundamental")
    c.add_argument("--morph-hz", type=float, default=0.02,
                   help="wavetable morph path rate")
    c.add_argument("-o", "--out")
    c.set_defaults(fn=cmd_render)

    c = sub.add_parser("picture", help="a genome → PNG")
    c.add_argument("target")
    c.add_argument("--size", default="1280x720")
    c.add_argument("--t", type=float, default=0.0,
                   help="the frozen slow-clock reading")
    c.add_argument("-o", "--out")
    c.set_defaults(fn=cmd_picture)

    c = sub.add_parser("info", help="genome anatomy, off the genome")
    c.add_argument("target")
    c.set_defaults(fn=cmd_info)

    c = sub.add_parser("champions", help="list the three hand-built genomes")
    c.set_defaults(fn=cmd_champions)

    c = sub.add_parser("modes", help="list the traversals")
    c.set_defaults(fn=cmd_modes)

    c = sub.add_parser("random", help="write a random founder genome json")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--mode", default="spectra", choices=MODES)
    c.add_argument("-o", "--out")
    c.set_defaults(fn=cmd_random)

    c = sub.add_parser("seed", help="breed: arms × seeds × generations")
    c.add_argument("-o", "--out", default="prism_out/b1")
    c.add_argument("--seeds", type=int, nargs="*", default=[0])
    c.add_argument("--gens", type=int, default=240)
    c.add_argument("--pop", type=int, default=24)
    c.add_argument("--births", type=int, default=12)
    c.add_argument("--arms", nargs="*",
                    default=["novelty", "random", "greedy"])
    c.add_argument("--organism", default="spectra", choices=MODES)
    c.add_argument("--strictness", type=float, default=0.5,
                    help="MC band dial: 0.1 narrow, 1.0 wide, 0.5 calibrated")
    c.add_argument("--eval-len", type=float, default=4.0)
    c.set_defaults(fn=cmd_seed)

    c = sub.add_parser("report", help="print a run's report")
    c.add_argument("run")
    c.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())