"""cli: eden run | report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_run(a):
    from .engine import run_experiment
    run_experiment(out=a.out, seeds=tuple(a.seeds), gens=a.gens,
                   pop=a.pop, births=a.births, arms=tuple(a.arms),
                   verbose=True, workers=a.workers)
    return 0


def cmd_report(a):
    from .engine import report_run
    summary = json.loads((Path(a.run) / "experiment.json").read_text())
    print(report_run(summary))
    return 0


def cmd_render(a):
    """Render one random world (or a seed's founder) for listening."""
    import numpy as np
    from .genetics import random_world
    from .world import render, SECONDS, SR
    from echo.probes import write_wav
    rng = np.random.default_rng(a.rng)
    for i in range(a.n):
        g = random_world(rng, seed=int(rng.integers(1 << 30)))
        audio, tele = render(g, seconds=SECONDS)
        out = Path(a.out) / f"world_{i:02d}_{g.archetype()}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out, audio, SR)
        print(f"{out}  agents={tele['agents']} "
              f"collisions={tele['collisions']} "
              f"path_len={tele['path_len']:.2f} "
              f"voice_nodes={tele['voice_nodes']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="eden",
        description="a little world of singing agents, heard through a "
        "wandering listener — spatial multi-agent synthesis evolved by "
        "echo-driven novelty search")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("run", help="breed worlds under both memories")
    c.add_argument("-o", "--out", default="eden_out/run1")
    c.add_argument("--seeds", type=int, nargs="*", default=[0])
    c.add_argument("--gens", type=int, default=200)
    c.add_argument("--pop", type=int, default=12)
    c.add_argument("--births", type=int, default=8)
    c.add_argument("--arms", nargs="*", default=["random", "novelty"])
    c.add_argument("--workers", type=int, default=None,
                   help="render pool size (default: cpu_count-2, capped 6; "
                        "1 = serial, no pool)")
    c.set_defaults(fn=cmd_run)

    c = sub.add_parser("report", help="print a run's report")
    c.add_argument("run")
    c.set_defaults(fn=cmd_report)

    c = sub.add_parser("render", help="render random worlds for listening")
    c.add_argument("-o", "--out", default="eden_out/worlds")
    c.add_argument("-n", type=int, default=4)
    c.add_argument("--rng", type=int, default=0)
    c.set_defaults(fn=cmd_render)

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1
    return a.fn(a)