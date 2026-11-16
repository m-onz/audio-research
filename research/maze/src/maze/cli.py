"""cli: maze run | report."""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_run(a):
    from .engine import run_experiment
    run_experiment(out=a.out, seeds=tuple(a.seeds), gens=a.gens,
                   pop=a.pop, births=a.births,
                   arms=tuple(a.arms), organism=a.organism, verbose=True)
    return 0


def cmd_report(a):
    from .engine import report_run
    import json
    summary = json.loads((Path(a.run) / "experiment.json").read_text())
    print(report_run(summary))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="maze",
        description="the audio maze: objective vs drift vs memory — "
        "a minimal voice organism bred by four selection policies; "
        "the one with echo's archive is the one that goes somewhere.")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("run", help="breed the four arms")
    c.add_argument("-o", "--out", default="maze_out/run1")
    c.add_argument("--seeds", type=int, nargs="*", default=[0])
    c.add_argument("--gens", type=int, default=40)
    c.add_argument("--pop", type=int, default=16)
    c.add_argument("--births", type=int, default=8)
    c.add_argument("--arms", nargs="*", default=["greedy", "random", "cell", "novelty"])
    c.add_argument("--organism", default="dx", choices=["dx", "bumps"],
                   help="dx = FM operators (vast nonlinear timbre space), "
                        "bumps = the original spectral-bump voice")
    c.set_defaults(fn=cmd_run)

    c = sub.add_parser("report", help="print a run's report")
    c.add_argument("run")
    c.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1
    return a.fn(a)