"""CLI: python -m oneshot_lab INPUT [-o OUTPUT] [options]"""
import argparse
import sys

from .engine import describe, plan, run_folder
from .registry import FAMILIES


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="oneshot_lab",
        description="Mutate a folder of oneshots into a bank of unique, weird "
                    "variations: time stretch (paulstretch, PV, wavetable), "
                    "pitch & harmonics, spectral transformations, glitch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", nargs="?", help="folder of source sounds "
                    "(default: ./oneshots)")
    ap.add_argument("-o", "--out", default="oneshot_out",
                    help="output folder")
    ap.add_argument("-t", "--transforms", default="all",
                    help="comma list of transform names, families "
                    f"({', '.join(sorted(FAMILIES))}), or 'all'")
    ap.add_argument("-r", "--random-chains", type=int, default=0,
                    help="extra random mutation chains per file "
                         "(2-3 stacked transforms, random params)")
    ap.add_argument("--variants", type=int, default=1,
                    help="variants per transform (v0 = canonical defaults, "
                         "extra variants get random params)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-in-s", type=float, default=25.0,
                    help="inputs longer than this are sliced to a window")
    ap.add_argument("--max-out-s", type=float, default=8.0,
                    help="cap on any output duration")
    ap.add_argument("-R", "--recursive", action="store_true",
                    help="scan input folder recursively (reprocess a bank "
                         "whose sounds live in transform subfolders)")
    ap.add_argument("--pick", type=int, default=None,
                    help="use only N randomly-picked input files (seeded)")
    ap.add_argument("--pick-per-folder", type=int, default=None,
                    help="use N randomly-picked files from each input "
                         "subfolder (balanced sampling across a bank)")
    ap.add_argument("-w", "--workers", type=int, default=None,
                    help="parallel workers (default: cpu-1)")
    ap.add_argument("--list", action="store_true", help="list all transforms")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and count jobs without running")
    args = ap.parse_args(argv)

    if args.list:
        print(describe())
        return 0
    if not args.input:
        ap.error("input folder required (e.g. ./oneshots)")

    names = [s.strip() for s in args.transforms.split(",") if s.strip()]
    if args.dry_run:
        files, jobs = plan(args.input, args.out, names,
                           args.random_chains, args.seed,
                           args.max_in_s, args.max_out_s, args.variants,
                           recursive=args.recursive, pick=args.pick,
                           pick_per_folder=args.pick_per_folder)
        per = {}
        for j in jobs:
            per[j["name"]] = per.get(j["name"], 0) + 1
        print(f"{len(files)} files, {len(jobs)} jobs planned:")
        for k in sorted(per):
            print(f"  {k:<24} {per[k]}")
        return 0

    try:
        run_folder(args.input, args.out, names, args.random_chains, args.seed,
                   args.max_in_s, args.max_out_s, args.variants,
                   workers=args.workers, recursive=args.recursive,
                   pick=args.pick, pick_per_folder=args.pick_per_folder)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())