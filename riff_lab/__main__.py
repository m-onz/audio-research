"""CLI:
    python3 -m riff_lab gen --n 6 [--bars 8] [--genre dnb] [--bpm 174]
    python3 -m riff_lab breed riffs/gen_001/riff_002 riffs/gen_001/riff_005 --n 6
    python3 -m riff_lab mutate riffs/gen_001/riff_003 --n 4
    python3 -m riff_lab pool --rebuild
"""
import argparse
import warnings

warnings.filterwarnings("ignore")          # short slices trip scipy nperseg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="riff_lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g1 = sub.add_parser("gen", help="random riff population (novelty search)")
    g1.add_argument("--n", type=int, default=6)
    g1.add_argument("--bars", type=int, default=None)
    g1.add_argument("--genre", default=None, choices=["dnb", "halftime"])
    g1.add_argument("--bpm", type=float, default=None)
    g1.add_argument("--candidates", type=int, default=3,
                    help="novelty search: generate N x this candidates "
                         "per generation (0 = no novelty)")
    g1.add_argument("--gens", type=int, default=1,
                    help="extended novelty run: N consecutive generations, "
                         "each scored against the growing archive")
    g1.add_argument("--strictness", type=float, default=None,
                    help="override the character-gate dial for this run "
                         "(0 = only technical defects rejected — fully "
                         "open; 1 = tightest). Riff genomes also evolve "
                         "their own strictness.")

    g5 = sub.add_parser("judge", help="calibrate the ears to YOUR taste")
    g5.add_argument("--fit", action="store_true",
                    help="fit taste thresholds from riffs/_taste/good/ "
                         "vs riffs/_taste/bad/ (drop wavs in while "
                         "auditioning)")
    g5.add_argument("--reset", action="store_true",
                    help="delete the learned taste")

    g2 = sub.add_parser("breed", help="crossover + mutate from riffs you like")
    g2.add_argument("parents", nargs="+")
    g2.add_argument("--n", type=int, default=6)
    g2.add_argument("--rate", type=float, default=0.12)

    g3 = sub.add_parser("mutate", help="variations of one riff")
    g3.add_argument("parent")
    g3.add_argument("--n", type=int, default=4)
    g3.add_argument("--rate", type=float, default=0.2)

    g4 = sub.add_parser("pool", help="rebuild the sound pool cache")
    g4.add_argument("--rebuild", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "gen":
        from . import ga
        ga.gen_random(args.n, bars=args.bars, genre=args.genre, bpm=args.bpm,
                      candidates_x=max(0, args.candidates), gens=args.gens,
                      strictness=args.strictness)
    elif args.cmd == "judge":
        from . import ears
        if args.reset:
            ears.TASTE_PATH.unlink(missing_ok=True)
            print("taste reset")
        if args.fit:
            ears.fit_taste()
    elif args.cmd == "breed":
        from . import ga
        ga.breed(args.parents, n=args.n, mutation_rate=args.rate)
    elif args.cmd == "mutate":
        from . import ga
        ga.mutate_one(args.parent, n=args.n, mutation_rate=args.rate)
    elif args.cmd == "pool":
        from . import pool
        pool.build(rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())