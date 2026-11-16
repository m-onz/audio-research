"""tomb command line.

    tomb run --batches 500 --tau 0.15 --seed 0
    tomb info tomb_out/checkpoint_000200/
"""

import argparse
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tomb",
                                 description="in-place audio novelty search (only the discriminable survive)")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run the coevolution")
    r.add_argument("-o", "--out", default="tomb_out")
    r.add_argument("--batches", type=int, default=500)
    r.add_argument("--tau", type=float, default=0.15,
                   help="strictness dial: higher = harder to admit (Soros)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--seconds", type=float, default=2.0,
                   help="sound duration")
    r.add_argument("--sound-capacity", type=int, default=250)
    r.add_argument("--ear-capacity", type=int, default=24)
    r.add_argument("--genesis-batches", type=int, default=50,
                   help="how many batches stay in genesis phase (structure-creators weighted)")
    r.add_argument("--checkpoint-every", type=int, default=200)
    r.add_argument("--founders", type=int, default=30)
    r.add_argument("-v", "--verbose", action="store_true")

    i = sub.add_parser("info", help="inspect a checkpoint")
    i.add_argument("checkpoint_dir")

    sub.add_parser("ops", help="list the material process operators")

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1

    if a.cmd == "ops":
        import tomb

        for fam, members in tomb.FAMILIES.items():
            names = ", ".join(n.strip("_") for n, _ in members)
            print(f"  {fam:10s} {names}")
        return 0

    if a.cmd == "info":
        import json

        m = json.loads((Path(a.checkpoint_dir) / "manifest.json").read_text())
        print(f"batch {m['batch']}: {m['n_sounds']} sounds, {m['n_ears']} ears")
        print(f"  species: {m['species']}")
        print(f"  complexity: {m['complexity_range'][0]}-{m['complexity_range'][1]} ops")
        print(f"  admitted: {m['sounds_admitted']} / {m['evals']} evals")
        wavs = sorted(Path(a.checkpoint_dir).glob("*.wav"))
        print(f"  sounds: {len(wavs)} wavs in {a.checkpoint_dir}/")
        return 0

    # run
    import tomb

    cfg = tomb.TombConfig(
        seed=a.seed, tau=a.tau, sound_capacity=a.sound_capacity,
        ear_capacity=a.ear_capacity, seconds=a.seconds,
        genesis_batches=a.genesis_batches,
        checkpoint_every=a.checkpoint_every,
    )
    t = tomb.Tomb(cfg, out_dir=a.out)
    t.genesis(n_founders=a.founders, verbose=a.verbose)
    result = t.run(batches=a.batches, verbose=a.verbose)
    print(f"final checkpoint: {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())