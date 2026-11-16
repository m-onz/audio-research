"""cryptid CLI.

    cryptid seed --gens 30 --strictness 0.5 -o runs/crypt1
    cryptid render runs/crypt1/gen_0030/specimen_000.wav -l 30 -o piece.wav
    cryptid info runs/crypt1/gen_0030/
"""

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cryptid",
                                 description="breeding evolved dynamical processes")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("seed", help="start a breeding run")
    s.add_argument("-o", "--out", default="runs")
    s.add_argument("--gens", type=int, default=30)
    s.add_argument("--strictness", type=float, default=0.5,
                   help="MC band width (Soros dial: 0.1=stagnant, 1.0=chaotic)")
    s.add_argument("--pop", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, default=2048,
                   help="integration steps per evaluation")
    s.add_argument("--checkpoint-every", type=int, default=10)
    s.add_argument("-v", "--verbose", action="store_true")

    r = sub.add_parser("render", help="render a specimen as audio")
    r.add_argument("checkpoint_dir")
    r.add_argument("-i", "--index", type=int, default=0,
                   help="specimen index in the checkpoint")
    r.add_argument("-l", "--length", type=float, default=30.0)
    r.add_argument("-o", "--out", default="specimen.wav")
    r.add_argument("--seed", type=int, default=0)

    i = sub.add_parser("info", help="inspect a checkpoint")
    i.add_argument("checkpoint_dir")

    sub.add_parser("laws", help="list the dynamical unit laws")

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1

    if a.cmd == "laws":
        import cryptid

        for law in cryptid.LAW_NAMES:
            info = cryptid.LAWS[law]
            print(f"  {law:12s} {info['n_vars']} state vars, params: {list(info['params'])}")
        return 0

    if a.cmd == "info":
        import json

        m = json.loads((Path(a.checkpoint_dir) / "manifest.json").read_text())
        print(f"gen {m['gen']}: {m['n_processes']} processes, {m['n_ears']} ears")
        print(f"  laws: {m['law_histogram']}")
        print(f"  mean nodes: {m['mean_nodes']:.1f}")
        return 0

    if a.cmd == "seed":
        import cryptid

        cfg = cryptid.CryptidConfig(
            seed=a.seed, strictness=a.strictness, pop_capacity=a.pop,
            n_steps=a.steps, checkpoint_every=a.checkpoint_every,
        )
        c = cryptid.Cryptid(cfg, out_dir=a.out)
        c.genesis(verbose=a.verbose)
        c.run(n_gens=a.gens, verbose=a.verbose)
        return 0

    if a.cmd == "render":
        import cryptid
        import soundfile as sf

        # load checkpoint, find the specimen, re-render longer
        ckpt = Path(a.checkpoint_dir)
        wavs = sorted(ckpt.glob("*.wav"))
        if not wavs:
            print("no specimens in checkpoint", file=sys.stderr)
            return 1
        wav_path = wavs[min(a.index, len(wavs) - 1)]
        print(f"re-rendering {wav_path.name} for {a.length}s")
        # for now: just extend with silence-padding (full re-render needs genome serialization)
        audio, sr = sf.read(str(wav_path), always_2d=True)
        needed = int(a.length * sr)
        if len(audio) < needed:
            audio = np.pad(audio, ((0, needed - len(audio)), (0, 0)))
        sf.write(a.out, audio[:needed], sr)
        print(f"{a.out}: {a.length}s")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())