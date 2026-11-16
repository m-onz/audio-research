"""cli: echo index | novelty | knn | encode | bench | info."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bank import EchoBank, EchoMetric
from .encode import BLOCK_NAMES, DIM, LAYOUT, encode, encode_blocks
from .probes import SR, decode

__all__ = ["main"]


def _bank(path: str) -> EchoBank:
    return EchoBank(path)


def _encode_file(path, sr: int = SR):
    return decode(path, sr, stereo=True), encode(decode(path, sr, stereo=True), sr)


def cmd_index(a):
    files = []
    for f in a.files:
        p = Path(f)
        if p.is_dir():
            files += sorted(p.glob("*.wav"))
        else:
            files.append(p)
    bank = _bank(a.out)
    vecs, metas = [], []
    for p in files:
        try:
            v = encode(decode(p, SR, stereo=True), SR)
        except Exception as e:                           # noqa: BLE001
            print(f"  skip {p}: {e}")
            continue
        vecs.append(v)
        metas.append({"file": str(p)})
        if len(vecs) % 25 == 0:
            print(f"  encoded {len(vecs)}/{len(files)}")
    if vecs:
        bank.fit(np.stack(vecs), save=a.save_metric)
        bank.add_many(vecs, metas, role="corpus", kind="wav")
    print(f"[echo] indexed {len(vecs)} sounds -> {a.out} "
          f"({DIM} dims, metric fitted: {a.save_metric})")
    return 0


def cmd_novelty(a):
    bank = _bank(a.bank)
    v = encode(decode(a.wav, SR, stereo=True), SR)
    nov = bank.novelty(v, k=a.k)
    near = bank.knn(v, k=min(6, len(bank)))
    print(f"novelty = {nov:.4f}  (k={a.k}, archive {len(bank)})")
    if near:
        print("nearest heard:")
        for n in near:
            print(f"  {n['dist']:8.4f}  {n['meta'].get('file', n['id'])}")
    return 0


def cmd_knn(a):
    bank = _bank(a.bank)
    v = encode(decode(a.wav, SR, stereo=True), SR)
    for n in bank.knn(v, k=a.k):
        print(f"{n['dist']:8.4f}  id={n['id']}  {n['meta']}")
    return 0


def cmd_encode(a):
    audio = decode(a.wav, SR, stereo=True)
    v = encode(audio, SR)
    if a.out:
        np.savez(a.out, vec=v, blocks=json.dumps(
            {k: list(x) for k, x in encode_blocks(audio, SR).items()}))
        print(f"[echo] {a.wav} -> {a.out} ({DIM} dims)")
    else:
        blocks = encode_blocks(audio, SR)
        pos = 0
        for name, dim, w in LAYOUT:
            print(f"{name:12s} dims {pos:3d}..{pos + dim - 1:3d}  "
                  f"w={w:.2f}  |first few| "
                  f"{np.round(blocks[name][:4], 3)}")
            pos += dim
        print(f"total: {DIM} dims, finite: {bool(np.isfinite(v).all())}")
    return 0


def cmd_bench(a):
    from .bench import full_bench
    full_bench(a.corpus, out_dir=a.out, n_perturb=a.n_perturb,
               n_distant=a.n_distant, seeds=tuple(a.seeds), gens=a.gens,
               pop=a.pop, births=a.births, pairs=a.pairs,
               seconds=a.seconds, strictness=a.strictness,
               extras=(None if a.extras == "auto" else a.extras == "on"),
               verbose=True)
    return 0


def cmd_validate(a):
    from .validate import validate_corpus
    validate_corpus(a.corpus, out_dir=a.out, limit=a.limit,
                    n_probes=a.probes, n_far=a.far, n_mp3=a.mp3,
                    verbose=True)
    return 0


def cmd_info(a):
    bank = _bank(a.bank)
    st = bank.stats()
    print(json.dumps(st, indent=1))
    print(f"blocks: {', '.join(f'{n}({d})' for n, d, _ in LAYOUT)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="echo", description="the memory organ: machine listening + "
        "vector archive for novelty search in audio evolution")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("index", help="encode a folder of wavs into the bank")
    c.add_argument("files", nargs="+", help="wavs or folders of wavs")
    c.add_argument("-o", "--out", default="echo.sqlite")
    c.add_argument("--save-metric", action="store_true", default=True)
    c.set_defaults(fn=cmd_index)

    c = sub.add_parser("novelty", help="have we heard something like this?")
    c.add_argument("wav")
    c.add_argument("-b", "--bank", default="echo.sqlite")
    c.add_argument("-k", type=int, default=15)
    c.set_defaults(fn=cmd_novelty)

    c = sub.add_parser("knn", help="nearest neighbors of a sound")
    c.add_argument("wav")
    c.add_argument("-b", "--bank", default="echo.sqlite")
    c.add_argument("-k", type=int, default=10)
    c.set_defaults(fn=cmd_knn)

    c = sub.add_parser("encode", help="inspect a sound's vector anatomy")
    c.add_argument("wav")
    c.add_argument("-o", "--out", default=None)
    c.set_defaults(fn=cmd_encode)

    c = sub.add_parser("bench", help="the effectiveness suite on a corpus")
    c.add_argument("corpus")
    c.add_argument("-o", "--out", default="echo_out")
    c.add_argument("--n-perturb", type=int, default=50)
    c.add_argument("--n-distant", type=int, default=60)
    c.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    c.add_argument("--gens", type=int, default=10)
    c.add_argument("--pop", type=int, default=12)
    c.add_argument("--births", type=int, default=6)
    c.add_argument("--pairs", type=int, default=2)
    c.add_argument("--seconds", type=float, default=3.0)
    c.add_argument("--strictness", type=float, default=0.5)
    c.add_argument("--extras", default="auto",
                   choices=["auto", "on", "off"],
                   help="librosa tier (echo_plus) in the comparison")
    c.set_defaults(fn=cmd_bench)

    c = sub.add_parser("validate", help="the at-scale proof: index a whole "
                      "corpus, family retrieval, discrimination, novelty scan")
    c.add_argument("corpus")
    c.add_argument("-o", "--out", default="echo_out/validate")
    c.add_argument("--limit", type=int, default=None,
                   help="only the first N files (smoke runs)")
    c.add_argument("--probes", type=int, default=60)
    c.add_argument("--far", type=int, default=100)
    c.add_argument("--mp3", type=int, default=10)
    c.set_defaults(fn=cmd_validate)

    c = sub.add_parser("info", help="what the bank holds")
    c.add_argument("bank")
    c.set_defaults(fn=cmd_info)

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1
    return a.fn(a)