"""loom command line.

    python -m loom compose corpus/ -o piece.wav --length 180 --preset storm --seed 3
    python -m loom presets
    python -m loom info corpus2/MOT-everything.wav
    python -m loom help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _json_arg(text: str | None, what: str):
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid JSON for {what}: {e}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="loom", description="algorithmic audio composition")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("compose", help="render a piece from sources")
    c.add_argument("sources", nargs="+", help=".wav file(s) or folder(s)")
    c.add_argument("-o", "--out", help="output .wav (default loom_out/<preset>_<seed>.wav)")
    c.add_argument("-l", "--length", type=float, default=None, help="length in seconds (default 120)")
    c.add_argument("--preset", default=None, help="preset name (see `loom presets`)")
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--sr", type=int, default=None, help="pipeline sample rate (default: first source's)")
    c.add_argument("--chunker", default=None, help="fixed | onset | random | beats")
    c.add_argument("--chunker-params", default=None, help="JSON dict of chunker params")
    c.add_argument("--select", default=None, help="catalog query, e.g. 'rms>0.7 and air<0.5'")
    c.add_argument("--arranger", default=None, help="sequence | cloud | euclid | layers")
    c.add_argument("--arranger-params", default=None, help="JSON dict of arranger params")
    c.add_argument("--processors", default=None, help="JSON chain spec, e.g. '[\"lowpass\", [\"echo\", {\"delay_s\": 0.3}]]'")
    c.add_argument("--passes", default=None, help="JSON list of re-processing passes")
    c.add_argument("--bits", type=int, default=None, choices=[16, 24])
    c.add_argument("--store", default=None, choices=["memory", "disk"])
    c.add_argument("--soft-clip", action="store_true")
    c.add_argument("--dry-run", action="store_true", help="arrange only; print the score, render nothing")
    c.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("presets", help="list presets")

    i = sub.add_parser("info", help="inspect wav sources")
    i.add_argument("sources", nargs="+")

    sub.add_parser("help", help="list processors, arrangers, features")

    a = p.parse_args(argv)
    if a.cmd is None or a.cmd not in ("compose", "presets", "info", "help"):
        p.print_help()
        return 1

    if a.cmd == "presets":
        from .presets import PRESETS

        docs = {
            "drift": "ambient granular: onset chunks, darkening lowpass, thinning cloud",
            "storm": "maximal-contrast collage: farthest-point order, tape-slow arc",
            "pulse": "rhythm: bass Euclid pulse with texture raining above",
            "ghost": "re-process loud chunks into a dark echo, walk it markov-style",
            "strata": "feature-separated layers: lows walk, mids pulse, highs drift",
            "collage": "random windows chased by similarity, drifting speed",
        }
        for name in sorted(PRESETS):
            print(f"{name:<10} {docs.get(name, '')}")
        return 0

    if a.cmd == "info":
        from .io import wav_info

        for src in a.sources:
            path = Path(src)
            if path.is_dir():
                wavs = sorted(path.glob("*.wav"))
                print(f"{path}/ : {len(wavs)} wav files")
                for w in wavs:
                    info = wav_info(w)
                    print(f"  {w.name:<40} {info['sr']}Hz {info['channels']}ch {info['seconds']:.1f}s {info['subtype']}")
            else:
                info = wav_info(path)
                print(f"{path}: {info['sr']}Hz {info['channels']}ch {info['seconds']:.1f}s {info['subtype']}")
        return 0

    if a.cmd == "help":
        from .arrangers import arranger_help
        from .chunkers import list_chunkers
        from .features import feature_help
        from .processors import processor_help

        print(processor_help())
        print()
        print("chunkers:", ", ".join(list_chunkers()))
        print()
        print(arranger_help())
        print()
        print(feature_help())
        return 0

    # compose
    from . import compose as _compose
    from .presets import PRESETS

    kw = {}
    if a.length is not None:
        kw["length"] = a.length
    if a.seed is not None:
        kw["seed"] = a.seed
    if a.sr is not None:
        kw["sr"] = a.sr
    if a.chunker is not None:
        kw["chunker"] = a.chunker
    if a.chunker_params is not None:
        kw["chunker_params"] = _json_arg(a.chunker_params, "--chunker-params")
    if a.select is not None:
        kw["select"] = a.select
    if a.arranger is not None:
        kw["arranger"] = a.arranger
    if a.arranger_params is not None:
        kw["arranger_params"] = _json_arg(a.arranger_params, "--arranger-params")
    if a.processors is not None:
        kw["processors"] = _json_arg(a.processors, "--processors")
    if a.passes is not None:
        kw["passes"] = _json_arg(a.passes, "--passes")
    if a.bits is not None:
        kw["bits"] = a.bits
    if a.store is not None:
        kw["store"] = a.store
    if a.out is not None:
        kw["out"] = a.out
    if a.soft_clip:
        kw["soft_clip"] = True

    try:
        result = _compose(
            sources=a.sources,
            preset=a.preset,
            verbose=a.verbose,
            dry_run=a.dry_run,
            **kw,
        )
    except Exception as e:
        print(f"loom: error: {e}", file=sys.stderr)
        raise

    print(result.summary())
    if a.dry_run and result.out is None:
        score_json = result.score.to_json()
        print(f"score written to memory; {score_json['n_events']} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())