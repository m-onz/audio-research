"""cli: quilt index | make | render | info | strategies | modes | doctor

    quilt index corpus/ -o quilt_out/bank.sqlite   # cut, encode, fit
    quilt make -b quilt_out/bank.sqlite --mode roam --seed 3
    quilt render piece.wav.route.json -o other.wav --crossfade 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from echo.probes import SR

from . import __version__, DEFAULT_STRATEGIES, STRATEGIES, MODES
from .split import split_corpus
from .index import build_bank, provenance_path
from .atlas import ChunkAtlas
from .navigate import walk
from .render import render_route, write_route_json, load_route

_REPO = Path(__file__).resolve().parents[3]           # .../izark_newtarn
_WOMB = _REPO / "experiments" / "womb"


# ------------------------------------------------------------------ commands

def cmd_strategies(a):
    print("the scissors:")
    for name, s in STRATEGIES.items():
        star = " *" if name in DEFAULT_STRATEGIES else ""
        print(f"  {name:10s} {s['doc']}{star}")
    return 0


def cmd_modes(a):
    print("the walkers:")
    for name, doc in MODES.items():
        print(f"  {name:10s} {doc}")
    return 0


def cmd_split(a):
    split_corpus(a.corpus, min_keep=a.min_keep,
                 fixed_dur=a.fixed_dur, max_files=a.max_files,
                 dry_run=a.dry_run, audit_path=a.audit,
                 workers=a.workers, log=lambda s: print(s, flush=True))
    return 0


def cmd_index(a):
    build_bank(a.corpus, a.out, max_files=a.max_files,
               force=a.force, workers=a.workers,
               log=lambda s: print(s, flush=True))
    return 0


def cmd_make(a):
    strat = None if a.strategy == "all" else [a.strategy]
    atlas = ChunkAtlas(a.bank, strategies=strat,
                       corpus_root=a.corpus_root, device=a.device)
    if a.novel is not None and a.novel < 1.0:
        atlas = atlas.novelty_view(keep=a.novel)
        print(f"novelty view: the {atlas.n} strangest pieces "
              f"(top {a.novel:.1%} by echo novelty)", flush=True)
    route = walk(atlas, mode=a.mode, seconds=a.seconds, seed=a.seed,
                exclude_window=a.exclude_window, temp=a.temp, k=a.k,
                speed=a.speed, contrast_window=a.contrast_window,
                branch=a.branch, amp=a.amp, voices=a.voices,
                grain_lo=a.grain_lo, grain_hi=a.grain_hi, inner=a.inner)
    out = Path(a.out) if a.out else (
        Path("quilt_out") / "pieces" /
        f"{a.mode}_{a.strategy}_{a.seed}_{int(a.seconds)}s.wav")
    route_path = Path(str(out) + ".route.json")
    write_route_json(route.to_json(atlas), route_path)
    rep = render_route(load_route(route_path), out,
                       crossfade_ms=a.crossfade, exact=not a.no_exact,
                       corpus_root=str(atlas.corpus_root))
    st = route.stats
    print(f"mode={route.mode} seed={route.seed} "
          f"pieces={st['n_steps']} total={st['total_dur']:.1f}s")
    print(f"step distance: median {st['dist_prev_median']} vs random pair "
          f"{st['random_pair_median']} (temperature {st['temperature']})")
    print(f"wrote {rep['out']} ({rep['duration_s']}s) + {route_path}")
    return 0


def cmd_render(a):
    route = load_route(a.route)
    rep = render_route(route, a.out or _default_out(route),
                       crossfade_ms=a.crossfade, exact=not a.no_exact,
                       corpus_root=a.corpus_root)
    print(f"wrote {rep['out']} ({rep['duration_s']}s, "
          f"{rep['pieces']} pieces, crossfade {rep['crossfade_ms']}ms)")
    return 0


def _default_out(route) -> str:
    src = Path(route.get("bank", "piece"))
    return str(Path("quilt_out") / "pieces"
               / f"render_{route.get('mode')}_{src.stem}.wav")


def cmd_info(a):
    atlas = ChunkAtlas(a.bank, strategies=None,
                       corpus_root=a.corpus_root, device=a.device)
    print(json.dumps(atlas.stats(), indent=1))
    prov = atlas.provenance
    if prov:
        print(f"indexed: {prov.get('files_seen')} files "
              f"({prov.get('files_failed')} failed), "
              f"{prov.get('elapsed_s')}s, "
              f"scissors {','.join(prov.get('strategies', []))}")
    return 0


def cmd_doctor(a):
    print(f"quilt {__version__} — the seamstress organ")
    ok = lambda b: "yes" if b else "NO"
    print(f"  ffmpeg:        {ok(_run_ok('ffmpeg', '-version'))}")
    print(f"  echo ear:      {ok(importable('echo'))} "
          f"(556-D vectors, fitted metric)")
    print(f"  librosa:       {ok(importable('librosa'))} "
          f"(the onset scissor)")
    # RAVE — the optional neural ear (see README rave). Not a dependency
    # of the gluing; reported because the ecosystem plans it as an
    # extension point (echo's NeuralBlock, womb's local training).
    rave_bin = _WOMB / "neural" / "bin" / "rave"
    print(f"  rave venv:     {ok(rave_bin.exists())} ({rave_bin})")
    ckpts = sorted(_WOMB.rglob("*.ckpt")) if _WOMB.exists() else []
    print(f"  rave ckpt:     {len(ckpts)} trained model(s) under womb/runs")
    if not ckpts:
        print("  -> quilt does not need RAVE: it glues original audio, it")
        print("     never decodes latents. To add RAVE latents as one more")
        print("     weighted block of the ear (echo's NeuralBlock plan):")
        print("     cd experiments/womb && womb rave preprocess corpus/ "
              "--db db && womb rave train db --name quilt_ear")
    return 0


def _run_ok(cmd, *args):
    try:
        subprocess.run([cmd, *args], capture_output=True, check=True)
        return True
    except Exception:
        return False


def importable(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="quilt", description="the seamstress organ: chop the corpus "
        "with several scissors, pin every piece in echo's feature space, "
        "walk the territory, and glue what you find — no processing, "
        "just re-ordering")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("strategies", help="list the scissors")
    c.set_defaults(fn=cmd_strategies)

    c = sub.add_parser("modes", help="list the walkers")
    c.set_defaults(fn=cmd_modes)

    c = sub.add_parser("split", help="cut the corpus in place -> the pile "
                      "(originals replaced by their chunks)")
    c.add_argument("corpus")
    c.add_argument("--min-keep", type=float, default=8.0,
                   help="continuous files up to this long stay whole (s)")
    c.add_argument("--fixed-dur", type=float, default=2.0,
                   help="window for the fixed scissor on long files (s)")
    c.add_argument("--max-files", type=int, default=None,
                   help="only the first N unsplit originals (resumable)")
    c.add_argument("--dry-run", action="store_true",
                   help="report what would happen; touch nothing")
    c.add_argument("--audit", default="quilt_out/split_audit.jsonl",
                   help="JSONL record of every decision")
    c.add_argument("--workers", type=int, default=None,
                   help="parallel split workers (default: auto)")
    c.set_defaults(fn=cmd_split)

    c = sub.add_parser("index", help="the pile -> feature-space bank "
                      "(no chunking: files are already pieces)")
    c.add_argument("corpus")
    c.add_argument("-o", "--out", default="quilt_out/bank.sqlite")
    c.add_argument("--max-files", type=int, default=None,
                   help="deterministic stride subsample (smoke runs)")
    c.add_argument("--force", action="store_true", help="replace the bank")
    c.add_argument("--workers", type=int, default=None,
                   help="parallel encoding workers (default: auto)")
    c.set_defaults(fn=cmd_index)

    c = sub.add_parser("make", help="walk the bank and sew a piece")
    c.add_argument("-b", "--bank", default="quilt_out/bank.sqlite")
    c.add_argument("--mode", default="chain", choices=sorted(MODES))
    c.add_argument("--seconds", type=float, default=180.0,
                   help="target piece duration (default 180 = 3 minutes)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--strategy", default="all",
                   help="walk only one scissor's pieces (or 'all')")
    c.add_argument("--exclude-window", type=int, default=12)
    c.add_argument("--temp", type=float, default=0.5,
                   help="chain/follow pick temperature (0 = greedy)")
    c.add_argument("--k", type=int, default=10,
                   help="candidate pool for chain/follow picks")
    c.add_argument("--speed", type=float, default=1.0,
                   help="roam step size in median-NN units")
    c.add_argument("--contrast-window", type=int, default=1,
                   help="pieces of recent past the contrast walker flees")
    c.add_argument("--branch", type=float, default=0.35,
                   help="follow: jump probability instead of continuing")
    c.add_argument("--amp", type=float, default=2.0,
                   help="attractor amplitude in PCA stds")
    c.add_argument("--novel", type=float, default=None,
                   help="walk only the top fraction most novel pieces "
                        "(e.g. 0.1 = the strangest 10%%)")
    c.add_argument("--voices", type=int, default=3,
                   help="hocket: interleaved voices")
    c.add_argument("--grain-lo", type=float, default=0.04,
                   help="hocket: smallest grain (s)")
    c.add_argument("--grain-hi", type=float, default=0.6,
                   help="hocket: largest grain (s)")
    c.add_argument("--inner", default="chain", choices=["chain", "roam"],
                   help="hocket: each voice's own walking policy")
    c.add_argument("--crossfade", type=float, default=0.0,
                   help="crossfade between pieces in ms (0 = the law: "
                        "butt-splice)")
    c.add_argument("--no-exact", action="store_true",
                   help="let the piece end on a chunk boundary (no trim)")
    c.add_argument("--corpus-root", default=None,
                   help="override where the corpus files live")
    c.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"],
                   help="atlas query device (auto: GPU for big banks)")
    c.add_argument("-o", "--out", default=None)
    c.set_defaults(fn=cmd_make)

    c = sub.add_parser("render", help="a route JSON -> audio (re-glue)")
    c.add_argument("route")
    c.add_argument("-o", "--out", default=None)
    c.add_argument("--crossfade", type=float, default=0.0)
    c.add_argument("--no-exact", action="store_true")
    c.add_argument("--corpus-root", default=None)
    c.set_defaults(fn=cmd_render)

    c = sub.add_parser("info", help="the bank's census")
    c.add_argument("-b", "--bank", default="quilt_out/bank.sqlite")
    c.add_argument("--corpus-root", default=None)
    c.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    c.set_defaults(fn=cmd_info)

    c = sub.add_parser("doctor", help="environment + RAVE status")
    c.set_defaults(fn=cmd_doctor)

    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help()
        return 2
    try:
        return a.fn(a)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"quilt: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())