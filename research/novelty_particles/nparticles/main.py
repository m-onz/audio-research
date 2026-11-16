"""Entry point.

  viewer (default): live gallery, evolution time-sliced into the render loop
      python3 -m nparticles.main
  headless: evolve as fast as possible, write the gallery to disk
      python3 -m nparticles.main --headless --gens 200 --out runs/mine
  render one saved genome to a still image
      python3 -m nparticles.main --render runs/mine/gallery/<uid>/genome.json still.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def do_render(genome_path: str, out_png: str, frames=320, n=420, size=760):
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from .genome import Genome
    from .render import Canvas, draw_step
    from .sim import Sim

    g = Genome.from_dict(json.loads(Path(genome_path).read_text()))
    sim = Sim(g, n=n, ttl=210, seed=3)
    cv = Canvas(size, size, spr_r=20, fade=0.994, exposure=1.3)
    for _ in range(frames):
        cv.tick()
        draw_step(cv, sim.step())
    # adaptive exposure so sparse/delicate systems don't come out pitch black
    hi = np.percentile(cv.fb, 99.9)
    cv.exposure = float(np.clip(0.55 / max(hi, 1e-6), 0.8, 8.0))
    u8 = (np.clip(cv.fb * cv.exposure, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(cv.frame()).save(out_png)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Novelty Particles — open-ended "
                                             "evolution of particle-system effects")
    ap.add_argument("--headless", action="store_true", help="no viewer, evolve full speed")
    ap.add_argument("--gens", type=int, default=10**9, help="(headless) generation limit")
    ap.add_argument("--pop", type=int, default=40, help="population size")
    ap.add_argument("--particles", type=int, default=128, help="particles per eval")
    ap.add_argument("--frames", type=int, default=170, help="frames per eval")
    ap.add_argument("--out", default="runs/default", help="run directory")
    ap.add_argument("--fresh", action="store_true", help="ignore previous state")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--movie", metavar="OUT_MP4", help="build an MP4 from the "
                                             "run's gallery and exit")
    ap.add_argument("--mode", choices=("reel", "slideshow"), default="reel",
                    help="movie style: live-animated clips or fingerprint catalog")
    ap.add_argument("--order", choices=("chrono", "novelty", "random"),
                    default="chrono", help="clip ordering")
    ap.add_argument("--count", type=int, default=0, help="number of clips (0 = all)")
    ap.add_argument("--clip-frames", type=int, default=0,
                    help="frames per clip (0 = mode default)")
    ap.add_argument("--no-reverb", action="store_true")
    ap.add_argument("--post", default=None, metavar="FX",
                    help="comma list of post effects with optional :arg — "
                         "excite[:drive] flanger[:rate] phasing[:eps] "
                         "ghost[:stretch] shimmer[:mix] tilt[:db] — "
                         "or 'none'. Default: excite:3,phasing:0.003,ghost:6")
    ap.add_argument("--fx-mod", type=float, default=1.0, metavar="SCALE",
                    help="genome-driven FX automation depth (0 = off, "
                         "default 1). Each effect's ANN plays the post "
                         "chain: flanger rate/depth, drive, shimmer, ghost "
                         "gain, phasing wobble, stretch rate")
    ap.add_argument("--render", nargs=2, metavar=("GENOME_JSON", "OUT_PNG"),
                    help="render a saved genome to a still image and exit")
    args = ap.parse_args()

    if args.render:
        do_render(args.render[0], args.render[1])
        return

    if args.movie:
        if not shutil.which("ffmpeg"):
            sys.exit("ffmpeg not found on PATH — install it first")
        from .media import build_movie
        from .post import DEFAULT_POST
        build_movie(args.movie, args.out, mode=args.mode, order=args.order,
                    count=args.count, clip_frames=args.clip_frames,
                    seed=args.seed or 3, reverb=not args.no_reverb,
                    post=args.post if args.post else DEFAULT_POST,
                    fx_mod=getattr(args, "fx_mod"))
        return

    from .evolution import Evolver

    ev = Evolver(args.out, pop_size=args.pop, seed=args.seed,
                 particles=args.particles, frames=args.frames,
                 resume=not args.fresh)
    print(f"run dir: {args.out}   archive: {len(ev.archive)}   "
          f"generation: {ev.generation}")

    if args.headless:
        t0 = time.perf_counter()
        while ev.generation < args.gens:
            events = ev.work(budget=10.0)
            if events:
                dt = time.perf_counter() - t0
                print(f"[{dt:7.1f}s] {ev.status_line()}   "
                      f"+{len(events)} new: "
                      + ", ".join(f"{e['uid']}(n{e['novelty']:.2f},{e['genome'].shape}"
                                  f"{'/trail' if e['genome'].trail else ''})"
                                  for e in events))
    else:
        from .viewer import run_viewer
        run_viewer(ev, args.out)


if __name__ == "__main__":
    main()