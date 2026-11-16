"""womb command line.

    womb metabolize piece.wav -o my_reel -n 3 --preset metabolism --seed 7
    womb sonify video.mp4 -o heard.wav --mode tones
    womb info my_reel
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="womb",
                                 description="where loom and doom meet (offline AV metabolism)")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="compose an original A/V piece from organs")
    g.add_argument("-o", "--out", default="womb_out/piece.mp4")
    g.add_argument("-l", "--length", type=float, default=60.0)
    g.add_argument("--piece", default=None, help="named organ piece (see `womb pieces`)")
    g.add_argument("--organs", default=None, nargs="+",
                   help="explicit organ sequence, e.g. --organs waveplate swarm")
    g.add_argument("--corpus", default=None,
                   help="folder of wavs as grain material (loom output, the EP...)")
    g.add_argument("--fps", type=float, default=24.0)
    g.add_argument("--size", default="720x1280", help="WxH (default: 9:16 portrait)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("pieces", help="list named organ pieces")
    sub.add_parser("organs", help="list organs")

    se = sub.add_parser("search", help="MAP-Elites novelty search over organ+lens genomes")
    se.add_argument("-o", "--out", default="womb_out/search")
    se.add_argument("--iters", type=int, default=120)
    se.add_argument("--seconds", type=float, default=4.0, help="sample length per candidate")
    se.add_argument("--cells", default="12x12", help="archive grid (BxC x BxC)")
    se.add_argument("--corpus", default=None)
    se.add_argument("--seed", type=int, default=0)
    se.add_argument("-v", "--verbose", action="store_true")

    wk = sub.add_parser("walk", help="compose a full piece from a search archive")
    wk.add_argument("archive", help="atlas.json from a search")
    wk.add_argument("-l", "--length", type=float, default=45.0)
    wk.add_argument("-o", "--out", default="womb_out/novelty_walk.mp4")
    wk.add_argument("--size", default="720x1280")
    wk.add_argument("--segments", type=int, default=None)
    wk.add_argument("--corpus", default=None)
    wk.add_argument("--seed", type=int, default=0)
    wk.add_argument("-v", "--verbose", action="store_true")

    rp = sub.add_parser("reprocess", help="re-lens an existing video through the lowpoly pass")
    rp.add_argument("video")
    rp.add_argument("-o", "--out", required=True)
    rp.add_argument("--grid", default="24x42", help="cells WxH")
    rp.add_argument("--jitter", type=float, default=2.0)
    rp.add_argument("--tear", type=float, default=0.15)
    rp.add_argument("--levels", type=int, default=16)
    rp.add_argument("--seed", type=int, default=0)

    m = sub.add_parser("metabolize", help="run the cross-modal feedback loop")
    m.add_argument("audio", help="seed audio (wav/mp3/anything ffmpeg reads)")
    m.add_argument("-o", "--out", required=True, help="reel directory to create")
    m.add_argument("-n", "--passes", type=int, default=3)
    m.add_argument("--preset", default="metabolism",
                   choices=["metabolism", "collapse", "hold"])
    m.add_argument("--doom-preset", default=None, dest="doom_preset",
                   help="override the doom visual preset")
    m.add_argument("--length", type=float, default=None,
                   help="seconds per pass (default: whole audio)")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("sonify", help="hear a video")
    s.add_argument("video")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--mode", default="spectra", choices=["spectra", "grains", "tones"])
    s.add_argument("--corpus", default=None, help="grain source for grains mode")
    s.add_argument("--length", type=float, default=None)
    s.add_argument("--seed", type=int, default=0)

    i = sub.add_parser("info", help="inspect a reel")
    i.add_argument("reel_dir")

    sub.add_parser("presets", help="list drift presets")

    rv = sub.add_parser("rave", help="local neural latent bank (optional venv)")
    rv_sub = rv.add_subparsers(dest="rave_cmd", required=True)

    cr = sub.add_parser("cryptid", help="bred sound-objects (see CRYPTID.md)")
    cr_sub = cr.add_subparsers(dest="cryptid_cmd", required=True)
    cr_sub.add_parser("list", help="list hand-built champions")
    crr = cr_sub.add_parser("render", help="render a champion or specimen to stereo wav")
    crr.add_argument("specimen", help="champion name or specimen.json")
    crr.add_argument("-l", "--length", type=float, default=60.0)
    crr.add_argument("-o", "--out", default=None)
    crr.add_argument("--hi", action="store_true",
                     help="simulate at output rate (no decimation aliasing; slower)")
    crr.add_argument("--dump", action="store_true",
                     help="also write the specimen json next to the output")
    cri = cr_sub.add_parser("info", help="print a cryptid's biography (specimen/champion) or a run's ledger")
    cri.add_argument("specimen")

    cs = cr_sub.add_parser("seed", help="run the MCC breeding loop (CRYPTID.md phase B)")
    cs.add_argument("-o", "--out", default="womb_out/crypt1")
    cs.add_argument("--pop", type=int, default=20)
    cs.add_argument("--ears", type=int, default=8)
    cs.add_argument("--gens", type=int, default=16)
    cs.add_argument("--strictness", type=float, default=0.5,
                    help="Soros' dial: 0.1 loose .. 1.0 strict (both extremes stagnate)")
    cs.add_argument("--pairs", type=int, default=2, help="ears/processes sampled per birth")
    cs.add_argument("--seconds", type=float, default=3.0, help="eval length per candidate")
    cs.add_argument("--champion-seeds", type=int, default=2)
    cs.add_argument("--rate", type=float, default=0.4, help="mutation jitter magnitude (log-sigma)")
    cs.add_argument("--seed", type=int, default=0)
    cs.add_argument("--resume", action="store_true")
    cs.add_argument("-v", "--verbose", action="store_true")

    cp = cr_sub.add_parser("preview", help="the listenable atlas: montage video of occupied cells")
    cp.add_argument("run_dir")
    cp.add_argument("-l", "--length", type=float, default=60.0)
    cp.add_argument("-o", "--out", default=None)
    cp.add_argument("--grid", default="texture", choices=["texture", "time", "space"])
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("-v", "--verbose", action="store_true")

    mx = cr_sub.add_parser("mixtape", help="the run's album: every cell rendered long, crossfaded")
    mx.add_argument("run_dir")
    mx.add_argument("--grid", default="texture", choices=["texture", "time", "space"])
    mx.add_argument("--seg", type=float, default=8.0, help="seconds per specimen")
    mx.add_argument("--fade", type=float, default=0.75, help="crossfade seconds")
    mx.add_argument("--max-segments", type=int, default=48)
    mx.add_argument("-o", "--out", default=None)
    mx.add_argument("--seed", type=int, default=0)
    mx.add_argument("-v", "--verbose", action="store_true")
    rp = rv_sub.add_parser("preprocess", help="corpus -> training database")
    rp.add_argument("corpus_dir")
    rp.add_argument("--db", default="rave_db")
    rp.add_argument("--sr", type=int, default=44100)
    rt = rv_sub.add_parser("train", help="train RAVE on the db (hours, resumable)")
    rt.add_argument("db_dir")
    rt.add_argument("--name", required=True)
    rt.add_argument("--batch", type=int, default=4)
    rt.add_argument("--gpu", type=int, default=-1,
                    help="-1 = CPU (macOS supported path)")
    rg = rv_sub.add_parser("generate", help="run audio through the trained model")
    rg.add_argument("model_dir", help="run dir or ckpt")
    rg.add_argument("--input", required=True)
    rg.add_argument("--out", required=True)
    rv_sub.add_parser("doctor", help="check the neural venv")

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1

    if a.cmd == "rave":
        from .flows import rave as rave_flow

        if a.rave_cmd == "doctor":
            print("RAVE venv:", "OK" if rave_flow.available() else
                  "missing — create with: python3 -m venv neural && "
                  "neural/bin/pip install torch==2.11.0 torchaudio==2.11.0 acids-rave")
            return 0
        if a.rave_cmd == "preprocess":
            rave_flow.preprocess(a.corpus_dir, a.db, sr=a.sr)
            print(f"database: {a.db}")
        elif a.rave_cmd == "train":
            rave_flow.train(a.db_dir, a.name, batch=a.batch, gpu=a.gpu)
        elif a.rave_cmd == "generate":
            out = rave_flow.generate(a.model_dir, a.input, a.out)
            print(f"generated: {out}")
        return 0

    if a.cmd in ("pieces", "organs"):
        import womb

        if a.cmd == "pieces":
            for n in womb.list_pieces():
                print(" ", n)
        else:
            print("womb organs:", ", ".join(womb.list_organs()))
        return 0

    if a.cmd == "search":
        import womb

        c1, c2 = (int(v) for v in a.cells.lower().split("x"))
        r = womb.search(out_dir=a.out, iters=a.iters, seconds=a.seconds,
                        cells=(c1, c2), corpus=a.corpus, seed=a.seed,
                        verbose=a.verbose)
        print(f"{r['archive']}: {r['cells_filled']}/{r['grid'][0] * r['grid'][1]} cells "
              f"in {r['elapsed']}s")
        return 0

    if a.cmd == "walk":
        import womb

        w, h = (int(v) for v in a.size.lower().split("x"))
        r = womb.walk(a.archive, length=a.length, out=a.out, size=(w, h),
                      seed=a.seed, corpus=a.corpus, n_segments=a.segments,
                      verbose=a.verbose)
        print(f"{r['path']}: {r['frames']} frames, {r['seconds']:.1f}s, "
              f"rendered in {r['elapsed']}s")
        return 0

    if a.cmd == "reprocess":
        import womb

        gx, gy = (int(v) for v in a.grid.lower().split("x"))
        out = womb.reprocess(a.video, a.out, grid=(gx, gy), jitter=a.jitter,
                             tear=a.tear, levels=a.levels, seed=a.seed)
        print(f"{out}")
        return 0

    if a.cmd == "generate":
        import womb

        if a.piece:
            plan = womb.PIECES[a.piece]
        elif a.organs:
            plan = [dict(organ=n) for n in a.organs]
        else:
            plan = womb.PIECES["cabinet"]
        w, h = (int(v) for v in a.size.lower().split("x"))
        r = womb.generate(plan=plan, length=a.length, out=a.out, fps=a.fps,
                          size=(w, h), seed=a.seed, corpus=a.corpus,
                          verbose=a.verbose)
        print(f"{r['path']}: {r['frames']} frames, {r['seconds']:.1f}s, "
              f"rendered in {r['elapsed']}s")
        return 0

    if a.cmd == "presets":
        import womb

        for name in sorted(womb.PRESETS):
            print(" ", name)
        return 0

    if a.cmd == "info":
        import womb

        reel = womb.Reel(a.reel_dir)
        print(reel.summary())
        print(reel.drift_report())
        return 0

    if a.cmd == "sonify":
        import womb

        audio, sr = womb.sonify(a.video, mode=a.mode, seed=a.seed,
                                corpus=a.corpus, length=a.length)
        womb.write_wav(a.out, audio, sr)
        print(f"{a.out}: {len(audio) / sr:.1f}s of {a.mode} (sr {sr})")
        return 0

    if a.cmd == "cryptid":
        import womb
        from womb.cryptid import (CHAMPIONS, build_champion, describe_genome,
                                  load_specimen, render as cryptid_render,
                                  save_specimen, viability)

        def _specimen(arg):
            if arg.endswith(".json"):
                return load_specimen(arg)
            return build_champion(arg)

        if a.cryptid_cmd == "list":
            for n in sorted(CHAMPIONS):
                print(" ", n)
            return 0
        if a.cryptid_cmd == "info":
            from pathlib import Path as _P
            if (_P(a.specimen) / "state.json").exists():
                from womb.breed import report_run

                print(report_run(a.specimen))
            else:
                print(describe_genome(_specimen(a.specimen)))
            return 0
        if a.cryptid_cmd == "seed":
            from womb.breed import seed_run

            r = seed_run(out=a.out, pop=a.pop, ears=a.ears, gens=a.gens,
                         strictness=a.strictness, pairs=a.pairs, seconds=a.seconds,
                         champion_seeds=a.champion_seeds, rate=a.rate, seed=a.seed,
                         resume=a.resume, verbose=a.verbose)
            print(f"{r['out']}: {r['specimens']} specimens, cells "
                  f"{r['cells']['texture']}/{r['cells']['time']}/{r['cells']['space']} "
                  f"in {r['elapsed']}s")
            return 0
        if a.cryptid_cmd == "preview":
            from womb.breed import montage as cryptid_montage

            outp = cryptid_montage(a.run_dir, out=a.out, total=a.length, grid=a.grid,
                                   seed=a.seed, verbose=a.verbose)
            print(outp)
            return 0
        if a.cryptid_cmd == "mixtape":
            from womb.breed import mixtape as cryptid_mixtape

            outp = cryptid_mixtape(a.run_dir, out=a.out, grid=a.grid, seg=a.seg,
                                   fade=a.fade, seed=a.seed,
                                   max_segments=a.max_segments, verbose=a.verbose)
            print(outp)
            return 0
        out = a.out or f"womb_out/cryptid_{a.specimen.replace('.json', '')}.wav"
        g = _specimen(a.specimen)
        stats = cryptid_render(g, length=a.length, out=out, hi=a.hi)
        if a.dump:
            save_specimen(g, str(Path(out).with_suffix(".json")))
        v = viability(stats["audio"], stats["sr"])
        print(f"{out}: {stats['seconds']:.0f}s stereo from sim {stats['sim_rate']:.0f}Hz "
              f"in {stats['elapsed']}s — rms {v['rms']}, flatness {v['flatness']}, "
              f"alive {v['alive_ratio']}")
        return 0

    if a.cmd == "metabolize":
        import womb

        reel = womb.metabolize(
            a.audio, out=a.out, passes=a.passes, preset=a.preset,
            seed=a.seed, doom_preset=a.doom_preset, length=a.length, verbose=a.verbose,
        )
        print(reel.summary())
        print(reel.drift_report())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())