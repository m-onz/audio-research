"""doom command line.

    python -m doom render piece.wav -o video.mp4 --preset glitchcore --seed 3
    python -m doom scenes
    python -m doom fx
    python -m doom info piece.wav
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="doom", description="audio-synced generative video")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("render", help="render a synced video for an audio file")
    r.add_argument("audio")
    r.add_argument("-o", "--out", default=None)
    r.add_argument("--preset", default=None)
    r.add_argument("--auto", action="store_true", help="let the audio direct the scenes")
    r.add_argument("--scene", default=None, help="render a single named scene")
    r.add_argument("--length", type=float, default=None, help="seconds (default: whole audio)")
    r.add_argument("--fps", type=float, default=24.0)
    r.add_argument("--size", default="960x540")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--x264-preset", default="medium", dest="x264_preset")
    r.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("scenes", help="list scene strategies")
    sub.add_parser("fx", help="list fx")
    sub.add_parser("presets", help="list presets")

    i = sub.add_parser("info", help="analyze an audio file's doom features")
    i.add_argument("audio")

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1

    if a.cmd == "scenes":
        from .scenes import list_scenes

        print("doom scenes:", ", ".join(list_scenes()))
        return 0
    if a.cmd == "fx":
        from .fx import list_fx

        print("doom fx:", ", ".join(list_fx()))
        return 0
    if a.cmd == "presets":
        from .presets import list_presets

        for name in list_presets():
            print(" ", name)
        return 0
    if a.cmd == "info":
        import numpy as np

        from .audio import analyze_audio, probe_duration

        f = analyze_audio(a.audio)
        print(f"{a.audio}: {probe_duration(a.audio):.1f}s")
        print(f"  rms   min {f.rms.min():.3f}  median {np.median(f.rms):.3f}  max {f.rms.max():.3f}")
        print(f"  bass  median {np.median(f.bass):.3f}   treble median {np.median(f.treble):.3f}")
        print(f"  centroid {np.median(f.centroid):.0f} Hz (median)")
        print(f"  onsets: {int(f.onset.sum())} ({f.onset.sum() / max(1, f.n_frames) * 100:.1f}% of frames)")
        return 0

    # render
    import doom

    w, h = (int(v) for v in a.size.lower().split("x"))
    out = a.out or (a.audio.rsplit(".", 1)[0].rsplit("/", 1)[-1] + "_doom.mp4")
    kwargs = dict(
        audio=a.audio, out=out, fps=a.fps, size=(w, h), seed=a.seed,
        crf=a.crf, x264_preset=a.x264_preset, verbose=a.verbose,
    )
    if a.length:
        kwargs["length"] = a.length
    if a.scene:
        kwargs["plan"] = [dict(scene=a.scene)]
        kwargs["preset"] = None
    elif a.preset:
        kwargs["preset"] = a.preset
    elif a.auto:
        kwargs["auto_plan"] = True
    try:
        result = doom.render(**kwargs)
    except Exception as e:
        print(f"doom: error: {e}", file=sys.stderr)
        raise
    print(f"{result['path']}: {result['frames']} frames, "
          f"video {result['video_seconds']:.2f}s / audio {result['audio_seconds']:.2f}s, "
          f"{result['size_bytes'] / 1e6:.1f} MB, rendered in {result['elapsed']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())