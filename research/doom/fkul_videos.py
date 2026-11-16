"""f**K_u_liebniz — the Newton death-mask video suite.

12 Instagram portrait videos (720x1280, 60 s, perfectly synced): for each of
the EP's 4 tracks, three 1-minute chunks (start / middle / end), each cut
from a different curated scene plan.

Aesthetic brief: Isaac Newton's black death mask — sculptural chiaroscuro
relief (`mask`), the Newton-Raphson basin fractal named after him
(`newton`), the PS2 low-poly look the artist liked in the `ps2` preset
(psx ordered dither on everything), and a wide rotation of the rest of
doom's scene registry for variety. Bass-heavy music drives the light sweeps
and fly-speeds; onsets flash the mask and jolt the fractal.

Run:  python3 fkul_videos.py [track_numbers...]     (default: all)
"""

import json
import sys
from pathlib import Path

import doom

EP = Path("f**K_u_liebniz")
OUT = Path("doom_out/fkul")
LENGTH = 60.0
SIZE = (720, 1280)  # 9:16 portrait
FPS = 24.0

# 12 curated plans, 3 x 20 s segments each; mask/newton lead throughout.
SCENE_PLANS = [
    [("mask", "black_bronze"), ("newton", "verdigris"), ("mask", "black_bronze")],
    [("newton", "ice"), ("lowpoly", "ps2_dusk"), ("spectro", "blood")],
    [("mask", "black_bronze"), ("worms", "vhs_night"), ("starfield", "ice")],
    [("lowpoly", "ps2_dusk"), ("mask", "black_bronze"), ("plasma", "inferno")],
    [("spectro", "amber"), ("newton", "verdigris"), ("feedback", "vhs_night")],
    [("plasma", "sunset_hdmi"), ("mask", "black_bronze"), ("mandel", "ice")],
    [("tunnel", "blood"), ("mask", "black_bronze"), ("rd", "ice")],
    [("mask", "black_bronze"), ("boids", "toxic"), ("newton", "verdigris")],
    [("flow", "toxic"), ("mask", "black_bronze"), ("lenia", "sunset_hdmi")],
    [("newton", "ice"), ("metaballs", "blood"), ("fire", "inferno")],
    [("mask", "black_bronze"), ("starfield", "ice"), ("lowpoly", "ps2_dusk")],
    [("feedback", "vhs_night"), ("mask", "black_bronze"), ("newton", "verdigris")],
]

FX_CHAINS = [
    # v2: every chain now carries the geometry-glitch layer — polytear,
    # affine_warp, vertex_jitter, depth_corrupt, trail, block_zoom — on top
    # of the psx/scanlines screen family. All onset/flux/bass reactive.
    ["psx(levels=24)", "polytear(block=18)", "scanlines", "glow"],
    ["psx(levels=16)", "affine_warp", "rgb_split", "grain"],
    ["psx(levels=12)", "depth_corrupt", "vertex_jitter(grid=28)", "scanlines"],
    ["psx(levels=8)", "trail(sustain=0.6)", "glow", "grain"],
    ["psx(levels=24)", "wireframe", "chroma_ripple", "scanlines"],
    ["vhs", "block_zoom", "psx(levels=20)", "interlace"],
    ["polytear(block=14)", "psx(levels=10)", "slices", "corrupt(amount=0.0003)"],
    ["vertex_jitter(grid=36)", "psx(levels=24)", "trail", "hue_cycle"],
]

TRANSITIONS = ["cut", "glitch", "dissolve"]


def plan_for(index: int) -> list[dict]:
    scenes = SCENE_PLANS[index % len(SCENE_PLANS)]
    fx = FX_CHAINS[index % len(FX_CHAINS)]
    plan = []
    for k, (scene, palette) in enumerate(scenes):
        plan.append(dict(
            scene=scene, params=dict(palette=palette), fx=fx,
            transition=TRANSITIONS[k % len(TRANSITIONS)],
        ))
    return plan


def main() -> None:
    tracks = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]
    OUT.mkdir(parents=True, exist_ok=True)
    suite_manifest = []
    for t in tracks:
        audio = EP / f"{t}.wav"
        dur = doom.probe_duration(audio)
        starts = [0.0, (dur - LENGTH) / 2, dur - LENGTH]
        for c, start in enumerate(starts):
            idx = (t - 1) * 3 + c
            plan = plan_for(idx)
            name = "_".join(s["scene"] for s in plan[:2])
            out = OUT / f"fkul_t{t}_c{c + 1}_{name}.mp4"
            r = doom.render(
                audio=str(audio), out=str(out),
                start=start, length=LENGTH, fps=FPS, size=SIZE,
                plan=plan, seed=t * 10 + c, crf=19,
            )
            scenes = [s["scene"] for s in plan]
            line = f"track {t} chunk {c + 1} [{start:6.1f}-{start + LENGTH:6.1f}s]  " \
                   f"{' > '.join(scenes)}  ->  {out.name}  " \
                   f"({r['frames']} frames, {r['elapsed']}s render)"
            print(line, flush=True)
            suite_manifest.append(dict(
                track=t, chunk=c + 1, window=[start, start + LENGTH],
                scenes=scenes, fx=plan[0]["fx"], out=str(out),
                frames=r["frames"], video_seconds=r["video_seconds"],
                onsets_synced=int(doom.analyze_audio(str(audio), fps=FPS)
                                  .window(start, LENGTH).onset.sum()),
            ))
    (OUT / "suite_manifest.json").write_text(json.dumps(suite_manifest, indent=1))
    print(f"\n{len(suite_manifest)} videos in {OUT}/")


if __name__ == "__main__":
    main()