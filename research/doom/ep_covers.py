"""f**K_u_liebniz — the 5000 x 5000 EP cover suite.

The Newton death mask in ~20 print-resolution variations: raking-light
bronze reliefs at several angles, topographic engravings, verdigris and
blood patinas, inverted molds, mirrored kaleidoscopes, and glitched
passes (slices, pixel sort, geometry-glitch, PSX dither) — all rendered
from the same heightfield, all deterministic.

Pipeline per cover: render the mask scene natively at 2500^2 -> nearest
x2 to 5000^2 -> apply the FX chain at full resolution -> save PNG.

Run:  python3 ep_covers.py            (writes doom_out/fkul_covers/)
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from doom.audio import AudioFeatures
from doom.fx import build_fx_chain
from doom.scenes import make_scene

OUT = Path("doom_out/fkul_covers")
RENDER = 2500   # native scene resolution (upscaled x2 to 5000)
FINAL = 5000


def static_feats(rms=0.72, bass=0.62, treble=0.18, flux=0.12, onset=False, n=4):
    """Neutral per-cover feature levels (deterministic, no file needed)."""
    return AudioFeatures(
        fps=24.0, sr=48000, duration=n / 24.0, n_frames=n,
        rms=np.full(n, rms, np.float32), bass=np.full(n, bass, np.float32),
        mid=np.full(n, 0.3, np.float32), treble=np.full(n, treble, np.float32),
        centroid=np.full(n, 500.0, np.float32), flux=np.full(n, flux, np.float32),
        onset=np.array([onset] + [False] * (n - 1)),
        wave=np.zeros(64, np.float32),
    )


VARIATIONS = [
    # name                     mask params                                   light  feats                       fx (at 5000)
    ("01_bronze_rake_left",   dict(palette="black_bronze"),                2.4,   dict(),                      []),
    ("02_bronze_rake_right",  dict(palette="black_bronze"),                0.6,   dict(),                      []),
    ("03_bronze_overhead",    dict(palette="black_bronze"),                4.6,   dict(),                      ["psx(levels=32)"]),
    ("04_bronze_deep_dark",   dict(palette="black_bronze"),                1.9,   dict(rms=0.35, bass=0.2),     []),
    ("05_bronze_loud_hot",    dict(palette="black_bronze"),                3.7,   dict(rms=1.0, bass=0.9),     ["glow"]),
    ("06_engrave_bronze",     dict(palette="black_bronze", mode="contour"),2.4,   dict(),                      []),
    ("07_engrave_mono",       dict(palette="mono", mode="contour"),        2.4,   dict(),                      []),
    ("08_engrave_verdigris",  dict(palette="verdigris", mode="contour"),   2.4,   dict(),                      []),
    ("09_engrave_amber",      dict(palette="amber", mode="contour", contour_levels=9), 2.4, dict(),           []),
    ("10_both_bronze",        dict(palette="black_bronze", mode="both"),    2.4,   dict(),                      []),
    ("11_verdigris_patina",   dict(palette="verdigris"),                   2.1,   dict(),                      ["psx(levels=24)"]),
    ("12_blood_relief",       dict(palette="blood"),                       2.9,   dict(rms=0.9, bass=0.8),     []),
    ("13_ice_relief",         dict(palette="ice"),                         2.4,   dict(),                      []),
    ("14_mold_inverted",      dict(palette="black_bronze", invert=True),   2.4,   dict(),                      []),
    ("15_glitched_bronze",    dict(palette="black_bronze"),                2.4,   dict(onset=True),             ["slices", "rgb_split", "pixel_sort"]),
    ("16_bitrot_bronze",      dict(palette="black_bronze"),                2.4,   dict(onset=True),             ["polytear(block=40)", "psx(levels=12)", "grain"]),
    ("17_vertex_bronze",      dict(palette="black_bronze"),                2.4,   dict(bass=0.95, onset=True),  ["vertex_jitter(grid=32)", "psx(levels=20)"]),
    ("18_interlaced_bronze",  dict(palette="black_bronze"),                2.4,   dict(),                      ["interlace", "scanlines", "psx(levels=16)"]),
    ("19_kaleido_mask",       dict(palette="black_bronze"),                2.4,   dict(),                      ["kaleido", "psx(levels=24)", "glow"]),
    ("20_crushed_4bit",       dict(palette="black_bronze"),                2.4,   dict(),                      ["psx(levels=4)", "grain(amount=0.12)"]),
]


def render_cover(name, mask_params, light_angle, feats_params, fx, seed=0):
    rng = np.random.default_rng(seed)
    mask_params = dict(mask_params)
    invert = mask_params.pop("invert", False)
    scene = make_scene("mask", RENDER, RENDER, seed=seed, div=1, **mask_params)
    scene.ang = float(light_angle)
    feats = static_feats(**feats_params)
    frame = scene.render(feats, 1)
    if invert:
        frame = 255 - frame  # inverse = the mold, not the cast
    # upscale x2 nearest -> 5000
    big = np.repeat(np.repeat(frame, 2, 0), 2, 1)
    # fx chain at full print resolution
    for fn, params in build_fx_chain(fx or []):
        big = fn(big, feats, 1, rng, **params)
    img = Image.fromarray(np.ascontiguousarray(big))
    img.save(OUT / f"{name}.png", optimize=True)
    return OUT / f"{name}.png"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for seed, (name, params, angle, feats, fx) in enumerate(VARIATIONS):
        t1 = time.time()
        p = render_cover(name, params, angle, feats, fx, seed=seed)
        mb = p.stat().st_size / 1e6
        print(f"{p.name:<28} {mb:5.1f} MB  ({time.time() - t1:.1f}s)", flush=True)
    print(f"\n{len(VARIATIONS)} covers in {OUT}/ ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()