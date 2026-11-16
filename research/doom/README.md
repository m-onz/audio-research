# doom

**Perfectly audio-synchronized generative video, in Python.** Point it at
any audio file; doom analyzes per-frame features (the sync clock), schedules
scenes, renders numpy frames, and pipes them through ffmpeg into an H.264
`.mp4` with the audio muxed in the same pass.

```python
import doom
doom.render(audio="piece.wav", out="video.mp4", preset="glitchcore")
```

```
audio ──► AudioFeatures (per-frame rms/bass/mid/treble/centroid/flux/onsets)  = THE SYNC CLOCK
                │
    segment plan ─► Scene registry (17 strategies) ──► FX chain (26 audio-reactive effects)
                │
        frame loop (t = i/fps, exact) ──► ffmpeg pipe (rgb24 → libx264) + audio mux ──► .mp4
```

Sync is structural, not tuned: frame *i* is drawn from the features of the
audio heard at frame *i*. Nothing "reacts" — both streams are generated
from the same clock. Same seed → same video bytes.

## Requires

- Python ≥ 3.10, numpy
- **ffmpeg + ffprobe on PATH** (tested with 8.0). Everything else —
  scene math, features, FX — is pure numpy; no imageio/cv2/PIL needed for
  video (PIL only for the still-image scripts).

## Install

```bash
cd doom && pip install -e .
doom render piece.wav --preset ps2 --seed 3     # or: python -m doom render ...
doom scenes / doom fx / doom presets / doom info piece.wav
```

## The scene registry (17)

| demoscene | Nature of Code | self-organizing | honest visualizers |
|---|---|---|---|
| `plasma` (domain-warped) | `boids` (flocking) | `rd` (Gray-Scott) | `spectro` (rolling spectrogram) |
| `tunnel` (polar UV) | `flow` (curl-field particles) | `lenia` (continuous CA) | `scope` (oscilloscope, actual samples) |
| `starfield` (3D streaks + nebula) | `worms` (Minsky harmonographs) | | `newton` (Newton-Raphson basins) |
| `metaballs` | `fire` (PSX-Doom CA) | | `mandel` (audio-driven zoom) |
| `lowpoly` (PS2 terrain, vertex snap, fog) | | | `mask` (**the death mask** — chiaroscuro relief, contour/engrave modes) |
| `feedback` (zoom-rotate video feedback) | | | |

Every scene is bass/treble/flux/onset driven: bass flies the tunnel and
swells the mask's features, onsets flash lights and jolt fractals.

## The FX registry (26), in three families

- **Glitch** (onset-triggered): `slices`, `corrupt`, `pixel_sort`,
  `rgb_split`, `block_glitch`, `strobe`
- **Geometry glitch** (built for the lowpoly/mask look — the chaos layer):
  `polytear` (block-band shearing), `affine_warp` (PS1 per-row shear),
  `vertex_jitter` (grid wobble), `depth_corrupt` (perspective folding),
  `wireframe` (Sobel overlay), `interlace`, `trail` (stateful frame-diff
  smear), `chroma_ripple`, `block_zoom` (onset punch-ins)
- **Screen & retro**: `scanlines`, `crt`, `vhs`, `grain`, `glow`, `shake`,
  `hue_cycle`, `psx` (ordered-dither into RGB555), `posterize`,
  `invert_pulse`, `kaleido`

Stateful FX (`trail`, `polytear`, `affine_warp`, ...) are registered as
classes and instantiated fresh per video — deterministic per-video state.
FX are composed as strings with inline params: `"psx(levels=8)"`,
`"polytear(block=14)"`.

## Plans & presets

A *plan* is segment list: `{"scene": "mask", "params": {...}, "fx": [...],
"start": 0, "end": 30, "transition": "cut|dissolve|glitch"}` (segments
without times split the piece evenly). `auto_plan=True` lets the audio's
energy arc direct the schedule. Presets: `demo`, `ps2`, `vhs`,
`glitchcore`, `scope_laboratory`, `noc`, `organism`.

**Windowed/chunk renders** are native: `doom.render(audio, start=120.0,
length=60.0)` — features, frame clock, and muxed audio all shift together
(the Instagram-chunk workflow).

## Research notes (what's laptop-viable in 2025, no LLMs)

- **Pipe-to-ffmpeg** is the standard offline pipeline: raw RGB24 on stdin,
  audio as input 1, `-shortest` for exact container length.
- **Demoscene canon** (plasma/tunnel/rotozoom/fire/CA) is trivially
  vectorizable in numpy at 0.2–25 ms/frame at 960×540.
- **PS1/PS2 look** = low internal resolution + nearest upscale + vertex
  snapping + affine UVs + RGB555 + 4×4 Bayer ordered dither + depth fog
  — all reproducible in numpy (see `lowpoly`, `mask`, `psx`).
- **ML, laptop-viable, no LLMs**: learned dynamical systems — Neural CA
  (Mordvintsev-style, ~17k params), Lenia creatures, Gray-Scott — train in
  minutes and run fast; `lenia`/`rd` ship as scenes, and the Scene protocol
  (`render(feats, i) -> HxWx3 uint8`) is the hook for any torch/MPS model.
  Diffusion video is not worth its render time here.

## Ship work in this repo

- `fkul_videos.py` — the **f**K_u_liebniz suite: 12 IG portrait videos
  (720×1280, 60 s windows) — `python3 fkul_videos.py 3 4`
- `ep_covers.py` — 20 × 5000×5000 PNG cover variations of the Newton death
  mask (reliefs, engravings, patinas, glitch passes)
- `tests/` — 15 tests incl. determinism, windowed sync, every scene/FX

## License

MIT — see LICENSE.