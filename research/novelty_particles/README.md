# Novelty Particles

**An open-ended, objective-free evolution of particle-system effects** —
NEAT-evolved ANN particle systems (Hastings, Guha & Stanley 2009,
*Interactive Evolution of Particle Systems for Computer Graphics and
Animation*) driven by **novelty search** instead of a human breeder.

The paper's IEC loop asks a human *"which of these 9 effects do you like?"*
and lets the chosen one reproduce. This project removes the human fitness
function entirely and asks a different question every generation:
*"which of these particle systems behaves like nothing we have ever seen?"*

There is no target effect, no fitness function, and no stopping condition.
A persistent archive of discovered behaviours grows forever; complexifying
neural networks (NEAT) keep expanding what "behaviour" can mean. The result
is an automatic concept generator: a window where the nine most recent
*discoveries* animate live, and a disk gallery that is a fossil record of
everything evolution ever found.

---

## How it maps to the paper

| NEAT Particles (paper)                          | Novelty Particles (this repo)                     |
|------------------------------------------------|---------------------------------------------------|
| ANN controls each particle: (x, y, dist-from-centre) → (velocity, colour) | same, plus a size output and a bias input |
| 8 activation functions per node (sin, cos, tan, sigmoid, tanh, gauss, ramp, step) | same |
| NEAT complexification: add-conn p=0.30, add-node p=0.20, weight jitter U(−0.5, 0.5) | same (asexual, so no speciation needed — the paper already notes human selection replaces it) |
| Generation shapes: sphere, point, line, circle  | evolvable **shape gene**: point / circle / line / disc |
| Effect classes: generic, plane, beam, rotator, trail | generic + **trail gene** (others are natural extensions) |
| User picks 1 of 9 → 8 mutants                     | **novelty = mean k-NN distance** to archive+population in behaviour space |
| fitness = human taste                            | fitness = *none* (pure divergent search) |
| —                                                | **minimal criterion** rejects degenerate systems (instant death, no motion, collapsed into one point) so novelty can't be gamed |
| —                                                | **behaviour descriptor** = 32×32 early/late occupancy fingerprints + 15 z-scored dynamics statistics |

### Why those behaviour descriptors?

* **Fingerprints** (the accumulated trail image over a headless run,
  split into early vs. late thirds) capture *what the effect looks like*
  — and the early/late split captures *temporal change* (an
  expanding-then-collapsing shell reads differently from a steady cloud).
* **Dynamics statistics** (mean/std speed, distance from centre, radial
  drift, angular momentum, spatial entropy, mirror symmetries, temporal
  change, colour dynamics, alive fraction, spread) capture *how it moves*
  even when two systems paint similar pictures.
* Novelty is the mean distance to the k=15 nearest neighbours among
  (archive ∪ current population). Everything is z-normalised online with
  floors, so no single statistic dominates.

### The key performance trick

Every particle in a system shares one ANN, so the network is activated
**once per frame for the whole particle batch** (numpy). A full behaviour
evaluation (170 frames, 128 particles, growing networks) costs ~40–60 ms,
which means evolution can be *time-sliced into the render loop*: the
viewer spends ~12 ms per frame evolving and still hits interactive
framerate. No threads, no multiprocessing.

## Install & run

```bash
pip3 install -r requirements.txt     # numpy, pygame, pillow

cd novelty_particles
python3 -m nparticles.main           # live gallery (default)
```

On first run you'll watch the first effects appear within seconds; on
later runs it **resumes** the run directory — open-endedness continues
where it left off.

### Viewer keys

| key      | action                                              |
|----------|-----------------------------------------------------|
| `SPACE`  | pause/resume evolution (animations keep running)   |
| `ENTER`  | follow the latest discovery                         |
| `←` `→`  | browse the whole archive                            |
| `1..9`   | zoom to a thumbnail                                 |
| `B`      | **boost-breed**: flood the population with mutants of the shown effect (a dash of human taste, IEC-style) |
| `S`      | save the shown genome                               |
| `H`      | toggle HUD                                          |
| `ESC`    | quit                                                |

### Headless mode

```bash
python3 -m nparticles.main --headless --gens 500 --pop 40 --out runs/overnight
```

Progress prints one line per generation (novelty, threshold, MC pass rate,
average network size, new archive members). ~1 s/generation at pop 40.

### Render a saved genome as a still

```bash
python3 -m nparticles.main --render runs/default/gallery/<uid>/genome.json effect.png
```

### Build an MP4 movie of the archive (with sonification)

```bash
python3 -m nparticles.main --movie reel.mp4                        # live-animated reel
python3 -m nparticles.main --movie reel.mp4 --count 40              # 40 evenly-spaced clips
python3 -m nparticles.main --movie catalog.mp4 --mode slideshow     # fingerprint catalog
python3 -m nparticles.main --movie reel.mp4 --order novelty          # most novel first
```

Requires `ffmpeg` on PATH. Clips default to chronological order — the
*story of discovery* — and every clip carries a caption (id, generation,
novelty, genes, network size).

**The soundtrack is a 3-layer sonification of the very things that made
each effect interesting:**

1. *Particles as instruments* — every rendered frame, the fastest live
   particles trigger short grains: distance-from-centre → pitch, speed →
   velocity, colour → carrier waveform *and* nested two-stage FM timbre,
   x → stereo pan. Each grain also carries a sub-octave and a two-tap
   resonant body. The audio follows the animation.
2. *The genome plays itself* — the evolved ANN *is* a synthesizer: fed a
   slow Lissajous traversal of its own input space, its colour outputs
   drive a detuned 7-voice supersaw drone (root/fifth/octave) with
   saturation drive and a brightness filter. Timbre variety is free:
   sin/cos nodes sound tonal, gauss nodes ring like bells, step nodes
   chatter, tan nodes grind.
3. *Behaviour-stat macros* — register from mean distance, event density
   from entropy, pentatonic quantisation for mirror-symmetric systems vs
   continuous glissando for asymmetric ones, FM index and detune spread
   from colour/speed statistics. Chronological reels also raise the root
   pitch with generation number, so complexification is audible as a
   slow ascent.

In `slideshow` mode the fingerprint images themselves are sonified as
rich spectrograms (rows → log-spaced fundamentals with 1x/2.01x/2.99x
harmonic stacks, columns → time).

**Post-processing** (`--post`, default `excite:3,phasing:0.003,ghost:6`):

| effect         | what it does |
|----------------|--------------|
| `excite[:drive]` | parallel tanh drive on the high band + even-harmonic term — spectral richness without boom |
| `flanger[:rate]` | LFO-swept multi-tap delay — moving notches (phaser/flanger), stereo-offset |
| `phasing[:eps]` | Steve Reich *tape phasing*: the texture bed's right channel plays at rate 1−ε, slowly sliding out of sync — evolving comb interference that never resolves |
| `ghost[:stretch]` | **paulstretch** (phase-randomised overlap-add) of the texture bed at Nx — extreme time-stretch mixed *underneath* the frame-synced events, so AV sync is never broken |
| `shimmer[:mix]` | granular octave-up copy of the ghost — upper-spectrum sparkle |
| `tilt[:db]` | one-FFT spectral tilt (brighten/darken) |

**Genome-driven FX automation** (`--fx-mod SCALE`, default 1, 0 = off):
the genome doesn't just *sound* — it **plays the mixing desk**. Each clip
runs a second, faster Lissajous performance of the same ANN
(`genome_mod_signals`), and the six outputs — smoothed, scaled by
`--fx-mod`, and *permuted per genome id* (so similar-sounding discoveries
process themselves differently) — become per-sample automation for:

  * flanger LFO rate & sweep depth (the notch sweeps themselves get played)
  * exciter drive (grit swells and relaxes)
  * shimmer mix and ghost gain
  * tape-phasing wobble (the drift breathes: almost resolves, then slips)
  * paulstretch rate — the ghost's stretch morphs *continuously*, its
    per-clip target set by the genome's entropy statistic

So a step-node genome makes the FX stutter, a gauss-node genome makes them
ping-pong, and a lazy bias-fed genome parks the knobs. Weirdness scales
with `--fx-mod` (try 2 or 3), and every animated parameter is clamped to
sane ranges so nothing explodes.

Example chains:

```bash
--post none                          # dry 3-layer sonification
--fx-mod 0                           # static post chain (no automation)
--post "excite:4,flanger:0.15,phasing:0.008,ghost:10,shimmer:0.4,tilt:2"   # deep ambient
--fx-mod 3 --post "excite:4,flanger:0.2,phasing:0.01,ghost:9,shimmer:0.5"  # really strange
```

All synthesis and post-processing is pure numpy (wavetables, additive/FM,
granular resynthesis, FFT) plus a vectorised Schroeder reverb; frames are
piped raw into a single ffmpeg encode, then muxed with the WAV.

## What lands on disk

```
runs/default/
├── gallery/<uid>/          one directory per discovered behaviour
│   ├── genome.json         the NEAT genome (any effect can be revived forever)
│   ├── fingerprint.png     32×32 RGB fingerprint ×10 (R=early, G=late, B=max)
│   ├── descriptor.npy      the behaviour vector (for novelty distance)
│   └── stats.json          novelty, generation, parent lineage, raw stats
├── state.json              population + generation (for resume)
└── normalizer.npz          online z-normalizer state
```

The fingerprint's red/green channels are the early/late occupancy maps:
**purple systems are early-heavy, green ones are late-heavy** — you can
read temporal structure straight from the gallery.

## Design notes & honest caveats

* **2D for v1.** The paper is 3D with billboards; this port is 2D to keep
  evaluation fast and rendering dependency-free. The ANN interface is
  position-in → velocity/colour-out, so a 3D port is a matter of adding a
  z input/output pair and a projection in the renderer.
* **FIFO archive.** The RAM archive is capped (~1200); the oldest entries
  are evicted from the *novelty metric* but never from disk. Consequence:
  ancient behaviours may eventually be rediscovered — a known, benign
  property of sliding-window novelty search (endless rediscovery in new
  network contexts). If it bothers you, raise `ARCHIVE_CAP`.
* **Asexual reproduction.** No crossover → no innovation-number bookkeeping
  beyond logging, no speciation (the paper itself notes speciation is
  unnecessary when selection pressure comes from outside the population).
* **MC thresholds are hand-set** (`sim.py`): alive fraction ≥ 0.10, mean
  speed, entropy, spread. They only exist to throw away degenerate
  systems; tuning them changes what counts as "viable", not what counts
  as "novel".

## Extensions worth trying

1. **Quality-Diversity switch**: keep the novelty loop but *also* maintain
   a MAP-Elites grid indexed by (mean-speed × symmetry) — a curated zoo
   instead of only a river of discoveries.
2. **Human-hybrid IEC**: the `B` (boost-breed) key is already a minimal
   version — make the viewer a full IEC interface where some thumbnails
   are user-nominated and some are novelty-nominated.
3. **More classes from the paper**: beam (Bézier control-point systems)
   and rotator (rotation outputs) drop straight into the same framework —
   add outputs and reinterpret them in `Sim.step`.
4. **CPPN inputs**: give the ANN angle-from-centre (atan2) as an extra
   input and watch rotational symmetries bloom.
5. **Video export**: `--movie` covers the archive reel; per-effect loops for
   game integration can reuse `media.build_movie` on a single genome.
6. **Sonification extensions**: map *novelty* itself to loudness/space
   (discoveries ring brighter as they get rarer), or drive the ANN-as-synth
   layer at control rate from live particle positions for tighter AV sync.
7. **3D**: add z; render with a rotating orthographic projection; the
   behaviour fingerprint becomes a projected occupancy (works unchanged).