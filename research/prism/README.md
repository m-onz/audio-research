# prism

**the pattern organism — one CPPN, many phenotypes.**

A compositional pattern producing network is a growable graph of
geometry functions (sin, gauss, abs, signed square, …). Point its
outputs at axes and it *is* the phenotype: the same genome draws the
picture, plays the spectrogram, walks the wavetable, and — read at
audio rate — is the waveform. Nothing is synchronized after the fact:
picture and sound are one function at different traversals.

```
                        ┌── picture   (x, y)         → PNG
   genome (CPPN) ──┬────┼── spectra  (time, log-f)  → magnitude field → iFFT frames
                   │    ├── wavetable (phase, morph) → band-limited morphing table
                   │    └── field    (seconds)       → the waveform itself
                   │
                   └── mutations: drift · add_conn · add_node · ADD-OUTPUT ·
                                 swap_fn · toggle · subgraph duplication
```

Phase A: the physics and the champions. Phase B (the breeding loop —
echo admission, viability bands, unbounded alphabet) is the next build.

## why this arena (the papers, weaponized)

- **the encoding is the environment.** cryptid's dynamical arena was
  texture-shaped: novelty filled it with crackle because crackle is
  what that manifold reaches. A CPPN's bias is compositional geometry:
  sin∘linear is a tone, sin∘sin is FM, gauss is a formant window,
  abs is mirror symmetry (even harmonics). Harmonic, *patterned*
  territory is cheap here — the reachable manifold is shaped like
  the sounds you want (the 2006 *Comparing Artificial Phenotypes*
  checklist: repetition-with-variation and imperfect symmetry are
  what composition emits by construction).
- **unboundedness is native.** add-output grows the output alphabet
  (brant_gecco19); add-node/add-conn grow the graph; subgraph
  duplication copies a motif and hooks it in as a parallel voice
  that diverges (voxelbuild's transition hook; morphological
  conventions). No soft ceiling at 4 nodes.
- **the humans stay out of the loop, not out of the world.** search
  pressure is echo-novelty + viability bands (maze validated the
  recipe); the artist curates *after* the run — judgment is
  subjective and stays with the beholder (the OEE2 subjectivity
  paper), but steering is mechanized.

## the genome

| part | detail |
|---|---|
| inputs (senses) | `x`, `y`, `d` (radial), `b` (bias), `t` (slow clock), `live_rms`, `live_centroid`, `live_flux` (installation inputs; zeros until Phase D) |
| functions (zoo) | `sin` `cos` `tanh` `gauss` `sig` `abs` `sq` (signed square) `sqrt` `lin` `step` |
| outputs (alphabet) | named, growable: `amp`, `wave`, `field`, `lum`, `pan`, `morph`, … more are born by `add_output` |
| evaluation | one linear feed-forward pass, vectorized over whole grids; float32; NaN-guarded at every node |
| mutations | weight/bias drift · add_conn · add_node (NEAT split) · **add_output** · swap_fn · toggle · **duplicate** (motif copy + parallel-voice hook) |

Every connection points strictly forward — an explicit node order
keeps evaluation a single pass and mutations honest (tested).

## the traversals

| mode | input mapping | phenotype |
|---|---|---|
| **spectra** | x=seconds, y=log-freq [-1,1] | magnitude field → per-frame inverse FFT with phase accumulation from **random initial phases** (dense additive synthesis — with hop=win/2 the per-bin advance is exactly πk, so zeroed phases realign every even frame into a windowed click train; random starts keep partials incoherent). The genome's picture is its score |
| **wavetable** | x=phase, y=morph, t=seconds | bank of 64 single-cycle tables, band-limited per harmonic against f0, crossfaded along a slow morph path (or a `morph` output) |
| **field** | x=seconds (weights are frequencies) | the waveform itself, evaluated at 4× and FFT-lowpassed before decimation — composition may alias; the readout is not allowed to |
| **picture** | x,y space (+ frozen `t`) | luminance tinted by two chroma outputs → PNG |

Two spectral-engine laws, learned the hard way (eden's review caught
them): **np.interp clamps, it does not silence** — the band mapping
zeroes everything outside the field's frequency range, or the top
band leaks into the ceiling bins; and **percentile normalization
belongs to the world, not the voice** — the atlas shares one range
across tiles so quiet specimens read quiet (tested, both).

The sole designed stage is the **safety limiter** (rms normalize →
fades → pan field → peak scale) — the house invariant.

## champions

| name | mode | what it is |
|---|---|---|
| **bellfield** | spectra | five log-frequency ridges bending at their own rates — a slow glass chord; the picture is the landscape the sound walks through |
| **glassvoice** | wavetable | a table that walks from near-pure sine to a reeded shadow and back, band-limited every step |
| **tide** | field | nested slow-drifting FM at audio rate; its weather is saturation depth, not an envelope |

Measured (60 s renders, this machine, post phase-fix):

| champion | wall time | rms | centroid | crest |
|---|---|---|---|---|
| bellfield | 0.6 s | 0.127 | 1187 Hz | 7.3 |
| glassvoice | 0.3 s | 0.127 | 247 Hz | 2.4 |
| tide | 1.2 s | 0.127 | 57 Hz | 1.9 |

Before the random-phase fix, a flat broadband field synthesized as a
windowed click train with crest ~60 (now 3.9 — the regression test
holds it there forever).

Determinism: seed = identity; two renders of one genome are
bit-identical (tested). A 6-generation mutation walk off bellfield
stays alive and drifts heritably (centroid 1078 → 1459 Hz); random
founders + 20 mutations render finite in all three modes.

## cli

```bash
prism champions                                  # the three, with blurbs
prism render bellfield -l 60 -o bell.wav         # a genome → audio
prism picture bellfield --size 1280x720          # the same genome → its score
prism render glass_voice.json -l 30 --mode wavetable --f0 110
prism info tide                                  # anatomy, off the genome
prism random --seed 7 -o founder.json            # a Phase B founder
```

Genome json: `{"genome": {...}, "mode": "spectra"}` — full round-trip.

## layout

```
src/prism/
  genome.py      # the CPPN: zoo, feed-forward order, mutations, json
  render.py      # the four traversals + wav/png writers + limiter
  champions.py   # three hand-built genomes + anatomy()
  cli.py         # render | picture | info | champions | modes | random
tests/           # 47 tests: invariants, mutations, determinism,
                 # band-limiting, pitch, pictures, the phase-fix and
                 # interp-leak regressions, atlas contrast, engine arms
```

## phases

- **A — this.** physics proven: three champions, determinism, speed,
  heritable mutation. *acceptance held.*
- **B — the loop.** viability MC (audible/alive/finite bands) + echo
  admission (novelty ≥ ½ current median) + rotating-lens parent
  selection; arms `novelty`/`random`/`greedy`; specimens carry
  biographies. Acceptance: cells still rising at gen 240, complexity
  (nodes/outputs) above founders and growing.
- **C — long runs + the strictness dial** (soros 0.1/0.5/1.0 sweep)
  and post-hoc curation → resume.
- **D — installation.** live inputs (`live_rms`…) wired to the room;
  hour renders; mp4 pieces through doom; `prism organ` for womb.

## requires

Python ≥ 3.10, numpy, pillow, ffmpeg on PATH. MIT — see LICENSE.