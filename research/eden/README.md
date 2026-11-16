# eden

**a little world of singing agents, heard through a reactive listener.**

A 3D torus populated by CPPN voices (prism genomes); a listener —
itself a reactive CPPN brain — wanders through, steered by what the
world is doing. The episode's audio is what the listener hears:
proximity, bearing, elevation, air. Interactions (collisions) retune
voices mid-episode and spike the world's event stream, which the
listener's brain consumes. echo's novelty score on each episode
drives the evolution; complexification (growing voice graphs, growing
societies) is measured, not assumed.

The phenotype is **sound over time** — scenes, arcs, approaches,
swerves — exactly what a chunked machine-listening vector exists to
hear.

```
   genome = { agents[CPPN voice + physics + reactions],
              listener[CPPN brain + hearing] }
        │
        ├─ phase 1: agents move, collide, react      (3D torus, 86 Hz)
        ├─ phase 2: voices = CPPN(time, log-f, neighbor-d)
        │            -> magnitude field -> additive frames   (prism's
        │            spectra traversal, with the phase fix)
        ├─ phase 3: the listener's brain, one batch pass:
        │            weather (rms/brightness/flux) + collision spikes
        │            (maestro's NEAT-Drummer decay frame) -> turn/speed/lift
        └─ phase 4: what the ear receives — falloff, azimuth pan,
                     ITD (<=0.57 ms), elevation tilt, air absorption
        │
        └─ echo encode -> novelty vs the archive -> which children live
```

```bash
pip install -e ../echo -e ../prism -e .
eden run --gens 200 --seeds 0 1          # -> eden_out/run1/report.md
eden render -o eden_out/worlds -n 3      # listen to raw founders
```

## the organism

**Every agent is a prism CPPN** read as an evolving spectrum: the
genome is evaluated on a (time × log-freq × neighbor-distance) grid
into a magnitude field, resynthesized frame by frame with accumulated
phase. The composition functions subsume any technique palette —
sin∘lin is a tone, sin∘sin is FM, gauss is a formant window, step is
percussion — and the **d input makes the voice a continuous function
of proximity**: voices change timbre as agents approach, before they
ever touch. The graphs GROW (prism's full NEAT op set: add_node,
add_conn, add_output, swap_fn, duplicate) — complexification is
intrinsic to the voice itself.

**Collisions** fire heritable reactions — `T_JUMP` (retune the read
elsewhere in the voice's own field), `BURST`, `ENTRAIN` (pull toward
the partner's read), `FLEE`, `ECHO` — and every touch wobbles the
runtime read. The genome itself stays sacred within the episode
(determinism depends on it); across generations, crossover and
mutation do the real genetics.

**The listener is a maestro-style score organism for movement**:
a second CPPN whose inputs are the world's global weather
(live_rms, live_centroid, live_flux) and a linearly-decaying spike
train of collision events — the NEAT Drummer encoding, transplanted
from maestro's audio scaffolds to navigation. Its outputs steer a 3D
heading (turn, lift, speed). Navigation is a *function of what is
heard*: g(p(t)), the maestro law, made spatial. Its brain grows by
the same mutations.

**Genetics**: mutation drifts the physics genes AND applies prism's
NEAT ops to every voice and the brain; crossover is per-agent
whole-package recombination (agents are coherent gene bundles —
society-level mixing, no graph splicing); agent birth/death is the
structural axis; the telemetry counts voice nodes/conns/outputs and
watches them grow.

## 3D audio localization — the investigation

The question was: do we need a game engine (Unreal et al.) for
proper roaming-in-3D audio? Short answer: **no — the cues that
matter are four cheap numpy ops**, and the engines cost more than
they give to a breeding loop:

| option | what it buys | cost | verdict |
|---|---|---|---|
| **numpy cues** (built) | interaural time difference (≤0.57ms gather), constant-power azimuth pan, distance air-absorption (14k→1.2k one-pole), elevation tilt | 0 deps, ~ms | **the loop lives here** |
| **pyroomacoustics** | image-source room acoustics, walls, reflections, RIRs, directivity | pip dep (numpy+C, offline, MIT); RIR convolution ~100ms+/episode | optional upgrade when *rooms* become the point — walls are a new interaction surface, not a localization fix |
| **SOFA HRTFs** (spaudiopy etc.) | true binaural 3D (pinna cues) | data files + dep | later, if elevation confuses listeners |
| **Unreal / Unity / Godot** | world-class spatializers (Steam Audio, Meta), live interactive scenes | 1-5GB toolchains, C++/blueprint, licensing, and worst: a headless batch renderer that cannot do thousands of deterministic cheap episodes | **wrong tool for breeding** — right tool for a phase-D artifact: walking through a champion world interactively |

The localization math in phase 4 is the standard cue set for
headphone listening; it is deliberately first-order (no pinna model)
but honest: sources sweep left-right with real delay, go muffled
with distance, tilt with height.

## a bug found in prism (and fixed here)

prism's `_render_spectra` initializes frame phases at **zero** with
hop = win/2, so every per-bin phase advance is exactly πk — after
every pair of frames, all bins realign into a single giant impulse.
The "spectra" mode is secretly a **windowed click train** whose
envelope is the CPPN field (crest ~60), not the intended dense
partial stack. eden fixes it with random initial phases (and
prism's own spectra mode may want the same one-line fix — check
whether its clickiness was ever intended).

## the measurement rig (unchanged from v1)

`random` vs `novelty` arms, identical crossover/mutation/viability/
budget; children compete by echo-novelty in the novelty arm.
Independent coverage on the 24×24 plain-feature grid, genome-derived
archetype census, and the complexification ledger (voice nodes,
agents, collisions per episode, listener path length). Artifacts:
`journey.wav` (the run's exploration in generation order),
`mixtape.wav` (final population), per-specimen wavs.

## layout

```
src/eden/
  world.py        # the organism: 3D torus, CPPN voices, reactive
                  # listener, spatialization, the episode
  genetics.py     # mutation (NEAT ops included), whole-package
                  # crossover, structure stats
  engine.py       # the arms loop (random | novelty), telemetry, artifacts
  cli.py          # eden run | render | report
  world_fm.py     # the FM-era world (v1), kept for the baseline comparison
  genetics_fm.py
tests/            # 12 tests: determinism, span, time-structure,
                  # collisions, brain reactivity, genetics, growth
```

## requires

Python ≥ 3.10, numpy, scipy (one-pole filters), ffmpeg, and the
siblings: **echo** (the memory), **prism** (the CPPN genome), maze
(the independent measurement). All offline. MIT.

## license

MIT — see LICENSE.