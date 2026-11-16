# research — the novelty-search experiments

This is the body of work the flagship `riff_lab` is a conclusion of. Every
project here is an experiment in **open-ended, objective-free evolution** —
different organisms, different phenotypes (sound, pattern, image, whole tracks),
but the same question underneath:

> *Does selecting for **novelty against a persistent memory** beat selecting
> for a goal (greedy) or for a coarse behaviour grid?*

The recurring, experimentally-supported answer across these projects is **yes** —
and that's why `echo` (the shared memory organ, one directory up) and `riff_lab`
exist at all.

Everything here is pure `numpy`/`scipy` at the core (two projects also use
`torch`/`librosa` — flagged below). Each subproject is a self-contained package
with its own `README.md`, `pyproject.toml`, and tests.

## The controlled experiment (the thesis)

`maze`, `scan`, `prism`, and `eden` are deliberately **the same selection engine
with a swappable memory policy** — run the identical organism under each of four
arms and compare where the lineage goes:

```
  greedy   — climb toward a fixed objective        (converges, collapses)
  random   — no selection                          (drifts nowhere)
  cell     — MAP-Elites: novelty vs a behaviour grid (coarse, collides)
  novelty  — sparseness vs echo's persistent archive (keeps finding new)
```

| project | organism / phenotype | deps |
|---|---|---|
| **`maze`** | a minimal spectral **voice synthesizer** (gaussian bumps + noise + lfo/vib + clicks) — the smallest legible organism | numpy + echo |
| **`scan`** | an evolvable **wavetable** synth (PPG-ROM / wavesequencing / vector mix / pitch path) — "frames born, not designed" | numpy + echo + **torch** |
| **`prism`** | one **CPPN genome** (NEAT) rendered four ways — spectra / wavetable / field / picture; includes rotating-lens novelty and a NEAT-Drummer branch | numpy + echo + pillow |
| **`eden`** | a **world of singing agents**, each a CPPN voice, heard through one reactive listener; society-level crossover | numpy + echo + maze |

## Minimal-criterion coevolution (MCC) & quality-diversity

Objective-free search where viability — not quality — is the only gate, and
generators coevolve with their critics.

| project | what it evolves | deps |
|---|---|---|
| **`womb`** | the MCC breeding loop + **MAP-Elites** quality-diversity search; audiovisual organ generators (waveplate / automata / swarm / kuramoto) + real codec-loss generational drift | numpy |
| **`cryptid`** | compact MCC: a growable **network of dynamical laws** (jerk / excitable / fire / osc / sand / map…) + an evolved ear readout — new timbres from dynamical systems | numpy |
| **`tomb`** | sounds × ears MCC, and a **second transform family** (`ops.py`: resonate / wavesort / diffuse / react / harmonic_seed / granular_sieve…). `riff_lab` auto-loads these ops as extra voice transforms | numpy |
| **`bomb`** | the **strictness-dial corpus-envelope judge** that `riff_lab/ears.py` descends from — measure character, don't ban it | numpy |
| **`novelty_particles`** | a clean, textbook **novelty search + minimal criterion + persistent archive** over particle-system effects (NEAT-lite genomes, FM sonification). The most readable implementation of the core algorithm | numpy + pillow |

```bash
# the clearest end-to-end demo of the core loop (headless):
cd research/novelty_particles
python3 -m nparticles.main --headless --gens 500 --pop 40 --out runs/overnight
```

## Corpus navigation & composition

Treating a sound corpus as a space to be *searched through* rather than a target
to optimise toward.

| project | idea | deps |
|---|---|---|
| **`quilt`** | **concatenative synthesis** over echo-space: cut a corpus into pieces, remember them as points, and sew them back along chosen walks — including a `contrast` walker that uses **novelty as a compositional force** | numpy + echo + **torch** + **librosa** |
| **`loom`** | a registry-based, reproducible **corpus → piece** engine (chunkers / rank-features / processors with per-event time-curves / arrangers) — the cleanest production foundation | numpy |
| **`maestro`** | full-track **orchestration**: a score-organism (NEAT-Drummer) over an explicit form/tempo-map, with damage placed by form position | numpy + echo |
| **`doom`** | audio-reactive **video** (scenes / fx / timeline, pure-numpy frame writer) — not audio-generative, but `womb` builds on its palettes/rendering, so it ships here as that dependency | numpy |

## The pattern & rhythm roots (`audio_lab`)

The standalone labs `riff_lab` grew out of — included as the DSP lineage:

- **`alien_rhythms.py`** — a full Python port of the Pure-Data **`alien` Lisp
  pattern language** (~40 ops) plus a grammar that *generates* and *mutates*
  pattern expressions, and per-hit stochastic processing. This is the
  intricate-motif engine that became `riff_lab/patterns.py`.
- **`dnb_chop_lab.py`** — "THE CHOP": onset-slice a corpus at its transients,
  role-classify, and pick kits by **farthest-point** with a usage penalty →
  became `riff_lab/chop.py`.
- **`rhythm_lab.py`**, **`grammar_lab.py`** (corpus-as-language n-gram texture),
  **`granular_lab.py`** (shared onset/feature backbone), **`curate_lab.py`**
  (the 84-D listening vector + farthest-point spread) — and a shelf of
  full-track assemblers (`storm` / `montage` / `fusion` / `club` / `edm` /
  `structure` / `terrain` …). See `audio_lab/TECHNIQUES.md`.

## Running the research projects

Unlike the flagship (which runs in place), these are **editable installs** — the
faithful way to resolve the `echo` / sibling dependencies they declare:

```bash
pip install -e ./echo            # the shared memory organ, first
pip install -e ./research/maze   # then whichever experiment you want
pip install -e ./research/eden   # (eden depends on maze, so install maze first)
```

`scan` and `quilt` additionally need `torch` (and `quilt` needs `librosa`);
`prism` / `scan` / `eden` / `novelty_particles` use `pillow` for image
phenotypes. Each project's own `README.md` has its run commands and the specific
experiment it answers.

> These are research experiments, not a polished product — some are rougher than
> the flagship. They're here so the *ideas* and the *evidence* ship with the
> code, not just the conclusion.
