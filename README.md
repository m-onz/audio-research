# audio-research


**Open-ended novelty search for generative audio.**

This is a working research kit for *evolving sound you have never heard before*
— not optimising toward a fixed target, but searching outward, away from
everything already made. It's built around one question, borrowed from Lehman &
Stanley's novelty search and every minimal-criterion / open-ended variant since:

> **Have we heard something like this before?**

Turn that into a number instead of a human judgement, drive selection by it
alone, and a population of little audio programs will keep drifting into new
territory on its own. No target, no fitness-toward-a-goal, no taste filter on
the search — just a permanent memory of where it has already been, and pressure
to leave.

Everything here is **pure-numpy DSP at the core, offline, deterministic**. No
cloud, no training, no GPU needed.

---

## The idea in one picture

```
   a population of genomes (little generative audio programs)
                        │
                 render each to audio
                        │
        ┌───────────────┴───────────────┐
        │                               │
   echo encodes it              patterns describe its groove
   (556-D perceptual vector)    (density, syncopation, entropy)
        │                               │
        └───────────────┬───────────────┘
                        │
         novelty(x) = how FAR is this from the
         k nearest things in the ARCHIVE of
         everything heard before?
                        │
          select the most novel ─► they reproduce
                        │
          admit the novel ones to the archive
                        │
                 (repeat — the frontier moves)
```

Selection is **pure novelty**. The only veto is a *minimal criterion* (MCNS):
a child with a technical defect — clicks, silence, NaN, clipping — or no
bassline at all is inviable. Nothing else, no "is it good", ever touches the
search. That's the whole point: quality is a trap for open-endedness; novelty
relative to a rich memory is not.

---

## What's in here

The flagship, two supporting libraries, and two tracks of experiments — all
built on the same shared DSP, arranged from "the thesis" down to "the toys that
fed it". (The flagship core needs no GPU; two `research/` experiments optionally
use `torch`/`librosa`, flagged where they appear.)

### `riff_lab/` — the flagship: novelty search that evolves riffs
A genetic algorithm whose individuals are **expression-based genomes** — small
generative programs (one per voice) that are re-evaluated every bar so the line
breathes. Each riff renders to a folder you can audition or drop into a live
[Pure Data](https://puredata.info/) performance patch. Selection is canonical
novelty search against a persistent archive; lineages *complexify* over
generations via a gentle saturating pressure on expression node-count.

```bash
# 50 generations, 5 riffs kept each, 4 novelty candidates per slot
python3 -m riff_lab gen --n 5 --gens 50 --candidates 4
```

The subcommands (`gen`, `breed`, `mutate`, `judge`, `pool`) and the three
novelty weights (audio / pattern / complexity) are the main things to play
with. Read [`docs/novelty-search.md`](docs/novelty-search.md) for the full
mechanism and the knobs.

### `echo/` — the memory organ
A standalone, pip-installable package: a chunked machine-listening **encoder**
(audio → 556-D temporal vector; event order survives) plus a **vector archive**
that answers `novelty(v)` and `min_dist(v)` as exact k-NN. This is the memory
that makes open-ended search possible — the component the whole repo is really
an argument for. Has its own README, tests, and MIT license.

### `oneshot_lab/` — the generative substrate
A general-purpose library for mutating folders of one-shots into banks of weird
variations: extreme time-stretch (paulstretch, phase-vocoder, wavetable scan),
pitch/harmonic effects, spectral surgery, and glitch/databend corruption. 45
transforms across 7 families, each chainable and fully logged with provenance.
`riff_lab` draws its timbres from here.

### `labs/` — machine-listening sound design
A second track: a chain of bass / neuro / dub / squelch one-shot synthesizers
(`bass_lab → bass_lab2 → bass_lab3 → dub_lab → squelch_lab`, and a
`neuro_lab` lineage) that render large candidate pools and then **curate by
farthest-point selection** over a spectral feature space — a close cousin of
novelty search, applied to timbre design rather than evolution. Plus utilities:
`uniq.py` (echo-based dedup of a folder), `compare.py`, `normalize.py`,
`hifi.py`. See [`labs/README.md`](labs/README.md).

### `research/` — the experiments the thesis rests on
The broader body of novelty-search research `riff_lab` is a conclusion *of*.
`maze` / `scan` / `prism` / `eden` are deliberately **the same evolutionary
engine with a swappable memory policy** (greedy / random / MAP-Elites grid /
novelty+archive) — the controlled experiment showing novelty-against-memory goes
somewhere the others don't. Alongside them: minimal-criterion coevolution
(`womb`, `cryptid`, `tomb`, `bomb`), the cleanest textbook implementation of the
core loop (`novelty_particles`), corpus-navigation composers (`quilt`, `loom`,
`maestro`), and the pattern/rhythm labs `riff_lab` itself grew from (`audio_lab`:
`alien_rhythms`, `dnb_chop_lab`). Each is a self-contained package with its own
README. See [`research/README.md`](research/README.md).

---

## Quick start

```bash
git clone <this repo> && cd audio-research
python3 -m pip install -r requirements.txt

# drop some .wav one-shots / samples into ./sounds (see sounds/README.md)
# then run the flagship:
python3 -m riff_lab gen --n 5 --gens 20 --candidates 4

# results land in ./riffs/gen_XXX/riff_XXX/ — audition the _preview.wav files;
# the novelty archive persists in riffs/_novelty_archive.json across runs.
```

No sounds yet? Any corpus of short samples works — breaks, field recordings,
synth one-shots. Point at one living elsewhere with
`AUDIO_RESEARCH_SOUNDS=/path/to/wavs`.

### Requirements
Python ≥ 3.10, `numpy` / `scipy` / `soundfile` (see `requirements.txt`).
`scikit-learn` is optional (echo uses it for fitted-metric block weights when
present; degrades gracefully without).

---

## Repo layout

```
audio-research/
├── riff_lab/      flagship: novelty-search GA over riff genomes
├── echo/          the memory organ — perceptual encoder + vector archive
├── oneshot_lab/   the transform engine — 45 chainable sound transforms
├── labs/          machine-listening sound-design chain + utilities
├── research/      the novelty-search experiments (maze/prism/eden/scan, MCC, …)
├── sounds/        your source corpus (empty; drop wavs here)
├── docs/          concept notes (start with novelty-search.md)
└── requirements.txt
```

## Where this comes from

This is distilled from a large personal research ecosystem exploring open-ended
search in audio. The recurring, experimentally-supported thesis across that work
is simple: **novelty-driven selection against a rich, persistent memory beats
greedy optimisation and beats coarse behaviour-grids** for producing genuinely
diverse, intricate sound. `riff_lab` is that conclusion applied to drum & bass /
dubstep riffs; `echo` is the memory it needs; `oneshot_lab` and the `labs` are
the material it grows from.

It's released so you can run it, break it, re-point it at your own sounds, and
lift the pieces into your own experiments. MIT licensed — go make something
nobody has heard.

## License

MIT — see [`LICENSE`](LICENSE).
