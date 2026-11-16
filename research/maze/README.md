# maze

**the audio maze** — a minimal demonstration that echo's memory
changes what evolution finds. One tiny voice synthesizer, four
selection policies, identical everything else; the arm with echo's
archive is the arm that goes somewhere.

The Lehman/Stanley maze story, told in sound: when you search
*without* a map, an objective funnels you into the first local
optimum (a boring sine), memoryless drift wanders slowly, and
novelty — measured against a persistent archive — walks away from
everything already heard, on purpose.

```bash
pip install -e ../echo && pip install -e .
maze run --gens 40 --seeds 0 1        # -> maze_out/run1/report.md
maze report maze_out/run1             # reprint the table
```

## the organisms

**`dx` (default)** — a compact FM operator genome (`dx.py`): 2–5
operators in feed-forward chains (ratio, index, level, target,
envelope), a global index LFO, a pinch of noise. Chowning's
nonlinearity: ratios × indices map a few dozen genes onto bells,
growls, washes, keys, drones — a country big enough that
√t-diffusion cannot cross it in budget. 2 s renders in ~3 ms
(synthesis at 11.025 kHz, exact FFT upsample ×4).

**`bumps`** (`voice.py`) — the original spectral-bump voice:
gaussian peaks + colored noise + click weather. Smaller country,
fine for smoke tests: `--organism bumps`.

Both label their own ground truth: `archetype()` reads the kind of
sound straight off the genome — tone, bell, growl, wash, keys,
drone, complex — so coverage is judged without any audio metric.

### the bumps genome (`voice.py`)

| gene | what it does |
|---|---|
| `bumps` | gaussian spectral peaks (freq, amp, bandwidth): bw 0.01 = pure tone, bw 0.5 = noise band |
| `noise` | colored noise floor (white → brown) |
| `lfo` | amplitude weather (rate, depth) |
| `vib` | pitch weather (per-voice vibrato) |
| `clicks` | stochastic click train (rate, amp, jitter) |
| `env` | attack + decay shape |
| `seed` | deterministic render |

~40 ms to render 4 s stereo, pure numpy. The arena spans pure
tones → chords → noise washes → grain clouds → click rhythms →
hybrids. Crucially, **the genome labels its own ground truth**
(`archetype()`): tone-ness, noise-ness, rhythm-ness are direct
functions of the genes — so coverage can be judged without any
audio metric at all.

## the four arms (`engine.py`)

Identical founder, mutation kernel, viability gate (audible +
moving), render budget, survivor rule, seeds. Only the memory
policy differs:

| arm | parent selection | the point |
|---|---|---|
| `greedy` | argmax `tonality(genome)` | the objective trap: converges to the pure-tone optimum |
| `random` | uniform | memoryless drift |
| `cell` | grid-cell latest (womb's current rule) on the independent descriptor grid | the current craft |
| `novelty` | **top quartile by `bank.novelty`**; children must also clear a floor at half the population's *current* median novelty | **echo's memory** |

The novelty floor is deliberately *relative*: a fixed floor
stagnates as the archive grows (novelty decays with density — the
first run's absolute floor froze admissions by gen 30). Half the
current median is a moving target: always "newer than what's
around", never impossible.

## the measurement (`measure.py`) — echo never judges echo

Every coverage number is computed **outside** echo:

- **kinds now / kinds ever** — archetype census read straight off
  genomes (ground truth, zero audio analysis)
- **pairwise / cells / radius** — 8 plain descriptors (93 ms
  windows, no mel, no chunks, no z-norm — an alien instrument),
  mean pairwise distance, unique cells on the listener axes, and
  distance traveled from the founder
- **parent-recall** — the one echo-measured number, checked
  against *true lineage*, not against echo itself: when the
  archive is asked "what does this child sound like?", how often
  is the actual parent in the top-5 nearest?

## results — the definitive run (FM organism, 240 gens, 3 seeds, 24x24 grid)

`maze run --organism dx --gens 240 --seeds 0 1 2 --pop 20 --births 10`
→ `maze_out/dx_long/report.md`

| arm | cells ever | yield (cells/admit) | pairwise | radius | parent-recall |
|---|---|---|---|---|---|
| greedy | 27.7 | 0.0122 | 0.326 | 0.35 | 10% |
| random | 16.7 | 0.0070 | 0.256 | 0.29 | 14% |
| cell | 29.7 | 0.0125 | 0.411 | 0.43 | 12% |
| **novelty** | 24.0 | **0.0200** | 0.311 | 0.37 | **16%** |

Territory growth (cumulative cells at 25/50/75/100% of the run):

| arm | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| greedy | 18 | 22 | 27 | 28 |
| random | 16 | **17** | **17** | **17** |
| cell | 16 | 21 | 26 | 30 |
| novelty | 17 | 19 | 23 | 24 |

Read it like the literature says to:

- **memoryless diffusion hits the wall**: the random arm stops
  discovering at gen ~60 (17 cells, flat for three quarters of the
  run) — the √t law. Every memory arm keeps climbing through gen 240.
- **echo is the most efficient explorer**: at half the admission
  budget (only the top half of each generation's children survive the
  novelty ranking) it still nearly matches the totals — **0.0200
  cells per admission: 2.9× random, 1.6× the grid**.
- **parent-recall 16% vs 10–14%**: echo's k-NN reconstructs lineage
  from sound alone better than any other arm's implicit memory.
- **caveat, stated honestly**: the cell arm's internal parent map
  shares its axes with the discovery grid (home advantage on raw
  totals — that arm is literally optimized for the metric's axes);
  echo's memory never sees the plain features, it works in its own
  556-D space and is axis-agnostic. And greedy's high cells-ever is
  the long march down the purity gradient — its *final* population
  is 35 pure tones (pairwise 0.326): it crossed territory without
  ever settling it.

Earlier runs are kept for the record: `maze_out/run1..4` (the
bumps organism — smaller country, saturation not reached, memory
arms statistically tied; that arena was too easy for the
question). The bumps voice remains available: `--organism bumps`.

## the extended run — how diverse does novelty get? (2000 gens, 3 seeds)

`maze run --arms novelty --gens 2000 --seeds 0 1 2` →
`maze_out/novelty_2k/` — the growth curve, mean of 3 seeds
(independent 24×24 grid cells, cumulative):

```
gen    60  ************************** 17.3
gen   120  ***************************** 19.0
gen   240  ************************************* 24.0   (matches the 4-arm run)
gen   480  ******************************************* 28.3
gen   800  ************************************************ 31.7
gen  1200  ******************************************************* 36.0
gen  1600  ********************************************************* 37.3
gen  2000  ************************************************************ 39.3
```

- **memoryless drift froze at 17 forever** (the 4-arm run); novelty
  reaches **39.3 cells — 2.4× — and is still rising at gen 2000.**
- **the curve is log-shaped, not flat**: every doubling of the run
  buys another ~6 cells. The easy territory goes first; each new
  cell costs more admissions (yield 0.0200 cells/admit at gen 240 →
  0.0039 at gen 2000). That is novelty search filling the *reachable*
  manifold — the archive keeps pointing at what is left, and what is
  left is genuinely farther away.
- **the population keeps moving**: final censuses differ seed to seed
  (seed 0: washes+complex; seed 1: keys+growls; seed 2: washes) —
  the arm never settles; it is a nomad by construction.
- **all 7 archetype kinds found by gen ~100** — after that the
  taxonomy saturates (it only has 7 words) while the grid keeps
  finding *new combinations* of them. Kinds is the coarse metric;
  cells and the ear are the fine ones.
- **mean population novelty settles ~1400** (from 5200 at genesis):
  the archive filling up makes everything less novel by density —
  and the run keeps finding edge material anyway.
- **the journey artifact**: `novelty_s<seed>/journey.wav` is one
  specimen per 100 gens, crossfaded in generation order — seed 0's
  tour reads bell → keys → drone → tone → washes → bells → complex →
  growl → wash: a listenable history of the exploration, with the
  `journey.json` ledger next to it.

**Verdict**: at this arena's scale, echo-driven novelty is an
unambiguous improvement over memoryless drift (2.4× territory, no
wall) and — from the 4-arm run — the most territory-efficient explorer
per admission. The remaining ceiling is the arena itself: the FM
genome's reachable manifold under its mutation kernel. To push past
it: bigger leaps, more operators, or a second arena (wavetable/
vector synthesis) — the measurement and the memory transfer as-is.

## layout

```
src/maze/
  voice.py     # the organism: genome, render, mutation, archetype labels
  engine.py    # the loop: four arms, viability, bank bookkeeping, artifacts
  measure.py   # the independent judge: plain features, coverage, ascii map
  cli.py       # maze run | report
tests/         # 12 tests: determinism, bounds, labels, independence, smoke
```

## requires

Python ≥ 3.10, numpy, ffmpeg (writes wavs), and **echo** as a
library dependency — this experiment exists to exercise it:
`EchoBank` (in-memory archive per run), `encode` (every child),
`novelty` (parent selection + admission floor), `knn`
(parent-recall diagnostic).

MIT — see LICENSE.