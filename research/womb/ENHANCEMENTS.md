# womb — enhancement plan (open-endedness pass)

**Read papers, then propose.** The root PDFs are the UCF/Stanley
open-endedness corpus: POET (paired co-generation of environments and
solutions), MCC (minimal criterion coevolution, both the original maze
paper and the unbounded benchmark), Chromaria (MC strictness), Voxelbuild
(major evolutionary transitions), multi-BC quality diversity (Pugh PPSN
2016), RAAHN (Hebbian learning from novel experiences), and the OEE2
subjectivity position paper. Every enhancement below takes one *mechanism*
from those papers and maps it to a concrete, offline, deterministic change
in womb. Goal, always: **more interesting videos with audio.**

Constraints unchanged: this machine, fully offline, pure numpy, seed =
identity, the artifact is always a finished video, registries stay open.

> **Refined follow-up:** the audio-diversity flagship now lives in
> `CRYPTID.md` — breeding sound as *measurement of evolved processes*
> (no synthesis grammar, no instruments), absorbing E1/E2/E4/E5/E6.

---

## E7. The craft pass (do first — every piece gets better today)

Not from the papers, but the goal is interesting *artifacts* and the
current ones have craft gaps that read as bugs:

1. **Stereo.** Every organ is already spatial — swarm agent positions,
   territory walker x, waveplate probe locations, automata row bands —
   but all audio collapses to mono at the writer. Add `emit()` stereo
   (or a pan helper keyed on organ state) and a 2-channel `AVWriter`.
   Biggest perceptual win per line of code in the whole plan.
2. **Transitions.** Segments hard-cut today: organs restart cold at
   every boundary (visual pop + audio click). Two mechanisms:
   - `--crossfade S`: blend frames and equal-power-fade audio across
     the last ~1s of each segment into the next.
   - param-morphing: when consecutive segments use the same organ,
   interpolate params over the boundary instead of re-instantiating.
3. **Master bus.** Currently peak-normalize only. Add a tiny numpy
   master chain: soft-knee limiter, 20ms fade in/out (kills the first
   frame click and dead-air ending), optional short convolution "glue"
   tail. Also kill handoff clicks inside `emit()` grain queueing.
4. **Arc composition in `walk()`.** Elites are currently sorted by
   brightness — a monotone sweep. Compose an energy *journey* instead:
   low → bloom → peak → decay (or pick arcs by seed), and let segment
   durations scale with elite fitness (better material holds longer).

Effort: small each; acceptance: `womb generate --piece colony` has no
clicks, is stereo, and crossfades; a fresh listen is obviously "produced".

---

## E5. Plastic organs — homeostasis + an inner novelty drive (RAAHN, Chromaria)

**From:** bowren_alife16 (RAAHN: plastic weights shaped online by the
organism's own *novel* experience collection) and soros_alife16
(the minimal criterion that each organism must keep satisfying to live).

Organs today are frozen dynamical systems: params set at birth, the
piece is a steady-state loop. Long videos stall. Two small mechanisms:

1. **Homeostasis** (`plasticity=True` param, default on for long pieces):
   each organ holds its own liveliness in a band — automata keep mean
   state in [0.3, 0.6] by drifting `feed/kill`; waveplate adapts strike
   amplitude to keep probe rms in range; kuramoto adapts coupling to
   hold the order parameter mid-band (the sync/async boundary, where
   the audible beating actually lives); swarm adapts attraction to keep
   region occupancy variance bounded. This is Soros' MC-strictness made
   *endogenous*: the organism's minimal criterion is its own liveliness.
2. **Inner novelty drive** (RAAHN's collection of novel experiences,
   aimed the other way): an exponentially-decayed memory of recent
   behavior descriptors (rms, centroid, motion); params receive gentle
   pushes *away* from the recent-mean behavior. The organ wanders its
   own behavior space *within* one piece — internal development, not
   just across searches.

Effort: small-medium (one `Homeostat` helper + per-organ hooks).
Acceptance: a 3-minute single-organ piece passes an attentive listen
without looping feel; telemetry (per-window rms/centroid/motion) shows
drift bands, not plateaus.

---

## E6. The beholder in the loop (stanley_oee2workshop16 — subjectivity)

**From:** the subjectivity paper's own conclusion: interestingness is in
the eye of the beholder, so the beholder belongs in the loop. The same
lab's answer in practice was Picbreeder; womb's atlas contact sheet *is*
the Picbreeder screen. Make selection real:

1. `womb search --resume atlas.json` — continue filling an existing
   archive instead of starting cold (currently every search is a new
   universe; resumability is what makes curation meaningful).
2. `womb curate atlas.json --likes 3:7 5:2 ...` — mark cells as liked;
   liked elites (and their descendants, by lineage id) get boosted
   parent-selection weight. The archive learns *your* taste, offline.
3. Lineage names: every genome tracks `lineage` strings from its
   ancestors; the atlas sheet labels cells with them — the search's
   *narrative* (Stanley & Soros: the story of how artifacts arise is
   part of what makes them interesting) becomes visible.

Effort: small. Acceptance: search → curate → search --resume yields an
archive measurably biased toward liked regions.

---

## E1. Multi-BC archives (pugh_ppsn16)

**From:** *Searching for Quality Diversity when Diversity is Unaligned
with Quality* — a single behavior characterization misaligned with
"quality" funnels search; drive search with several BCs simultaneously.

womb currently uses ONE BC pair: audio brightness × visual activity.
Proposed BC bank (all computable from the existing `render_sample`):

| BC name | axis | what it spans |
|---|---|---|
| `tone` | centroid × flux | brightness vs. busyness of the sound |
| `rhythm` | onset density × onset regularity | sparse-irregular → dense-metronomic |
| `texture` | visual entropy × luma variance | flat/quiet → dense/hyper |
| `coherence` | audio↔visual correlation | free-running → structurally locked |

Mechanism per the paper: one evaluation, insertion into *every* BC grid;
parent selection picks a BC uniformly, then an elite from it — so the
population is pulled by whichever lens the offspring happens to land in.
One `atlas-<bc>.png` per grid; `walk()` gains `--bc coherence` to tour a
chosen lens. `coherence` is the one womb is *about* — today nothing in
the search optimizes the audiovisual lock that is the library's thesis.

Effort: medium (search.py only). Acceptance: same iteration budget fills
more distinct genomes; a coherence-archive walk yields visibly tighter
sound-image behavior than a tone-archive walk.

---

## E2. Minimal-criterion drift mode — the MCC flagship (pap140s3, brant_gecco19, soros_alife16)

**From:** MCC — no fitness, no novelty metric, no archive; two populations
paired by a mutual minimal criterion, diverging by drift. Soros supplies
the dial: MC too strict OR too relaxed both stagnate; intermediate
strictness yields orderly divergence. Brant/gecco19 supplies the
unbounded encoding lesson: complexity only grows if the genome *can* grow
(see E4).

womb's MCC pair writes itself — **organs × lenses** (content ×
reprocessing):

- A population of **organ genomes** (organ sequence + params).
- A population of **lens genomes** (lowpoly params + degrade ops).
- An organ genome is viable iff at least one lens renders it
  *watchable* (the viability band: rms in range, luma_std alive, no
  white/blackout) — "every maze is solved by someone".
- A lens genome is viable iff at least one organ survives it intact —
  "every navigator solves some maze". Destructive lenses (bitrot-heavy,
  2-bit palettes) lose their hosts and die.
- Offspring = mutations of both; no ranking, no cells — just the MC gate
  plus a perceptual-distance floor from the parent (drift, not novelty
  search: distance must merely exceed ε, it is not maximized).

Expose Soros' dial as `--mc-strictness 0.1..1.0` (scales the viability
band widths) and *log population diversity per generation* — the run
itself becomes the Chromaria experiment: find where this system's
"orderly divergence" regime lives.

Deliverables: `womb drift --pop 48 --gens 40 --mc-strictness 0.5
-o womb_out/drift1` → a lineage tree (each genome's ancestry + the
telemetry that admitted it), a population contact sheet, and a piece
rendered by a random *surviving lineage* (`womb drift --render <lineage>`).

Effort: medium. Acceptance: at moderate strictness, admitted genomes
per generation stay flat-to-rising while descriptor diversity rises
(orderly divergence); at 0.1 and 1.0, diversity collapses (stagnation) —
reproducing the paper's result inside womb.

---

## E3. POET-population metabolism (1901.01753v3 — POET)

**From:** POET — many environments evolve in parallel with their
solutions; solutions *transfer* between environments when they help;
challenges are kept "hard but not too hard" by adaptive difficulty.

`metabolize` currently runs ONE lineage chain. POET-ize it:

- `womb metabolize --population R` — R parallel reels, each its own
  drift trajectory (preset, sonify rotation, seed).
- **Transfer:** each pass, with probability `--transfer p`, lineage A
  may inject lineage B's *stem* into its own environment (its degrade
  preset + sonify mode). The transfer is *kept* only if the receiving
  lineage's next-pass telemetry stays within viability — fortuitous
  stepping stones survive, bad grafts are rejected. (This is the
  mechanism POET showed was essential; it is also exactly how new
  material enters an art practice.)
- **Adaptive difficulty (the POET difficulty knob):** per lineage,
  scale degrade intensity from measured drift — if drift accelerates
  toward collapse (rms falling, entropy dying), soften the codecs;
  if drift goes flat (< ε per pass, the `hold` failure mode), harden.
  Presets stop being fixed ladders and become *target drift bands*.

Deliverable: `womb metabolize --population 6 --transfer 0.25 --adaptive`
→ six reels + one population ledger; `womb info` gains a
population-level drift report comparing lineages, marking transfers
adopted/rejected (the story again). The final artifact can be a *cut*
across the surviving lineages (the montage is E7-arc-composition applied
to reels).

Effort: medium (metabolize.py; the per-pass machinery all exists).
Acceptance: a 6-population, 4-pass run shows at least one adopted
transfer in the ledger, and adaptive lineages outlive fixed-preset
lineages at equal passes under `collapse`.

---

## E4. Complexifying organ genomes — the unbounded-encoding move (brant_gecco19, pugh_ecal17)

**From:** the unbounded maze benchmark (complexity grows only when the
encoding can grow) and Voxelbuild (major transitions: structures jump
in complexity when the building genome passes a threshold).

Today organs are fixed-structure with scalar knobs; the ceiling is
hand-authored. Grow past it:

- **Organ-graph genome**: nodes are organ instances, edges are
  couplings — one organ's telemetry modulates another's params.
  Real couplings already have obvious wiring: waveplate energy →
  kuramoto coupling (strikes synchronize the field); automata births →
  swarm spawn points (the colony births its inhabitants); territory
  walker position → waveplate strike location (the tour plays the
  plate); kuramoto order parameter → automata feed (sync controls growth).
- **Mutations:** add/remove node, add/remove/rewire edge, jitter node
  params. **Minimal criterion (the "every maze is solved" analog):**
  a node must *contribute* — ablate it; if audio/visual distance from
  the parent render is below ε, it is a passenger and pruned. Every
  module earns its keep or dies.
- **The transition moment:** allow a fusion mutation that merges two
  coupled organs into one composite organ class — the first step toward
  organs that *emerge* instead of being hand-written. (This is where
  "nobody would have designed this" hybrids come from — the actual
  Voxelbuild payoff.)

Build order: v1 = a registered `symbiont` organ (exactly two children +
one coupling, hand-wired) to prove the audiovisual coupling idea; v2 =
graph genomes in the search/drift loops with add-node/ablate mutations.
Effort: v1 small-medium, v2 large. Acceptance: a search that can add
nodes discovers ≥2-node genomes with nonzero ablation-measured
contribution per node — hybrids that out-listen every single organ.

---

## E8. Bridge mode: organs inside the metabolism (POET, inverted)

The library's two modes (organs; metabolism) currently never touch.
One small bridge closes the loop: `metabolize --organ-mode` renders each
pass's stem through an *organ* whose params are set by the previous
pass's drift telemetry (onset rate → strike rate / grain density, rms →
gains, centroid → palette/pitch bias). Sonify remains the audio inverse
channel; this is the *render* inverse — the metabolism stops being
doom-only and the whole library becomes one loop. Optional, last.

---

## Suggested order (value per effort)

| # | enhancement | effort | why here |
|---|---|---|---|
| 1 | E7 craft pass | small | every artifact improves immediately |
| 2 | E5 plastic organs | small-med | long pieces develop instead of loop |
| 3 | E6 curate + resume | small | the subjectivity paper, operationalized |
| 4 | E1 multi-BC | medium | archive fills with genuinely different material |
| 5 | E2 MCC drift | medium | the research flagship; reproduces the papers inside womb |
| 6 | E3 POET metabolism | medium | transfers + adaptive difficulty; the second flagship |
| 7 | E4 organ graphs | large | the ceiling-lifter; major transitions |

Each step ships independently; the drift/POET modes build on E6's
resume and E1's descriptors, and E4 builds on everything.