# tomb — research digest

Papers read (all in ../): pap140s3-file1 (MCC), brant_gecco19 (unbounded
MCC), 1901.01753v3 (POET), soros_alife16 (MC strictness), bowren_alife16
(Novelty-RAAHN/TANE), pugh_ppsn16 (multi-BC QD), pugh_ecal17 (major
transitions), stanley_oee2workshop16 (subjectivity position).

Code read: ./MinimalCriterionCoevolution (C#/SharpNeat) — MCCAlgorithmContainer,
QueueEvolutionAlgorithm, per-species FIFO queues, evaluator baselines.

## 1. Minimal Criterion Coevolution — Brant & Stanley (GECCO '17)

Two coevolving populations (mazes ↔ maze-solving neural agents) with **no
fitness function, no behavior characterization, no novelty archive**.
The only selection pressure:

- **MC (agent)**: solve at least one maze in the current maze queue.
- **MC (maze)**: be solved by at least one agent in the current agent queue.

Everyone who satisfies the MC enters a **fixed-capacity FIFO parent queue**
and is guaranteed at least one reproduction; overflow removes the oldest.
No ranking exists — divergence comes purely from **drift** under the MC.
Bootstrap: novelty search evolves seeds that satisfy the MC.
Result (control): convergent runs; Result (**speciated variant**): genuine
open-ended divergence — mazes and solutions of steadily growing variety and
complexity. Speciation is **genetics-based** (no behavior description):
queue capacity split evenly across species; per-species FIFOs; oldest-
removed per species. Complexity ratchets reciprocally: harder mazes force
better agents; better agents admit harder mazes.

## 2. Unbounded MCC — Brant & Stanley (GECCO '19)

Removes the fixed-size maze ceiling: a waypoint encoding lets mazes
**expand indefinitely** (expand-mutation grows the outer boundary). MCC
then produces an unending succession of larger, more complex mazes that
remain solved — the first benchmark of sustained open-endedness outside
an alife world. Lesson: give both populations an **unbounded
complexification axis** and the reciprocal ratchet keeps turning.

## 3. POET — Wang, Lehman, Clune, Stanley (2019)

Builds on MCC: paired populations of *environments* and *solutions*, each
environment optimized by its own agent, plus two extra mechanisms:
- **Transfer / goal switching**: periodically try each agent on every other
  environment; adopt better pairings. Stepping stones cross-pollinate.
- **Niching of environments** (novelty among environments) so the problem
  population stays diverse.
Lesson for tomb: transfer = letting sounds audition against *every* ear,
not just their paired one; and new "environments" (ears/tests) should be
generated *from* the current solution population.

## 4. MC strictness — Soros, Cheney, Stanley (ALIFE '16, Chromaria)

The MC is a **dial between order and chaos**: too strict → stagnation
(genetic homogeneity); absent → degeneracy and "verges on chaos" noise
(no cohesion, no correlation — indistinguishable from random search).
Moderate strictness yields **orderly divergence**. Any selective force
that caps a % of reproducers behaves like an over-strict MC.
Lesson: tomb's MC thresholds (discrimination distance, queue capacities)
are its most sensitive parameters — expose them as one dial.

## 5. Unaligned QD / multi-BC — Pugh, Soros, Stanley (PPSN '16)

When diversity is unaligned with quality, a single BC fails; drive search
with **multiple BCs simultaneously** and quality survives. Lesson: don't
pick one "ear" — a society of different perceivers is a multi-BC engine.

## 6. Novelty-RAAHN — Bowren, Pugh, Stanley (ALIFE '16)

Hebbian learning + autoencoder + **novelty-filtered experience buffer**:
an agent collects only experiences novel under its (learned) feature space.
Lesson: the *ear* itself can be learned, and novelty collection can gate
memory. Upgrade path for tomb's ears: random projections → learned tiny
autoencoders (fits the 16 GB laptop).

## 7. Subjectivity position — Stanley & Soros (OEE workshop)

Open-endedness is partly **in the eye of the beholder**; no metric fully
captures it. Lesson: the artist is a legitimate evaluation function —
human curation of checkpoints is not cheating, it is the missing metric.

## 8. Major transitions — Pugh et al. (ECAL '17, Voxelbuild)

Long QD runs show sudden **innovation explosions** (Cambrian-like bursts)
between long stable periods. Lesson: tomb runs should be long and
logged; expect (and look for) phase transitions in the telemetry.

## 9. From the MCC codebase (C#/SharpNeat)

- `MCCAlgorithmContainer<T1, T2>`: alternates one generation per
  population, then **`UpdateEvaluationBaseline`** — each evaluator receives
  the other population's decoded phenotypes. The MC lives in evaluators.
- `QueueEvolutionAlgorithm`: species-local FIFO; asexual reproduction;
  parents dequeued from the front, re-enqueued at the back **before**
  children are admitted; children admitted only on MC satisfaction;
  `RemoveOldestFromOverfullSpecies` (age-sorted, per species).
- BodyBrain domain generalizes: morphologies ↔ controllers (different
  encodings per population) — proof that MCC populations need not share
  a genome type.