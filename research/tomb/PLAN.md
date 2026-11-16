# tomb — plan v1.2 (genesis phase, feature ears, unique-contribution MC)

**tomb: an in-place audio novelty search.** loom weaves it, doom renders it,
womb births it — **tomb is where only the discriminable survive.**

No fitness. No behavior archive. No global novelty metric. Two coevolving
populations — **sounds** and **ears** — each held to a minimal criterion
defined *by the other*:

- a **sound** lives only if at least one ear in the current population can
  *distinguish* it from every other sound in the current population
- an **ear** lives only if it can distinguish at least one pair of sounds
  that the population currently holds

Everything else is drift. The objective is not encoded anywhere — it is
emergent: the sounds that keep getting admitted are, by construction,
those that some way of hearing finds *new* in the presence of everything
else alive. "Interesting" is never measured; it is *survived*.

## 1. Why this fits the papers exactly

| paper | role in tomb |
|---|---|
| MCC (GECCO '17) | the core loop: dual queues, MC-only selection, speciation by genome, drift |
| Unbounded MCC (GECCO '19) | complexification: sounds may grow longer/layered; ears may grow filters |
| Soros strictness (ALIFE '16) | the τ dial (order ↔ chaos) is tomb's main exposed parameter |
| POET (2019) | transfer: every sound auditions against *every* ear; new ears born from the current soundworld |
| Pugh multi-BC (PPSN '16) | the ear population IS a multi-BC ensemble — many ways of being heard |
| Novelty-RAAHN (ALIFE '16) | upgrade path: ears from random projections → tiny learned autoencoders |
| Stanley subjectivity (OEE ws) | the artist curating checkpoints is a legitimate part of the algorithm |
| Major transitions (ECAL '17) | long runs + telemetry to watch for Cambrian bursts |

## 2. The two populations

### 2.1 sounds — the material

- **Origin: WHITE NOISE.** The seed corpus is pure noise (a few noise blobs
  of 1–4 s). This is the maximum-entropy start: nothing about "instruments"
  or synthesis grammar is baked in, and the MC — being discriminable by
  some ear — becomes the *only ordering principle* between noise and sound.
  (Optional: any folder can be imported as extra origin material later.)
- **Genome**: a lineage — `(origin noise ref → op list)`, never raw audio.
  Materializes on evaluation into 1–4 s of audio.
- **Self-referential loop**: parents of every child are *living population
  members* — the population is its own corpus. Nothing outside the tomb
  (besides the noise origin) is ever heard.
- **Mutations are MATERIAL PROCESSES, not synthesis** (the womb lesson —
  its best organs were physics/state readouts, not oscillators; tomb's ops
  treat sound as *matter* — fields, statistics, orderings, decay):

  **FIELD ops** (spectrogram = 2D matter; inherited from womb's organs):
  - `diffuse` — anisotropic blur/diffusion of the spectrogram (womb
    waveplate's damping, applied to spectra)
  - `propagate` — one FDTD wave-equation pass across the spectrogram
    (womb waveplate as a mutation operator: spectral ripples)
  - `react` — a few Gray-Scott reaction-diffusion steps on the log-magnitude
    field (womb automata/doom RD as audio shaper)
  - `automate` — one continuous-CA step on the magnitude field

  **STAT ops** (population as its own filter bank — self-reference):
  - `morph` — remap this sound's spectral envelope toward a random living
    member's envelope (breeding through timbre transfer, not DSP)
  - `convmate` — convolve with (or autoconvolve against) a living member:
    self-convolution is the literal "self-convoluting process"
  - `rephase` — keep the magnitude, randomize phase; same matter, new
    creature (cheap, powerful drift op)
  - `histogram` — match the sample-value histogram to another member

  **ORDER ops**:
  - `wavesort` — sort samples or spectral rows within windows
    (pixel-sorting, heard)
  - `shufflewin` — permute windows locally

  **LOSS ops** (generational decay — from womb's degrade, kept verbatim):
  - `mp3_round`, `resample_war`, `crush`, `bitrot` — real codec/generation
    loss; the tomb's sounds age like tapes

  **STRUCTURE ops** (complexification axis, GECCO '19):
  - `splice`, `reverse`, `layer` (mix with a living member), `expand`
    (append a mutated copy — sounds grow into multi-part creatures)

  **GENESIS ops** (structure-creators — the only ops that break the noise
  fixed point; weighted heavily during the genesis phase, rarer after):
  - `resonate` — noise through a feedback delay network (Karplus-Strong
    territory: noise becomes tone)
  - `comb` — layer delayed copies; autocorrelation combs → pitched buzz
  - `decimate` — sample-hold; noise → steppy tonal grit
  - `logistic` — samples through the logistic map (deterministic chaos
    at specific parameters → structure)
  - `gate`/`env` — impose a temporal envelope (flat spectrum, real
    temporal structure)
  - `wavesort` doubles as a genesis op: sorting random samples within
    windows creates monotonic ramps — a genuine symmetry break

- **Mutation rate**: one or two ops per child; op choice weighted by a
  per-run recipe or uniform. Each op appends to the lineage.
- **Speciation**: genetics-based, per the paper — the *lineage op histogram*
  (field/stat/order/loss/structure counts), not a timbre descriptor.
  No BC anywhere. Queue capacity split evenly across species.

### 2.2 ears — the perceivers

- **Genome**: parameters of a perceiver. v0: **filterbank ears** — each ear
  is m (band, time-window) energy probes → a descriptor in R^m. New ears are
  born in two ways: random probes, and **probes positioned at spectral
  peaks of living sounds** (POET's environment-from-solutions: ears grow
  *out of the soundworld they must hear*). Mutated by moving/adding/
  removing probes.
- **Discrimination**: sound s is *distinguishable* under ear e if
  `dist_e(s, nearest other sound) > τ`. Ear e is viable if it separates
  at least `k` current sound pairs at that same τ.
- **Strictness dial** (the Soros result, made explicit): τ controls
  order↔chaos. High τ (hard to distinguish) → stagnation. τ→0 (everything
  distinct) → degenerate drift, effectively random search. The run UI is
  mostly this one knob.
- **Upgrade path** (phase 2): ears become tiny autoencoders trained on
  the current population (RAAHN-style): "understandable by some ear" =
  reconstruction error below an adaptive threshold. Learned perception
  keeps the MC *organically recalibrating* — exactly Chromaria's insight.

### 2.3 the loop (faithful to the C# code)

```
bootstrap: seed sounds from corpus; spawn random ears until k of them
           discriminate seed pairs (novelty-search-free bootstrap)
loop:
  for pop in (sounds, ears):            # alternate, one batch each
    parents = pop.dequeue(batch)       # species-proportionate, FIFO
    children = mutate(parents)          # asexual; parents re-enqueued to back
    for c in children:
      if MC(c) with respect to the OTHER population's current queues:
        pop.enqueue(c)
    pop.remove_oldest_over_capacity()   # per species, age-sorted
  transfer (POET): each new sound evaluated against every ear, not one
  checkpoint every N batches (render sounds, snapshot state)
```

### 2.2.5 The four refinements (v1.2 — why the naive design would fail)

The v1.1 design had three predicted failure modes, all drawn from the
papers' own logic:

1. **The noise fixed point.** Almost every FIELD/STAT op is a no-op on
   white noise (blur a flat field → flat field; rephase random phase →
   random phase; convolve two random spectra → *whiter* noise). The ops
   are structure-transformers, and the population starts with no
   structure to transform → eternal drift in noise-space.
   **Fix: the genesis phase** — a bootstrap (mirroring the paper's own
   novelty-search bootstrap) where founders are white noise weathered by
   GENESIS ops (structure-creators). Genesis ops stay in the recipe
   after bootstrap but at reduced weight; FIELD ops become powerful once
   structure exists.
2. **Vacuous discrimination.** Filterbank band-energies are what noise
   flattens — ears would discriminate by meaningless fluctuation
   (Soros' "verges on chaos" regime).
   **Fix: feature-space ears** — random projections over loom's 14
   proven features (rms, flatness, centroid, pitch_conf, transient...)
   which are *meaningfully different* for noise vs. structured sound.
   Autoencoder ears remain the phase-2 upgrade, now with structured
   training data to learn from.
3. **Double-drift.** Both MCs as originally written are trivially
   satisfiable at the noise stage (any probe set separates some
   fluctuation pair) → both populations random-walk.
   **Fix: unique-contribution ear MC** — an ear is viable only if it
   discriminates a pair *no other ear currently discriminates* (Pugh
   multi-BC pressure). Ears must earn their queue slot with unique
   perception.
4. **Speciation is non-negotiable** — the paper's control converged;
   only the speciated variant diverged.

**Success experiment (defined before building):** after a 1-hour run,
listen blind to 20 random admitted sounds. Success = >= 5 keepers you
would actually use in a piece. If that fails, iterate the op recipe and
ear features — not the MCC skeleton.

## 3. What the artist touches

- `tomb run --corpus ../loom/loom_out --tau 0.8 --hours 3`
- checkpoints: every N batches, tomb renders the current sound population
  as a folder of wavs (species-tagged, lineage in filenames) + a
  population JSON snapshot. **You listen, you pick.** Picked sounds are
  re-queued (transplanted — the subjective metric, per the position paper).
- `tomb telemetry` — the paper's quantitative lenses:
  - sounds admitted / evaluations (the MC filter rate)
  - population diversity per ear (mean pairwise descriptor distance)
  - speciation counts over time (transitions watch)
  - complexity trend: mean lineage length / expansion depth over time
- All offline, numpy + loom's processors, no ML required for v0.

## 4. What I expect to happen (the research questions)

1. Does speciation matter here as it did in mazes? (paper: yes, strongly)
2. Does the ear population prevent the sound population from collapsing
   into one timbral niche — the audio analog of "similar trajectories"?
3. Where is the τ sweet spot, and does it drift as populations age
   (Soros' implicit-MC warning)?
4. Do we see major transitions (ECAL '17) — long stasis then bursts?
5. Does layered/expanded sound complexity ratchet upward without being
   rewarded (GECCO '19's reciprocal complexification)?

## 5. Build phases

- **Phase 1 — the core** (pure numpy + loom): genomes, queues, speciation,
  filterbank ears, MC evaluation, bootstrap, CLI, checkpoints, telemetry,
  tests. *Acceptance: a 30-minute run on this machine produces ≥ 200
  admitted sounds with non-collapsing speciation and a positive complexity
  trend, reproducible from its seed.*
- **Phase 2 — learned ears** (torch, optional venv like womb's): tiny
  autoencoder ears, reconstruction-based MC. Compare telemetry vs v0.
- **Phase 3 — expansion**: layer/expand ops, POET-style ear birth *from*
  soundworld statistics, artist-picking CLI, a "bestiary" renderer
  (contact-sheet of the population's spectra over time).

## 6. Decisions

1. ~~Corpus~~ — **ANSWERED: white noise origin, self-referential population**
   (no instrument/synthesis bias; the population is its own corpus).
2. **Ear style for v0** — filterbank perceivers (fast, transparent) or start
   with tiny autoencoders (learned, slower)?
3. **Run scale** — minutes-scale tinkering or overnight runs?
4. **Sound duration range** — 1–4 s snippets, or bigger 8–16 s creatures?

## 7. Why this escapes instrument/synthesis thinking

Every traditional "audio mutation" (oscillators, filters with musical
grammar, note-based ops) smuggles in an instrument worldview. Tomb's
operator families are imported from *state physics and statistics*:

- womb's waveplate proved 2D wave equations sound like metal and thunder —
  as a *mutation*, propagation turns any spectrogram into resonant matter
- womb's automata/RD proved CA on fields yields organic texture — as a
  *mutation*, reaction-diffusion on the log-magnitude field is erosion
- womb's degrade proved real lossy codecs are beautiful — the LOSS family
  means tomb sounds *age*, acquiring generational patina
- `morph`/`convmate` mean every new sound is shaped by *the population
  itself* — the corpus is alive, and no sound is ever processed by a
  filter that didn't come from another sound
- `rephase`/`wavesort`/`histogram` are pure statistics — matter
  redistribution, no musical grammar at all

White noise → field physics → statistics → decay: the sound is never
"played," it is *weathered*.