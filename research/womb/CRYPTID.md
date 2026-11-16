# womb cryptid — the sound-object breeder (plan v2, audio-first)

> Refines ENHANCEMENTS.md into one flagship aimed at the refined goal:
> **diverse audio — no synthesis technique, no instruments — sounds
> traditional synthesis cannot reach.**

**Thesis.** Traditional synthesis — additive, subtractive, FM/PM, wavetable,
sampling, granular, physical modeling of instruments — shares a single
grammar: *a designed oscillation, then designed shaping*. Everything it
can produce is bounded by what a designer can specify. The way out is the
corpus of papers in this project's root: don't design the generator at
all. **Breed processes, and measure them.** Sound = the observation of an
evolved dynamical system, the way a seismograph is the "synthesizer" of
plate tectonics. Nobody asks what instrument the Lorenz system plays.

**The invariant (what makes it non-traditional by construction):**

> The only signals that exist are states of evolved processes. There are
> no oscillators, no notes, no scales, no envelopes-as-music-tech, no
> filter banks as tone shapers, no instrument samples, no corpus
> playback. The measurement may select, gate, differentiate, integrate,
> delay-embed, mix, and coordinate-project — it never adds energy of
> its own. The sole designed stage is a safety limiter.

Everything below defends that invariant while staying: this machine,
fully offline, pure numpy, seed = identity, artifact = finished audio.

---

## 1. What falls out of process-measurement that no synth grammar produces

Each phenomenon is *free* in some process family and *expensive or
impossible* in designed oscillation:

| phenomenon | falls out of | why synthesis can't |
|---|---|---|
| simultaneous tonality + noise (chimera states) | nonlocal oscillator networks with evolved heterogeneity | you can fake it by mixing a sine with hiss; you cannot get the wandering coherence boundary that coupling produces |
| power-law event streams with causal structure (1/f clicks, "avalanche weather") | self-organized criticality (sandpile nodes) | noise has the spectrum but no events; granular synths have events but designed statistics |
| signals with deterministic memory that never repeat (hyperchaos) | coupled chaotic units (multiple positive Lyapunov exponents) | oscillators are periodic or drifting-by-design; noise has no memory |
| inharmonic timbre that morphs by attractor geometry, not envelope | evolved jerk systems (Sprott's coefficient search is the precedent) | formant sweeps without a single filter |
| rhythms with cyclic structure but no metric grid | random Boolean / NK network attractors (Kauffman) | sequencers are metric by construction |
| glitch discontinuities timed by *state transitions*, not clocks | bitwise packing of network state | bitcrush quantizes a designed signal; this discontinuity carries dynamics underneath |
| multi-timescale nonstationarity (breath-like behavior over seconds/minutes) | heterogeneous evolved time constants per node | automation curves are drawn; here the drift is the physics |

Precedent for the whole approach: Sprott (1994) found his catalog of
chaotic flows by *blind search over jerk coefficients* — nobody designed
them; they were discovered. Cryptid generalizes that move: search over
networks of laws, then let the beholder keep the strange ones.

---

## 2. Architecture — four parts, one per paper mechanism

```
  process genome  --(states)-->  ear genome  --(measurement)--> wav
        ^        mutual minimal criterion (MCC)         ^
        |            every process needs an ear               |
        +------------ every ear needs a process ---------------+
  drift floor: child must differ perceptually from parent (not maximized)
  character archive: several descriptor grids (multi-BC), none is quality
  curation: the beholder is the only quality pressure (subjectivity)
```

### 2.1 The process genome — the organisms (unbounded encoding)

A growable directed network of heterogeneous units. The genome is the
adjacency + per-node law + per-node params + per-node dt. It can always
grow — the brant_gecco19 lesson: complexity only rises if the encoding
can. Voxelbuild supplies the transition hook (see mutations).

**Node laws (the zoo — each ~30–80 lines numpy):**

| law | unit behavior | sonic role |
|---|---|---|
| `jerk` | evolved Sprott-style chaos: x‴ = a·x″ + b·x′ + c·x + d·x³ | inharmonic morphing voices |
| `excitable` | FitzHugh–Nagumo cousin: rest / spike / refractory | arrhythmic pulse trains |
| `fire` | integrate-and-fire with evolved threshold | spiking rhythms, sync bursts |
| `osc` | Kuramoto unit (phase only; never a bare sine readout) | chimeras when nonlocally coupled |
| `map` | logistic/tent/sine map at evolved rates | sub-audio LFOs; aliased audio-rate = harsh material |
| `sand` | BTW sandpile: SOC avalanches | 1/f event weather |
| `bool` | Kauffman NK unit: Boolean regulatory input | non-metric cyclic gates |
| `delay` | delayed state feedback with evolved lag | echo-chaos, long-term memory |
| `clamp` | diode-like nonlinear transfer, evolved quantizer steps | the aliasing aesthetic as a *component*, not a post-effect |
| `bytefeed` | any file's bytes as a state stream (the PDFs, code, the ledger itself) | databending species — data *as dynamics*, never as playback |

Couplings: signed weighted edges with per-edge gain and per-node dt —
heterogeneous time constants are a major diversity lever. Integration:
simultaneous (vectorizable), Heun 2nd order, float32, dt per node from a
genome-bounded set. NaN/blow-up = death: the stability requirement IS
part of the minimal criterion — survival is non-negotiable.

**Mutations (all seeded, all logged):**
- param drift (log-scale jitter), law swap, node birth/death
- edge add/remove/rewire/gain-jitter, dt drift
- **subgraph duplication** — copy a connected motif; copies diverge
  later (the Voxelbuild major-transition move: new body plans appear
  when a duplicated part takes on a new role)
- **graft / transfer** — transplant a subnetwork from another lineage
  (POET's transfer of stepping stones between environments, made genetic)
- homeostatic option (RAAHN nod): node params drift online to keep their
  own state in a liveliness band, so long renders don't die into fixed
  points — the organism's minimal criterion made endogenous

### 2.2 The ear genome — the observers (the second population)

Two ears on the same process = two different species — so the ear is a
first-class evolvable genome, not a hard-coded readout. MCC pairing:
a **process** is viable iff ≥1 ear hears it alive; an **ear** is viable
iff it hears ≥1 process alive. Mazes and navigators, exactly.

Allowed ear operators (measurement, never generation): state selection
(which nodes), differentiation / integration, delay-difference /
Takens delay-embedding (a nonlinear observation, not a filter bank),
event gating (threshold crossings → spikes and their intervals),
coordinate projection (rotate the 3D attractor, take a plane — evolving
formant sweeps without filters), bitwise packing of several states,
state-derived stereo (attractor phase → field position; L/R
incoherence becomes an evolvable character). Safety: fixed limiter.

### 2.3 The minimal criterion — MCC, not fitness

- **No objective.** No "sounds good" function anywhere. (pap140s3;
  the subjectivity paper's whole point.)
- Viability bands: audible (rms in band), alive (descriptor variance
  across windows > floor — excludes silence *and* dead-white hiss
  *only at the extreme*), finite (no NaN), and within safety limits.
- **Drift floor:** a child must differ from its parent by > ε perceptual
  distance — divergence by drift subject to the MC, never novelty
  maximization.
- **The strictness dial** (soros_alife16): `--strictness 0.1..1.0`
  scales all band widths. Soros' result — stagnation at both extremes,
  orderly divergence in between — becomes a measurable property of
  *this* system: log population descriptor-diversity per generation
  and find the regime. The breeding run IS the Chromaria experiment.

### 2.4 The character archive — multi-BC without quality (pugh_ppsn16)

Several descriptor grids; each axis is a *character*, none is a rank.
Candidate is inserted into every grid; parents are chosen by picking a
grid, then a cell — search is pulled by whichever lens the offspring
lands in.

Proposed character axes (all numpy-computable on 2–6 s windows):
event density · aperiodicity (spectral flatness / autocorr depth) ·
spectral slope · temporal asymmetry · self-similarity across
timescales · modulation rate · stereo incoherence.

**Curation is the only quality pressure** (subjectivity → Picbreeder
precedent): listen to the preview montage, like specimens, liked
lineages breed more. Offline, in this machine, forever.

---

## 3. CLI & artifacts

```
womb cryptid seed   --pop 48 --gens 30 --strictness 0.5 -o runs/crypt1
womb cryptid preview runs/crypt1 -l 60 -o crypt1_previews.mp4   # listenable atlas
womb cryptid curate runs/crypt1 --likes 4 7 11                  # beholder pressure
womb cryptid seed   runs/crypt1 --resume --gens 20              # continue the universe
womb cryptid render runs/crypt1/specimen_04.json -l 120 -o voice.wav
womb cryptid info   runs/crypt1/specimen_04.json                 # its biography
womb cryptid organ  runs/crypt1/specimen_04.json                 # wrap as a womb organ
womb cryptid metabolize specimen_04.json -n 3                   # degrade a cryptid
```

Artifacts per run: `specimens/*.json` (genome + lineage + admission
telemetry), `atlas-*.png` contact sheets (one per BC grid), and the
**preview montage** — the atlas you can *listen to*: 3 s per elite,
spectrogram + attractor portrait on screen, sound below. Every specimen
carries its biography: *"born gen 14, lineage BOLT, from parent ZIZ by
law-swap node 7 (jerk→excitable); admitted at rms 0.21, flatness 0.62,
events 3.1/s."* The story of how an artifact arose is part of what
makes it interesting — the subjectivity paper, kept as data.

---

## 4. Determinism & performance (this machine, fanless)

- float32, simultaneous node update, fixed integration order — seed
  fully determines a specimen; a genome rendered twice is identical.
- Evaluation at a decimated fixed rate (e.g. 11025 Hz), readout resampled
  to 44.1k at render time — deterministic, and the resample is part of
  the measurement, not the physics. Rendering at 44.1k is a per-specimen
  choice (`--hi`).
- Budgets: nets start 8–64 nodes; genome growth is energy-priced (node
  count soft-cap) so the fanless machine survives; every run resumable.
- A 48×30 breeding pass at ~1–2 s per short evaluation ≈ 30–50 min,
  resumable in chunks — sized for this laptop, like loom/doom/womb.

---

## 5. Failure modes and their answers

| failure | answer |
|---|---|
| everything collapses to hiss or silence | the alive-band + NaN-death MC; Soros' dial; curation |
| everything sounds the same | multi-BC grids + drift floor + grafts |
| degenerate fixpoints win | homeostasis option; death-on-NaN keeps selection honest |
| the weird ones are unlistenable | they still land somewhere in the archive (character ≠ quality); the montage makes pruning a 5-minute listen |
| it's "just noise" criticism | the archive's structure proves otherwise: cells occupied across flatness, event density, asymmetry — and the biography shows provenance |
| runaway compute | energy-priced genomes, resumable runs, decimated eval |

---

## 6. Build phases

> **Phase B long run** (`womb_out/crypt_long`, review-pass after the
> first short experiments were judged not yet usable): 48 generations,
> pop 32 × 12 ears, pairs 3, strictness 0.5 — **710 specimens in 14.2
> minutes**, cells 15/47/26 (88 of 192), deepest lineage 23 generations.
> Complexity grew the way the unbounded-encoding papers predict:
> founders averaged 3.0 nodes, late-generation children 4.8 (at the
> 6-node soft cap), edges up to 7; the population's laws settled around
> excitable/map/bool/sand/fire networks (jerk stays rare — chaos is a
> sliver, as the Phase A search showed). The review pass also fixed:
> the founder loop archived the *last* pairing instead of the *viable*
> one (NaN crash — regression test added), resume replayed the original
> mutation rng (now advanced by generation), the spread ledger now
> measures the last 40 admissions (the whole-archive std diluted), and
> `_finish` sanitizes blown-up simulations. New usable artifacts:
> `womb cryptid mixtape` (the run's album — every cell rendered long and
> crossfaded) joins the preview montage.

> STATUS: **Phase A built** (laws zoo + ear ops + champions, 15 tests;
> acceptance held: 60 s renders 2–8 s wall, bit-identical determinism).
> **Phase B built** (`src/womb/breed.py`): the MCC loop — mutual-MC
> processes×ears, growable genomes, drift floor, Soros' strictness
> dial, multi-BC character archive (texture/time/space grids),
> per-generation ledger, resumable checkpoints, atlas PNGs, and the
> *listenable* preview montage. Acceptance — the Chromaria experiment
> (three regimes, same seed, 12 generations, `scripts_check/chromaria.py`):

| regime | admissions | cells t/x/s | spread | mean rms |
|---|---|---|---|---|
| loose 0.1 | 83 (junk floods) | 21 → 43 | −22% | 0.215 → 0.178 (sinking) |
| **calibrated 0.5** | 72 | **21 → 53 (+152%)** | **−14%** | 0.213 → **0.38** (bloom) |
| strict 1.0 | 49 | 15 → 32, texture frozen at 4 | **−33%** | 0.284 → 0.227 |

> Mid-strictness dominates every diversity axis (Chromaria's orderly
> divergence, reproduced). Honest nuance: at these settings the
> extremes stagnate rather than die — loose by drifting into the
> silent basin, strict by *convergence* (births continue but into an
> ever-narrower slice: the texture grid never left 4 cells). Notes
> from the build: the drift floor is calibrated so mid-strictness
> rejects only silent mutations (real mutant drift measures
> 0.01–0.04); the `events` descriptor needed a median-relative
> threshold + 50 ms envelope (near-constant envelopes cross their own
> std constantly — drones would count as click weather).

**Phase A — prove the physics (small/medium).** Node zoo + 3 hand-built
champion cryptids: a chimera drone (osc-network), a sandpile clickstorm,
an evolved-jerk voice. Acceptance: each passes a "never heard that"
listen; 60 s renders in < 30 s wall; determinism test passes (two
renders bit-identical).

**Phase B — the MCC loop (medium/large).** Growable genomes, mutual-MC
processes×ears breeding, strictness dial, drift floor, multi-BC atlas,
preview montage. Acceptance: at mid-strictness, occupied cells and
descriptor spread rise with generations; at 0.1 and 1.0, diversity
collapses — Chromaria's result reproduced inside womb.

**Phase C — the beholder (small).** Curate / resume / lineage names /
biographies. Acceptance: a curate→resume cycle measurably biases the
archive toward liked character cells.

**Phase D — integration (medium).** `cryptid organ` wraps a champion as
a womb organ: the same states drive picture (attractor portrait, network
glow, avalanche maps) and sound — the womb thesis, applied to bred
fauna; and `cryptid metabolize` feeds a champion through mp3 rounds /
resample wars / bitrot (generational loss on hyperchaos = textures no
one can design). Acceptance: one full womb piece whose every sound is
cryptid-bred and structurally synced to its visuals.

---

## 7. Relation to ENHANCEMENTS.md

This is the audio-diversity flagship, absorbing E4 (unbounded organ
genomes), E2 (MCC drift), E1 (multi-BC), E6 (curation), E5
(homeostasis). E7's craft (limiting, fades, stereo discipline) still
applies at the render stage. The metabolism (E3/POET) inherits cryptids
as new material rather than new machinery.