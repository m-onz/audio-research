# scan

**the wavetable arena** — a second country for the maze. One evolvable
wavetable rig — table content, morph path, vector mix, and pitch path
all born, not designed — bred under four selection policies, identical
everything else. maze proved echo's memory beats memoryless drift in
the FM country; scan asks whether the result *transfers* to the
wavetable/vector country, whose mutation topology is completely
different (harmonic frames that morph smoothly, instead of FM's
discontinuous index regimes).

```
   genome = { frames[harmonic partials, inharm],   the medium — born, not designed
              morph path,                          how you move through it
              vector voices + mix path,            how tables combine (Prophet VS)
              pitch path (f0 over time)            how it is read out  ← evolvable
            }
        │
        ├─ render: mip-mapped band-limited tables, phase-accumulated
        │          read, bilinear (position × frame) interpolation,
        │          vector crossfade — pure numpy, ms per clip
        │
        └─ echo encode → novelty vs the archive → which children live
```

**STATUS: PHASE E BUILT — the substrate organism (DeepHyperNEAT,
repurposed) + the extended arc.** The wt organism (phase A), the
loop (B), the definitive wt run (C) and now the sub organism —
CPPN-decoded substrate frames, torch-backed, GPU-pluggable — exist
and are tested (41 tests). The 2000-gen extended arc runs in
minutes, not days.

---

## why this arena (the repo's own evidence)

1. **maze says so, explicitly.** The definitive run's verdict ends:
   *"The remaining ceiling is the arena itself: the FM genome's
   reachable manifold under its mutation kernel. To push past it:
   bigger leaps, more operators, or **a second arena (wavetable/vector
   synthesis)** — the measurement and the memory transfer as-is."*
   This is that arena.
2. **prism proved the physics but froze the pitch.** prism's
   wavetable mode builds 64 band-limited single-cycle tables from a
   CPPN and scans them — but `f0` is a *render parameter* (110 Hz
   default), never a gene. The one axis the user actually hears most
   (pitch) is outside the search. scan makes the readout evolvable,
   which forces the one real engineering upgrade over prism:
   **mip-mapped band-limiting against a time-varying f0** (below).
3. **cryptid's complement.** cryptid asks "can we go beyond the
   traditional synthesis grammar?" and answers it with dynamical
   processes. scan asks the mirror question: *"can novelty search
   fill the traditional grammar's country — wavetable + vector, the
   PPG/Prophet VS/Wavestation grammar — with zero designer input?"*
   The FM maze answered for Chowning's grammar; this is the sibling
   country, with different geography: wavetable space is full of
   *smooth morphing gradients* (interpolating frames), which drift
   explores well, plus discrete *law swaps* (step-mode morph =
   wavesequencing) that need leaps.
4. **The medium thesis.** The frames ARE the medium — evolution
   sculpts matter; the paths are the performance of that matter.
   Same split as eden (voices vs listener) and maestro (organism vs
   spine), transplanted to a single organism.

## the grammar being bred (the research, condensed)

| synth | mechanism | what evolution gets |
|---|---|---|
| **PPG Wave 2** (Wolfgang Palm, 1981) | 30 wavetables × 64 single-cycle waves; a position control/envelope/LFO sweeps the table; 8-bit DAC made the sweep audibly stepped (character), analogue filter smoothed it | the **morph path** — timbre evolution without an envelope |
| **Prophet VS** (Sequential, 1986) | 4 digital oscillators each picking one of 127 ROM waves; a 2-D joystick (recordable, automatable) crossfades them | the **vector mix** — timbre as *mixing*, not scanning |
| **Korg Wavestation** (1990) | wave sequencing: up to 255 steps, each a PCM/single-cycle wave with own duration and crossfade time (0–998), loopable | **step-mode morph** — a wavetable that is also a *sequence* |
| modern engines (Serum/Vital class) | mip-mapped band-limiting: per-frame harmonic-truncated variants selected by instantaneous playback pitch | **alias-free pitch sweep over a fixed table set** — the piece prism lacks |

scan takes all four: frames (the medium) + morph path (PPG) +
vector voices (VS) + step mode (Wavestation) + mip-mapped readout
(modern). All four axes are genes.

## the organism

### genome (~50–80 genes, growable to ~200)

```
WtGenome:
  f0            base frequency, 30–2000 Hz (log drift)
  pitch_path    {type: fixed|glide|vib|walk|step, rate, depth, end}
  frames        K = 4–64 single-cycle tables, each:
                  partials  P = 2–32 amplitudes (log-scale), born/die
                  inharm    stretch exponent (0 = harmonic, 1 = clangorous)
                  phases    random per partial (eden's lesson: never zero-align)
  morph_path    {type: lfo|curve|step, rate, depth, shape, duty}
                  lfo   = smooth scan (the PPG sweep)
                  curve  = free breakpoint path
                  step   = wavesequencing (holds + jumps, Wavestation)
  voices        1–4 vector voices, each: {detune, level, pan, morph_offset}
  vector_path   2-D mix path over the simplex of voices (the VS joystick):
                  two slow LFOs or a free curve
  env           per-voice {attack, decay_tau} — minimal, one per voice
  seed          deterministic render
```

### render (pure numpy, measured 14 ms per 2 s clip)

1. **Decode frames**: `frame_k[n] = Σ_p a[k,p] · sin(2π·p·(p+1)^inharm_k·n/N + φ[k,p])`,
   N = 2048. Harmonic frames via masked inverse FFT (exact, one per
   level); inharmonic frames by direct partial sum. DC never exists;
   phases are seeded-random per (frame, partial) — eden's lesson:
   never zero-align.
2. **The integer mip**: every frame is pre-materialised at *every*
   partial count 1..P — a single cumsum in the partial domain makes
   all variants cost one render. At playback the instantaneous f
   selects `v_lo = ⌊edge/f⌋` and blends toward `v_lo+1` with weight
   `b = edge/f − v_lo`, **hard-gated to zero when the blend partial
   would cross Nyquist**. A high note reads a short table; a sweep
   reads them in order; the only softness is one transition partial,
   and the gate kills even that. (Better than the power-of-two mips
   the design first sketched — Vital-class engines do this per
   partial; the cumsum trick gets the same effect from tables.)
3. **Phase-accumulate readout**: `phase += f_v(t)/sr`; bilinear
   interpolation across (sample, frame) — the morph position
   `m_v(t) = morph_path(t) + voice_offset`; equal-power crossfade
   between adjacent frames; step mode holds each frame and applies
   `duty` as a segment-edge crossfade (the Wavestation's crossfade
   parameter — duty 0 is hard wavesequencing).
4. **Vector mix**: 1–4 voices on the corners of a square, bilinear
   joystick weights from the vector path, sqrt-weighted (equal
   power), per-voice detune (±1200 c), level, pan, envelope.
5. **The safety limiter** (the house invariant, prism's `_polish`):
   rms-normalise, edge fades, peak scale — the only designed stage.
   Synthesis at 11.025 kHz, exact FFT upsample ×4 (maze dx's trick).

### mutation kernel (drift + birth/death + law swaps + leaps)

| op | p | what it does |
|---|---|---|
| partial drift | — | log-scale amp jitter (dx `_jit`) |
| partial birth/death | 0.10 | P grows/shrinks within 2..32 (the unbounded-ish axis) |
| inharm drift | — | harmonic ↔ metallic continuum |
| frame interpolate | 0.08 | insert a crossfaded frame between neighbours — K grows; the *smooth-complexification* move |
| frame duplicate+diverge | 0.04 | subgraph duplication, wavetable edition: copy a frame, hook it in, let it drift |
| frame birth/death | 0.06 | random frame / delete a frame (K in 4..64) |
| morph drift | — | rate/depth/shape jitter |
| morph law swap | 0.05 | lfo ↔ curve ↔ step (the Wavestation toggle) |
| pitch drift | — | f0 jitter; path params jitter |
| pitch law swap | 0.04 | fixed ↔ glide ↔ vib ↔ walk ↔ step |
| voice drift / birth / death | 0.06 | detune, pan, level; voices 1..4 |
| **the leap** | 0.02 | one axis jumps hard: a frame re-randomised, morph law swapped at max rate, f0 octave jump, vector path re-drawn |

### archetype labels (ground truth off the genome, zero audio)

`tone` (one partial, static morph) · `organ` (many harmonics, static) ·
`metal` (inharm high) · `formant` (partials clustered) · `scan` (deep
slow morph) · `sequence` (step-mode morph) · `ensemble` (≥3 detuned
voices) · `glide` (pitch path ≠ fixed) · `wash` (fast morph + detune) ·
`chord` (voices at near-integer ratios). Ten words, read straight off
the genes like dx's seven — the census needs no ears.

## the experiment (maze's engine, adopted — phase B, built)

Identical founder, mutation kernel, viability gate (audible + the
4-block static-drone gate), budget, seeds. Only the memory differs:

| arm | parent selection |
|---|---|
| `greedy` | argmax `tonality(genome)` — pitch purity (few partials, zero inharm, zero detune): converges to the sine trap, honest and audible |
| `random` | uniform — the √t diffusion control |
| `cell` | grid-latest on the independent plain-feature grid — womb's rule |
| `novelty` | top quartile by `bank.novelty` + relative floor at half the current median — echo's memory |

Measurement adopts maze's `measure.py` untouched in spirit: 24×24
independent plain-feature grid (no mel, no chunks, no z-norm), pairwise
distance, radius from founder, archetype census, and the one
echo-measured number — parent-recall against true lineage. Artifacts:
`journey.wav` (exploration in generation order), `mixtape.wav` (final
population), per-specimen wavs + genome JSON, `report.md`.

### smoke run (10 gens × 2 seeds, `scan_out/smoke/`)

| arm | kinds ever | cells ever | yield | pairwise | radius | parent-recall |
|---|---|---|---|---|---|---|
| greedy | 4.50 | 12.5 | 0.1562 | 0.199 | 0.33 | 44% |
| random | 5.00 | 11.0 | 0.1383 | 0.267 | 0.20 | 45% |
| cell | 6.00 | 10.5 | 0.1313 | 0.303 | 0.24 | 37% |
| **novelty** | 4.00 | 11.0 | **0.2750** | 0.232 | 0.20 | **54%** |

Ten generations is weather, not climate — but the early signal is
the maze signal: **novelty's yield per admission is 2× every other
arm** (it admits only half its children, so raw totals undercount
it), parent-recall is the best (the archive reconstructs lineage
from sound), and the census tells the story — greedy drifts toward
`chord` (the purity objective's gravity), novelty's population is
`scan/metal/wash/sequence`: the morphing country. The independent
grid shows all four arms' final populations interleaved — no arm owns
the axes; the difference is *which* territory gets settled.

Artifacts per arm per seed: per-specimen wavs (archetype-tagged),
`mixtape.wav` (final population), `journey.wav` + `journey.json`
(the run's exploration in generation order — novelty_s0's tour reads
wash → chord → scan). `experiment.json` carries the full telemetry.

### the definitive run (phase C ✓ — `scan_out/wt_long/`, 240 gens × 3 seeds)

| arm | kinds ever | cells ever | yield | pairwise | radius | parent-recall |
|---|---|---|---|---|---|---|
| greedy | 7.33 | 16.3 | 0.0068 | 0.138 | 0.27 | 9% |
| random | 7.67 | 15.3 | 0.0064 | 0.305 | 0.28 | 8% |
| cell | 7.33 | 17.0 | 0.0071 | 0.404 | 0.40 | 9% |
| **novelty** | 6.67 | 16.3 | **0.0136** | 0.247 | 0.26 | 9% |

Read honestly — three findings, one per hypothesis:

- **H1 half-held**: novelty's yield per admission is **2× every
  other arm** (0.0136 vs 0.0064–0.0071 — maze's ratio, replicated),
  but total cells tie at ~16 because **the arena saturated**: every
  arm's territory curve is flat by 25% of the run. The maze lesson
  fires exactly as the honest limits predicted — when the country is
  small, memories can't differ in totals, only in efficiency. The
  next move is the kernel, not the budget (sharpen below).
- **H4 confirmed, spectacularly**: greedy's final census is
  **tone×60** — sixty specimens, all pure tones. The purity objective
  walked the whole population into the fixed point and stayed there.
- **The novelty arm chose a country**: its census is
  **sequence×59, scan×1** — evolution dove into wavesequencing, the
  rhythmic-timbre region no other arm settled (cell found
  glide/sequence, random found chord/sequence). Novelty didn't just
  explore more per admission; it explored a *different place* — the
  listenable proof is `wt_long/novelty_s*/journey.wav`.
- parent-recall ~9% for all arms at 240 gens (vs 54% in the 10-gen
  smoke): with ~2400 bank entries the parent is rarely in a child's
  top-5 — maze's definitive run showed the same 10–16% collapse.
  Small banks recall; big archives discriminate.

**The saturation diagnosis** (the kernel, not the budget): the plain
grid axes (brightness/noisiness/events) don't resolve the morph
country — a stepper and a scanner can sit in the same spectral cell —
and the law-swap probabilities (morph 0.05, pitch 0.04 per birth)
mean grammars change slowly. Next moves, in order of expected effect:
raise law-swap weights 2–3×; add inharm/scan-depth axes to the
independent grid's cell key; or run the 2000-gen long arc (phase D)
where the log-curve test lives.

### hypotheses (falsifiable, before the run)

- **H1 (transfer)**: in the wavetable arena novelty ≥ random on
  cells-ever and yield/admission, replicating maze's 2×–3× yield gap.
- **H2 (topology matters)**: smooth-frame mutations (interpolate,
  partial drift) dominate early discovery; law swaps and leaps are
  where the novelty arm pulls ahead late (drift alone fills the local
  basin, the archive forces the discontinuous jumps).
- **H3 (the pitch axis earns its genes)**: renders with frozen f0
  (a control arm) cover measurably fewer cells than pitch-path genomes
  at equal budget — the axis prism left outside the search is real
  territory.
- **H4 (greedy trap re-fires)**: the purity objective converges to
  single-partial tones at fixed pitch, the analogue of maze's 35 pure
  sines.

## the rig — technology survey (researched, decided)

| option | what it buys | cost | verdict |
|---|---|---|---|
| **numpy rig** (chosen) | full control of every gene→sample step; ms renders; bit-deterministic; prism's wavetable code is an in-house reference (band-limiting, morph crossfade, the phase lessons); zero new deps | the mip-map readout is the only new physics (~60 lines) | **the loop lives here** — the breeding inner loop, like maze/prism/eden |
| **Pure Data headless** (`pd -nogui -batch`, `[writesf~]`, or libpd/pylibpd) | a real patch ecosystem; graphical rig design; `pd-offline` (ruby/libpd) proves offline batch rendering works | patch generation is its own text DSL; control scheduled at 64-sample blocks makes precise morph automation clunky; per-render process spawn ~50–100 ms; float-locale parsing pitfalls; bit-determinism across Pd versions is a risk | wrong tool for the inner loop; right tool later for a **playable phase-D rig** (breed offline, perform live) |
| **Csound** (`ctcsound` API) | genuinely good headless API: `compileOrc` from a Python string, numpy↔function-table via `tableCopyIn`, offline render to file; GEN10 *is* additive partial specification; world-class filters/reverb | per-instance startup cost; codegen round-trips; text orchestra | **the export/mastering organ** — phase D: render champions through Csound filters, export evolved frames as GEN10 tables or 16-slot wavs; not the inner loop |
| **SuperCollider NRT** (`scsynth -N score.osc _ out.wav 44100 WAVE int16`) | VOsc/VOsc3 are *literally* vector-synthesis wavetable interpolation UGens; the OSC score-file format is simple enough to write directly from Python (documented); buffers fill at time 0 (async ops are sync in NRT) | per-render server spawn (~100 ms+); float32 engine; another runtime to keep deterministic | viable offline path, but 10–50× slower per specimen than numpy and one more moving part; revisit if a SC-native instrument is wanted |
| **C++** | fastest possible render | weeks of engineering; the ecosystem loses its numpy determinism/independence | only if render cost ever dominates — at ~3–10 ms per clip it never will |
| **osc_gen** (pip, GPL-3) | handy utilities: morph(), from_wav slicing/resynthesis, .h2p export | GPL-3 license friction vs this ecosystem's MIT; not needed for the core | optional, for the *export* step only, if license is acceptable |
| **sprechstimme / other synth libs** | batteries included | heavyweight, not built for headless batch determinism at the gene level | no |

The decision is the ecosystem's standing law: **ears select, processes
energize, numpy renders, ffmpeg writes, nothing goes online** (echo's
README, maze, prism — all of them). A numpy rig is also the only option
where the mutation kernel and the renderer share floats exactly —
which is what makes mutation-monotonicity (echo bench's claim) testable.

## performance (measured on this machine, M4 / 16 GB)

The first smoke run took 1m55s; a 240-gen × 3-seed definitive run
projected ~70 minutes. Profiling one arm-generation found the truth:
**echo's encode was 72.6% of the loop (113 ms per child), and it was
slow for a pathological reason** — homebrew's arm64 OpenBLAS ships a
generic `SANDYBRIDGE` kernel with busy-spinning threads that stall
at every tiny matmul (the mel projection is 169×1025 @ 1025×64):

| measurement | spinning BLAS | pinned (`OPENBLAS_NUM_THREADS=1`) |
|---|---|---|
| echo's mel matmul | 66–92 ms | **0.9 ms** |
| echo encode, per clip | 113 ms | **9.4 ms** (12×) |
| 10-gen smoke (2 seeds, 4 arms) | 1m55s | **7.6s** with parallelism |
| 240-gen × 3-seed definitive | ~70 min est. | **5m11s** |

Two fixes, both now defaults:

1. **BLAS pinned to one thread** (`scan/__init__.py` sets it before
   numpy loads; spawned workers inherit it). The loop's matrices are
tiny; threads only spin. Renders stay bit-deterministic (no BLAS
   in the render path at all — FFT, gathers, cumsum); echo vectors
   are deterministic within an environment.
2. **The (arm, seed) grid runs as independent processes** — 12 runs
   on 10 cores, each with its own bank, artifacts and seeded rng
   streams. `test_parallel_equals_serial` holds the science to
   bit-identity (elapsed telemetry aside), so the speedup is free.
   `--jobs 1` is always available for serial runs.

**Why not the GPU** (checked, measured, declined): MPS does not
support float64 at all (torch raises on the first tensor), the
per-op dispatch overhead (~0.1–1 ms across dozens of small ops)
exceeds the compute in a 22-MFLOP matmul, and porting echo to torch
would break the ecosystem's numpy-determinism law for zero net
gain — after the BLAS fix the loop is latency-bound in table
gathers and small FFTs, not BLAS. The GPU is the right tool for
phase-D exports (doom video, RAVE latents), not the breeding loop.

## phase E — the substrate organism (DeepHyperNEAT, repurposed)

`./DeepHyperNEAT` (neat-python derived) evolves a CPPN that is
DECODED into a substrate — a deep network whose weights are painted
by querying the CPPN at geometric coordinates — plus the Deep
mutations that grow the substrate's depth and breadth. Its Python-
loop, global-RNG execution was not directly usable (it evaluates per
point, unseeded, ~1000x too slow for a breeding loop); what is
repurposed is the SEMANTICS, in house style:

| DeepHyperNEAT | scan's sub organism |
|---|---|
| `query_cppn(source_coord, target_coord)` per connection, in a Python loop | one batched CPPN query per layer pair (all src×dst pairs at once), torch |
| weight = clamp(cppn out)·max_weight | identical convention (MAX_WEIGHT 3.0) |
| substrate = task network (xor, mnist...) | substrate = the FRAME-CONTENT generator: maps (partial_pos, morph_pos) → partial amplitude |
| `mutate_increment_depth` / `increment_breadth` | `op_inc_depth` / `op_inc_breadth` (bounded 8 sheets × 24 wide) |
| neat-python population/species machinery | scan's engine: echo-novelty admission, maze's four arms |
| global `np.random` (unseeded) | seeded numpy generators, bit-deterministic per device |

**The hinge**: the substrate's output lands EXACTLY in scan's
harmonic-domain frame format, so the whole rig is inherited — the
integer-mip band-limiting, the seeded phases, the morph/vector/pitch
paths, viability, artifacts, the measurement. The genome = CPPN
(prism's full mutation set: drift, add_conn, add_node, add_output,
swap_fn, toggle, duplicate) + substrate geometry + the grid genes
(P 8..64 partials, K 4..128 frames) + scan's paths. Frames are never
genes — they are materialised (and cached) from the substrate.

`--organism sub` selects it; champions: **rampbank** (a CPPN that
ignores geometry — constant answer → all-ones weights → a saturation
V-valley along the p+m diagonal; spectra brighten along morph),
**comb** (w = sin(4d): weights oscillate with geometric distance —
interference spectra), **stepper** (the comb played as a Wavestation).

### measured — the sub definitive run (240 gens × 3 seeds, `scan_out/sub_long/`)

| arm | kinds ever | cells ever | yield | growth 25/50/75/100% |
|---|---|---|---|---|
| greedy | 7.00 | 15.0 | 0.0063 | 14 15 15 15 (flat — the trap) |
| random | 8.33 | 17.0 | 0.0071 | 14 15 16 17 (still rising) |
| cell | 8.33 | 17.0 | 0.0072 | 15 16 16 17 |
| **novelty** | 7.00 | **18.0** | **0.0150** | **15 16 18 18 (still rising)** |

Against the wt definitive run, three differences that matter:

- **no saturation**: every memory arm's territory curve is still
  rising at gen 240 (wt's were flat by 25%) — the unbounded content
  country behaves like one. novelty ends at 18 cells vs wt's 16.3.
- **yield doubles again** (0.0150 vs 0.0063–0.0072) — maze's ratio,
  replicated in the second arena, second organism.
- **kinds ever 9** for random/cell (wt: ~7.3) — the formant and
  ensemble territories open up; only `metal` stays unreachable
  (substrate frames are exactly harmonic; inharm needs the growable
  output channel — the honest limit below). greedy converged to
  organ/glide/tone — the trap bends to the P≥8 floor but still
  converges and stays boring; novelty's census spreads across
  scan/glide/sequence/wash/formant.

### the GPU, honestly (measured, M4 / 16 GB)

The substrate forward and decode are torch (float32), device-
pluggable. The measurements that decided the defaults:

| workload | cpu | mps |
|---|---|---|
| champion render, shapes warm | 7–13 ms | 7–13 ms (tie) |
| fresh founder, NEW shape | **16 ms** | 99 ms (shader compile per shape) |
| batched siblings sharing shape | — | the win case (one bmm chain per group) |

A breeding population constantly drifts P/K/widths — every child is a
new shape — so **auto = cpu**; MPS pays ~6× per child at v1 sizes.
The GPU's honest paths, both built: `render_batch` groups siblings
that share a substrate shape into ONE batched matmul chain (torch.bmm)
— the shape-stable case where MPS wins — and `--device mps` is always
available for scale-up (bigger sheets, finer grids). CPU bmm is
bit-stable: a genome's amp depends only on its genes (tested); MPS is
deterministic per run (tested). Determinism note: float32 on GPU is
per-device; the breeding loop pins one device per run.

### the extended arc (phase D, sub organism)

```bash
scan run --organism sub --arms novelty --gens 2000 --seeds 0 1 2 \
       --pop 20 --births 10 -o scan_out/sub_2k        # ~12 min on M4
```

maze's 2000-gen log-curve test, now in the unbounded country: if the
novelty arm keeps a rising curve where wt's arena saturated, the
substrate organism is the open-ended one. Results land in
`scan_out/sub_2k/report.md` + per-seed journey wavs.

## phases

- **A — the organism.** ✓ genome + render + mutations + archetype
  labels + three champions + determinism/bounds tests. Acceptance
  held (see the champion table above).
- **B — the loop.** ✓ maze's engine + measure ported (4 arms,
  viability, bank bookkeeping, artifacts); smoke run 10 gens × 2
  seeds — novelty 2× yield, best parent-recall. 27 tests passing,
  including parallel-equals-serial bit-identity.
- **C — the definitive run.** ✓ 240 gens × 3 seeds × 4 arms in
  `scan_out/wt_long/` (5m11s after the performance work): greedy →
  tone×60 (the trap, confirmed), novelty → sequence×59 (it chose the
  wavesequencing country), 2× yield — and arena saturation by
  gen 60, diagnosed above. The 2000-gen long arc is the next move.
- **D — the long run + export.** novelty-only 2000 gens (the
  log-curve test); champions exported as 16/64-slot wavetable wavs
  (loadable in any sampler) + GEN10 CSD manifests + a mixtape.
  Optional Csound mastering pass; optional libpd performance patch
  generated from the champion JSON.

### champions (phase A's known answers — built, measured)

| name | wiring | known answer |
|---|---|---|
| **organpipe** | 1 timbre, 12 harmonics at 1/k, four identical frames, fixed f0, single voice | the additive reference — a static reedy organ; the viability gate's near-miss (rms moves only 1.1× across the clip) |
| **surfer** | 32 frames morphing sine → bright PWM-ish stack, sine-LFO morph at 0.3 Hz, glide 110 → 55 Hz | the PPG sweep — timbre evolution with zero designed envelopes |
| **sequencer** | 8 unrelated frames (sine, organ, formant, bright, metal, hollow, dense), step morph 2 Hz with duty 0.15, step pitch jumping fifths, 2 voices ±7 c | the Wavestation — rhythmic timbre from table order alone |

Measured (8 s renders, this machine):

| champion | wall | rms | centroid | crest | archetype | centroid trajectory (8 blocks) |
|---|---|---|---|---|---|---|
| organpipe | 31 ms | 0.179 | 185 Hz | 3.0 | organ | static — the medium at rest |
| surfer | 37 ms | 0.158 | 184 Hz | 6.0 | scan | 146 → 212 → 66 — the sweep, audible in numbers |
| sequencer | 50 ms | 0.087 | 526 Hz | 10.9 | sequence | 9 → 587 → 5 — the steps |

**Phase A acceptance, held by tests (19 passing):** bit-identical
renders from seed; all three morph grammars audibly distinct
(cross-correlation < 0.95, the stepped scan block-variance > 1.3×);
the integer-mip kills the high-partial alias (f0 2000 × 32 partials:
leak < 0.1% of energy in the alias band) while keeping low-pitch
harmonics intact; the pitch axis earns its difference (fixed vs glide
decorrelate); **14 ms per 2 s render** on viable random founders; a
20-mutation walk keeps 100% of children viable (bar ≥ 60%); the
greedy trap is a true fixed point (structural tonality — partial
count cannot fall below 2, so the optimum is stable); json round-trips
bit-identically. `scan audition` renders a listening batch — you are
the judge.

## layout

```
src/scan/
  organism.py   # WtGenome, the integer-mip render, mutate + the
                 # structural ops (factored, tested), archetype, tonality
  champions.py   # organpipe / surfer / sequencer + stats()
  engine.py      # the four arms (maze's, adopted), bank bookkeeping,
                 # mixtape + journey artifacts, the report
  measure.py     # the independent judge (maze's, adopted): plain
                 # features, coverage, radius, ascii map
  cli.py         # scan champions | render | random | audition | info |
                 #          run | report
  __main__.py
tests/           # 26 tests: determinism, bounds, mip anti-alias + its
                 # mirror, mode distinctness, the trap fixed point,
                 # bounds-respecting ops, the mutation walk, speed, io,
                 # the engine smoke, budget fairness, founder identity
scan_out/        # champions/, audition/, smoke/ land here
```

## usage

```bash
pip install -e ../echo -e scan                # numpy + echo are the deps
pip install torch                             # optional: the sub organism

scan champions                                  # the three, with blurbs
scan render surfer -l 8 -o scan_out/surfer.wav # a champion -> audio
scan render comb -l 8 --device mps             # a sub champion, on GPU
scan render founder.json -l 4 --hi             # full-rate alias check
scan random --seed 7 -o founder.json           # a founder
scan audition --seed 5 -n 8 -o scan_out/audition  # listen; you judge
scan info sequencer                             # anatomy, off the genome
scan run --gens 10 --seeds 0 1 -o scan_out/smoke       # phase B smoke
scan run --gens 240 --seeds 0 1 2 -o scan_out/wt_long  # phase C: the run
scan run --organism sub --gens 240 --seeds 0 1 2 \
       -o scan_out/sub_long                            # phase E: substrate
scan run --organism sub --arms novelty --gens 2000 --seeds 0 1 2 \
       -o scan_out/sub_2k                              # the extended arc
scan report scan_out/sub_long                    # reprint the table
```

Python: `from scan import WtGenome, render, random_genome, mutate,
viable, tonality` — the same surface maze's engine expects from an
organism module — which is why the port was a port.
## requires

Python ≥ 3.10, numpy, ffmpeg on PATH, **echo** as a library
(`pip install -e ../echo`). Everything offline, deterministic from
seed. MIT.

## honest limits (written before the code, kept honest after)

- **The integer mip's one softness**: the blend partial (v_lo+1) sits
  at most `f` above the edge during transitions — and the Nyquist
  gate zeroes the blend whenever it would cross. What remains is
  sub-Nyquist edge content, bounded, heritable character. `--hi`
  (full-rate render, edge 19.8 kHz) is the checking path.
- **Step-mode morph can click** — hard frame jumps are discontinuities
  unless duty crossfades them; duty is the gene that decides, and the
  click itself is heritable character either way (the PPG was loved
  for exactly this).
- **Frame count is bounded** (4..64), partials (2..32), voices (1..4) —
  soft ceilings like dx's op count. The unbounded feel comes from the
  law swaps (morph × pitch × vector types: 3 × 5 × 2 grammars) and
  the leap.
- **The arena may be too easy** — maze's bumps organism saturated;
  if 240-gen runs fill the grid by gen 60, the mutation kernel needs
  more law-swap weight (the maze lesson: make the country bigger, not
  the budget).
- **Random founders pass viability easily** (8/8 in the audition) —
  the gate bites later, when evolution discovers the static-drone
  basins; the walk test is the honest measure.
- **prism interop is answered**: scan's sub organism IS the prism
  hybrid the design flagged — CPPN-grown frames inside a scan genome,
  as a separate organism (`--organism sub`) so the memory comparison
  stays clean. wt frames and sub frames render through the same
  readout; a single run mixing both is the obvious sequel.
- **Substrate frames are exactly harmonic** (inharm = 0) — the `metal`
  archetype is unreachable for `sub` until a second CPPN output
  channel (inharm per partial) is wired; the alphabet hook
  (`mut_add_output`) already grows the outputs, decode just reads `w`.
- **`tone` is unreachable for `sub`** (P ≥ 8 partials by design) — the
  10-word census is wt's; sub's censuses read organ/glide/scan/
  sequence/wash/formant/chord/ensemble. Labels are per-organism
  ground truth, not a shared taxonomy.
- **Batch low-bits**: `amp_batch` groups by exact shape; on CPU a
  genome's result is bit-identical solo or batched (tested), on MPS
  batched bmm may differ from single in float low-bits — runs are
  still bit-reproducible end to end (batches are a pure function of
  the generation's children), but `scan render` of an exported genome
  on mps can differ inaudibly from its engine render. Pin per-device
  expectations accordingly.
- **The mutation kernel's rng draw order changed in phase E** (the
  shared `drift_params` refactor): runs made before it are not
  stream-reproducible under the new code; runs are deterministic
  within a code version, which is the claim the tests hold.