# quilt

**the seamstress organ** — the corpus cut *in place* into a pile of
pieces, every piece pinned into echo's feature space, and a walker
that takes a trip through the territory and sews what it finds into
one continuous piece of audio. No synthesis, no processing, no cloud:
every sample in the output is a sample from the corpus. The only
thing quilt ever changes is the order.

```
corpus/ ──split (in place: onset | silence | fixed, chosen per file)──> the pile
   pile ──index (echo.encode, one piece per file, provenance in names)──> the bank
   bank ──walkers (chain · contrast · roam · attractor · drift · follow)──> route
   route ──glue (pure concatenation)──> the piece          [+ GPU under the queries]
```

The premise: echo already answered "have we heard this before?" —
every sound is a point in a fitted, weighted, perceptually-informed
556-dimensional space. quilt asks the follow-up question nobody asked
the memory: *if the corpus is a territory, what does it sound like to
travel through it?* A three-minute piece is a path; the walkers are
policies for choosing paths; the glue is honest.

This is **corpus-based concatenative synthesis** — the cataRT lineage
(Diemo Schwarz, IRCAM: segment a corpus, describe every unit, store
it in a database, *select units by navigating descriptor space*) —
rebuilt offline on echo's fitted metric, with the latent-navigation
paradigm of RAVE mapped onto a discrete space (see [§rave](#rave--research-verdict)
for why RAVE itself is the cousin, not the engine).

---

## scope — what quilt IS

1. **The scissors, in place** (`split`): the corpus folder becomes
   the pile. Every original is cut where its material says so, its
   chunks written next to it, verified, and only then is the original
   deleted — the folder grows in file count, not bytes:

   | scissor | wins when | cut |
   |---|---|---|
   | `onset` | event-dense (median piece < 3 s — librosa's phantom onsets in steady drones are refused) | at the hits |
   | `silence` | real loud/quiet structure (the plan must have *holes* — dropped quiet; a subdivided long loud run is not silence structure) | at the gates |
   | `fixed` | continuous material longer than `--min-keep` (8 s) | 2 s windows |
   | (keep) | short files — a one-shot is already a piece | — |

   Naming carries provenance: `orig.wav` → `orig__onset_0000.wav,
   orig__onset_0001.wav, …` — the scissor and the chunk's temporal
   order live in the filename (that is how `follow` walks the
   corpus's own sequences and how `--strategy` filters a walk).
   Idempotent (chunk-marked files are skipped), audited
   (a JSONL records every decision), `--dry-run` reports without
   touching, and a failure never deletes its original.

2. **The memory seam** (`index`): the pile → the bank. One file =
   one piece: read (libsndfile fast path on the pile's native
   44.1 kHz wav), `echo.encode`, pin into an `EchoBank` with the
   filename's provenance (original, scissor, index). No chunking
   here — the scissors already ran. The metric is fitted on the
   whole population, so distances mean what they mean on this
   corpus. Parallel across files (`--workers`), deterministic
   (sorted files, ordered results — identical bank for any worker
   count). A `.provenance.json` sidecar carries the census.

3. **The walkers** (`navigate`): six policies for traversing the
   space, each leaving a measurable trace (the *temperature* of the
   piece: median step distance vs the corpus's median random-pair
   distance):

   | walker | policy | trace |
   |---|---|---|
   | `chain` | next = a near neighbor of the last, temperature-biased | cold — "similar in sequence" |
   | `contrast` | next = the farthest piece from the recent past | hot — "the most different next" |
   | `roam` | a brownian virtual point drifts at median-NN speed; piece = nearest real chunk | lukewarm — free wandering |
   | `attractor` | a Lorenz trajectory steers through the top-3 PCA axes | chaotic but structured roaming |
   | `drift` | a straight line between two distant poles, walked chunk by chunk | a 3-minute dissolve A → B |
   | `follow` | mostly the pile's own temporal successor (`orig__onset_0007` after `…_0006`), with near-neighbor branches | passages that branch away |

   All walkers share one law: the recent past is excluded (no piece
   recurs within the exclusion window), relaxing only when a tiny
   bank corners them.

4. **The glue** (`render`): butt-splice, sample-exact, stereo, 16-bit
   wav out. An optional crossfade exists (off by default) only because
   hard splices click at discontinuities; use it if the clicks offend,
   or keep the law and let them be part of the material. Every piece
   ships with its **route JSON** — the complete, re-renderable
   provenance of the walk: every chunk's file, scissor, step distance,
   and the walker's note.

## scope — what quilt is NOT (deliberately)

- **Not a processor.** No DSP on the output, ever. No fades (unless
  asked), no EQ, no time-stretch, no cross-synthesis. If a piece
  clicks, that click is the corpus's own truth at that boundary.
- **Not a generator.** quilt cannot make a sound the corpus doesn't
  contain. This is by design: the experiment is *navigation*, and its
  honesty lives in the constraint.
- **Not destructive without a record.** `split` replaces originals —
  but every decision is audited (`quilt_out/split_audit.jsonl`), the
  pile is the same bytes (splitting, never duplicating), and the run
  is idempotent.
- **Not a judge.** No walker knows "good"; they know near, far, and
  their own policies. Quality lives in the corpus and the listener.

## the metal — GPU support, measured

This machine: Apple M4, torch 2.5, MPS. Everything degrades cleanly
to numpy when the metal or torch is absent.

- **Atlas queries run on the GPU** (the walkers' hot path): the
  scaled matrix lives as a float32 MPS tensor; every
  distance-to-everything is one GEMV, median-NN one GEMM, the PCA
  covariance one GEMM. Measured on the walker's own shapes:
  45x vs one CPU thread at 500k pieces. `--device auto` (default)
  engages it for banks ≥ 50k pieces; `--device cpu` forces the exact
  float64 numpy path (always the reference; the tests pin both).
- **The GPU ear** (`GpuEar`): echo's 556-D encoder ported 1:1 to
  batched torch — same framing, same cached mel/DCT matrices, same
  block order — float32 on MPS, measured equivalent (vector cosine
  > 0.9999 vs the numpy reference, nearest-neighbor picks preserved,
  run-to-run deterministic). **Verdict, measured: not used for
  indexing by default** — 8 CPU workers beat the M4's GPU on this
  material (GPU x2.7 vs one thread, but 0.33x vs the pool), because
  the pile's pieces are short (encode ~1-2 ms CPU). It stays in the
  kit, tested, for corpora where deep same-length batches make the
  metal pay.
- Two float eras exist (numpy f64 reference, MPS f32 for queries) —
  same device + same seed = same route, and routes are *not* promised
  bit-identical across devices. Pick one for a series of pieces.

## usage

```bash
pip install -e echo                      # the ear + bank (once)
pip install -e quilt

quilt split corpus/                      # IN PLACE: originals -> chunks
#   --dry-run to just report; --max-files N to resume; audit JSONL written

quilt index corpus/ -o quilt_out/bank.sqlite    # pile -> feature-space bank
#   ~1-3 ms per piece on CPU (8 workers); --force replaces; --workers N

quilt make -b quilt_out/bank.sqlite \
           --mode roam --seconds 180 --seed 3        # walk + sew a piece
#   -> quilt_out/pieces/roam_all_3_180s.wav (+ .route.json)
#   --device auto|cpu|mps chooses where the queries run

quilt render quilt_out/pieces/roam_all_3_180s.wav.route.json \
           -o softened.wav --crossfade 10           # re-glue, faded

quilt info -b quilt_out/bank.sqlite                 # the census
quilt strategies | quilt modes | quilt doctor       # the inventories
```

Python-side:

```python
from quilt import split_corpus, build_bank, ChunkAtlas, walk, render_route

split_corpus("corpus")                            # once (idempotent)
build_bank("corpus", "quilt_out/bank.sqlite")      # once
atlas = ChunkAtlas("quilt_out/bank.sqlite")        # the territory
route = walk(atlas, mode="contrast", seconds=180, seed=7)
render_route(route.to_json(atlas), "piece.wav")    # the glue
```

`make` prints the piece's temperature trace, e.g.:

```
mode=contrast seed=7 pieces=91 total=183.2s
step distance: median 41.9 vs random pair 19.4 (temperature 2.2)
```

chain runs cold (< 1), contrast hot (> 1), roam in between — the
walkers' policies are auditable numbers, not vibes.

## rave — research verdict

**What RAVE is.** RAVE (Realtime Audio Variational autoEncoder,
Caillon & Esling, IRCAM, arXiv:2111.05011; implementation
`acids-rave`) learns a compact continuous latent space from a corpus
via a two-stage VAE — representation learning, then adversarial
fine-tuning — and *decodes audio from latents*, 20× faster than
realtime on CPU. Its artistic paradigm: encode your corpus once, then
perform by navigating the latent space (interpolations, random walks,
attractor paths), with the decoder as the voice.

**Why it is the cousin, not the engine, here.** This experiment
inherits RAVE's *navigation paradigm* but not its decoder, because
the law of the house is "no processing, just gluing": a RAVE output
sample is a neural network's re-synthesis — beautiful, but it is not
the corpus's sample. quilt's space is *discrete*: every point is a
real piece of real audio, and the walk is unit selection in a
descriptor space — precisely the cataRT/Schwarz lineage, with echo's
fitted 556-D metric as the descriptors (measured: AUC 0.99 family
retrieval at corpus scale — see echo's validate report).

**The dependency question.** RAVE is deliberately NOT a dependency of
the core: it adds hours of CPU training (womb's venv has acids-rave
and a smoke run, but no trained checkpoint), its latents need a model
that doesn't exist locally yet, and the offline law means no pretrained
weights from the cloud. Where RAVE *does* earn a place is as an
optional **second ear**: echo's README already plans the
`NeuralBlock` — statistics of your own locally-trained RAVE latent
appended as one more weighted block of the vector. When womb trains
one (`womb rave train`), quilt inherits it for free, because it
consumes whatever space the bank speaks. `quilt doctor` reports the
RAVE status and the exact path. Until then, the DSP ear is complete
on its own.

## honest limits

- **`split` is destructive by design** — originals are replaced by
  their chunks after verification. The audit JSONL is the record;
  `--dry-run` is the preview; a failed or interrupted file keeps its
  original (idempotent rerun resumes; interrupted *partial* chunks
  are named like chunks and will be skipped, so inspect the audit if
  a run was killed mid-write).
- **Hard splices click** where the boundary is discontinuous — the
  cost of the no-processing law. `--crossfade ms` exists; the route
  JSON means a piece can be re-glued faded without re-walking.
- **The bank is an index, not a copy**: rendering needs the pile in
  place. If a file moves, re-render fails loudly with the path.
- **Scale**: the 27-hour corpus splits into a pile of ~10⁵ pieces
  (a few minutes at 8 workers) and indexes in minutes; a 500k-piece
  bank is ~1.1 GB sqlite and ~4.4 GB RAM to walk (echo's bank keeps
  its own caches; the atlas adds a 1.1 GB GPU tensor). Brute force
  remains the search — one GEMV per step, exact, sub-millisecond on
  the metal.
- **GPU routes are device-stable, not cross-device-stable**: float32
  on MPS orders near-ties slightly differently than float64 numpy.
  Same device + seed = same piece; compare pieces within one era.
- **Onset on noise** fires often; the scissors clamp it (merge short,
  subdivide long, refuse non-event-sized plans). Dense rhythmic
  material simply yields more, smaller pieces — the bank tags
  everything; `--strategy onset` walks one grain.
- **`whole`-kept pieces are long** (up to 8 s): a walk using all
  scissors occasionally lets a long piece dominate a stretch —
  feature, not bug; `--strategy` filters it away.
- **Temperature is measured, not guaranteed musical**: the trace
  numbers are the honest reading of each walker's policy.
- **Roam's step has a resolution floor**: the step is speed x 0.1 x
  the territory's macro scale (median random-pair distance), because
  the NN median on a duplicate-dense pile is microscopic and steps
  scaled to it never leave the start (found by the extended-run
  survey: every speed walked bit-identical routes). Steps smaller
  than the local Voronoi cell size are still indistinguishable —
  speed 0.25 and 1.0 coincide in sparse regions, 4.0 does not.
  Same lesson, second half: the dial only speaks above the territory's
  own grain.

## layout

```
src/quilt/
  io.py         # fast audio io (libsndfile) with ffmpeg fallback
  naming.py     # the pile's provenance convention in filenames
  chunking.py   # the scissors: fixed | onset | silence | whole
  split.py      # the in-place processor: corpus folder -> pile
  index.py      # pile -> bank (one piece per file, parallel)
  gpu.py        # the metal: GpuEar (batched torch port) + device utils
  atlas.py      # the territory: scaled matrix, queries, PCA (CPU or GPU)
  navigate.py   # the walkers: chain contrast roam attractor drift follow
  render.py     # the glue: route -> wav (butt-splice, optional fade)
  cli.py        # quilt split | index | make | render | info | doctor
tests/          # 72 tests: known-answer cuts, in-place safety + audit,
                 # twin retrieval, walker traces, exact renders, GPU
                 # equivalence + determinism (skipped without MPS)
quilt_out/       # bank.sqlite (+ .provenance.json), pieces/, split audit
```

## license

MIT — see the ecosystem root.