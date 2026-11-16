# echo

**the memory organ** — machine listening + a vector archive for
novelty search in audio evolution. Every sound ever heard is encoded
into a dense, temporal feature vector; the archive answers the only
question open-ended search actually asks:

    HAVE WE HEARD SOMETHING LIKE THIS BEFORE?

```
audio ──encode──> 556-D temporal vector ──k-NN──> novelty score
                        │                            │
                        └── vectorDB (sqlite) ◄───────┘  the memory of a breeding run
```

The premise: novelty search (Lehman & Stanley) and every MCC variant
in this ecosystem measure "new" against *some* memory — a population,
a grid of cells, a single parent. Those memories are thin. Grids are
six scalars binned into 8×8 collisions; drift floors compare a child
only to its parent through a whole-clip spectral average with no time
axis; tomb's ears are lossy 4-D projections with no persistence. The
result is evolution that cannot tell a genuinely new sound from a
re-arranged old one — "not exciting," exactly as diagnosed. echo is
the missing organ: a rich, chunked, perceptually-weighted
characterization that **keeps its own memory** and turns "have we
heard this?" into a number instead of a cell collision.

Everything offline, deterministic, pure numpy at the core. No cloud,
ever.

---

## scope — what echo IS

1. **A machine-listening encoder.** Audio → vector, chunked over
   temporal segments so the *order of events survives*: "click then
   drone" and "drone then click" are different points. MFCCs (HTK mel
   scale, 64 bands), the proven spectral scalars (rms, crest, zcr,
   centroid, bandwidth, flatness, rolloff, flux), stereo life, chroma,
   envelope weather, and womb's 64-band silhouette — combined into one
   anatomy with per-block weights.

2. **A vector archive (the vectorDB).** Single sqlite file, in-memory
   float32 matrix, exact k-NN. `novelty(v)` = mean distance to the k
   nearest archive entries (Lehman & Stanley's score). `min_dist(v)` =
   tomb's τ question ("is it discriminable?") asked against
   *everything ever heard*, not just the living population. Roles
   (`corpus` / `pop` / `archive`) keep your material, the live
   population, and history separate in one bank.

3. **A measurement harness** (`echo bench`) that scores *any*
   candidate encoding of "have we heard this?" against the metrics
   the ecosystem already uses — four experiments, one report, so the
   encoder can be tuned against evidence instead of vibes.

## scope — what echo is NOT (deliberately)

- **Not a generator.** It never makes sound; it only measures. The
  ecosystem's law: ears select, processes energize.
- **Not a fitness function.** No quality anywhere in here. Novelty is
  not quality (Pugh: diversity can be unaligned with quality); echo
  must never become a judge of "good."
- **Not the minimal criterion.** Viability ("alive?") stays in the
  host system. echo answers "new?". MCNS's structure — a viability
  gate plus novelty pressure — needs both, and echo is only the
  second half.
- **Not a general audio-ML toolkit.** No transcription, no beat
  tracking, no source separation in the public API. hpss exists only
  inside one optional block.
- **Not neural by default.** The core is DSP: fast (~20 ms per 3 s
  clip), dependency-light (numpy + ffmpeg), bit-reproducible. Neural
  latents are an *extension point* (below), not a requirement.

---

## the vector — anatomy of "like"

`encode(audio, sr) -> (556,)` — one call, the whole listening. Blocks
in order, each chunk-aggregated over **8 temporal segments**:

| block | dims | weight | what it hears |
|---|---|---|---|
| `mfcc_mean` | 8×24 | 1.00 | timbre trajectory (64-mel HTK, DCT-II) — the course |
| `mfcc_std` | 8×24 | 0.35 | intra-chunk texture: crackle vs steady drone with the same mean |
| `scal_mean` | 8×10 | 0.80 | trajectory of the trusted scalars (rms crest zcr centroid bandwidth flatness rolloff flux side corr) |
| `silhouette` | 64 | 0.80 | womb's proven log-magnitude signature, kept as a block — the archive sees what the old drift floor saw |
| `dynamics` | 13 | 0.90 | modulation spectrum (0.5–16 Hz, 8 bands), onset rate, centroid drift, rms CV, rise/fall asymmetry, flux ratio — movement the crackle gates never saw |
| `chroma` | 12 | 0.40 | magnitude-weighted pitch classes |
| `spatial` | 3 | 0.30 | stereo side energy + inter-channel correlation life |

**The distance** is weighted L2 over corpus-standardized dimensions:
each dim is z-normalized against a fit (`EchoMetric.fit` on your
corpus — the axes that never move go inert), then blocks carry
weights, so MFCCs don't drown chroma. Fitted on *your* material, the
metric is data-driven without being a black box: every dimension has
a name and a meaning you can inspect (`echo encode <wav>` prints the
anatomy).

**Why chunked matters:** a whole-clip average cannot distinguish
"loud burst then drone" from "drone then loud burst" — the ablation
row `echo_flat` (N_CHUNK=1) in the bench exists to prove the time axis
earns its keep.

## the two tiers

| tier | call | cost/3s clip | dims | use |
|---|---|---|---|---|
| **core** | `encode(audio, sr)` | ~20 ms | 556 | the breeding loop's ear (numpy only) |
| **plus** | `encode_plus(audio, sr)` | ~300–600 ms | 592 | corpus indexing, run atlasing, curation — adds librosa blocks: CQT chroma (12), spectral contrast (7), hpss harmonic/percussive ratio per chunk (8), onset-strength + pulse (9) |

`echo_plus` needs `librosa` + `scipy` (optional install). The bank
pins its anatomy in sqlite (`kv` layout record): a 556-bank refuses
592-dim vectors — banks never silently mix eras.

**Neural extension point (planned, offline rule intact):** a
`NeuralBlock` — statistics of *your own* RAVE latent (womb already
trains it locally on your corpus) appended as one more weighted block.
CLAP/PANNs-style encoders are explicitly out unless their weights
live on this machine; the ecosystem's no-cloud law wins over
convenience. RAVE latents are the offline-compliant neural ear: your
model, your machine, zero downloads.

## the bank

```python
from echo import encode, EchoBank

bank = EchoBank("run.sqlite")            # or ":memory:" for a single run
bank.fit(corpus_vectors)                 # z-norm on your material
bank.add(encode(child_audio), meta={"gen": 4}, role="archive")

nov  = bank.novelty(encode(candidate))   # Lehman score, k=15 nearest
tau  = bank.min_dist(encode(candidate))  # discriminability vs everything
near = bank.knn(encode(candidate), k=5)   # the actual neighbors
```

- **Persistence**: one sqlite file per run (or per corpus); the
  metric + layout travel inside it. Reopen and the memory is intact —
  runs are resumable, archives are comparable.
- **Novelty semantics**: `novelty = mean(k-NN distance)`. Empty bank
  → `inf` (nothing heard yet — everything is new). k adapts: pass
  `k=min(15, len(bank))` while small.
- **Scale**: exact brute force in float32 — 0.6 ms per query at 10k
  entries, comfortable to ~100k (a few hundred generations). Beyond
  that the backend swaps (hnswlib/FAISS) behind the same interface;
  the vector and the question never change.

## using echo in a breeding loop

The contract: encode once per evaluation, ask the archive twice —
once for admission ("is it new enough?"), once for parent bias.

```python
# inside your MCC loop, after viability:
vec = encode(audio, SR)
if bank.novelty(vec, k=15) < novelty_floor:      # too similar to history
    continue                                     # not new: the archive remembers
bank.add(vec, meta={"gen": gen, "mutation": mut}, role="archive")
```

The floor is the Soros dial of this layer: adaptive (a low quantile
of the novelties you have admitted) keeps a steady discovery rate;
fixed floors stagnate or flood exactly like strictness does. Two
integration recipes for this ecosystem:

- **womb/bomb (`breed.py`)** — keep everything; replace the drift
  floor's `signature`/`distance` parent-check with
  `bank.novelty(child_vec) > floor`, and add `vec` to every archive
  entry. The grid atlas becomes a *view* of the bank (still works —
  it reads `desc`), while the bank carries the real memory.
- **tomb** — the MC becomes global: `min_dist(candidate) > tau`
  against the persistent bank replaces "some ear distinguishes it
  from the living population". Ears can stay as *explorers* of the
  feature space, but the memory outlives every population.

## measuring effectiveness — `echo bench` + `echo validate`

**`echo bench`** (calibration, on a designed corpus): four
experiments, scored against the ecosystem's current metrics (womb's
64-band signature, tomb's 16 scalars, tomb-style 4-D ears):

1. **Pair discrimination** — known-answer pairs: near-duplicates
   (gain, ±15 ms shift, EQ tilt, −40 dB noise, **320 kbps mp3
   round-trip**, resample round-trip, and *mutation children* of the
   champions that the current drift floor itself scores as "close")
   must land close; distant corpus regions, cross-law champions, and
   synthetic archetypes must land far. AUC / d′ / EER per encoder,
   plus ablations (`echo_flat` = no time axis, `echo_mfcc` = MFCC
   trajectory alone).
2. **Mutation monotonicity** — the evolution claim: distance from a
   parent must grow with mutation size (Spearman), and a different
   law must be farther than any parameter drift.
3. **Corpus locality** — is a chunk's nearest neighbor its neighbor
   in the source recording? Recall@k + separation.
4. **A/B selection** — three memories, one loop, identical laws,
   judge, seeds: `drift` (MCC alone), `cell` (womb's grid-parent
   pull), `novelty` (the echo archive). Coverage in womb's grid
   cells *and* in dense echo space.

**`echo validate`** (production, on any pile of audio): index the
whole corpus, then four measurements — leave-one-out **family
retrieval** (is a file's nearest neighbor its sibling from the same
source folder?), **discrimination at scale** through the fitted
metric, a **novelty scan** of every entry, and **spot checks** you
read like a listener.

### measured — calibration (269-chunk bomb corpus, seeds 0/1)

| encoder | AUC | d′ | EER |
|---|---|---|---|
| **echo** | **0.972** | 2.40 | 0.057 |
| echo_flat (no chunks) | 0.980 | 2.24 | 0.034 |
| echo_mfcc (MFCCs only) | 0.991 | 2.50 | 0.034 |
| echo_plus (librosa tier) | 0.972 | 2.37 | 0.057 |
| womb_sig (current drift floor) | 0.965 | 2.12 | 0.080 |
| tomb16 (current features) | 0.910 | 1.98 | 0.138 |
| tomb_ear (current ears) | 0.902 | 1.68 | 0.138 |

- mp3 round-trips score 0.7 (near) vs 29 (far median): codec
  weathering reads as *the same sound, aged* — the metabolism's
  generational loss will not be chased as fake novelty.
- mutation monotonicity: rho 1.00; a different law (35–53) is
  farther than any parameter mutation (≤26) — the gradient
  evolution needs exists.
- corpus locality: separation 0.228 — nearest-neighbor distance is
  ~4× tighter than random pairs.
- A/B over 10 gens at pop 12: all three memories converge on grid
  cells (31–34 of 192) — **the 8×8 grid saturates before the memories
  differ**; that is the grid's resolution limit, which is precisely
  echo's reason to exist. The dense echo-space metrics are where the
  comparison must be read.

### measured — at scale (the whole 2264-file corpus, 22 hours)

Indexed in 15 min (32 ms median encode, one sqlite bank, 7 decode
failures logged-not-fatal):

- **family retrieval** (2254 queries, 57 source folders):
  recall@1 **0.42**, recall@5 **0.78**, median rank **2**. Kicks find
  kicks (`UKG_Kick_09` → `UKG_Kick_10` first), claps find claps
  (R@1 0.81), dnb chops 0.89, woodblocks 1.00.
- **discrimination** through the fitted metric: AUC **0.991**, d′
  3.21, EER 0.050.
- **duplicate detection**: identical files filed in two different
  folders land at distance **0.000** — the archive knows what it
  has heard before, exactly.
- **novelty landscape**: median 12.75; the most embedded sounds are
  the corpus's clichés (duplicated fusion pieces), the strangest
  are cryptid specimens and modular bleeps (49 vs 13) — the memory's
  opinion matches a listener's.
- **speed**: 0.24 ms per k-NN query at 2602 entries.

`echo_out/bench_report.md` and `echo_out/validate/validate_report.md`
carry the full current numbers; every encoder tweak rescores on the
same yardstick.

## honest limits

- **Brute force** is the search: ~100k entries is the comfortable
  ceiling (a backend swap is the fix; the interface holds).
- **EQ tilts** are heard as fairly different (they reshape every mel
  band); if "same mix, new master" should count as heard-before,
  drop `silhouette` weight or loudness-normalize first — the block
  weights are the taste knobs and the bench is the proof.
- **Locality truth is approximate** — adjacent chunks are not
  guaranteed-similar material; the separation ratio is the robust
  number, recall@k the fuzzy one.
- **Champions are seed-invariant** (bit-identical renders), so seed
  jitter is not a probe; mutation children are.
- **ffmpeg** decodes/encodes files (the ecosystem's io); the pure
  in-memory path needs nothing but numpy.

## layout

```
src/echo/
  melbank.py    # HTK mel filterbank + orthonormal DCT (cached)
  encode.py     # the ear: framing, per-frame features, chunking, blocks
  bank.py       # EchoMetric (z-norm + weights) + EchoBank (sqlite, k-NN, novelty)
  extras.py     # the librosa tier: encode_plus (CQT chroma, contrast, hpss, rhythm)
  probes.py     # known-answer pair generation (perturbations + archetypes)
  bench.py      # the four experiments + report writer
  cli.py        # echo index | novelty | knn | encode | bench | info
tests/          # 27 tests: anatomy, determinism, order-sensitivity, k-NN
                # exactness, persistence, stat fns, A/B smoke
echo_out/       # bench_report.md + results.json live here
```

```bash
pip install -e echo                 # as a dependency of any project

echo index  corpus/ -o bank.sqlite  # encode a folder, fit the metric
echo novelty new.wav -b bank.sqlite # have we heard something like this?
echo knn new.wav -b bank.sqlite -k 5
echo encode new.wav                 # print a sound's vector anatomy
echo bench  corpus/ -o echo_out    # the full effectiveness suite
```

## license

MIT — see LICENSE.