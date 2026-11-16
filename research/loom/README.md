# loom

**A modular library for algorithmic audio composition.** Point it at a
`.wav` or a folder of `.wavs`; it chunks, processes, re-processes, and
recombines the corpus into a finished piece of a defined length —
deterministically, from a single seed.

```python
import loom

result = loom.compose(
    sources="corpus2/MOT-oct26-everything.wav",   # one file or a folder
    preset="drift",                               # or build from scratch
    length=180.0,                                 # exact output length
    seed=3,
)
print(result.summary())
```

Every render produces a `.wav` **and** a `.manifest.json` recording the
fully-resolved configuration, chunk lineage, and output checksum. Same
seed → byte-identical file.

```
Source(s) ──► Chunker ──► Catalog ──► Passes ──► Select ──► Arranger ──► Score ──► Renderer ──► .wav
  (lazy)      strategy    chunks +     re-process   query     strategy     events    mixing     + manifest
              (metadata)  feature         + tag     language              as data   + fades
                          ranks         + lineage
```

---

## Why

This library distills a lab of ~20 one-off experiment scripts
(`audio_lab/`) into reusable abstractions. The experiments proved a set of
ideas — quantile-ranked machine-listening features, parameters as
time-curves, manifests for reproducibility, seedable determinism — but
each script re-implemented its own wav I/O, chunking, and rendering, and
none of them could compose with each other. loom keeps the ideas and adds
the missing discipline:

- **Lazy sources.** Chunking and analysis stream hour-long files in
  blocks; chunks are *metadata* until the moment they're needed. A 906 MB
  / 55-minute corpus renders a 3-minute piece in ~12 s at well under
  1 GB RAM (see `examples/02_single_big_file.py`).
- **Everything is a strategy in an open registry** — chunkers,
  processors, arrangers. A learned ML model plugs in exactly where a DSP
  transform does (see *ML extensions* below).
- **Composition produces data, not audio.** An arranger emits a
  `Score`: a flat list of JSON-serializable events. Render it, edit it by
  hand, archive it, re-render it.
- **Determinism is a contract.** Every stochastic decision draws from one
  seeded RNG tree. The manifest includes the output's SHA-256.

## Install

```bash
cd loom
pip install -e .            # numpy is the only required dependency
pip install -e .[all]       # optional: soundfile (better wav IO), scipy, scikit-learn
```

`soundfile` is strongly recommended (the stdlib `wave` fallback can't read
float wavs and only writes 16-bit). Without scikit-learn, clustering and
the ML example fall back to dependency-free implementations.

## The pipeline, piece by piece

### 1. Sources

`loom.open_sources(path)` accepts a `.wav`, a folder of `.wavs`, or a list
of either (folders scan non-recursively, sorted — determinism is a
requirement, not a convenience). All sources are resampled on the fly to
one pipeline rate (default: the first source's own rate, so a lone 48 kHz
corpus is never resampled). Reads are always stereo `(n, 2)` float32.

### 2. Chunking

Chunkers slice sources into `ChunkRef`s — pure metadata `(source index,
start, dur)`. No audio is decoded yet.

| chunker | params | notes |
|---|---|---|
| `fixed` | `seconds`, `overlap`, `jitter` | uniform windows |
| `onset` | `min_seconds`, `max_seconds`, `threshold` | energy-onset segmentation; **streams** in 30 s blocks, safe on hour-long files; falls back to fixed windows on flat material |
| `random` | `n`, `min_seconds`, `max_seconds` | N random windows per source |
| `beats` | `bpm`, `bars` | tempo-locked windows |

### 3. Catalog

Every chunk is analyzed once (rms, peak, crest, zcr, spectral centroid /
rolloff / flatness, bass/mid/air band shares, autocorrelation f0 +
confidence, transient-ness — `loom feature_help()`). Selection never uses
raw values: the catalog stores **quantile ranks**, so `"rms>0.8"` means
"the loudest 20%" on any corpus, loud or quiet.

The query language (`loom.compose(select=...)`, pass selection, and voice
selection in `layers`):

```
all                          everything
rms>0.8                      loudest quintile
bass>0.6 and air<0.3         woody but not airy
tag=dark or tag=crushed      derived families from passes
src>=1                       only the second-and-later sources
```

`and` binds tighter than `or`. Numbers are always quantiles in [0, 1].

### 4. Re-processing (passes)

Passes transform chunks — the "re-processing" loop:

```python
passes=[
    dict(select="rms>0.5",
         processors=[["lowpass", {"cut_hz": 800}], "reverse"],
         tag="dark"),                      # derive new chunks, tagged
    dict(select="tag=dark and rms>0.6",
         processors=["crush(bits=5)", "echo"],
         tag="crushed"),                   # passes compose via tags
    dict(select="all",
         processors=["trim"],
         replace=True),                    # transform in place instead
]
```

Derived chunks keep their **lineage** (the processor chain) and **parent**;
later passes and the final selection query them like any other chunk. For
big corpora with many passes use `store="disk"` so derived audio spills to
a temp dir instead of RAM.

### 5. Processors

Chains run per event; any parameter can be a **curve** — a number is a
constant, a list is piecewise-linear over piece time `[0, 1]`:

```python
processors=["normalize",
            ["lowpass", {"cut_hz": [6000, 400]}],   # darkens across the piece
            ["gain", {"amount": [1.0, 0.3]}],       # fades away
            "fades"]
```

`loom help` prints the full list. Built-ins: `gain`, `normalize`,
`reverse`, `speed` (varispeed), `stretch` (pitch-preserving granular
time-stretch), `stutter` (re-triggered slice, pitch shifting each repeat,
 direction up/down/random), `chop` (Euclid gating inside a chunk —
micro-silence), `morph` (diversity engine: holds a pool of processor specs
and applies ONE per event, seeded — a chain slot that behaves differently
every couple of seconds), `lowpass`, `highpass`, `bandpass`
(FFT-domain, raised-cosine edges), `ring`, `drive`, `crush`, `corrupt`
(byte-level databending), `echo` (finite geometric taps), `repeat`
(stutter with pitch jitter), `fades`, `env`, `gate`, `trim`. Inline specs
work too: `"crush(bits=5)"`.

### 6. Arrangement

Arrangers turn a selection into a `Score` of events (chunk, time, dur,
speed, gain, pan, fades):

| arranger | shape |
|---|---|
| `sequence` | end-to-end; `order` ∈ `random`, `by:feature`, `markov[:k]` (corpus grammar), `harmonic` (consonant-interval walk over pitched chunks), `farthest` (max contrast), `chase` (similarity pursuit); `gap` (negative = overlap), `speed`, `gain`, `pan` curves |
| `cloud` | grains scattered at a `density` curve (events/min); free overlap — granular textures |
| `euclid` | chunks on a Euclidean rhythm grid (`bpm`, `steps`, `pulses`, `rotate`, `gate`, `accent`) |
| `layers` | recursive: voices = `[{select, arranger, params, gain, chain}]`, each voice its own arranger, query — and its own processor `chain` — over its own query |

Orderings deserve a note: `markov` fits an order-1 chain over cluster
labels **on the corpus's own temporal order** — the corpus's phrasing —
and walks it. `farthest` greedily maximizes timbral distance (the
anti-boring ordering). `chase` does the opposite: each next chunk is the
feature-nearest to the last.

### 7. Rendering

Mixing is boring on purpose: float64 master, per-event chain → varispeed
→ truncation → cosine fades → equal-power pan → overlap-add → peak
normalize (or `soft_clip`) → PCM 16/24 bit. Events that overrun the piece
are clipped, never wrapped. NaN/Inf can never reach the output.

## Presets

`loom presets` lists them. Each is a dict of compose() defaults in
`src/loom/presets.py` — copy, tweak, or add your own. Explicit kwargs
always override preset values.

| preset | idea |
|---|---|
| `drift` | ambient granular: onset chunks, darkening lowpass, thinning cloud |
| `storm` | maximal-contrast collage: farthest-point order, tape-slow arc |
| `pulse` | rhythm: bass Euclid pulse, texture raining above |
| `ghost` | re-process loud chunks into a dark echo, walk it markov-style |
| `strata` | feature-separated layers: lows walk, mids pulse, highs drift |
| `collage` | random windows chased by similarity, drifting speed |

## CLI

```bash
loom compose corpus/ -o piece.wav --length 180 --preset storm --seed 3 -v
loom compose big.wav --preset ghost --store disk --bits 24
loom compose corpus/ --chunker onset --arranger euclid \
     --arranger-params '{"bpm": 140, "steps": 16, "pulses": 7}'
loom presets          # list presets
loom info corpus2/    # inspect sources
loom help             # processors, arrangers, features reference
```

`--dry-run` arranges and prints the score without rendering.

## Extending: custom strategies and ML

Registries are the extension surface. Any callable with the right
signature is a strategy (`examples/04_custom_strategies.py`):

```python
from loom.processors import Processor, register_processor

@register_processor
class Shimmer(Processor):
    name = "shimmer"
    def process(self, audio, sr, t, rng, rate_hz=6.0, depth=0.5):
        n = audio.shape[0]
        lfo = 0.5 * depth * (1 - np.cos(2 * np.pi * np.arange(n) * rate_hz / sr))
        return audio * (1.0 - depth + lfo)[:, None]

loom.compose(..., processors="shimmer")     # or ["shimmer(depth=0.8)", ...]
```

**ML integration follows the same seam** (`examples/05_ml_codebook.py`):
fit a model, pass the *instance* into the chain — it runs inline during
offline rendering, exactly where a DSP transform would. The library ships
one worked reference: `loom.ml.CodebookProcessor` fits a k-means codebook
over corpus STFT magnitudes and rewrites each chunk's spectra to the
nearest entry (keeping the chunk's phase — a miniature corpus-resynthesis
model). With scikit-learn absent it falls back to a ~30-line numpy
k-means. The same pattern fits any model: an autoencoder resynthesis, a
tiny transformer over spectral tokens, a DDSP decoder — fit offline, then
`compose(processors=[model, "fades"])`.

Design rules for learned processors: implement
`process(audio, sr, t, rng, **params) -> (n, 2) float32`, never produce
non-finite values, accept curves for `t`-dependent params, and stay
deterministic given `rng`.

## Architecture notes (for offline-rendering reference)

- **Lazy everywhere.** `Source.read(start, dur)` seeks and decodes only
  that window, resampling per read (no whole-file loads; target rate
  defaults to the source's own rate so a lone corpus is untouched).
  `Source.iter_blocks()` is the streaming pattern for whole-file
  analysis — onset detection is the worked example.
- **Metadata-first chunks.** A `Chunk` is a `ChunkRef` (source window)
  or a store key (derived), plus features, tag, lineage, parent. Audio
  materializes exactly once, at analysis or render.
- **Curves.** `loom.Curve` normalizes constants, numeric strings, lists,
  and callables to `f(t) in [0,1] -> value`. This is the library's main
  expressive axis: nearly every time-varying parameter accepts one.
- **Score/render separation.** Arrangers emit JSON-serializable events;
  rendering resolves audio. Edit scores by hand; archive them; re-render.
- **Seeded RNG tree.** `loom.Seeded(seed)` spawns independent child
  generators (numpy `SeedSequence`) for chunking, each pass, arrangement,
  and each event — deterministic without shared-state coupling.

## Repository layout

```
loom/
  pyproject.toml         # numpy-only core; soundfile/scipy/sklearn optional
  src/loom/
    io.py               # wav read/write (soundfile or stdlib), resample
    sources.py          # lazy seekable Sources; streaming blocks
    curves.py           # parameter-curves
    rng.py              # seeded RNG tree
    features.py         # machine-listening feature set
    chunks.py           # Chunk/ChunkRef (metadata-first)
    chunkers.py         # fixed | onset | random | beats
    catalog.py          # ranks, query language, clustering, orderings
    store.py            # memory/disk backends for derived chunks
    processors.py       # DSP registry + chains
    arrangers.py        # sequence | cloud | euclid | layers
    score.py            # Event/Score (JSON round-trip)
    render.py           # mixing, fades, panning, normalization
    pipeline.py         # compose(): the whole pipeline in one call
    presets.py          # named starting points
    ml.py               # learned-processor reference (codebook)
    cli.py              # `loom` / `python -m loom`
  tests/                # pytest suite incl. big-file and reproducibility
  examples/             # 6 worked examples
```

## Testing

```bash
pip install -e .[test]
pytest                 # 70+ tests: io, chunkers, catalog, arrangers,
                       # passes, ml, end-to-end, byte-identical seeds
```

## License

MIT — see LICENSE.