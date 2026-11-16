# audio_lab — Technique Audit & New-Process Proposals

## 1. How each existing experiment creates sound

| script | sound source | actual sound *creation* | structuring | "ML"/classification |
|---|---|---|---|---|
| trim_silences | field recordings | none (analysis only) | — | energy thresholding |
| granular_lab | recordings | interpolation resampling (hold/linear/hermite/sinc), envelopes | parameter curves, Euclid/Fibonacci/expo/cycling orders, polymeter | hand-crafted spectral features + quantile bucketing |
| rhythm_lab | corpus + drum bank | rms-matched replay + stochastic per-hit processing (pitch/drive/decimate/tilt), sine sub-thump | genre templates, grids, swing, ratchets | spectral ranks → drum roles |
| alien_rhythms | drum bank + corpus | per-hit processing, sub/riser synth, lisp pattern algebra | generated & mutated expression trees, per-bar re-eval | brightness ranks (tex/air) |
| flux_rhythms | drum bank + trimmed | same | 1–3s phrase regeneration, rate tables | same |
| montage_lab | all corpus | speed change, drive/tilt/reverse | movement strategies, similarity/contrast transitions | centroid transition policies |
| storm_lab | all corpus | vari-speed phasor (tape warp), orthogonal echo replication, hard-cut pitch contrast | weather phases, gaps, energy-window picking | brightness buckets |
| edm_lab | bank + corpus + pads | sine sub-bass, noise risers/impacts, ducking, filter automation | EDM arc template, energy thresholds | quantile roles |

## 2. The honest diagnosis

Why the final output can feel boring / too similar to its corpus:

- **~95% of every piece's spectral content is corpus replay.** The only true
  synthesis is sine sub-bass and noise risers/impacts. Everything else is a
  slice of existing audio, re-ordered, re-pitched, filtered. The corpus is
  therefore also the *timbral ceiling*: no process can produce a spectrum the
  corpus doesn't already contain.
- **Structures are hand-designed templates** (genre skeletons, the EDM arc,
  weather-phase templates). They are good templates — but they are ours, not
  the data's.
- **Classification is all quantile binning of hand-crafted features.** Robust,
  but it never learns anything new about the material.

## 3. Proposed new processes (all standard DSP/ML, no LLMs)

**A. Binary sonification / databending** — interpret arbitrary binaries as
PCM: direct byte-cast, bit-plane slicing (bytes → 2-bit planes), G.711 μ-law
decode, differential decode (delta), nibble-split, endian pairs. Binaries
(executables, bytecode, images, even the corpus wavs themselves) are an
infinite source of spectra *no* corpus contains.

**B. Corruption operators** — bit-rotation, XOR stream-cipher (LFSR) corruption,
block byte-swapping, packet-style zero-runs, elementary **cellular automata**
(Wolfram rules; the rule number read from the binary itself) as
gate/mute/reverse controllers.

**C. NMF timbre stamping** *(standard ML)* — learn a spectral dictionary
(k components) from corpus STFTs by non-negative matrix factorisation;
project the sonified stream onto the dictionary; resynthesize with the
stream's phase. The binary's *structure* now wears the corpus's *timbres*.
Multiplicative updates run in seconds on a laptop; torch can put it on GPU.

**D. LPC stamps** — learn linear-predictive resonance models (Levinson-Durbin)
from corpus sounds; drive them with a whitened binary stream. The binary
"speaks through" the corpus's formants — vocoder-adjacent but rawer.

**E. Spectral convolution stamp** — cross-synthesis: binary phase × corpus
magnitude envelope. Cheaper cousin of C, very characterful.

**F. Entropy-derived structure** — the binary IS the score: windowed byte
entropy becomes the energy arc; entropy mean-shifts become section
boundaries; entropy variance becomes the chaos dial; zero-byte runs become
silence; the whole piece's *shape* is read out of the data instead of a
template.

**G. Wavetable scanning** — wrap 1024-sample windows of stamped binary as
single-cycle wavetables, scan through the file at entropy-derived pitch
curves. Buzzy digital tonality unique to each binary.

**H. Arbitrary interleaving** — stamped streams + corpus chunks classified
into one shared voice space, sequenced by the alien-lisp pattern algebra,
CA-gated, silence-structured per (F).

**Future menu (not built yet):** k-means codebook resynthesis (cluster corpus
spectral frames, rebuild audio from the codebook), HMM chains over corpus
spectral frames (learn the corpus's transition grammar and re-walk it),
Karplus-Strong strings excited by corpus impulses, matched-filter "chasing"
(one sound hunts another through the corpus).

## 4. What this fixes

Sonification + stamping breaks the corpus ceiling: the spectral content is now
*derived from data the corpus has never seen*, then pulled toward the corpus's
identity by learned (NMF/LPC) timbre models — novel but coherent. Entropy-derived
structure removes the last hand-made template. Everything remains
deterministic, seedable, and runs comfortably on a laptop.