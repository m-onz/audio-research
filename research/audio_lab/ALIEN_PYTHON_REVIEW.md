# Review: /Users/monz/Desktop/alien-python — relevance to audio_lab

## What it is

A 40-experiment research program in **novelty-search / open-ended audio
generation**, plus an outer loop (`program.py`) that compounds diversity:
run an engine episode -> harvest -> featurize every artifact with an 84-d
machine-listening set (spectral bands, envelope, flux, coarse cepstra) ->
**farthest-point curation** in robust-whitened feature space -> the curated
"spread set" becomes the corpus for the next episode. Deliberately LLM-free
(40_llm_mutator is excluded by its own research direction). GPU engines run
batched on Apple MPS. Engines mostly take a `corpus/` folder of wavs.

Current state: code intact, corpora mostly empty, ~60MB — the engines are
**corpus-hungry**. audio_lab has ~200 generated pieces and endless corpus
production. That is the headline: **symbiosis**.

## A. Things we already built equivalents of (validates the direction)

| their engine | our equivalent |
|---|---|
| 05_nmf_stems (NMF stems + mutant remix) | binary_lab's NMF stamping |
| 21_embedding_wavetable (embeddings as oscillators) | binary_lab's wavetable scanner |
| 24_beat_slicer (onset slices on a grid, Markov resequence) | rhythm_lab / edm_lab |
| 26_info_dynamics (surprise curve drives generation) | binary_lab's entropy-derived structure |
| 17/18 (GA/DE sound matching) | — (but see C below) |

## B. The "future menu" from TECHNIQUES.md — already written and tested here

These fill our documented gaps. Port them as audio_lab stages:

- **08_codebook_lm** — k-means spectral-frame codebook + n-gram LM +
  Griffin-Lim decode = miniature AudioLM from OUR corpus. Brings us
  **Griffin-Lim phase reconstruction** (we have nothing like it) and
  the corpus-grammar generator. Deps: sklearn (have).
- **10_hmm_grammar** — hmmlearn HMM over PCA frames learns the corpus's
  *temporal* grammar (attack/sustain/rattle states + transitions).
  This is exactly the HMM I proposed. Deps: hmmlearn (tiny, not installed).
- **07_corpus_markov** — Markov chains over clustered units = the corpus's
  own ordering grammar (complements our hand-designed interleaves).
- **06_gmm_dream / 04_spectral_dream** — GMM sampling over PCA spectral
  frames: "dreaming" new material that respects the corpus's timbre modes.

## C. Genuinely new capabilities worth adopting

1. **The curation layer (program.py `views()` + `curate()`)** — the single
   highest-value import. Run it over audio_lab's entire output: 84-d features,
   whitening, farthest-point selection -> a maximally-spread curated set.
   Feeding THAT back into granular/rhythm/binary stages directly attacks
   "the final output is boring": boring = too-similar material recombined;
   a diversity-maximal source pool changes what every downstream stage hears.
2. **25_selfsim_structure** — self-similarity matrix + novelty-curve section
   boundaries: CLONE the structure of any track (e.g. a storm/edm piece)
   onto corpus units. A more musical structure-derivation than entropy.
3. **09_terrain_walk** — t-SNE map of our ~200-piece archive + composed paths
   (Lissajous/spiral/noise-walk) = a composer that walks the whole lab.
4. **20_spectral_mangling** — image ops on spectrograms (median/dilate/
   erode/sobel/dither). scikit-image is missing but scipy.ndimage covers
   nearly all of it. Zero-training novel mangling; instant port.
5. **38_latent_novelty / 39_token_flywheel** — the open-ended flywheel:
   novelty search in an autoencoder latent space (38) and a tiny token
   transformer that RETRAINS on its own admitted archive (39). Drop our
   material in `corpus/` and the flywheel dreams our lab onward.
6. **36_gpu_novelty** — MPS-batched render/STFT/featurize/k-NN novelty: the
   GPU path. 37_hybrid_events adds ring-mod/AM/spectral-morph hybrid genes
   and event genes (phrases, rhythm masks, pentatonic walks).
7. **27-31 QD toolbox** — novelty search, MAP-Elites, MCNS viability gates
   (the anti-junk filter), IMGEP goal sampling. The *gates* idea applies to
   audio_lab: admission criteria for generated pieces.
8. **33_output_analysis / 35_diversity_trajectory** — diversity audits,
   plateau detection over time. Run against OUR output folders to quantify
   whether our stages are really producing spread.
9. **14_micro_ddsp** — f0/harmonic/loudness analysis -> oscillator bank ->
   controls.csv for Pd + A/B morphs. Bridges straight into their Pd alien.
10. **15_distortion_fit** — fit a tiny WaveNet to a Pd chain, then feed it
    inputs the chain couldn't survive. "Corruption" as a learned model.

## D. Practical notes

- 16_cak_emergent: README only, no main.py (incomplete).
- 40_llm_mutator: excluded by their own design (and by our no-LLM rule).
- consolidate.py: a whole-lab kit builder (featurize everything, select
  spread set, organize into pitch-tier/timbre-cluster folders, Pd-ready,
  then DELETE the rest). The "one pool, select the best, delete the rest"
  philosophy is directly applicable to audio_lab's growing output.
- Sample-rate mismatch: their SR=48000, ours 44100 — both codebases already
  resample; fine.
- Missing deps in our env: hmmlearn (tiny), scikit-image (avoidable via
  scipy.ndimage), umap (optional; sklearn t-SNE suffices).

## E. Recommended integration order

1. **curate_lab.py** (port of the curation layer) — biggest impact, no new
   deps. Harvest audio_lab -> spread set -> feed back as curated corpus.
2. **08_codebook_lm + 10_hmm_grammar ports** (needs hmmlearn) — the corpus
   grammar generators; break the replay ceiling from the corpus side
   (binary_lab broke it from the data side).
3. **25_selfsim_structure port** — structure cloning for arrangement.
4. **09_terrain_walk** — walk the whole archive as a landscape.
5. Run **38/39/36** (their engines) with OUR archive as corpus/ — the
   symbiosis experiment.
6. **33/35 audits** on our output; adopt MCNS-style admission gates in our
   generators.