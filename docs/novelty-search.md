# Novelty search, as implemented in `riff_lab`

This is the mechanism behind `python3 -m riff_lab gen`. It's a faithful,
runnable take on **novelty search** (Lehman & Stanley, 2011) and **minimal
criterion novelty search** (MCNS), applied to generative audio. If you've read
those papers this will be familiar; if not, this doc is self-contained.

## The premise

Optimising toward a fixed objective ("make it sound good") collapses diversity:
the population converges on whatever climbs the gradient fastest, and
open-endedness dies. Novelty search throws the objective away. Instead it
rewards individuals for being **different from everything the search has already
produced**. With a good behaviour characterisation and a persistent memory, that
pressure alone drives an ever-expanding frontier of genuinely new artefacts.

The catch everyone hits: *different measured how, against what memory?* Thin
memories — a single parent, a few binned scalars, an 8×8 behaviour grid — can't
tell a truly new sound from a rearranged old one. So the heart of this repo is a
**rich** memory (`echo`) and a **structural** one (pattern descriptors), combined.

## Behaviour = three signals

For each rendered candidate riff, novelty is measured on three axes, all
relative to the archive of everything heard before (`riff_lab/novelty.py`):

```
novelty(x) = W_AUDIO   · audio_sparseness      (0.50)
           + W_PATTERN · pattern_sparseness    (0.35)
           + W_COMPLEX · complexity             (0.15)
```

1. **Audio sparseness.** The rendered preview (drums included) is encoded by
   `echo` into a 556-D temporal vector — MFCCs, spectral scalars, chroma,
   stereo life, envelope "weather", a 64-band silhouette — *chunked over time so
   event order survives* ("click then drone" ≠ "drone then click"). Distances
   are taken in a **fitted metric space** (per-dim z-norm + block weights) so no
   single Hz-scaled axis dominates, and averaged over rotating **lenses**
   (timbre / time / pitch). Sparseness = mean distance to the *k=15* nearest
   neighbours.

2. **Pattern sparseness.** A structural fingerprint of the groove — per-voice
   density, off-beat fraction, inter-onset-interval variance, position entropy —
   so two riffs with similar *timbres* but different *rhythms* still read as far
   apart. z-normed Euclidean k-NN.

3. **Complexity.** Total node-count of the evolvable expressions (melodic motifs
   + drum fill), a gentle **saturating** pressure (saturates at ~120 nodes) so
   lineages ratchet toward more intricate motifs instead of only chasing new
   timbres. This is the *complexification* axis.

Each axis is normalised and blended. **Selection is rank by this number.**
Nothing else.

## The minimal criterion (MCNS)

Pure novelty will happily select broken outliers — silence *is* different. So a
single **viability floor** gates children before they're scored
(`riff_lab/ga.py`, `riff_lab/ears.py`):

- `ears.defects()` — clicks, leading silence, DC offset, digital clipping, NaN:
  objective technical failures, always rejected, no dial.
- a bassline must actually play (≥ 3 bass hits) or it isn't a DnB riff.

That's the entire veto. **No quality or taste objective touches selection** —
the gate is about technical validity, not aesthetics.

## The archive

`NoveltyArchive` (`riff_lab/novelty.py`) is the persistent memory, saved to
`riffs/_novelty_archive.json` and reloaded across runs — so novelty is measured
against *everything you have ever generated*, not just this session.

- **Admission** is thresholded: a child joins the archive if its novelty clears
  `rho_min`, plus a 2 % stochastic trickle so the memory never ossifies.
- **`rho_min` adapts**: if a generation admits nothing it relaxes (×0.95,
  "we're stuck, lower the bar"); if it floods (>4 admits) it tightens (×1.20).

## The generational loop

```
population (persists across --gens, capped at 48)
   │
   ├─ reproduce: tournament-of-two (low selection pressure) → crossover / mutate
   │             + 15 % fresh-random immigration (keeps the gene pool honest)
   │
   ├─ evaluate: render each child → echo-encode → DROP the audio
   │            (only the 556-float vector + genome are kept, so memory stays
   │             flat no matter how large --n / --candidates)
   │
   ├─ score: blended novelty vs archive ∪ population ∪ batch
   │
   ├─ select: top --n are built to disk; winners + runners-up rejoin the pop
   │
   └─ admit winners to the archive; adapt rho_min; prune pop to cap
```

Behaviour is measured against the **union** of the permanent archive, the live
population, and the current sibling batch — the standard novelty-search
formulation, so the search never collapses because "new" is always relative.

## The knobs

| where | knob | effect |
|---|---|---|
| CLI | `--n` | riffs kept per generation |
| CLI | `--gens` | consecutive generations (lineages persist & complexify) |
| CLI | `--candidates` | novelty candidates generated per kept slot |
| CLI | `--strictness` | 0 = defects-only (fully open) … 1 = tightest character gate |
| `novelty.py` | `W_AUDIO / W_PATTERN / W_COMPLEX` | the three-axis blend |
| `novelty.py` | `K` | neighbourhood size for sparseness |
| `novelty.py` | `COMPLEX_SAT` | node-count where the complexity ratchet saturates |
| `novelty.py` | `LENSES` | which echo sub-blocks define each distance lens |

Genomes also carry their **own** `strictness` gene, so the character gate can
itself evolve when you let it.

## Teaching the gate your ears (optional)

`ears` can be calibrated to your taste without ever touching the *search*: drop
sounds you like into `riffs/_taste/good/` and ones you reject into
`riffs/_taste/bad/`, then

```bash
python3 -m riff_lab judge --fit     # learn per-metric thresholds
python3 -m riff_lab judge --reset    # forget them
```

This only moves the *character* thresholds (the taste/validity envelope), not
the novelty objective — the open-ended search stays objective-free by design.

## Further reading

- Lehman & Stanley, *Abandoning Objectives: Evolution through the Search for
  Novelty Alone* (2011).
- Lehman & Stanley, *Minimal Criterion Coevolution* and the broader open-ended
  / quality-diversity (MAP-Elites, novelty+local-competition) literature.
- `echo/README.md` for the encoder and vector-archive internals.
