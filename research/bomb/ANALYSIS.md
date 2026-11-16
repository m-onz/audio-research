# bomb — why these sounds are interesting (measured)

The complaint that started this experiment: *womb's bred output is just
crackles and quite boring.* The reference material — `sounds.wav` (12 min),
`sounds2.wav` (17.7 min), `fusion_pieces/` (59 pieces) — is the opposite.
This document measures WHY, then uses the measurement as the minimal
criterion for breeding (the corpus is the environment).

**Corpus built by silence-trim:** 269 chunks, 1554 s kept of 1782 s
(87% — the material barely rests; even the gaps are short).

## 1. The numbers

Per-file spectral profiles (2048-sample windows, mid channel; table shows
medians over files; centroid_std = how much the spectral center moves):

| set | rms | flatness | centroid Hz | tonal | flux | slope dB/oct | crest | side | centroid_std |
|---|---|---|---|---|---|---|---|---|---|
| **corpus** (sounds+sounds2) | 0.128 | 0.126 | 4859 | 0.212 | 0.051 | −4.8 | 2.83 | 0.32 | 2601 |
| **fusion_pieces** (59) | 0.066 | 0.408 | 4723 | 0.170 | 0.103 | −3.3 | 3.35 | 0.275 | 1777 |
| womb champions (rejected) | 0.11-0.8 | 0.001-0.14 | 153-2151 | 0.16-0.72 | 0.003-0.04 | −7..−18 | **1.1-15** | 0.11-0.15 | 15-1935 |
| womb bred, late-gen (rejected) | 0.02-0.24 | 0.02-0.11 | 272-1781 | 0.12-0.31 | 0.003-0.06 | −9..−11 | **2-15** | 0.0-0.11 | 100-978 |

## 2. Why the reference material is interesting — five findings

1. **Density, not events.** Corpus windows sit at crest 1.7–4.4 (p4–p96):
   *every* window carries sustained energy. Crackle lives at crest 10–15
   — peaks over silence. Your material never rests; crackle only taps.
2. **Harmonic-noise hybrid.** Window flatness 0.005–0.28: windows are
   pitched AND noisy at the same time (layered material). Pure drones
   (womb's chimera: flatness 0.001) are *below* the family — too clean.
   Hiss is above it. The interesting zone is the mix.
3. **Brightness.** Centroid median ~4.7–4.9 kHz: the corpus lives high.
   Womb's bred output medians ~1.2 kHz — dark by comparison.
4. **Spectra that move.** centroid_std 2600 Hz (corpus) / 1777 (fusion);
   flux 0.05–0.10 and rising in the composed pieces. The spectral center
   wanders on a seconds timescale. Bred crackle has near-zero flux
   outside its clicks.
5. **Stereo is alive.** Side-energy ratio 0.27–0.32 — significant L/R
   decorrelation, i.e. real stereo movement. Womb's output: 0.0–0.15.

Two families live inside the corpus (breeding can drift between them):
`sounds.wav` — bright, noisy-structured, loud (rms 0.2–0.4, flatness
0.25–0.55, centroid 2–5 kHz); `sounds2.wav` — darker, harmonic,
sustained (flatness 0.02–0.3, centroid 600–1200 Hz).

## 3. What the measurement did to the minimal criterion

The old MC (loud + alive + finite) admitted crackle because crackle is
loud *somewhere* and its blocks vary. The corpus MC measures whether a
candidate's **steady-state windows could belong to your material**:
inside the envelope on all nine axes (rms, flatness, centroid, slope,
tonal, flux, crest, side, centroid-drift), with per-axis floors.

Calibration (strictness 0.5): corpus self-kinship 82% pass; all three
rejected champions die at the gate — **clickstorm on crest (0.0: pure
crackle signature)**, chimera on rms/centroid (a too-loud, too-dark,
too-pure drone), jerk_voice on loudness (0.44 rms vs corpus window
p96 = 0.27). And **60/60 random law networks fail** — single cryptid
voices are too pure (flatness 0.0) or too spiky (crest 0.0) to sound
like layered material.

## 4. Consequence: the genesis had to be layered

Founders were hand-tuned against the gate (tune_founders.py) — three
families, all multi-law networks with multi-op ears:

| founder | what it is | kinship |
|---|---|---|
| **floor** | a dense-driven sandpile (avalanche *bed*, not clicks) + tonal ring above | 0.97–0.98 |
| **buzz** | fast integrate-and-fire pulse train (bright buzz) + ring + quantizer shadow | 0.90–0.94 |
| **choir** | fast excitable pulses under a bright inharmonic ring, map-modulated | 0.93 |

The same law zoo as cryptid — but the environment your corpus provides
only admits layered organisms. From there, MCC drift explores *inside
your material's family*, and the drift floor keeps children moving.

## 5. Run results (bomb1, strictness 0.5, 24 generations)

To be read from `runs/bomb1/state.json` — see README for commands.
Admission rates under the corpus gate run ~25–50% (the old gate: ~90%):
the environment is genuinely selective, exactly as designed.