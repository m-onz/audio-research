# bomb

**the corpus-anchored experiment.** The premise, born from an honest
complaint (*womb's bred output was just crackles and quite boring*):
the minimal criterion decides what evolves — so let **your material**
decide it.

1. `sounds.wav` + `sounds2.wav` are silence-trimmed into a corpus
   (269 chunks, 25.9 min).
2. The corpus's spectral character is measured window by window —
   nine axes (rms, flatness, centroid, slope, tonal energy, flux,
   crest, stereo side, centroid drift). **ANALYSIS.md** is the write-up
   of why the material is interesting: dense (crest 1.7–4.4), harmonic-
   noisy (flatness 0.005–0.28), bright (~4.8 kHz), always moving
   (centroid_std 2600 Hz), stereo-alive (side 0.32).
3. That character envelope becomes the **minimal criterion** for
   breeding cryptid sound-objects: a candidate is viable iff its
   steady-state windows could belong to your corpus. No fitness — still
   MCC; the corpus is the environment.
4. The old output dies at the gate (clickstorm's crest = pure crackle
   signature); random single-law voices are too pure to pass; the
   genesis is three analysis-informed **layered** families
   (floor / buzz / choir), hand-tuned against the gate.

```bash
python3 bomb.py corpus  ../sounds.wav ../sounds2.wav -o corpus
python3 bomb.py analyze corpus --extra ../fusion_pieces -o analysis.json
python3 bomb.py calibrate corpus --strictness 0.5
python3 bomb.py breed corpus -o runs/bomb1 --gens 24 -v
python3 bomb.py breed corpus -o runs/bomb1 --resume --gens 12      # keep going
python3 bomb.py mixtape runs/bomb1 --grid texture --seg 8         # the album
```

Run artifacts are womb-format (specimens, atlases, ledger, montage) —
`womb cryptid info runs/bomb1` and `womb cryptid preview runs/bomb1`
work on bomb runs. Everything offline, deterministic from seed.

## files

```
bomb.py            # corpus trim, spectral profile, kinship judge, CLI
tune_founders.py   # the genesis tuning session (provenance)
corpus/            # 269 trimmed chunks + manifest.json
corpus_envelope.json  # the measured character envelope (the MC)
analysis.json      # per-file profiles (corpus + fusion_pieces)
ANALYSIS.md        # the findings — read this first
runs/bomb1/        # breeding runs
```