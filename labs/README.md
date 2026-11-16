# labs — machine-listening sound design

A second research track alongside the flagship `riff_lab`. Where `riff_lab`
*evolves* riffs by novelty search, these labs **synthesize large candidate
pools of one-shots and then curate them by farthest-point selection** over a
spectral feature space — a close cousin of novelty search applied to timbre
design rather than evolution. "Render a thousand, keep the hundred most
different from each other."

They all stand on `oneshot_lab` (the transform engine, one directory up) and
read their raw source material from the shared `../sounds` corpus.

## The chain

The bass/dub labs register their synthesis + transform modules into a shared
patcher, so each lab inherits everything below it. Import order encodes the
dependency:

```
hifi            resampling / filtering / limiting helpers (no deps)
  └─ bass_lab        squelchy sub one-shots: spectral-stamp a carrier
       └─ bass_lab2       + comb/modal/Duffing/cross-synthesis modules
            └─ bass_lab3       + Chebyshev / xenharmonic mapping modules
                 └─ dub_lab          + wavetable-scan, waveset, vowel-seq, glitch
                      └─ squelch_lab       + phase-distortion / phaser / flanger / STFT phase

neuro_lab           COMPOSED evolving bass GESTURES (rhythm+pitch+timbre arcs)
  └─ neuro_lab2          + convolution / impulse-response colouring
       └─ neuro_lab3          + multiband / resampling finishing
```

Each writes a curated bank to its own `labs/<name>_out`-style folder (e.g.
`m-onz_neuro/`) — all gitignored.

```bash
cd labs
python3 neuro_lab.py --bpm 150 --keep 100     # pool -> curate 100 gestures
python3 squelch_lab.py                         # wet phase-family one-shots
python3 bass_lab.py --list-picks               # just rank the corpus by bassiness
```

Run `python3 <lab>.py --help` (or read the module docstring) for each lab's
dials — most take `--bpm`, a pool size, and a `--keep` count.

## Getting source material in

`process_m_onz.py` is the front of the pipeline: it slices long textures /
field recordings from `../sounds` into clean one-shots (multi-scale windows +
farthest-point diverse subset) and then runs them through the whole
`oneshot_lab` registry.

```bash
python3 process_m_onz.py --dry-run    # plan slice counts
python3 process_m_onz.py              # slice + process into m-onz_oneshots/
```

The bass labs then read `m-onz_oneshots/{raw,processed}`.

## Utilities

| script | what it does |
|---|---|
| `uniq.py` | de-duplicate a folder of sounds by `echo` similarity (exact k-NN) — the same memory organ `riff_lab` uses for novelty |
| `compare.py` | A/B spectral comparison of two sounds / folders |
| `normalize.py` | loudness / peak normalise a bank |
| `hifi.py` | shared resample / filter / limiter helpers (imported by the labs) |

## Conventions

- Source corpus: `../sounds` (override with `AUDIO_RESEARCH_SOUNDS`).
- Everything is pure-numpy + scipy + soundfile, offline and deterministic.
- Outputs are gitignored; redistribute only sounds you have the rights to.
