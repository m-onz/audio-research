# womb

**where loom and doom meet.** Original audiovisual generation, fully
offline: **organs** — evolving systems projected to both senses at once.
The picture and the sound are the same state; sync is structural, not
driven. The artifact is always a finished video.

```bash
womb generate --piece colony -l 60 --corpus ../loom/loom_out -o piece.mp4
```

## Organs (joint A/V generators)

| organ | what it is | what you hear vs see |
|---|---|---|
| **waveplate** | 2D wave equation on a struck plate | probe readouts — the visible standing modes ARE the tones (Chladni physics) |
| **automata** | continuous cellular automaton | cell births fire corpus grains (pitched by row); the state's column shape plays as a live spectrum |
| **swarm** | agents on a region grid | occupancy = voice gains (pentatonic), boundary crossings = grains — hear the traffic you see |
| **kuramoto** | coupled phase oscillators | the phase field seen directly and summed as voices — sync waves become beating |
| **territory** | your corpus as a 2D map of its own features | a walker tours YOUR material — trajectory heard as corpus chunks, seen as the map |

`--corpus` points at any folder of wavs (loom output, the EP, field
recordings) — it becomes the organs' grain bank. Without a corpus the
organs run on their own synthesis. Register new organs with
`@womb.register_organ`.

Named pieces: `plate`, `colony`, `territory`, `cabinet` (one of each).
Or pass your own sequence: `womb generate --organs waveplate swarm kuramoto`.

## The metabolism (feedback as a second mode)

Beyond generation-from-scratch, womb can also breed generations:
**sonify** hears video back into sound (frames-as-spectrograms,
motion-gated grains, flow drones), and **metabolize** runs the loop with
real generational loss (mp3/mp4 codec rounds, resample wars, bitrot) and
a drift ledger:

```
stem_0 --doom--> video_0 --sonify--> stem_1 --degrade--> video_1 --> ...  --> final.mp4
```

```bash
womb metabolize piece.wav -o my_reel -n 3 --preset metabolism --seed 7
womb info my_reel          # the drift ledger
womb sonify video.mp4 -o heard.wav --mode spectra
```

## cryptids — bred sound (phase A of CRYPTID.md)

**Sound-objects born, not designed.** No oscillators, no notes, no
scales, no filters-as-tone, no samples: the only signals that exist
are states of evolved processes (chaotic jerk systems, chimera
oscillator rings, self-organized-critical sandpiles), read by an
*ear* — a measurement genome. Three hand-built champions prove the
physics; their params were found by search on this machine (the
coefficient search lives in `scripts_check/`, the search itself is
the provenance):

| champion | what it is | what you hear |
|---|---|---|
| **chimera** | 128+128 Kuramoto units, ring coupling | a coherence boundary that wanders the ring — tonal where coherent, drifting whistles where not; the stereo image IS the boundary position |
| **clickstorm** | a 40×40 BTW sandpile at SOC | 1/f click weather: chaotic maps choose where/when grains fall, every avalanche is a click with power-law amplitude and causal pan |
| **jerk_voice** | an evolved jerk attractor (the one Lyapunov-positive set in 8000 samples) | an inharmonic voice that never repeats, doubled by a quantizer shadow, damping breathing ±10% on a slow chaotic map |

```bash
womb cryptid list
womb cryptid render chimera -l 60 -o womb_out/cryptid_chimera.wav
womb cryptid info jerk_voice        # the biography: born, lineage, how it is heard

# phase B — the MCC breeding loop (CRYPTID.md)
womb cryptid seed --strictness 0.5 --gens 16 -o womb_out/crypt1
womb cryptid seed womb_out/crypt1 --resume --gens 20          # keep breeding
womb cryptid preview womb_out/crypt1 --grid time -l 120     # the listenable atlas
womb cryptid mixtape womb_out/crypt1 --grid texture --seg 8  # the run's album
womb cryptid info womb_out/crypt1    # the generation ledger + regime hint
```

Breeding pairs two populations under a **mutual minimal criterion** —
processes must be heard alive by ≥1 evolved ear, ears must hear ≥1
process alive — with no fitness anywhere and Soros' `--strictness`
dial scaling the bands (loose floods, strict converges, mid diverges:
Chromaria reproduced, logged per generation). Every admission lands in
three character grids (texture / time / space — flatness×slope,
events×modrate, side×asym), gets a specimen JSON you can render
yourself, and appears in the preview montage: scrolling spectrogram,
waveform, pan trace, Takens portrait, and the six character bars, one
segment per occupied cell. Runs are resumable (`--resume`).

Ten laws in the zoo (`jerk osc_pop map sand excitable fire bool delay
clamp bytefeed`) + five ear ops (`tap diff embed events chimera`)
compose arbitrary specimens as JSON — breed your own. Renders are
bit-identical from the seed, stereo, and fast (a 60 s champion renders
in 2–8 s on this machine). `womb cryptid render <x> --hi` simulates at
the output rate when decimation aliasing is unwanted.

## Requires

- Python ≥ 3.10, numpy, **ffmpeg/ffprobe on PATH**
- sibling installs: `pip install -e ../loom -e ../doom` (the metabolism
  renders through doom; analysis through doom's feature engine)
- **everything runs on this machine, offline. No cloud, ever.**

## The inverse channel: `sonify`

doom's `spectro` scene turns audio into a scrolling spectrogram;
womb turns frames *back* into sound. Three pure-numpy modes:

- **spectra** — frames as spectrogram columns: log-spaced partial bank,
  additive synthesis with phase continuation. The mirror of doom's
  spectrogram scene.
- **grains** — motion-gated granular resynthesis: the video's
  frame-difference energy scatters corpus grains (doom's cloud arranger,
  in audio).
- **tones** — block-displacement drone: coarse optical flow drives detune
  and level of an oscillator bank; stillness is silence (loom's gap
  philosophy, video-driven).

## Drift presets

- **metabolism** — gentle, listenable generations: mp3 rounds at falling
  bitrates, palette crush, rotating sonify modes (spectra → grains → tones)
- **collapse** — each pass loses fidelity on purpose: resample wars,
  bitrot, frame drops, 3-bit palettes
- **hold** — the slowest possible rot (160 kbps mp3 rounds only)

Every pass's telemetry (rms, centroid, luma, motion, audio/video drift
from the seed) lands in the reel's `manifest.json`; `drift_report()`
prints the ledger.

## RAVE (optional, still offline)

A local neural latent bank — trained on *your* corpus, run from Python
(no Max/MSP anywhere). Because RAVE's torchaudio lags this machine's
torch, it lives in an isolated venv inside womb:

```bash
cd womb && python3 -m venv neural \
  && neural/bin/pip install torch==2.11.0 torchaudio==2.11.0 acids-rave

womb rave doctor                       # check the venv
womb rave preprocess corpus/ --db db   # corpus -> training database
womb rave train db --name my_rave      # hours on CPU (macOS path); resumable
womb rave generate runs/my_rave --input seed.wav --out heard.wav
```

Verified on this machine: preprocess + train loop run on CPU. Then feed
the reconstruction back into the metabolism — latent round-trips as drift.

## Layout

```
src/womb/
  video.py       # frame decode + telemetry (luma, motion, entropy, flow)
  sonify.py      # the inverse channel (spectra | grains | tones)
  degrade.py     # real generational loss (codec rounds, wars, bitrot)
  reel.py        # the artifact: lineage dir + manifest + final.mp4
  metabolize.py  # the loop + drift ledger
  cryptid.py     # bred sound-objects: laws zoo + ears + champions
  breed.py       # the MCC loop: mutual-MC breeding, strictness dial, archive
  flows/rave.py  # local neural latents (isolated venv)
  cli.py         # womb metabolize | sonify | info | rave | cryptid | presets
scripts_check/  # the searches that found the champions (provenance)
neural/          # optional RAVE venv (not created by default)
tests/           # 15 cryptid tests + 14 metabolism tests
```

## License

MIT — see LICENSE.