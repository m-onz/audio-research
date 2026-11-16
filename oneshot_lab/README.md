# oneshot_lab

A general-purpose, reusable Python library for mutating folders of oneshots
(or any sounds) into banks of unique, weird variations: extreme time stretch
(paulstretch, phase vocoder, wavetable scan), pitch & harmonic effects,
spectral transformations, and glitch / databend corruption.

Built by extracting and generalizing the DSP developed across this project's
experiments (`audio_lab`, the `*_reprocess` scripts: databend / NMF / LPC
stamping, wavetable voices, storm vari-speed, granular grain-fx, crossover
possession, harmonic resonators & shimmer), plus newly requested effects:
paulstretch, wavetable & oscillating time stretch, automated harmonic/pitch
effects, spectral transformations, glitch effects.

## Quick start (batch a folder)

```bash
# mutate ./oneshots into ./oneshot_out — every transform, one bank per transform
python -m oneshot_lab ./oneshots -o ./oneshot_out

# curated selection + 2 random mutation chains per file, tighter caps
python -m oneshot_lab ./oneshots -o ./oneshot_out \
    --transforms paulstretch,glitch,harmonics --random-chains 2 --max-out-s 6

# extra random-parameter variants of a few transforms
python -m oneshot_lab ./oneshots -t spectral_nmf,paulstretch --variants 4

# see everything registered
python -m oneshot_lab --list
```

Output layout: `oneshot_out/<transform>/<source>__<transform>__<params>.wav`
plus `_manifest.json` / `_manifest.jsonl` with full provenance
(source, transform, params, seed, duration) for every output.

## Generational reprocessing (banks of banks)

A bank is itself a folder of sounds, so it can be reprocessed into a new
generation — mutations of mutations, with the lineage readable in the
filenames (`<hit>__echo_replicate__def__spectral_smear__...`). Scan it
recursively and sample it in a balanced way:

```bash
# generation 2: 2 sounds picked from each transform folder of the bank,
# mutated through every transform again
python -m oneshot_lab ./oneshot_out -o ./oneshot_out2 \
    --recursive --pick-per-folder 2 --random-chains 1 --seed 7

# or pick N sounds total, seeded
python -m oneshot_lab ./oneshot_out -o ./oneshot_out2 --recursive --pick 60
```

## Library use

```python
import numpy as np
from oneshot_lab import core, engine, timestretch

x, sr = core.read_audio("hit.wav")          # any soundfile format, (n,2) float32
rng = np.random.default_rng(0)

# one transform, explicit params
y = timestretch.paulstretch(x, sr, rng, stretch=30.0, window_s=0.4)

# a fully random mutation chain (2-3 stacked transforms, random params)
y, chain = engine.mutate(x, sr, rng)

# batch a whole folder (multiprocessed)
engine.run_folder("./oneshots", "./oneshot_out", random_chains=2)
```

Every transform shares one signature:

    fn(x, sr, rng, **params) -> np.ndarray

so you can stack them freely (chain them, randomize them, add your own and
register them in `registry.py`).

## Transforms

### timestretch
- **paulstretch** — Paul Nasca's Extreme Sound Stretch (canonical algorithm,
  public domain): large windows, magnitudes kept, phases randomized each
  frame — extreme stretches become ambient smear. Extensions: `blur`
  (spectral smoothing), `glide` (crossfade magnitudes between input frames),
  `phase_coherence` (partially evolving phases: tonal drones).
- **paulstretch_hyper** — 0.6s windows, blur, coherence: deep ambient.
- **pv_stretch** — phase vocoder: duration stretches, pitch holds.
- **granular_stretch** — pitch-holding grain smear with jitter/reverse/pitch.
- **wavetable_stretch** — TIME-STRETCH WAVETABLE: the sound becomes a stack of
  single-cycle wavetables; an oscillator at the hit's detected f0 reads them
  while a scan head walks through the original (forward / pingpong / random
  walk) — time re-cast as wavetable position.
- **oscillating_timestretch** — OSCILLATING TIME STRETCH: phasor-LFO
  vari-speed (rate AND pitch breathe together, tape style), with wobble
  storms. From the storm vari-speed.
- **tape_stop** — vinyl brake; time+pitch dive, dead-slow tail.
- **freeze_loop_stretch** — loudest window machine-loops with micro-jitter,
  releasing back into the real tail.

### spectral
- **spectral_freeze** — hold a magnitude frame forever, phases drifting — a
  frozen ghost following the original envelope.
- **spectral_smear** — temporal blur of magnitudes; transients melt.
- **spectral_shuffle** — STFT time-blocks permuted, some reversed (spectral
  glitch).
- **spectral_reverse** — envelope backwards, waveform forwards.
- **spectral_binroll** — rotate the frequency axis (metallic alien recast).
- **spectral_hybrid** — spectral convolution with a self variant (reverse /
  shuffle / roll) — cross-synthesis.
- **spectral_nmf** — learn a dictionary from the sound's OWN spectrogram and
  rebuild it — self-requantized texture.
- **spectral_morph** — magnitudes morph original → variant → original with
  held phases (the crossover "possession chain").
- **lpc_reanimate** — the hit re-speaks through its own LPC formant filter,
  excited by glottal buzz / noise / its whitened self.
- **comb_bank** — tuned feedback combs at the hit's f0 harmonics.

### glitch
- **databend** — the wav file's RAW BYTES (headers included) sonified through
  many encodings (mu-law, delta, nibbles, 16-bit structures, MSB bodies, LSB
  dither) and optionally corrupted (swap / zero / xor / rotate).
- **crush**, **ringmod**, **am_gate**, **waveshape** (tanh / fold / cheby /
  asymmetric), **samplehold_wobble** (drunk tape skitter).
- **stutter** — machine-gun loop-lock with pitch ladders and dropped hits.
- **granular_scatter** — micro-grain explosion, each grain independently
  reversed / crushed / decimated / ring-modded, scattered across stereo.
- **echo_replicate** — echoes at irrational multiples of the hit's own length
  (grid-orthogonal glitch, from storm glitch_replicate).
- **ca_chop** — Wolfram cellular-automaton gated chop (chaotic rules only),
  each active step randomly pitched / reversed / driven / filtered.
- **reverse** — plain reverse.

### harmonics
- **octaver** — interval stacks (octaves/fifths), detuned, panned.
- **detune_chorus** — drifting detuned voice swarm.
- **shimmer_harmonizer** — pitched delay-tap feedback cascade (octave/fifth
  shimmer, from the harmonic lab).
- **reich_phase** — copies at mismatched loop speeds, drifting in/out of sync.
- **resonator** — measure the hit's own harmonic series and re-ring it as
  decaying detuned partials.
- **sub_octave** — tuned sub sine following the hit's envelope.

### pitch
- **pitch_shift_pv** — pitch shift, duration preserved.
- **pitch_shift_tape** — tape pitch: pitch and time bend together.
- **glissando** — gliding pitch sweep through the hit.
- **vibrato** — oscillating pitch warble.

## Engine notes

- Deterministic: every (file, transform, variant) gets a seed derived from
  the master `--seed` — the same run is reproducible.
- Inputs longer than `--max-in-s` are sliced to a random window; every output
  is capped at `--max-out-s`, peak-normalized, DC-stripped, edge-faded and
  written as 16-bit PCM.
- Failures are logged per job (stderr + manifest), never abort the batch.
- Add a transform: write `fn(x, sr, rng, **params)` in a module and add one
  entry to `TRANSFORMS` in `registry.py` (function, family, defaults,
  needs_bytes, blurb, random-param generator). The CLI/engine pick it up.

## Oneshot warp batch (huge variation banks)

`oneshot_warp.py` (project root) is the oneshot-production workflow: it
samples the curated corpus + ./oneshots + the *_out track folders, splits
everything on energy onsets (sustained material is chunked), and renders
thousands of ONESHOT-scale `stretch_modulate` variations (natural endings,
trimmed tails) into `warp_ones/<recipe>/`:

```bash
python3 oneshot_warp.py --dry-run                    # plan + counts
python3 oneshot_warp.py --corpus-sources 1500 \
    --chopped-sources 700 --variants 3 --seed 11    # full batch
```

Recipes (parameter styles): ramp_short, brake_short, wave_short,
breath_short, pulse_short, grainy_short, silk_short, wild_short — each
segment gets a seeded pick of `--variants` distinct recipes.

## Requirements

numpy, scipy, soundfile (`pip install numpy scipy soundfile`).