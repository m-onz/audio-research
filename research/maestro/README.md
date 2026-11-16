# maestro

**the orchestration organ** — a score organism over a spine. A CPPN
(the same organism design as prism's genome; genomes are JSON and
travel between the two) is sampled over a tick grid whose inputs are
a **spike-encoded spine** — existing audio, measured — so every voice
is a *function of the spine's state at each tick*. The phenotype is
an agnostic event score: `(voice, tick, velocity, duration)` in JSON,
knowing nothing about audio, mapping onto any sound material you own.
Previews are listenable today; the loom mapper renders phase D; echo
breeds phase E.

```
    the problem it exists to kill: events that don't know
    about each other. markov walks, euclid grids, greedy
    orderings are all LOCAL — none of them make a whole piece.

    the cure (NEAT Drummer, EvoMUSART 2008, best paper):
    never generate f(t). generate g(p(t)) — a part as a
    FUNCTION of a scaffold, and inherit its structure.
```

```
spine.wav ──scaffold──> spike trains s0..s4 + weather ──feeds──┐
   (any audio: a corpus chunk, an eden world, a cryptid         │
    specimen, a fusion piece, another maestro voice)            ▼
                                  CPPN genome — ONE memoryless pass
                                  over a FORM's tick grid (sections:
                                  slow bits, incredible fast BPM —
                                  density follows tempo for free)
                                          │
                                    voice × tick level matrix (velocity)
                                                               │
                              strike|gate, threshold ──> events (JSON)
                                                               │
        preview (pings, for the ear)      ──┤ you listen
        palette: voice -> corpus sound     ──┤ spine mix + duck
        post: stutters in the fast sections, stretches in the slow
             bits, tape, reverses, gates  ──> piece.wav + manifest
```

**The three-level cure for independent events.** Within a voice:
every event comes from the same network, so events are correlated
through its structure. Across voices: every voice reads the same
spike streams, so the ensemble moves together. Across time: the
spine's own pattern drives the ticks — when the spine diverges, every
voice diverges with it, purposefully (the Johnny Cope measure-23
effect). None of this is enforced by rules; it is what a *function*
means.

## the scaffold — the paper's fig. 2, for audio

NEAT Drummer fed MIDI notes into the CPPN as spikes that decay
LINEARLY over the note's duration: the value at any tick says where
the network is WITHIN the note. That decay is a **temporal coordinate
frame** — patterns can vary across a note's course, eighths read
differently from quarters, and answers can live mid-decay. Audio has
no note events, so maestro derives them:

| sense | what it is | how measured |
|---|---|---|
| `s0` | the master rhythm | full-band spectral flux, peak-picked |
| `s1..s4` | registers: low / mid / high / air | per-band envelope rises (30–200, 200–1500, 1.5k–5k, 5k–17k Hz) |
| `live_rms` `live_centroid` `live_flux` | the weather: loudness, brightness, change | robust-normalized, tick-sampled |

Every onset becomes a spike decaying until the *next onset in its
stream* — the inter-onset interval is audio's note duration, clamped
to [0.05, 2] s. Onset provenance travels inside every score
(`spine.onsets`) — every spike is auditable. One blockwise STFT pass
(echo's window sizes), ~35 ms per measured second of spine.

## the traversal — the paper's mechanics, kept

Sample the network at discrete ticks. Every output is a voice. The
output LEVEL at a tick is the velocity of a strike, rectified and
clipped to [0, 1]. Two consecutive audible ticks are two separate
strikes; a pause must be **produced** (output below threshold). No
memory, no recurrence — one vectorized `evaluate()` per whole score.

- **strike** — the paper's drums: every audible tick is an event of
  one tick. Velocity carries the nuance; machine-gun runs are the
  honest cost.
- **gate** — for sustained material: a rising edge opens an event
  held while audible; velocity is the run's peak.

**Pure by default.** The senses are `spikes,weather` plus bias —
nothing else. `t` (time) and `x` (place in piece) exist but are
*impure*: structure not inherited from the spine, the crack through
which independent events creep back. Opt in deliberately
(`--senses spikes,weather,time`) and treat it as an experiment arm.

## the score as data

```json
{"maestro": "score", "version": 1,
 "grid":   {"bpm": 120, "ticks_per_beat": 4, "tick_hz": 8.0, "n_ticks": 1457},
 "spine":  {"path": "...", "sha256": "...", "onsets": {"s0": [...]}},
 "senses": ["spikes", "weather"], "threshold": 0.12, "mode": "strike",
 "genome": { ... the whole organism, JSON ... },
 "events": [{"voice": "low", "tick": 12, "t": 1.5, "dur": 0.125,
              "velocity": 0.73}],
 "matrix": {"low": [... per-tick levels ...]}}
```

Agnostic means agnostic: the score never mentions audio. The same
score maps onto drums, corpus grains, eden worlds, cryptid specimens
— whatever the mapper resolves `voice` to (phase D: loom's catalog
query language; the score was shaped so a `layers` voice with a
`chain` drops straight in). Same score, different mapper: the same
organism, a different piece.

## champions — known answers, the cryptid precedent

| name | wiring | known answer |
|---|---|---|
| **follower** | `step(s0 − 0.35)` | strikes exactly at the spine's attacks — attack-lock |
| **answerer** | `gauss(6·s0 − 3)` | fires MID-DECAY, between onsets — the coordinate frame made audible: only the spike encoding has a "mid-note" to find |
| **splitter** | `step(sk − 0.3)` per band | register-locked arrangement: low/mid/high voices follow their own bands |

`random_score_genome` is the breeding founder: prism's minimal
NEAT-style start (loose hidden chain, two outputs) but wired onto
the scaffold senses — structure comes from the spine from generation
zero. The mutation zoo is the full prism set: drift, add_conn,
add_node (NEAT split), add_output (the unbounded alphabet,
brant_gecco19), swap_fn, toggle, subgraph duplicate.

## what exists / what's next

- **phase A — the organism** ✓ genome (CPPN), mutations, JSON.
- **phase B — the scaffold + traversal** ✓ spike-decay encoder,
  tick traversal, strike/gate, score JSON, `--bpm auto` (tempo
  estimated from the spine's own onsets).
- **phase C — the artifacts** ✓ preview: staged pings over an
  rms-balanced spine mix, deterministic bytes; atlas.png piano rolls.
- **phase D — the mapper + palette** ✓ `maestro map` / compose:
  voices → sounds (any audio file) with gain/pan/varispeed/truncate,
  velocity as gain, spine mixable — plus AUTO palette from the
  corpus manifest: percussive roles by filename keywords (the bank
  folders mix roles; names carry them), textures by family pools,
  deterministic by seed.
- **phase E — the loop** ✓ `maestro run`: score organisms bred over
  a spine under the ecosystem's law — prism's engine, adopted:
  viability + strictness dial (the two failure modes of score
  generation are the two gates: sparse → rms floor, machine-gun →
  static-drone gate), rotating-lens novelty over echo's blocks
  (pugh_ppsn16 multi-BC), maze's relative floor, 2 mutations per
  birth (leaps), ledger.jsonl + biographies + mixtape + journey +
  atlas + report; three arms (novelty / random / greedy — the
  objective trap kept in the study). Measured independently:
  audio-plain cells AND score-space cells (density/pacing —
  arrangement territory, never audio).
- **phase F — the composition** ✓ `maestro compose`: spine + FORM +
  organism + palette + POST → the finished piece. Forms are tempo
  maps — sections with their own BPM (tectonic: 92 grind → 320
  insane; accel: a 70→330 ladder; flip: fast slams against slow
  bits) — and because strike events are per-TICK, density follows
  tempo for free: the 320bpm section rolls at ~44 events/s while
  the 92bpm grind sits at ~13. POST is the mastering hand,
  section-aware and seeded (the audio_lab, distilled): club_lab
  stutters (const/ramp/ladder) land in the FAST sections, granular
  time-stretches hang the SLOW bits, tape speed, reverses, euclid
  gates; the duck sidechain glues the floor. One command, one wav,
  one manifest naming everything.
- **the kit** ✓ `--champion kit`: six voices — kick, rimshot, two
  percussion, hihat, open hihat — layered POLYMETRICALLY and
  evolving, as pure functions of the spine. Three clocks (the
  spine's bass / mid / full-flux bands run at their own meters),
  subdivision multipliers (sin(4pi*s2) doubles the mid clock,
  sin(8pi*s0) quadruples the master — quarter-IOI spacing that
  phases as the spine's meters drift), and a weather gate: the
  openhat opens only in the spine's loud passages, its phase
  sliding with loudness (0.3/s in the quiet break, 3.4/s in the
  loud rush of the tectonic demo). The palette casts the roles from
  the corpus by name — real kicks, rimshots, open hats — with a
  default kit image (hats left, rims right, floor center).
- **the club** ✓ `maestro club`: the night-club pipeline — a DRIVING
  floor and an EVOLVING top, the audio_lab/loom recipes made maestro:
  the driver is the hand-designed dancefloor skeleton (four-on-the-
  floor kick on the form's beats, accented downbeats, double-time in
  rush sections, none in the break, rimshot offbeats, closed-hat
  16ths, crash impacts at section drops, one spine-locked polymetric
  perc); the top is a maestro pattern BRED on this very spine
  (`maestro run` -> `best_specimen`), quantized onto the form's
  half-beat grid (peak velocity per window, the loudest N voices —
  club_lab's slot discipline pointed at the evolved phenotype: the
  organism decides WHICH voices crest WHEN and HOW loud; the grid
  makes it danceable). club_lab's duck (0.35/0.14) on every driver
  kick. The `club` form: 122 jackin' stretches, a 74 break, a 150
  rush.
- **layered generation** (the paper's future work): voice 2 takes
  voice 1's rendered audio — or its event matrix — as an additional
  spine. The scaffold need not be human: eden worlds, cryptid
  specimens, even another maestro score. The ecosystem can generate
  its own scaffolds; that is the open-ended move.

### the prism merge (what was adopted, with credit)

prism's engine.py, measure.py and its test spec are the breeding
  discipline maestro's phase E is built on, adopted nearly verbatim:
  `viability` + the strictness dial, `tonality`, `floor_ok`,
  `LENS_INDEX`/`lens_rank` (rotating-lens novelty over echo blocks),
  `MUTS_PER_BIRTH`, the ledger/biography/mixtape/journey/atlas/run
  artifact set, `plain_features`/`plain_coverage`/`plain_radius`,
  `complexity`, `anatomy`. prism breeds audio organisms
  (spectra/wavetable/field traversals); maestro breeds arrangements
  over spines. Same genome, same law, different phenotype — the two
  projects are designed to interlock: prism organisms can be maestro
  voice material (mapper) or spines (scaffold), and maestro scores
  can seed prism-style audio phenotypes later.

## the NEAT Drummer mapping

| NEAT Drummer (2008) | maestro / this ecosystem |
|---|---|
| MIDI song channels | a spine: any wav — corpus, eden, cryptid, fusion |
| note spikes (decay = duration) | onset spikes (decay = inter-onset interval) |
| drum outputs | named voices, grown by `add_output` |
| output level = strike volume | event velocity (continuous, rectified) |
| ticks (4 per beat) | the tick grid (`--bpm/--tpb` or `--tick-hz`) |
| the human selector (IEC) | echo novelty + you (eden's "You are the judge") |
| style rules | none — inherited from the spine, exactly as claimed |
| NEAT + CPPN | the prism genome, unchanged |

## honest limits

- **Onset picking is heuristic** — flux peaks above a relative
  floor (max of the 95th percentile and a quarter of the peak; the
  quantile alone collapses on sparse trains) with an 80 ms
  refractory. Bad on rubato swells, great on beat music. The
  provenance (`spine.onsets`) makes every miss inspectable, and
  `echo bench`-style A/B is the way to tune it.
- **−60 dB log floor** — band novelty is measured in the log domain
  with a floor at 10⁻³ of the global peak: quieter rises are
  invisible, so near-silent registers generate no phantoms (a pure
  sine's spectral leakage never fires the air train) but very quiet
  real events in loud material are missed too.
- **Tempo is estimated, not tracked** — `--bpm auto` scores beat
  periods against inter-onset intervals (long spans 1/k,
  subdivisions 1/m, near-ties break toward the beat). IOIs alone
  cannot resolve tempo octaves: uniform eighths read as 180, not 90
  — the honest limit, locked as a test. Rubato and long rubato
  swells will wobble; pass an explicit `--bpm` when you know it.
- **Strike mode machine-guns** a voice whose output stays high —
  that is the paper's semantics (a pause must be produced), not a
  bug; the threshold and gate mode are the dials.
- **The last event decays over `max_decay`** (2 s) — there is no
  next onset to bound it. Clamp is a parameter.
- **Sustained, onset-free spines** (pure drones) produce empty
  spike trains — pure genomes go silent on them; the weather senses
  still carry loudness/brightness, and impure genomes (t/x wired)
  will move, but the scaffold itself is absent. Detect onsets on a
  rhythmic spine, or feed drones as weather.
- **Preview pings are not the piece** — they prove the coupling is
  audible; `maestro map` is where sound happens.
- **A silent founder is legal** — the engine retries founders (200
  tries) and its gates reject what the strictness dial says is dead.
- **Mapper varispeed is linear-interp** — fine for ±octave;
  aliasing lives at the extremes (a loom render through `speed`
  processors is the hi-fi path when it matters).
- **Granular stretch has sidebands** — big grains (8192/2048) keep
  them tight but they are there; phase-locked stretching is the
  hi-fi upgrade when it matters.
- **Post is seeded damage, not taste** — the plan places events by
  section character and density, but whether a specific stutter
  lands at 47.2s is the seed's opinion. Curate by re-seeding or
  editing the manifest's post ledger (it is JSON, re-applyable).
- **Palette is name-driven** — percussive roles come from
  word-boundary filename keywords (kick/hat/rim/open...): a corpus
  file with a misleading name mis-roles, and files named "3.wav"
  fall back to their family pools. `--voice name=file` overrides.
- **live_rms is PASSAGE weather** — ~0.75 s of smoothing, anchored
  to the spine's own dynamic range; material with no dynamics reads
  flat zero (no loud passages = gates stay shut). Gates keyed to it
  hear "how loud is this passage", never "how loud is this frame".
- **A founder can be thin** — a random (or evolved) organism over a
  given spine may leave voices silent (the paper's honesty: a
  function of this spine, only). Survey a few seeds (`maestro score
  --random ...`) or breed on the spine (`maestro run`) before
  committing a 3-minute piece.
- **Genomes travel between maestro and prism structurally** (JSON
  round-trips) but the SENSE SETS differ — maestro wires s0..s4;
  prism reverted to geometry-only inputs. A maestro genome evaluated
  under prism gets inert nodes, not an error; rewire senses before
  expecting the same phenotype.

## usage

```bash
pip install -e ../echo && pip install -e .

maestro score --spine corpus/dubstep.wav --champion swell --bpm auto -o out.json
maestro map out.json --voice swell=kick.wav --voice flow=air.wav \
        --spine-gain 0.55 -o piece.wav                 # the final piece
maestro run --spine corpus/dubstep.wav --arms novelty random greedy \
        --gens 240 --bpm auto -o maestro_out/run1      # the breeding loop
maestro report maestro_out/run1
maestro preview out.json -o pv.wav --spine-gain 0.6
maestro info out.json
```

Champions: `follower` (attack-lock), `answerer` (mid-decay — the
coordinate frame audible), `splitter` (register lock), `swell`
(velocity follows the decay curve — the paper's nuance), `kit`
(six-voice ensemble: three clocks, subdivision multipliers, a
weather-gated open hat). Python:
`load_spine -> Scaffold`, `score_from(genome, scaf)` -> score dict,
`render_mapped(score, voices)` -> the piece, `run_experiment(...)` ->
the arms study. Everything offline, deterministic from seed — same
genome + same spine = byte-identical score, preview, and piece.

## layout

```
src/maestro/
  genome.py      # the CPPN (prism's organism, standalone) + mutations
  scaffold.py    # the spine: STFT, onset picking, spike trains, weather,
                 # tempo estimation (bpm auto), form grids + spine looping
  form.py        # the tempo map: sections (tectonic / accel / flip / json)
  score.py       # the traversal: one pass -> matrix -> events -> JSON
  champions.py   # follower/answerer/splitter/swell + the score founder
  preview.py     # the listenable artifact (staged pings + spine mix)
  mapper.py      # voices -> sounds -> the mix (sampler, memory or wav)
  palette.py     # voice -> corpus sound: roles by filename keywords,
                 # textures by family pools, deterministic by seed
  post.py        # the mastering hand: stutters/stretches/tape/reverse/
                 # gates, section-aware + seeded, duck sidechain
  compose.py     # the piece: spine + form + organism + palette + post
  club.py        # the night club: driving floor + evolved top
                 # (grid skeleton, quantize_top, best_specimen)
  measure.py     # the independent judge (prism's, adopted) + score-space
                 # descriptors + piano-roll atlas
  engine.py      # phase E: the breeding loop (prism's, adopted)
  cli.py         # maestro compose | score | map | run | report | forms |
                 #          preview | info | champions
tests/           # 97 tests: known-answers, gates, lenses, forms, post,
                 # mapper, compose end-to-end, determinism, io, cli
maestro_out/     # scores, pieces, previews, runs land here
```

## requires

Python ≥ 3.10, numpy, pillow (the atlas), ffmpeg on PATH, **echo** as
a library (`pip install -e ../echo`) — decode/wav-write, the window
sizes, and the breeding loop's memory (EchoBank + encode + LAYOUT).
Everything runs on this machine. No cloud, ever.

## license

MIT — see LICENSE.

## the paper

Amy K. Hoover, Michael P. Rosario, Kenneth O. Stanley.
*Scaffolding for Interactively Evolving Novel Drum Tracks for
Existing Songs.* EvoMUSART 2008 (best paper).
http://eplex.cs.ucf.edu/neatdrummer — the law this organ obeys:
parts as functions of parts; g(p(t)) over f(t); style and structure
are inherited, never specified.