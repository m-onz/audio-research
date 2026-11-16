# womb — plan v1

**loom** weaves sound. **doom** renders sight. **womb** is where they meet:
a library whose artifact is always a finished audiovisual video, generated
jointly — sound-first, feedback-driven, degradation-embracing.

> loom → doom → **womb**: the rhyme closes, and so does the loop.

---

## 0. Why this library exists (synthesis of the research)

The pasted research on joint AV models (LTX-2.5, MOVA, JavisDiT++,
DreamX-Creator) is accurate, and its honest conclusion for our hardware is:
**the big joint models are a cloud stage, not a workstation core.** 16 GB
unified memory cannot hold LTX-2.5 (~100 GB bf16; even the 4-bit LTX-2.3
MLX stack is ~19.5 GB). What the research *recommends* — and what no
off-the-shelf tool provides — is exactly where womb positions itself:

1. **Sound-first A2V** — make the sound with loom, grow the visuals from it.
2. **Feedback loops between modalities** — each pass drifts; degradation is
   the material (the NIME `degr d t n` lineage).
3. **Latent-space performance** — RAVE on our own corpus, mapped to control.
4. **Own-aesthetic LoRAs** — the anti-"everyone's output looks the same" move.
5. **Seam-abuse** — mismatched cross-attention, hallucinated foley, glitchy
   sync as deliberate aesthetics.

Of these, **2 is achievable today with zero neural models** — because we
already own both ends of the loop: loom composes audio from corpora, doom
renders synced visuals from audio. The missing piece is the **inverse
channel**: a way to *hear* doom's output and feed it back into loom.
That's womb's core invention (see F2 below). The neural flows (1, 3) run
locally at small scale; the heavy joints (LTX-2.5 A2V, MMAudio foley) are
queued cloud jobs with resume — womb treats them as *one optional stage
in a DAG*, never as the identity.

## 1. Hardware contract (verified on this machine)

| capability | status |
|---|---|
| arm64, macOS, 10 cores, **16 GB unified**, ~25 GB free disk | ✅ this plan targets this |
| torch 2.14 + **MPS** | ✅ local neural inference viable |
| ffmpeg 8.0.1 (libx264/aac) | ✅ all rendering pipes through it (doom's writer) |
| Fanless sustained load | ⚠️ design rule: every flow is **resumable + checkpointed**; batches sized to avoid thermal throttling; long renders tolerate interruption |

**Local / cloud split (fixed in design):**

- **LOCAL, now**: metabolism flows (F1–F3, pure numpy), RAVE *inference*
  (~20× realtime on CPU), StreamDiffusion on MPS (documented M1+ 16 GB OK),
  LTX-Video 2B distilled short clips (verify on day one; MPS 16 GB is
  plausible at 512×384, few steps).
- **CLOUD, queued**: RAVE *training* (hours of GPU), LTX-2.5 A2V / I2V,
  MMAudio / ThinkSound V2A, LoRA fine-tunes. Job spec + retry + download +
  automatic doom post-pass. No cloud is *required* — a local-only womb is a
  complete art practice.

## 2. Core concepts

- **Reel** — the artifact object: one audio stem + one video + a lineage
  manifest (loom seed, doom seed, pass numbers, drift metrics, model ids).
  Always renderable to MP4 (doom writer) or WAV + frame folder.
- **Flow** — a named, composable pipeline step: `av`, `va`, `degrade`,
  `rave`, `stream`, `cloud`. A **score** (DAG of flows) renders a Reel.
- **Drift** — first-class measured quantity: after every feedback pass,
  womb logs loudness/spectral stats, frame entropy, cross-pass SSIM-ish
  distance. Drift is steered, not accidental — presets target drift rates
  ("hold", "bloom", "collapse").

## 3. Module map

```
womb/
  src/womb/
    reel.py          # Reel: audio+video+manifest; render(); lineage
    features.py      # shared machine-listening (reuses loom.audio ideas)
    sonify.py         # F2: video -> audio (the inverse channel)
    flows/
      av.py           # F1: audio -> video (productized loom→doom bridge)
      feedback.py     # F3: pass orchestration + drift metrics + degrade ops
      rave.py         # F4: corpus -> RAVE latents -> audio (train cloud/run local)
      stream.py       # F5: audio feats -> StreamDiffusion keyframes (MPS)
      ltx_local.py    # F5b: LTX-Video 2B short clips (MPS, if verified)
      cloud.py        # F6/F7: job queue for LTX-2.5 A2V, MMAudio foley, LoRA
    degrade.py       # loss operators: codec rounds, resample wars, bitrot,
                      #   frame-drop, generational wav->mp3->wav, etc.
    score.py         # DAG composition of flows; checkpoint/resume
    presets.py        # named drift studies (see §6)
    cli.py            # `womb render score.yaml`, `womb sonify`, `womb drift`
  tests/
  examples/
```

Dependencies: numpy + loom + doom (pip -e siblings) + ffmpeg. Neural flows
import torch lazily; cloud flows import nothing heavy (job specs are YAML).

## 4. The flows (concrete specs)

### F1 `av` — sound-first audio→video (productize what we've been doing)
Input: wav or loom compose-config. Runs loom (if config) → doom render
(auto_plan or explicit) → Reel. This exists scattered in session scripts
today; womb makes it one call with a manifest that records both seeds.
**Effort: small.**

### F2 `va` — sonification: video→audio (the missing inverse channel)
The heart of womb. Three modes, all pure numpy/ffmpeg, all deterministic:

1. **spectra** — treat the video's own frames as spectrograms: decode →
   luminance → log-frequency columns → ISTFT with loom's phase machinery.
   The doom `spectro` scene's mirror image — the two are designed to
   meet in the middle.
2. **motion-grains** — frame-difference energy gates a granular
   recomposition of a chosen corpus (loom's chunker does the grain
   cutting; motion decides where).
3. **flow-tones** — horizontal block displacement (optical-flow proxy) maps
   to detune/pan of sustained tones; quiet frames → silence (loom's
   `gap` philosophy, video-driven).

Output is a WAV (or loom-ready corpus folder) — i.e., **doom videos can
be heard by loom**. This single capability turns the two libraries into a
cross-modal metabolism. **Effort: medium. This is the flagship.**

### F3 `feedback` — drift loops
`reel_n = F1(va(degrade(reel_{n-1})))` for N passes. Pass ops and drift
telemetry land in the manifest; a `drift_report()` prints the arc
(loudness, spectral centroid, frame entropy vs pass index). Degrade
operators (§ `degrade.py`) are selectable per pass. **Effort: small once
F2 exists.**

### F4 `rave` — audio latent bank (local inference / cloud training)
- `womb rave encode corpus/` → trains (queued to cloud by default; local
  MPS possible for tiny corpora) → checkpoint + 16-d latent bank.
- `womb rave walk --seed` → latent interpolation / feature-conditioned
  walks → audio stem → feeds F1. Latents map to loom features (rms/bass/
  centroid) so *one gesture moves both sound and picture* — the IRCAM
  IML direction from the research, made concrete.

### F5 `stream` — audio-conditioned keyframe stylization (MPS)
StreamDiffusion img2img at ~512px, driven per keyframe (1 fps) by audio
features → doom interpolates between keyframes (existing scenes) and
composites. Runs local; the big joints stay cloud.

### F6/F7 `cloud` — the heavy joints, queued
YAML job spec: `{model: ltx-2.5-a2v, audio: stem.wav, prompt, seconds,
resolution}` → submit (RunPod/Lambda abstracted behind one `Provider`
interface) → poll → download → **automatic doom post-pass** (sync check,
optional glitch chain) → Reel. MMAudio V2A inverse jobs for foley study.
Resume-safe: jobs persist; interrupted renders restart cleanly.

## 5. What deliberately stays OUT

- No real-time/live engine (that's Max/Pd + nn~ territory; womb is a
  *rendered-artifact* practice — the research's "rendered vs live"
  question answered for v1).
- No browser/server; CLI + scripts only, like loom/doom.
- No attempt to run LTX-2.5 locally, ever — it's a cloud stage by contract.

## 6. Presets (named drift studies)

- `metabolism` — pure loom↔doom loop, gentle degradation, 6 passes, drift
  report. The thesis piece.
- `collapse` — aggressive degrade ops (mp3 rounds, bitrot, frame-drop) each
  pass; the piece audibly/visibly dies by design.
- `sound-first` — loom seed → LTX-2.5 A2V (cloud) → doom post → Reel.
- `seam-abuse` — cloud A2V at deliberately mismatched conditioning +
  foley hallucinations composited as texture, not correction.
- `latent-organ` — RAVE bank walks controlling sound + doom parameters.

## 6.5 Repositioning (per artist feedback)

womb is a GENERATIVE engine first: organs — evolving systems projected to
both senses (waveplate / automata / swarm / kuramoto / territory) —
compose original pieces with optional corpus grains. The metabolism
(sonify + feedback loops) remains as the second mode.

## 7. Build plan

> STATUS: Phase 1 built (metabolism + organs + pieces, 19 tests) —
> local-only per the artist's decisions (womb, offline, this machine,
> pure Python; RAVE optional via isolated venv, verified pure-Python).

**Phase 1 — Metabolism (local, pure numpy; the novel core).**
Reel/artifact, F1 bridge, F2 sonify (3 modes), F3 feedback + drift
metrics, `degrade.py` core ops, CLI, tests incl. determinism + a full
loopy regression (a→v→a→v where v2a(v1(a)) roundtrip is measured).
*Acceptance: `womb preset metabolism --seed 7` renders a 3-pass Reel with
drift report, fully offline, <10 min on this machine.*

**Phase 2 — Local neural.** RAVE inference wrapper + latent walks;
StreamDiffusion keyframe stylizer (MPS); LTX-Video 2B local clip test.
*Acceptance: one Reel whose audio passed through RAVE latents and whose
keyframes passed through StreamDiffusion, rendered on this laptop.*

**Phase 3 — Cloud queue.** Provider interface, LTX-2.5 A2V + MMAudio
adapters, resume, auto-post. *Acceptance: submit→download→post for one
A2V job without babysitting.*

## 8. Open decisions (need from you)

1. **Name**: `womb` (loom+doom meet; birth metaphor) — or `vroom` (A/V in
   one word, playful)? My vote: **womb**.
2. **Cloud in v1?** Phase 3 can be deferred indefinitely; local-only womb
   is already a full practice. Provider preference (RunPod/Lambda/Colab)?
3. **16 vs 24 GB** — I design for 16 (this machine, the safe subset).
4. **Priority order** if you want neural sooner: RAVE-sound or
   StreamDiffusion-picture first?