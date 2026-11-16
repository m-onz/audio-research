# sounds — your source corpus

This directory is where you drop the raw material. It ships empty on purpose:
the code is public, the sounds are yours.

Everything under here is scanned **recursively** for `.wav` files, so the
layout is up to you — a flat folder of one-shots works fine.

```
sounds/
├── anything.wav
├── a_subfolder/more.wav
└── drums/              # only needed by riff_lab's drum kits (see below)
    ├── kick/*.wav
    ├── snare/*.wav
    ├── hihat/*.wav
    ├── openhh/*.wav
    └── perc/*.wav
```

## What reads from here

- **`riff_lab`** builds its sound pool (`pool.py`) and transient-sliced chop
  library (`chop.py`) from every `.wav` under `sounds/`, classifying each into
  bass / noise / fill **by spectral analysis, not folder name**. Its drum kits
  (`drums.py`) look for categorized hits under `sounds/drums/<category>/`; a
  missing category is simply dropped from the kit.
- **The `labs/` chain** reads its raw source textures from here too
  (`process_m_onz.py` slices + processes them into one-shot banks).

## Overrides

| variable | meaning | default |
|---|---|---|
| `AUDIO_RESEARCH_SOUNDS` | source corpus root | `./sounds` |
| `AUDIO_RESEARCH_SOUNDS_CAP` | max files scanned (0 = no cap) | 4000 (pool), 1500 (chop) |
| `AUDIO_RESEARCH_DRUMS` | drum-kit library root | `./sounds/drums` |

```bash
# point at a corpus living elsewhere, no copying
AUDIO_RESEARCH_SOUNDS=~/samples/breaks python3 -m riff_lab gen --n 5 --gens 50
```

Nothing you put here is committed (see `.gitignore`) — redistribute only
sounds you have the rights to.
