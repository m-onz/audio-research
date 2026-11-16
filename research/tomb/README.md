# tomb

**In-place audio novelty search — only the discriminable survive.**

Sounds and ears coevolve under minimal criteria (MCC, GECCO '17):
a **sound** lives only if some ear distinguishes it from every other
living sound; an **ear** lives only if it discriminates a pair no other
ear discriminates. No fitness, no behavior archive, no global novelty
metric — just drift under the MC.

```
white noise → GENESIS ops (structure-creators) → founders
         ↓
    MCC loop:  sounds mutate (material processes) ↔ ears drift (feature projections)
         ↓         population = its own corpus (self-referential)
    checkpoints:  rendered wavs, species-tagged, lineage in filename
         ↓         you listen, you pick (the subjective metric)
```

```bash
tomb run --batches 500 --tau 0.15 --seed 0    # run the coevolution
tomb info tomb_out/checkpoint_000200/          # inspect a checkpoint
tomb ops                                       # list the operator families
```

## The τ dial (Soros)

`--tau` is the strictness of the MC (the distance threshold for
"distinguished"). Higher = harder to admit → stagnation. Lower = easier →
degeneracy. The sweet spot produces orderly divergence.

## Operator families (sound as matter, never played)

| family | ops | role |
|---|---|---|
| **genesis** | resonate, comb, decimate, logistic, gate, wavesort | structure-creators — break the noise fixed point |
| **field** | diffuse, propagate, react, automate | spectrogram physics (womb's organs as mutations) |
| **stat** | morph, convmate, rephase, histogram | population-as-filterbank (self-reference) |
| **order** | wavesort | sorting = symmetry breaking |
| **loss** | mp3_round, resample_war, crush, bitrot | generational decay (sounds age like tapes) |
| **structure** | splice, reverse, layer, expand | complexification axis (GECCO '19) |

## Speciation

Genetics-based (op-family histogram of the lineage), per the MCC paper —
no behavior characterization. A lineage heavy in `field` ops is a different
species from one heavy in `loss` ops, even if they sound similar. Queue
capacity is split evenly; each species has its own FIFO.

## The success experiment

After a 1-hour run, listen blind to 20 random admitted sounds.
**Success = ≥5 keepers you would actually use in a piece.**
If that fails, iterate the op recipe and ear features — not the MCC skeleton.

## Layout

```
src/tomb/
  ops.py       # 22 material-process operators in 6 families
  sounds.py    # genomes (op lineages), species-wise FIFO queues
  ears.py      # feature projections (loom's 14 features + temporal stats)
  engine.py    # the MCC loop: genesis, MC evaluation, checkpoints, telemetry
  cli.py       # tomb run | info | ops
tests/         # 11 tests: fixed-point break, determinism, full run
```

## Requires

- Python ≥ 3.10, numpy, ffmpeg on PATH
- Fully offline, pure Python, no ML required for v0

## License

MIT — see LICENSE.