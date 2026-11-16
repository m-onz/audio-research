# cryptid

**Breeding evolved dynamical processes — the sound IS the state.**

No oscillators, no notes, no filters, no envelopes — just laws, coupling,
and measurement. The genome is a growable network of heterogeneous
dynamical units; the ear is an evolvable measurement that decides which
slice of the state you hear.

```
    process genome  --(states)-->  ear genome  --(measurement)--> wav
          ^        mutual minimal criterion (MCC)         ^
          |            every process needs an ear               |
          +------------ every ear needs a process ---------------+
```

```bash
cryptid seed --gens 50 --strictness 0.5 -o runs/crypt1
cryptid info runs/crypt1/gen_0050/
```

## The laws (dynamical unit zoo)

| law | behavior | sonic character |
|---|---|---|
| `jerk` | x‴ = a·x″ + b·x′ + c·x + d·x³ (Sprott chaos) | inharmonic morphing voices |
| `excitable` | FitzHugh–Nagumo: rest/spike/refractory | arrhythmic pulse trains |
| `fire` | integrate-and-fire with evolved threshold | spiking rhythms |
| `osc` | Kuramoto phase unit | chimeras: simultaneous tonality + noise |
| `sand` | BTW sandpile: SOC avalanches | 1/f event weather |
| `map` | logistic/tent map at evolved rates | sub-audio LFOs, aliased audio-rate |
| `bool` | Kauffman NK Boolean regulatory | non-metric cyclic gates |
| `delay` | state feedback with evolved lag | echo-chaos, long-term memory |
| `clamp` | diode-like nonlinear quantizer | aliasing as component |
| `bytefeed` | any file's bytes as state stream | databending species |

## Mutations

param drift (log-scale) · law swap · node birth/death · edge add/remove/rewire ·
**subgraph duplication** (copies diverge — the major-transition hook) ·
homeostatic params (RAAHN: params drift online to keep own state alive)

## The ear (evolvable measurement, never generation)

select which state variables · gain · differentiate (hear the change) ·
gate (threshold) · stereo projection · safety limiter (the only designed stage)

## MCC pairing

- **Process MC**: some ear hears it alive (audible, finite, has variance)
- **Ear MC**: it hears some process alive

## The strictness dial (Soros)

`--strictness` controls the MC band width: 0.1 = stagnant (too strict),
1.0 = chaotic (too loose). 0.5 is the orderly divergence sweet spot.

## Why this matters

Traditional synthesis — additive, subtractive, FM, wavetable, granular,
physical modeling — shares one grammar: *a designed oscillation, then
designed shaping*. Everything it can produce is bounded by what a designer
can specify. Cryptid removes the designer: **breed processes, measure them,
let the beholder keep the strange ones.** Sprott found his chaotic flows
by blind search over coefficients; cryptid generalizes that move to
networks of laws.

The results are sounds traditional synthesis cannot reach:
- simultaneous tonality + noise from coupling (chimeras)
- inharmonic timbre that morphs by attractor geometry
- rhythms with cyclic structure but no metric grid
- glitch discontinuities timed by *state transitions*, not clocks
- multi-timescale nonstationarity (breath-like drift)

## Layout

```
src/cryptid/
  process.py   # the 10 dynamical unit laws
  genome.py    # growable networks + mutations
  ear.py       # evolvable measurement
  engine.py    # the MCC loop: genesis, viability, checkpoints
  cli.py       # cryptid seed | render | info | laws
```

## License

MIT — see LICENSE.