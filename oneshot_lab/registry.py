"""Registry: every transform, its family, default params, and a random
parameter generator. The engine drives everything from here — extend the
library by adding one entry here."""
import numpy as np

from . import glitch, harmonics, novel, pitchfx, spectral, timestretch, warp

# entry = (function, family, defaults dict, needs_bytes, blurb, rand(rng)->params)
TRANSFORMS = {
    # ------------------------------------------------ timestretch
    "paulstretch": (
        timestretch.paulstretch, "timestretch", dict(stretch=8.0, window_s=0.15),
        False, "Paul's Extreme Sound Stretch — extreme ambient smear",
        lambda r: dict(stretch=float(r.uniform(4, 25)),
                      window_s=float(r.uniform(0.08, 0.5)),
                      blur=float(r.uniform(0, 0.7)),
                      glide=float(r.uniform(0.2, 1.0)),
                      phase_coherence=float(r.uniform(0, 0.85))),
    ),
    "paulstretch_hyper": (
        timestretch.paulstretch, "timestretch",
        dict(stretch=40.0, window_s=0.6, blur=0.4, phase_coherence=0.5),
        False, "Hyper paulstretch — 0.6s windows, blurred, coherent drones",
        lambda r: dict(stretch=float(r.uniform(20, 90)),
                      window_s=float(r.uniform(0.4, 1.2)),
                      blur=float(r.uniform(0.3, 0.8)),
                      glide=float(r.uniform(0.3, 1.0)),
                      phase_coherence=float(r.uniform(0.3, 0.9))),
    ),
    "pv_stretch": (
        timestretch.pv_stretch, "timestretch", dict(factor=4.0),
        False, "Phase vocoder stretch — pitch holds, harmonics bloom",
        lambda r: dict(factor=float(r.uniform(2, 7))),
    ),
    "granular_stretch": (
        timestretch.granular_stretch, "timestretch", dict(factor=3.0),
        False, "Granular stretch — pitch holds, jittering grain smear",
        lambda r: dict(factor=float(r.uniform(1.5, 8)),
                       grain_ms=float(r.uniform(30, 140)),
                       jitter=float(r.uniform(0.1, 0.8)),
                       reverse_p=float(r.uniform(0, 0.4)),
                       pitch_p=float(r.uniform(0, 0.3))),
    ),
    "wavetable_stretch": (
        timestretch.wavetable_stretch, "timestretch", dict(scan_per_s=8.0),
        False, "Time-stretch wavetable — the sound scans through itself as "
               "single-cycle wavetables",
        lambda r: dict(scan_per_s=float(r.uniform(2, 20)),
                      scan_mode=str(r.choice(["forward", "pingpong", "random"])),
                      vib_cents=float(r.uniform(0, 20))),
    ),
    "oscillating_timestretch": (
        timestretch.oscillating_timestretch, "timestretch",
        dict(depth=0.25, rate_hz=0.6),
        False, "Oscillating time stretch — phasor-LFO vari-speed, tape wobble",
        lambda r: dict(depth=float(r.uniform(0.1, 0.5)),
                       rate_hz=float(r.uniform(0.2, 4.0)),
                       dur_mult=float(r.choice([1.0, 2.0, 3.0]))),
    ),
    "tape_stop": (
        timestretch.tape_stop, "timestretch", dict(floor=0.02),
        False, "Tape stop / vinyl brake",
        lambda r: dict(floor=float(r.uniform(0.01, 0.08)),
                       hold=float(r.uniform(0.1, 0.6)),
                       shape=float(r.uniform(1.5, 4))),
    ),
    "freeze_loop_stretch": (
        timestretch.freeze_loop_stretch, "timestretch", dict(factor=6.0),
        False, "Freeze lock-groove — the loudest window machine-loops, "
               "micro-jittered, releasing into the real tail",
        lambda r: dict(factor=float(r.uniform(2, 10))),
    ),
    # ------------------------------------------------ warp (stretch modulation)
    "stretch_modulate": (
        warp.stretch_modulate, "warp", dict(curve="ramp", max_stretch=10.0,
                                            voices=3, rate_hz=0.15),
        False, "Pitch-preserving stretch with a modulated stretch factor + "
               "drifting voices (Ableton-style tempo automation shimmer)",
        lambda r: dict(curve=str(r.choice(["ramp", "brake", "wave", "breathe"])),
                      max_stretch=float(r.uniform(3, 30)),
                      curve_shape=float(r.uniform(0.8, 2.5)),
                      rate_hz=float(r.uniform(0.05, 1.5)),
                      depth=float(r.uniform(0.1, 0.45)),
                      voices=int(r.integers(1, 6)),
                      voice_drift=float(r.uniform(0.002, 0.02)),
                      grain_ms=float(r.uniform(25, 150)),
                      jitter_ms=float(r.uniform(0, 8)),
                      out_s=float(r.uniform(4, 10))),
    ),
    # ------------------------------------------------ spectral
    "spectral_freeze": (
        spectral.spectral_freeze, "spectral", dict(at=0.35, dur_s=6.0),
        False, "Spectral freeze — frozen ghost drone following the envelope",
        lambda r: dict(at=float(r.uniform(0.1, 0.7)),
                       dur_s=float(r.uniform(3, 10)),
                       shimmer=float(r.uniform(0.05, 0.35)),
                       detune=float(r.uniform(0, 0.01))),
    ),
    "spectral_smear": (
        spectral.spectral_smear, "spectral", dict(blur_s=0.35, mix=0.75),
        False, "Temporal spectral blur — transients melt into ghosts",
        lambda r: dict(blur_s=float(r.uniform(0.1, 0.9)),
                       mix=float(r.uniform(0.4, 1.0))),
    ),
    "spectral_shuffle": (
        spectral.spectral_shuffle, "spectral", dict(block_ms=120.0),
        False, "Spectral glitch — STFT time blocks permuted, some reversed",
        lambda r: dict(block_ms=float(r.uniform(40, 300)),
                       reverse_p=float(r.uniform(0.1, 0.6)),
                       mag_only=bool(r.random() < 0.5)),
    ),
    "spectral_reverse": (
        spectral.spectral_reverse, "spectral", dict(),
        False, "Envelope backwards, waveform forwards — possession reverse",
        lambda r: dict(bin_flip_p=float(r.uniform(0, 1))),
    ),
    "spectral_binroll": (
        spectral.spectral_binroll, "spectral", dict(shift_bins=37),
        False, "Frequency-axis rotation — comb-metallic alien recast",
        lambda r: dict(shift_bins=int(r.integers(-80, 80)),
                       drift_bins=int(r.integers(0, 25))),
    ),
    "spectral_hybrid": (
        spectral.spectral_hybrid, "spectral", dict(ref="reverse", mix=0.6),
        False, "Spectral convolution with a self variant — cross-synthesis",
        lambda r: dict(ref=str(r.choice(["reverse", "shuffle", "roll"])),
                       mix=float(r.uniform(0.4, 1.0))),
    ),
    "spectral_nmf": (
        spectral.spectral_nmf, "spectral", dict(k=24, mix=1.0),
        False, "NMF self-stamp — rebuilt from its own learned dictionary",
        lambda r: dict(k=int(r.integers(8, 48)), mix=float(r.uniform(0.5, 1.0))),
    ),
    "spectral_morph": (
        spectral.spectral_morph, "spectral", dict(target="reverse", cycles=2.0),
        False, "Magnitude morph A->B->A with held phases — the crossover possession",
        lambda r: dict(target=str(r.choice(["reverse", "roll"])),
                       cycles=float(r.uniform(1, 4))),
    ),
    "lpc_reanimate": (
        spectral.lpc_reanimate, "spectral", dict(excitation="pulse"),
        False, "LPC reanimation — the hit re-speaks through its own formants",
        lambda r: dict(excitation=str(r.choice(["pulse", "noise", "self"])),
                       dur_s=float(r.uniform(2, 8))),
    ),
    "comb_bank": (
        spectral.comb_bank, "spectral", dict(n_combs=6, feedback=0.72),
        False, "Tuned feedback comb bank at f0 harmonics — metallic ring wash",
        lambda r: dict(n_combs=int(r.integers(3, 9)),
                       feedback=float(r.uniform(0.5, 0.9)),
                       ratios=None if r.random() < 0.7 else
                       [1, 1.41, 2, 2.33, 3, 4.05, 5.07, 6.5][:int(r.integers(3, 8))]),
    ),
    # ------------------------------------------------ novel (partial-level, analysis-driven)
    "spectral_dna": (
        novel.spectral_dna, "novel", dict(cross=0.8, residual_mix=0.45),
        False, "Partial-level crossbreeding: A's peak structure + B's amplitudes",
        lambda r: dict(cross=float(r.uniform(0.5, 1.0)),
                      residual_mix=float(r.uniform(0.25, 0.7)),
                      tol_cents=float(r.uniform(30, 120))),
    ),
    "partial_resequence": (
        novel.partial_resequence, "novel", dict(),
        False, "Each partial's amplitude trajectory independently re-sequenced",
        lambda r: dict(),
    ),
    "env_splice": (
        novel.env_splice, "novel", dict(graft=0.45),
        False, "A's spectral body wearing B's envelope (+ stolen attack graft)",
        lambda r: dict(graft=float(r.uniform(0, 0.8))),
    ),
    "formant_walk": (
        novel.formant_walk, "novel", dict(),
        False, "Spectral envelope glides into B's formant field (possession)",
        lambda r: dict(one_way=bool(r.random() < 0.4),
                      hold_end=float(r.uniform(0.2, 0.5))),
    ),
    "harmonic_grid": (
        novel.harmonic_grid, "novel", dict(),
        False, "Re-tune f0 to a scale, shear odd/even partials, octave shadow + sub",
        lambda r: dict(scale=str(r.choice(sorted(novel.SCALES))),
                      root=str(r.choice(sorted(novel.ROOTS))),
                      shear=float(r.uniform(0.15, 0.5)),
                      octave_shadow=float(r.uniform(0.15, 0.5))),
    ),
    # ------------------------------------------------ glitch
    "databend": (
        glitch.databend, "glitch", dict(corrupt_p=0.6), True,
        "Raw-byte databend — the wav file's own bytes, sonified and corrupted",
        lambda r: dict(corrupt_p=float(r.uniform(0.2, 1.0))),
    ),
    "crush": (
        glitch.crush, "glitch", dict(),
        False, "Bitcrush + drunk sample-hold decimation",
        lambda r: dict(),
    ),
    "ringmod": (
        glitch.ringmod, "glitch", dict(),
        False, "Ring mod / AM with drifting metallic carrier",
        lambda r: dict(side=str(r.choice(["both", "both", "upper", "lower"]))),
    ),
    "am_gate": (
        glitch.am_gate, "glitch", dict(),
        False, "Square/sine AM gate — machine-gun tremolo",
        lambda r: dict(),
    ),
    "waveshape": (
        glitch.waveshape, "glitch", dict(),
        False, "Waveshaping — tanh / fold / cheby / asymmetric power",
        lambda r: dict(),
    ),
    "stutter": (
        glitch.stutter, "glitch", dict(),
        False, "Machine-gun loop-lock with pitch ladders and drops",
        lambda r: dict(),
    ),
    "granular_scatter": (
        glitch.granular_scatter, "glitch", dict(),
        False, "Micro-grain explosion — per-grain reverse/crush/decimate/ringmod",
        lambda r: dict(count=int(r.integers(40, 160))),
    ),
    "echo_replicate": (
        glitch.echo_replicate, "glitch", dict(),
        False, "Echo loops at irrational multiples — grid-orthogonal glitch",
        lambda r: dict(),
    ),
    "ca_chop": (
        glitch.ca_chop, "glitch", dict(),
        False, "Cellular-automaton gated chop with random hit processing",
        lambda r: dict(wild=float(r.uniform(0.3, 1.0))),
    ),
    "samplehold_wobble": (
        glitch.samplehold_wobble, "glitch", dict(),
        False, "Drunk sample-and-hold skitter — broken-tape wobble",
        lambda r: dict(base_hz=float(r.uniform(3, 12)),
                       depth=float(r.uniform(0.3, 0.8))),
    ),
    "reverse": (
        glitch.reverse, "glitch", dict(),
        False, "Plain reverse", lambda r: dict(),
    ),
    # ------------------------------------------------ harmonics
    "octaver": (
        harmonics.octaver, "harmonics", dict(),
        False, "Interval stack — detuned panned octaves/fifths",
        lambda r: dict(intervals=list(r.choice([
            (-12, 12), (-12, -5, 12), (-12, 7, 19), (12, 19, 24), (-19, -12, 12)])),
            gains=tuple(float(r.uniform(0.2, 0.5)) for _ in range(3))),
    ),
    "detune_chorus": (
        harmonics.detune_chorus, "harmonics", dict(voices=3),
        False, "Wide drifting detune chorus — the hit blooms into a swarm",
        lambda r: dict(voices=int(r.integers(2, 5)),
                       detune_cents=float(r.uniform(4, 20))),
    ),
    "shimmer_harmonizer": (
        harmonics.shimmer_harmonizer, "harmonics", dict(),
        False, "Pitched delay-tap feedback cascade — harmonizer shimmer",
        lambda r: dict(fb=float(r.uniform(0.6, 0.88)), iters=int(r.integers(6, 16))),
    ),
    "reich_phase": (
        harmonics.reich_phase, "harmonics", dict(),
        False, "Loops at mismatched speeds drifting in and out of sync",
        lambda r: dict(),
    ),
    "resonator": (
        harmonics.resonator, "harmonics", dict(),
        False, "Re-ring the hit's own measured harmonic series",
        lambda r: dict(n_partials=int(r.integers(5, 12)),
                       decay=float(r.uniform(0.8, 4.0))),
    ),
    "sub_octave": (
        harmonics.sub_octave, "harmonics", dict(),
        False, "Tuned sub sine following the hit's envelope",
        lambda r: dict(div=int(r.choice([1, 2, 2, 4])), gain=float(r.uniform(0.4, 0.9))),
    ),
    # ------------------------------------------------ pitch
    "pitch_shift_pv": (
        pitchfx.pitch_shift_pv, "pitch", dict(),
        False, "Pitch shift, duration preserved (phase vocoder)",
        lambda r: dict(),
    ),
    "pitch_shift_tape": (
        pitchfx.pitch_shift_tape, "pitch", dict(),
        False, "Tape pitch shift — pitch and time bend together",
        lambda r: dict(),
    ),
    "glissando": (
        pitchfx.glissando, "pitch", dict(),
        False, "Gliding pitch sweep through the hit",
        lambda r: dict(),
    ),
    "vibrato": (
        pitchfx.vibrato, "pitch", dict(),
        False, "Oscillating pitch warble",
        lambda r: dict(rate_hz=float(r.uniform(2, 10)),
                       depth_cents=float(r.uniform(15, 70))),
    ),
}

FAMILIES = {}
for name, (fn, fam, defaults, needs_bytes, blurb, rand) in TRANSFORMS.items():
    FAMILIES.setdefault(fam, []).append(name)
    TRANSFORMS[name] = (fn, fam, defaults, needs_bytes, blurb, rand)


def resolve(names, families=()):
    """Resolve a list of transform names and/or family names to names."""
    out = []
    for n in names:
        if n in TRANSFORMS:
            out.append(n)
        elif n in FAMILIES:
            out.extend(FAMILIES[n])
        else:
            raise KeyError(f"unknown transform or family: {n} "
                           f"(known families: {sorted(FAMILIES)})")
    seen, dedup = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            dedup.append(n)
    return dedup