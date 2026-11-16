"""engine: the MCC loop — processes and ears coevolve under minimal criteria.

    process MC:  some ear hears it alive (audible + finite + has variance)
    ear MC:      it hears some process alive

No fitness, no behavior characterization, no archive — just drift
under the MC. The population is its own corpus. Complexity can always
rise (the genome is a growable network). Curation is the only quality
pressure (subjectivity paper).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .ear import HEAR_STEPS, EarGenome, mutate_ear, random_ear
from .genome import ProcessGenome, mutate_genome, random_genome
from .process import LAW_NAMES

__all__ = ["Cryptid", "CryptidConfig", "SR"]

SR = 44100


class CryptidConfig:
    def __init__(self, seed=0, strictness=0.5, pop_capacity=40, ear_capacity=12,
                 n_steps=2048, genesis_pop=20, genesis_ears=6,
                 drift_floor=0.01, checkpoint_every=10):
        self.seed = seed
        self.strictness = float(strictness)      # Soros dial
        self.pop_capacity = int(pop_capacity)
        self.ear_capacity = int(ear_capacity)
        self.n_steps = int(n_steps)              # integration steps per eval
        self.genesis_pop = int(genesis_pop)
        self.genesis_ears = int(genesis_ears)
        self.drift_floor = float(drift_floor)    # min perceptual distance from parent
        self.checkpoint_every = int(checkpoint_every)


class Cryptid:
    """The breeding ground. Processes and ears, drifting under the MC."""

    def __init__(self, config: CryptidConfig, out_dir: str | Path = "runs"):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.population: list[ProcessGenome] = []   # viable processes
        self.ears: list[EarGenome] = []              # viable ears
        self.gen = 0
        self._next_genome_id = 0
        self._next_ear_id = 0
        self.telemetry: list[dict] = []

    # ---------------------------------------------------------------- viability
    def _process_alive(self, g: ProcessGenome, ear: EarGenome,
                       audio: np.ndarray) -> bool:
        """The MC: audible, finite, has temporal variance."""
        if audio is None:
            return False
        if audio.shape[0] < 64:
            return False
        # audible
        rms = float(np.sqrt(np.mean(audio ** 2)))
        lo, hi = 0.01, 0.5
        band = (hi - lo) * self.cfg.strictness
        if not (lo <= rms <= lo + band + lo):  # widened by strictness
            if not (lo * 0.5 <= rms <= lo + band * 2 + lo):
                return False
        # has variance (not silence, not dc)
        if float(np.std(audio)) < 0.005:
            return False
        # spectral variance (not dead-flat noise either)
        if audio.shape[0] > 256:
            S = np.abs(np.fft.rfft(audio[:256, 0] * np.hanning(256)))
            if float(np.std(S)) < 0.1:
                return False
        return True

    def _ear_hears_anything(self, ear: EarGenome) -> bool:
        for g in self.population:
            state = g.step(self.cfg.n_steps, self.rng)
            if state is None:
                continue
            audio = ear.hear(state, SR)
            if self._process_alive(g, ear, audio):
                return True
        return False

    # ---------------------------------------------------------------- the loop
    def genesis(self, verbose=False):
        """Bootstrap: random processes + random ears until some pair works."""
        if verbose:
            print(f"[cryptid] genesis: {self.cfg.genesis_pop} processes, "
                  f"{self.cfg.genesis_ears} ears")

        # generate candidate processes
        candidates = []
        for _ in range(self.cfg.genesis_pop * 3):
            g = random_genome(self._new_genome_id(), self.rng,
                              n_nodes=int(self.rng.integers(3, 8)),
                              n_edges=int(self.rng.integers(3, 12)))
            state = g.step(self.cfg.n_steps, self.rng)
            if state is not None:  # survived integration (no NaN)
                candidates.append((g, state))

        # generate candidate ears
        ear_candidates = []
        total_vars = max(g.total_state_vars() for g, _ in candidates) if candidates else 10
        for _ in range(self.cfg.genesis_ears * 3):
            e = random_ear(self._new_ear_id(), self.rng, total_vars)
            ear_candidates.append(e)

        # find pairs that work: at least one ear hears at least one process
        for e in ear_candidates:
            for g, state in candidates:
                audio = e.hear(state, SR)
                if self._process_alive(g, e, audio):
                    g.birth_gen = 0
                    self.population.append(g)
                    e.birth_gen = 0
                    self.ears.append(e)
                    break
            if len(self.ears) >= self.cfg.genesis_ears:
                break

        # also admit processes that no ear hears yet (they may find an ear later)
        heard = {g.genome_id for g in self.population}
        for g, state in candidates:
            if len(self.population) >= self.cfg.pop_capacity:
                break
            if g.genome_id not in heard:
                g.birth_gen = 0
                self.population.append(g)

        if verbose:
            print(f"[cryptid] genesis: {len(self.population)} processes, "
                  f"{len(self.ears)} ears")

    def step(self, verbose=False) -> dict:
        """One MCC generation: mutate processes, mutate ears, evaluate."""
        self.gen += 1
        stats = {"gen": self.gen}

        # --- process generation ---
        if self.population:
            parent = self.population[int(self.rng.integers(len(self.population)))]
            child = mutate_genome(parent, self.rng, rate=0.25)
            child.genome_id = self._new_genome_id()
            child.birth_gen = self.gen

            state = child.step(self.cfg.n_steps, self.rng)
            if state is not None:
                # drift floor: child must differ from parent perceptually
                if self.ears:
                    ear = self.ears[int(self.rng.integers(len(self.ears)))]
                    audio_child = ear.hear(state, SR)
                    parent_state = parent.step(self.cfg.n_steps, self.rng)
                    if parent_state is not None:
                        audio_parent = ear.hear(parent_state, SR)
                        if audio_child is not None and audio_parent is not None:
                            d = float(np.abs(audio_child - audio_parent).mean())
                            if d < self.cfg.drift_floor:
                                stats["process_admitted"] = False
                                stats["reason"] = f"drift {d:.4f} < floor"
                                self.telemetry.append(stats)
                                return stats

                    # try all ears: at least one must hear it alive
                    heard = False
                    for e in self.ears:
                        audio = e.hear(state, SR)
                        if self._process_alive(child, e, audio):
                            heard = True
                            break
                    if heard:
                        self.population.append(child)
                        if len(self.population) > self.cfg.pop_capacity:
                            # remove oldest
                            self.population.pop(0)
                        stats["process_admitted"] = True
                        stats["mutation"] = child.mutation
                    else:
                        stats["process_admitted"] = False
                        stats["reason"] = "no ear hears it"
                else:
                    # no ears yet: admit anything that survives integration
                    self.population.append(child)
                    stats["process_admitted"] = True
            else:
                stats["process_admitted"] = False
                stats["reason"] = "NaN death"

        # --- ear generation (every 3rd gen, slower) ---
        if self.gen % 3 == 0 and self.ears:
            parent_ear = self.ears[int(self.rng.integers(len(self.ears)))]
            child_ear = mutate_ear(parent_ear, self.rng, rate=0.2)
            child_ear.ear_id = self._new_ear_id()
            child_ear.birth_gen = self.gen

            if self._ear_hears_anything(child_ear):
                self.ears.append(child_ear)
                if len(self.ears) > self.cfg.ear_capacity:
                    self.ears.pop(0)
                stats["ear_admitted"] = True
            else:
                stats["ear_admitted"] = False

        # --- telemetry ---
        n_laws = {}
        for g in self.population:
            for l in g.laws:
                n_laws[l] = n_laws.get(l, 0) + 1
        stats.update({
            "pop": len(self.population),
            "ears": len(self.ears),
            "law_histogram": n_laws,
            "mean_nodes": float(np.mean([g.n_nodes for g in self.population]))
                         if self.population else 0,
        })
        self.telemetry.append(stats)

        if verbose and self.gen % 5 == 0:
            print(f"[cryptid] gen {self.gen}: {stats['pop']} processes, "
                  f"{stats['ears']} ears, {stats['mean_nodes']:.1f} nodes mean, "
                  f"laws: {stats['law_histogram']}")

        if self.gen % self.cfg.checkpoint_every == 0:
            self.checkpoint()
        return stats

    def run(self, n_gens: int, verbose=True) -> dict:
        t0 = time.time()
        for _ in range(n_gens):
            self.step(verbose=verbose)
        elapsed = time.time() - t0
        self.checkpoint()
        if verbose:
            print(f"[cryptid] {n_gens} gens in {elapsed:.0f}s, "
                  f"final: {len(self.population)} processes, {len(self.ears)} ears")
        return {"gens": n_gens, "elapsed": elapsed, "checkpoint": str(self.out)}

    # ---------------------------------------------------------------- checkpoint
    def checkpoint(self) -> Path:
        """Render a listening atlas: every viable process through a random ear."""
        ckpt = self.out / f"gen_{self.gen:04d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        import os

        for i, g in enumerate(self.population):
            if not self.ears:
                break
            ear = self.ears[int(self.rng.integers(len(self.ears)))]
            state = g.step(self.cfg.n_steps, self.rng)
            if state is None:
                continue
            audio = ear.hear(state, SR)
            if audio is None or audio.shape[0] < 64:
                continue

            # render: audio + spectrogram frame
            path = ckpt / f"specimen_{i:03d}_{g.mutation or 'seed'}.wav"
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            inter = np.empty(len(audio) * 2, np.int16)
            inter[0::2], inter[1::2] = pcm[:, 0], pcm[:, 1]
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le",
                 "-ar", str(SR), "-ac", "2", "-i", "pipe:0", str(path)],
                input=inter.tobytes(), check=True)

        manifest = {
            "gen": self.gen,
            "n_processes": len(self.population),
            "n_ears": len(self.ears),
            "law_histogram": {l: sum(1 for g in self.population if l in g.laws)
                             for l in LAW_NAMES},
            "mean_nodes": float(np.mean([g.n_nodes for g in self.population]))
                         if self.population else 0,
        }
        (ckpt / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return ckpt

    # ---------------------------------------------------------------- render
    def render_specimen(self, genome: ProcessGenome, ear: EarGenome,
                        seconds: float, out: str) -> str:
        """Render a full piece from one process through one ear."""
        n_total = int(seconds * SR)
        chunks = []
        remaining = n_total
        while remaining > 0:
            n = min(remaining, HEAR_STEPS * 4)
            state = genome.step(n, self.rng)
            if state is None:
                break
            audio = ear.hear(state, SR)
            if audio is None:
                break
            chunks.append(audio)
            remaining -= audio.shape[0]
        if not chunks:
            raise RuntimeError("specimen died during render")
        full = np.concatenate(chunks)[:n_total]
        if len(full) < n_total:
            full = np.pad(full, ((0, n_total - len(full)), (0, 0)))
        # normalize
        peak = float(np.abs(full).max())
        if peak > 0:
            full = full / peak * 0.9
        pcm = (np.clip(full, -1, 1) * 32767).astype(np.int16)
        inter = np.empty(len(full) * 2, np.int16)
        inter[0::2], inter[1::2] = pcm[:, 0], pcm[:, 1]
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le",
             "-ar", str(SR), "-ac", "2", "-i", "pipe:0", str(out)],
            input=inter.tobytes(), check=True)
        return str(out)

    # ---------------------------------------------------------------- helpers
    def _new_genome_id(self) -> int:
        self._next_genome_id += 1
        return self._next_genome_id

    def _new_ear_id(self) -> int:
        self._next_ear_id += 1
        return self._next_ear_id