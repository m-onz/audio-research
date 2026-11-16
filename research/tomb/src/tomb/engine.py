"""engine: the MCC loop — sounds and ears coevolve under minimal criteria.

    sound MC:  at least one ear distinguishes it from all living sounds (τ)
    ear MC:    discriminates a pair that no other living ear discriminates

Everything else is drift. No fitness, no behavior characterization, no
novelty archive — novelty is in-place: distinct w.r.t. the current
population through at least one current way of hearing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .ears import EarQueue, ear_from_sounds, features_vector, random_ear
from .ops import random_op_recipe, SR
from .sounds import SoundGenome, SoundQueue

__all__ = ["Tomb", "TombConfig"]


class TombConfig:
    def __init__(self, seed=0, tau=0.15, sound_capacity=250, ear_capacity=24,
                 sound_batch=40, ear_batch=4, seconds=2.0, genesis_batches=50,
                 genesis_ops_per_sound=2, checkpoint_every=200, ear_min_pairs=1):
        self.seed = seed
        self.tau = float(tau)                  # strictness dial (Soros)
        self.sound_capacity = sound_capacity
        self.ear_capacity = ear_capacity
        self.sound_batch = sound_batch
        self.ear_batch = ear_batch
        self.seconds = seconds
        self.genesis_batches = genesis_batches
        self.genesis_ops_per_sound = genesis_ops_per_sound
        self.checkpoint_every = checkpoint_every
        self.ear_min_pairs = ear_min_pairs


class Tomb:
    def __init__(self, config: TombConfig, out_dir: str | Path = "tomb_out"):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.sounds = SoundQueue(config.sound_capacity)
        self.ears = EarQueue(config.ear_capacity)
        self.batch = 0
        self.telemetry: list[dict] = []
        self._desc_cache = None
        self._stats = {"evals": 0, "sounds_admitted": 0, "ears_admitted": 0}

    # ---------------------------------------------------------------- genesis
    def genesis(self, n_founders: int = 30, n_ears: int = 8, verbose: bool = False):
        """Bootstrap: white noise weathered by GENESIS ops (structure-creators)
        until founders exist that ears can discriminate. Mirrors the MCC
        paper's novelty-search bootstrap."""
        if verbose:
            print(f"[tomb] genesis: {n_founders} founders from white noise x "
                  f"{self.cfg.genesis_ops_per_sound} GENESIS ops each")

        # founders: noise + genesis ops
        for _ in range(n_founders):
            noise = self.rng.standard_normal(int(self.cfg.seconds * SR)).astype(np.float32) * 0.5
            gid = self.sounds.new_genome_id()
            ops = []
            for _ in range(self.cfg.genesis_ops_per_sound):
                name, params = random_op_recipe(self.rng, phase="genesis")
                ops.append((name, None, params))
            g = SoundGenome(genome_id=gid, origin="noise", origin_audio=noise,
                            ops=ops, birth_batch=0)
            self.sounds.admit(g)
            self._stats["sounds_admitted"] += 1
            self._desc_cache = None

        # ears: random projections until some discriminate founder pairs
        feat_mat = self._feature_matrix()
        attempts = 0
        while self.ears.size < n_ears and attempts < 200:
            attempts += 1
            if self.ears.size > 0 and self.rng.random() < 0.4:
                e = ear_from_sounds(self.rng, feat_mat, m=4)
            else:
                e = random_ear(self.rng, m=4)
            e.ear_id = self.ears.new_ear_id()
            e.birth_batch = 0
            if self._ear_unique_contribution(e):
                self.ears.admit(e)
                self._stats["ears_admitted"] += 1
        if verbose:
            print(f"[tomb] genesis: {self.sounds.size} sounds, {self.ears.size} ears")

    # ---------------------------------------------------------------- evaluation
    def _feature_matrix(self) -> np.ndarray:
        audio_map = {g.genome_id: g.materialize({}) for g in self.sounds.all_genomes()}
        return np.stack([features_vector(audio_map[g.genome_id])
                         for g in self.sounds.all_genomes()])

    def _descriptors(self, ear, audio_map) -> np.ndarray:
        return np.stack([ear.hear(audio_map[g.genome_id])
                         for g in self.sounds.all_genomes()])

    def _sound_mc(self, candidate_audio, population_audio_map) -> bool:
        """MCNS: viability gate + novelty MC.
        
        Viability (must pass ALL — the 'solve the maze' of audio):
            flatness < 0.6   (has tonal structure, not noise)
            centroid < 5000  (has spectral weight below the shriek range)
            rms in [0.02, 0.5]  (audible but not deafening)
            temporal_cv in [0.05, 5.0]  (has movement, not static or chaotic)
        
        Novelty: at least one ear distinguishes it from all living sounds.
        """
        if not self._viability(candidate_audio):
            return False
        if self._desc_cache is None:
            self._desc_cache = {}
            for ear in self.ears.all_ears():
                for g in self.sounds.all_genomes():
                    self._desc_cache[(ear.ear_id, g.genome_id)] = ear.hear(
                        population_audio_map[g.genome_id])
        for ear in self.ears.all_ears():
            d_cand = ear.hear(candidate_audio)
            dists = []
            for g in self.sounds.all_genomes():
                d_other = self._desc_cache.get((ear.ear_id, g.genome_id))
                if d_other is None:
                    d_other = ear.hear(population_audio_map[g.genome_id])
                    self._desc_cache[(ear.ear_id, g.genome_id)] = d_other
                dists.append(np.linalg.norm(d_cand - d_other))
            if dists and min(dists) > self.cfg.tau:
                return True
        return False

    def _viability(self, audio: np.ndarray) -> bool:
        """Listenability gates: the functional MC (MCNS viability)."""
        x = audio.astype(np.float64)
        n = len(x)
        if n < 512:
            return False
        rms = float(np.sqrt(np.mean(x * x)))
        if not (0.02 <= rms <= 0.5):
            return False

        seg = x[: min(8192, n)]
        nfft = len(seg)
        S = np.abs(np.fft.rfft(seg * np.hanning(nfft))) + 1e-9
        flat = float(np.exp(np.mean(np.log(S))) / np.mean(S))
        if flat > 0.6:
            return False  # still too noise-like

        fr = np.fft.rfftfreq(nfft, 1.0 / SR)
        centroid = float((fr * S).sum() / (S.sum() + 1e-9))
        if centroid > 5000:
            return False  # too bright / thin

        win = max(8, n // 100)
        env = np.array([np.sqrt(np.mean(x[i:i + win] ** 2))
                        for i in range(0, n - win, win)])
        tcv = float(np.std(env) / (np.mean(env) + 1e-9))
        if not (0.05 <= tcv <= 5.0):
            return False  # static or chaotic

        return True

    def _ear_unique_contribution(self, candidate_ear) -> bool:
        """The candidate discriminates a pair that NO other ear discriminates."""
        audio_map = {g.genome_id: g.materialize({}) for g in self.sounds.all_genomes()}
        if len(audio_map) < 2:
            return True  # trivially unique with < 2 sounds

        # candidate's pairwise distance matrix
        descriptors = [candidate_ear.hear(a) for a in audio_map.values()]
        cand_dists = []
        ids = list(audio_map.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cand_dists.append(np.linalg.norm(descriptors[i] - descriptors[j]))
        if not cand_dists:
            return False

        cand_sep = np.array(cand_dists) > self.cfg.tau
        if not cand_sep.any():
            return False  # can't discriminate anything

        # check no existing ear discriminates the SAME pair
        for ear in self.ears.all_ears():
            d = [ear.hear(audio_map[i]) for i in ids]
            ear_dists = []
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    ear_dists.append(np.linalg.norm(d[i] - d[j]))
            ear_sep = np.array(ear_dists) > self.cfg.tau
            overlap = (cand_sep & ear_sep).any()
            if overlap:
                return False  # someone already discriminates what we would
        return True

    # ---------------------------------------------------------------- the loop
    def step(self, verbose: bool = False) -> dict:
        """One MCC batch: alternate sound and ear generations."""
        self.batch += 1
        stats = {"batch": self.batch}

        # --- sound generation ---
        parents = self.sounds.dequeue_batch(min(self.cfg.sound_batch, self.sounds.size))
        audio_map = {g.genome_id: g.materialize({}) for g in self.sounds.all_genomes()}

        phase = "main"
        if self.batch <= self.cfg.genesis_batches:
            phase = "genesis"

        admitted = 0
        for parent in parents:
            name, params = random_op_recipe(self.rng, phase=phase)
            member_id = None
            if name in ("morph", "convmate", "histogram", "layer") and self.sounds.size > 1:
                member_id = self.rng.choice([g.genome_id for g in self.sounds.all_genomes()
                                              if g.genome_id != parent.genome_id])
            gid = self.sounds.new_genome_id()
            child = SoundGenome(
                genome_id=gid, origin=parent.origin,
                origin_audio=parent.origin_audio,
                ops=parent.ops + [(name, member_id, params)],
                birth_batch=self.batch, parent_id=parent.genome_id,
            )
            self._stats["evals"] += 1
            child_audio = child.materialize(audio_map)
            if self._sound_mc(child_audio, audio_map):
                self.sounds.admit(child)
                audio_map[gid] = child_audio
                self._desc_cache = None  # population changed
                admitted += 1
        self.sounds.requeue(parents)
        self._stats["sounds_admitted"] += admitted
        stats["sounds_admitted"] = admitted
        stats["sound_admit_rate"] = admitted / max(1, len(parents))

        # --- ear generation ---
        if self.batch % 10 == 0:  # ears evolve slower than sounds
            ear_parents = self.ears.dequeue_batch(self.cfg.ear_batch)
            for parent_ear in ear_parents:
                if self.rng.random() < 0.5:
                    child = random_ear(self.rng, m=4)
                else:
                    feat = self._feature_matrix()
                    child = ear_from_sounds(self.rng, feat, m=4)
                child.ear_id = self.ears.new_ear_id()
                child.birth_batch = self.batch
                child.parent_id = parent_ear.ear_id
                if self._ear_unique_contribution(child):
                    self.ears.admit(child)
                    self._stats["ears_admitted"] += 1
            self.ears.requeue(ear_parents)

        # --- telemetry ---
        species = self.sounds.species_counts
        complexities = [g.complexity for g in self.sounds.all_genomes()]
        stats.update({
            "n_sounds": self.sounds.size,
            "n_ears": self.ears.size,
            "species": species,
            "complexity_mean": float(np.mean(complexities)) if complexities else 0,
            "complexity_max": int(max(complexities)) if complexities else 0,
            "evals": self._stats["evals"],
        })
        self.telemetry.append(stats)

        if verbose and self.batch % 10 == 0:
            print(f"[tomb] batch {self.batch}: {stats['n_sounds']} sounds "
                  f"({stats['sound_admit_rate']:.0%} admitted), {stats['n_ears']} ears, "
                  f"complexity {stats['complexity_mean']:.1f}/{stats['complexity_max']}, "
                  f"species {stats['species']}")

        if self.batch % self.cfg.checkpoint_every == 0:
            self.checkpoint()
        return stats

    # ---------------------------------------------------------------- checkpoint
    def checkpoint(self) -> Path:
        """Render the current population as wavs + a manifest."""
        ckpt_dir = self.out / f"checkpoint_{self.batch:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        import subprocess

        for g in self.sounds.all_genomes():
            audio = g.materialize({})
            path = ckpt_dir / f"sound_{g.genome_id:04d}_{g.species}_c{g.complexity}.wav"
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le",
                 "-ar", str(SR), "-ac", "1", "-i", "pipe:0", str(path)],
                input=pcm.tobytes(), check=True,
            )
        manifest = {
            "batch": self.batch,
            "n_sounds": self.sounds.size,
            "n_ears": self.ears.size,
            "species": self.sounds.species_counts,
            "complexity_range": [min(g.complexity for g in self.sounds.all_genomes()),
                                 max(g.complexity for g in self.sounds.all_genomes())],
            "evals": self._stats["evals"],
            "sounds_admitted": self._stats["sounds_admitted"],
        }
        (ckpt_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return ckpt_dir

    def run(self, batches: int = 1000, verbose: bool = True) -> dict:
        t0 = time.time()
        for _ in range(batches):
            self.step(verbose=verbose)
        elapsed = time.time() - t0
        final = self.checkpoint()
        if verbose:
            print(f"[tomb] done: {batches} batches in {elapsed:.0f}s, "
                  f"checkpoint at {final}")
        return {"batches": batches, "elapsed": elapsed, "checkpoint": str(final)}