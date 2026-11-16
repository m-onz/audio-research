"""ear: the evolvable measurement — cryptid's second population.

An ear decides *which slice of a process's state becomes the sound*.
It never generates energy: it selects, gates, differentiates, projects,
and packs. The same process heard through two different ears is two
different species (MCC's maze/solver pairing, made sonic).

Ear operators (measurement, never generation):
    select       which state variables to observe
    gain          per-variable amplification
    diff          differentiate (hear the change, not the level)
    gate          threshold + zero below
    project       random low-D projection of the selected states
    pack          bitwise pack selected states (the aliasing aesthetic)
    stereo_split  L/R from two projections
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

__all__ = ["EarGenome", "random_ear", "mutate_ear", "HEAR_STEPS"]


@dataclass
class EarGenome:
    ear_id: int = 0
    select: np.ndarray = None       # (k,) indices of state vars to observe
    gains: np.ndarray = None         # (k,) per-variable gains
    diff: np.ndarray = None         # (k,) bool: differentiate this var?
    gate_thresh: float = 0.0        # gate threshold
    project: np.ndarray = None      # (2, k) stereo projection matrix
    birth_gen: int = 0
    parent_id: int | None = None

    def hear(self, state_matrix: np.ndarray, sr: int) -> np.ndarray:
        """Process state matrix (n_steps, total_vars) -> stereo audio (n, 2).

        This is the MEASUREMENT — no energy is added, only selected,
        gated, differentiated, projected, and safety-limited.
        """
        n_steps = state_matrix.shape[0]
        total_vars = state_matrix.shape[1]
        sel = np.clip(self.select, 0, total_vars - 1)  # clip to valid range
        selected = state_matrix[:, sel]  # (n_steps, k)
        selected = np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)

        # differentiate selected vars
        for j in range(len(self.diff)):
            if self.diff[j]:
                selected[:, j] = np.diff(selected[:, j], prepend=selected[0, j])

        # gain
        selected = selected * self.gains[None, :]

        # gate: zero below threshold
        if self.gate_thresh > 0:
            loud = np.abs(selected).max(axis=1) > self.gate_thresh
            selected = selected * loud[:, None]

        # stereo projection
        L = selected @ self.project[0]  # (n_steps,)
        R = selected @ self.project[1]

        # safety limiter (the only designed stage)
        peak = max(float(np.abs(L).max()), float(np.abs(R).max()))
        if peak > 1.0:
            L, R = L / peak, R / peak

        stereo = np.stack([L, R], axis=1).astype(np.float32)
        return stereo


HEAR_STEPS = 512  # how many integration steps per hear() call


def random_ear(ear_id: int, rng: np.random.Generator, total_vars: int,
               k: int = None) -> EarGenome:
    """A random measurement: k of the total_vars, random projection."""
    k = k or max(2, min(total_vars, int(rng.integers(2, min(total_vars + 1, 10)))))
    k = min(k, total_vars)
    select = rng.choice(total_vars, size=k, replace=False)
    gains = rng.uniform(0.1, 3.0, k)
    diff = rng.random(k) < 0.3
    gate = float(rng.uniform(0.0, 0.3))
    # orthogonal-ish stereo projection
    proj = rng.normal(0, 1, (2, k))
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    proj = proj / (norms + 1e-9)
    return EarGenome(ear_id=ear_id, select=select, gains=gains, diff=diff,
                     gate_thresh=gate, project=proj)


def mutate_ear(e: EarGenome, rng: np.random.Generator, rate: float = 0.2) -> EarGenome:
    """Mutate: jitter gains, flip diff flags, move the gate, re-roll the projection."""
    child = copy.deepcopy(e)
    child.parent_id = e.ear_id
    child.ear_id = 0

    n_mut = 1 + int(rng.random() < 0.3)
    for _ in range(n_mut):
        op = rng.choice(["gain", "diff", "gate", "proj", "select"], p=[0.3, 0.15, 0.15, 0.25, 0.15])

        if op == "gain":
            child.gains = child.gains * (10 ** rng.normal(0, rate, len(child.gains)))
            child.gains = np.clip(child.gains, 0.01, 20.0)

        elif op == "diff":
            i = int(rng.integers(len(child.diff)))
            child.diff[i] = not child.diff[i]

        elif op == "gate":
            child.gate_thresh = float(np.clip(
                child.gate_thresh * (10 ** rng.normal(0, rate)), 0.0, 0.8))

        elif op == "proj":
            child.project = rng.normal(0, 1, child.project.shape)
            norms = np.linalg.norm(child.project, axis=1, keepdims=True)
            child.project = child.project / (norms + 1e-9)

        elif op == "select" and len(child.select) > 2:
            # swap one selected index for a new one (stays in valid range)
            i = int(rng.integers(len(child.select)))
            child.select[i] = int(rng.integers(max(int(child.select.max()) + 1, 4)))

    return child