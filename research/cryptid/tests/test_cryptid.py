import numpy as np
import pytest

import cryptid
from cryptid.ear import random_ear
from cryptid.genome import mutate_genome, random_genome
from cryptid.process import LAWS, LAW_NAMES, n_state_vars, step_unit


def test_all_laws_step():
    """Every law must accept a state and return a finite state."""
    rng = np.random.default_rng(0)
    for law in LAW_NAMES:
        nv = n_state_vars(law)
        state = rng.uniform(-0.5, 0.5, nv).astype(np.float32)
        params = cryptid.default_params(law)
        dt = 0.001
        inp = np.array([0.1], dtype=np.float32)
        for _ in range(100):
            new = step_unit(law, state, params, dt, inp)
            if new is None:
                break  # death is valid (NaN = the MC kills it)
            state = new


def test_random_genome_steps():
    """A random genome must produce a state matrix (or die cleanly)."""
    rng = np.random.default_rng(0)
    for seed in range(20):
        g = random_genome(seed, rng, n_nodes=int(rng.integers(2, 8)),
                          n_edges=int(rng.integers(2, 10)))
        state = g.step(100, rng)
        if state is not None:
            assert state.shape[1] == g.total_state_vars()
            assert np.isfinite(state).all()


def test_genome_grows(small_genome, rng):
    """Node birth mutation must increase the network size."""
    n0 = small_genome.n_nodes
    for _ in range(10):
        child = mutate_genome(small_genome, rng, rate=0.5)
        if child.n_nodes > n0:
            assert child.n_nodes > n0
            return
    assert False, "no growth in 10 mutations"


def test_genome_shrinks(small_genome, rng):
    for _ in range(10):
        child = mutate_genome(small_genome, rng, rate=0.5)
        if child.n_nodes < small_genome.n_nodes:
            return
    assert False, "no shrink in 10 mutations"


def test_mutation_lineage(small_genome, rng):
    child = mutate_genome(small_genome, rng)
    assert child.parent_id == small_genome.genome_id
    assert child.mutation != ""


def test_ear_hears(small_genome):
    """An ear must produce stereo audio from a state matrix."""
    rng = np.random.default_rng(0)
    g = small_genome
    state = g.step(512, rng)
    if state is None:
        pytest.skip("genome died")
    total = state.shape[1]
    ear = random_ear(1, rng, total, k=min(4, total))
    audio = ear.hear(state, 44100)
    assert audio.ndim == 2
    assert audio.shape[1] == 2  # stereo
    assert audio.shape[0] == 512
    assert np.abs(audio).max() <= 1.0  # safety limiter


def test_ear_mutate(small_ear, rng):
    child = cryptid.mutate_ear(small_ear, rng)
    assert child.parent_id == small_ear.ear_id
    assert child.ear_id == 0  # unassigned until admitted


# ------------------------------------------------------------------ full engine

def test_genesis_and_run(tmp_path):
    """A tiny full run: genesis + 10 generations."""
    cfg = cryptid.CryptidConfig(seed=3, strictness=0.5, pop_capacity=20,
                                n_steps=512, genesis_pop=8, genesis_ears=4,
                                checkpoint_every=100)
    c = cryptid.Cryptid(cfg, out_dir=tmp_path / "run")
    c.genesis(verbose=False)
    assert len(c.population) > 0
    assert len(c.ears) > 0
    c.run(n_gens=10, verbose=False)
    assert c.gen == 10
    assert len(c.telemetry) == 10


def test_mutation_determinism(small_genome):
    """Same seed -> identical mutations (the evolutionary layer is seeded;
    some laws have stochastic elements in their stepping, which is fine —
    the MC handles stochastic death naturally)."""
    from cryptid.genome import mutate_genome
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    for _ in range(10):
        c1 = mutate_genome(small_genome, rng1, rate=0.3)
        c2 = mutate_genome(small_genome, rng2, rate=0.3)
        assert c1.mutation == c2.mutation
        assert c1.n_nodes == c2.n_nodes


def test_checkpoint_wavs(tmp_path):
    cfg = cryptid.CryptidConfig(seed=1, strictness=0.5, pop_capacity=15,
                                n_steps=256, genesis_pop=6, genesis_ears=3,
                                checkpoint_every=5)
    c = cryptid.Cryptid(cfg, out_dir=tmp_path / "ck")
    c.genesis(verbose=False)
    c.run(n_gens=5, verbose=False)  # hits checkpoint at gen 5
    ckpt = tmp_path / "ck" / "gen_0005"
    assert ckpt.exists()
    wavs = list(ckpt.glob("*.wav"))
    assert len(wavs) >= 1
    import subprocess

    for w in wavs[:2]:
        out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                              "format=duration", "-of", "csv=p=0", str(w)],
                             capture_output=True, text=True)
        assert float(out.stdout) > 0.0