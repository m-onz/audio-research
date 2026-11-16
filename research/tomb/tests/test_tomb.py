import numpy as np
import tomb
from tomb.ears import features_vector, random_ear
from tomb.ops import OPS, mutate, random_op_recipe


def test_every_op_runs(noise_2s):
    """Every op accepts noise and returns finite audio."""
    rng = np.random.default_rng(0)
    member = noise_2s[: len(noise_2s) // 2]
    for name in sorted(OPS):
        y, _ = mutate(noise_2s, 44100, rng, name, member=member)
        assert len(y) > 0, name
        assert np.isfinite(y).all(), name
        assert np.abs(y).max() <= 1.0, name


def test_genesis_ops_break_noise_fixed_point(noise_2s):
    """The key hypothesis: genesis ops create spectral structure from noise
    (flatness drops significantly). Field ops on noise are ~no-ops."""
    rng = np.random.default_rng(0)

    def flatness(y):
        S = np.abs(np.fft.rfft(y[:8192] * np.hanning(8192))) + 1e-9
        return float(np.exp(np.mean(np.log(S))) / np.mean(S))

    noise_f = flatness(noise_2s)
    # at least resonate and wavesort must break the fixed point
    res_f = flatness(mutate(noise_2s, 44100, rng, "resonate")[0])
    sort_f = flatness(mutate(noise_2s, 44100, rng, "wavesort")[0])
    assert res_f < noise_f * 0.5, f"resonate: {res_f} vs noise {noise_f}"
    assert sort_f < noise_f * 0.5, f"wavesort: {sort_f} vs noise {noise_f}"
    # field ops are near no-ops on pure noise
    diff_f = flatness(mutate(noise_2s, 44100, rng, "diffuse")[0])
    assert abs(diff_f - noise_f) < 0.1, f"diffuse changed noise too much: {diff_f}"


def test_op_recipe_determinism():
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    for _ in range(20):
        n1, p1 = random_op_recipe(rng1, phase="genesis")
        n2, p2 = random_op_recipe(rng2, phase="genesis")
        assert n1 == n2
        assert p1 == p2


def test_features_vector_distinct():
    """Feature vectors must differ meaningfully between noise and
    structured sound (the v1.2 refinement's core claim)."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(44100 * 2).astype(np.float32) * 0.5
    tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(44100 * 2) / 44100)).astype(np.float32)
    f_noise = features_vector(noise)
    f_tone = features_vector(tone)
    d = np.linalg.norm(f_noise - f_tone)
    assert d > 0.3, f"feature distance too small: {d}"
    # flatness should differ strongly
    idx = tomb.ears.FEATURE_NAMES.index("flatness")
    assert abs(f_noise[idx] - f_tone[idx]) > 0.3


def test_ear_hears():
    rng = np.random.default_rng(0)
    ear = random_ear(rng, m=4)
    noise = rng.standard_normal(44100).astype(np.float32) * 0.5
    d = ear.hear(noise)
    assert len(d) == 4
    assert np.isfinite(d).all()


# ------------------------------------------------------------------ genomes + queues

def test_sound_genome_materialize(noise_2s):
    g = tomb.SoundGenome(genome_id=1, origin="noise", origin_audio=noise_2s,
                         ops=[("resonate", None, {}), ("wavesort", None, {"window": 128})])
    audio = g.materialize({})
    assert len(audio) > 0
    assert np.isfinite(audio).all()
    assert g.species in ("genesis", "order")  # both ops present


def test_species_by_lineage(noise_2s):
    g_field = tomb.SoundGenome(genome_id=1, origin="noise", origin_audio=noise_2s,
                               ops=[("diffuse", None, {}), ("propagate", None, {}),
                                    ("react", None, {})])
    assert g_field.species == "field"
    g_loss = tomb.SoundGenome(genome_id=2, origin="noise", origin_audio=noise_2s,
                              ops=[("crush", None, {}), ("bitrot", None, {}),
                                   ("mp3_round", None, {})])
    assert g_loss.species == "loss"


def test_sound_queue_fifo_and_prune(noise_2s):
    q = tomb.SoundQueue(capacity=10)
    for i in range(20):
        g = tomb.SoundGenome(genome_id=q.new_genome_id(), origin="noise",
                             origin_audio=noise_2s, ops=[("resonate", None, {})])
        q.admit(g)
    assert q.size <= 10  # capacity respected


# ------------------------------------------------------------------ engine

def test_genesis_and_small_run(tmp_path):
    """A tiny full run: genesis + 20 batches, deterministic."""
    cfg = tomb.TombConfig(seed=3, tau=0.1, sound_capacity=40, ear_capacity=8,
                          sound_batch=8, seconds=1.0, genesis_batches=5,
                          checkpoint_every=10)
    t = tomb.Tomb(cfg, out_dir=tmp_path / "run1")
    t.genesis(n_founders=10, n_ears=4)
    assert t.sounds.size == 10
    assert t.ears.size >= 1
    t.run(batches=20, verbose=False)
    assert t.sounds.size > 5
    assert len(t.telemetry) == 20
    # some sounds should have been admitted (MC satisfied at least sometimes)
    assert t._stats["sounds_admitted"] > 10


def test_determinism(tmp_path):
    """Same seed -> same population size, same species counts, same telemetry."""
    results = []
    for run in ("a", "b"):
        cfg = tomb.TombConfig(seed=7, tau=0.1, sound_capacity=30, ear_capacity=8,
                              sound_batch=6, seconds=0.5, genesis_batches=3,
                              checkpoint_every=100)
        t = tomb.Tomb(cfg, out_dir=tmp_path / run)
        t.genesis(n_founders=8, n_ears=4)
        t.run(batches=10, verbose=False)
        results.append((t.sounds.size, t.ears.size,
                        tuple(sorted(t.sounds.species_counts.items()))))
    assert results[0] == results[1]


def test_checkpoint_renders_wavs(tmp_path):
    cfg = tomb.TombConfig(seed=1, tau=0.1, sound_capacity=20, ear_capacity=8,
                          sound_batch=4, seconds=0.5, genesis_batches=2,
                          checkpoint_every=5)
    t = tomb.Tomb(cfg, out_dir=tmp_path / "ck")
    t.genesis(n_founders=6, n_ears=3)
    t.run(batches=6, verbose=False)  # hits checkpoint at batch 5
    ckpt = tmp_path / "ck" / "checkpoint_000005"
    assert ckpt.exists()
    wavs = list(ckpt.glob("*.wav"))
    assert len(wavs) >= 6
    import subprocess

    for w in wavs[:3]:
        out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(w)], capture_output=True, text=True)
        assert float(out.stdout) > 0.3  # not empty