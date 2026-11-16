"""render tests: sanity, determinism, band-limiting, pictures, io."""

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from prism.champions import champion
from prism.genome import random_genome
from prism.render import MODES, SR, picture, render, write_png, write_wav


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


@pytest.mark.parametrize("mode", MODES)
def test_render_smoke(mode):
    """A random founder must render: finite, shaped, peak-safe. rms
    can be tiny when the genome's field is a click train — the limiter
    is doing its job; viability gates are Phase B's business."""
    rng = np.random.default_rng(11)
    g = random_genome(rng, n_hidden=3)
    audio = render(g, length=2.0, mode=mode, seed=0)
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert audio.shape[0] == int(2.0 * SR)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 0.99
    assert _rms(audio) > 0.0005           # polish guarantees non-silence


@pytest.mark.parametrize("name,mode", [("bellfield", "spectra"),
                                       ("glassvoice", "wavetable"),
                                       ("tide", "field")])
def test_champions_deterministic(name, mode):
    g = champion(name)
    a = render(g, length=2.0, mode=mode, seed=4)
    b = render(champion(name), length=2.0, mode=mode, seed=4)
    assert np.array_equal(a, b)           # bit-identical across builds


def test_field_is_band_limited():
    g = champion("tide")
    audio = render(g, length=2.0, mode="field", seed=0)
    x = audio[:, 0].astype(np.float64)
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / SR)
    hi = X[f > 21_000.0].sum()
    assert hi / X.sum() < 1e-3            # nothing above the ceiling


def test_wavetable_is_pitched():
    g = champion("glassvoice")
    audio = render(g, length=1.0, mode="wavetable", f0=110.0, seed=0)
    x = audio[:, 0].astype(np.float64)
    # the table plays its note: autocorrelation period ≈ 1/f0
    # (the strongest *bin* may be an upper harmonic — that is timbre)
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    lo, hi = int(SR / 220), int(SR / 55)
    lag = lo + int(ac[lo:hi].argmax())
    period = lag / SR
    assert abs(period - 1.0 / 110.0) < (1.0 / 110.0) * 0.03


def test_spectra_is_tonal():
    g = champion("bellfield")
    audio = render(g, length=3.0, mode="spectra", seed=0)
    x = audio[:, 0].astype(np.float64)
    X = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    f = np.fft.rfftfreq(x.size, 1.0 / SR)
    band = (f > 100.0) & (f < 2000.0)
    energy = X[band].sum() / max(X.sum(), 1e-12)
    assert energy > 0.5                   # most energy in the ridge band


def _flat_genome():
    """A genome whose amp output is a constant — the flattest,
    broadbandest field a spectra render can face."""
    from prism.genome import Genome, INPUT_IDS
    g = Genome()
    out = g.add_output("amp", fn="lin")
    g.conn(INPUT_IDS["b"], out, 0.5)
    return g


def test_spectra_broadband_is_dense_not_clicky():
    """Regression: hop = win/2 makes the per-bin phase advance exactly
    πk, so zero-initialized phases realign every even frame into one
    giant windowed impulse — a flat field would synthesize as a click
    train (crest ~60). Random initial phases keep it dense additive
    synthesis: broadband noise, crest < 8."""
    g = _flat_genome()
    audio = render(g, length=4.0, mode="spectra", seed=0)
    x = audio[:, 0].astype(np.float64)
    x = x - x.mean()
    rms = float(np.sqrt((x ** 2).mean()))
    peak = float(np.abs(x).max())
    assert rms > 0.01                       # audible after polish
    assert peak / rms < 8.0                 # dense, not a click train


def test_spectra_no_out_of_band_energy():
    """Regression: np.interp CLAMPS outside its range, so bins above
    the field's top band inherited the top band's magnitude — a leak
    into 20.3–21.2 kHz. The band mapping must silence, not clamp."""
    g = _flat_genome()
    audio = render(g, length=4.0, mode="spectra", seed=0)
    x = audio[:, 0].astype(np.float64)
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    total = X.sum() + 1e-12
    leak_hi = X[f > 20500.0].sum() / total
    assert leak_hi < 0.001                  # nothing above the field's edge
    # and the low edge too: no sub-40Hz rumble from the bottom clamp
    leak_lo = X[f < 35.0].sum() / total
    assert leak_lo < 0.001


def test_picture():
    g = champion("bellfield")
    img = picture(g, width=160, height=90)
    assert img.shape == (90, 160, 3) and img.dtype == np.uint8
    img2 = picture(champion("bellfield"), width=160, height=90)
    assert np.array_equal(img, img2)      # deterministic pixels


def test_atlas_one_world_one_contrast(tmp_path):
    """Regression: per-tile percentile normalization stretched every
    voice to the same dynamic range — a near-silent tile read as loud
    as a blast. The atlas shares ONE range across tiles."""
    from PIL import Image
    from prism.measure import atlas_png
    loud = np.zeros((SR, 2), np.float32)
    rng = np.random.default_rng(0)
    loud[: SR // 4] = rng.standard_normal((SR // 4, 2)) * 0.8
    quiet = (rng.standard_normal((SR, 2)) * 0.002).astype(np.float32)
    p = atlas_png(tmp_path / "atlas.png", [loud, quiet], cols=1)
    assert Path(p).exists()
    sheet = np.asarray(Image.open(p), np.float64)
    th = sheet.shape[0] // 2
    # the sheet is flipud'd for display: tile 0 (loud) sits at the
    # bottom, tile 1 (quiet) at the top
    quiet_max = float(sheet[:th].max())
    loud_max = float(sheet[th:].max())
    assert loud_max > quiet_max + 20    # the quiet tile stays quiet


def test_write_wav_and_png(tmp_path: Path):
    g = champion("tide")
    audio = render(g, length=1.0, mode="field")
    wav = write_wav(tmp_path / "t.wav", audio)
    img = picture(champion("bellfield"), 64, 32)
    png = write_png(tmp_path / "t.png", img)
    assert Path(wav).exists() and Path(wav).stat().st_size > 44100
    assert Path(png).exists()
    assert np.array_equal(np.asarray(Image.open(png)), img)


@pytest.mark.parametrize("name,mode", [("bellfield", "spectra"),
                                       ("glassvoice", "wavetable"),
                                       ("tide", "field")])
def test_champion_render_speed(name, mode):
    g = champion(name)
    t0 = time.time()
    audio = render(g, length=4.0, mode=mode)
    dt = time.time() - t0
    assert dt < 10.0, f"{name} took {dt:.1f}s for 4s of audio"
    assert _rms(audio) > 0.05


def test_cli_roundtrip(tmp_path: Path):
    from prism.cli import main
    g = champion("glassvoice")
    gp = tmp_path / "g.json"
    gp.write_text(__import__("json").dumps({"genome": g.to_dict(),
                                            "mode": "wavetable"}))
    out = tmp_path / "cli.wav"
    assert main(["render", str(gp), "-l", "1.0", "-o", str(out)]) == 0
    assert out.exists()
    # and the info command runs clean
    assert main(["info", str(gp)]) == 0
    assert main(["champions"]) == 0