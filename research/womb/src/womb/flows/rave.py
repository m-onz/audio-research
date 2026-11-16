"""flows.rave: the local neural latent bank (optional, pure Python).

RAVE (IRCAM) trains a compact variational autoencoder on YOUR corpus and
gives you a 16-d latent space you can walk, morph and sample — entirely
from Python, no Max/MSP anywhere. Verified on this machine: acids-rave
2.3.1 preprocesses and trains on CPU (its CLI doesn't select MPS on
macOS yet; `--gpu -1` = CPU is the supported path).

Because RAVE's torchaudio pin lags this machine's torch, it lives in an
ISOLATED venv at womb/neural (torch 2.11 + torchaudio 2.11 + acids-rave
— created once, used by every womb rave command). The main environment
is never touched.

Typical session (offline, overnight training on CPU):
    womb rave preprocess corpus_dir --db db_dir --sr 44100
    womb rave train db_dir --name my_rave --gpu -1     # hours; resumable
    womb rave generate run_dir --input seed.wav --out heard.wav
    womb metabolize heard.wav -o reel ...              # feed the loop
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

__all__ = ["RAVE_BIN", "rave", "preprocess", "train", "generate", "available"]

_PKG_ROOT = Path(__file__).resolve().parents[3]        # .../womb (repo root)
NEURAL_VENV = _PKG_ROOT / "neural"
RAVE_BIN = NEURAL_VENV / "bin" / "rave"


def available() -> bool:
    """Is the neural venv installed?"""
    return RAVE_BIN.exists()


def rave(*args, check: bool = True) -> subprocess.CompletedProcess:
    """Run the venv's `rave` CLI with args (streamed to the terminal)."""
    if not available():
        raise RuntimeError(
            "RAVE venv not found. Create it once with:\n"
            "  cd womb && python3 -m venv neural && "
            "neural/bin/pip install torch==2.11.0 torchaudio==2.11.0 acids-rave"
        )
    cmd = [str(RAVE_BIN), *args]
    return subprocess.run(cmd, check=check)


def preprocess(corpus_dir, db_dir, sr: int = 44100, channels: int = 1) -> None:
    """corpus folder of wavs -> training database."""
    db = Path(db_dir)
    db.mkdir(parents=True, exist_ok=True)
    rave("preprocess", "--input_path", str(corpus_dir),
         "--output_path", str(db), "--sampling_rate", str(sr),
         "--channels", str(channels))


def train(db_dir, name: str, channels: int = 1, batch: int = 4,
          gpu: int = -1, **extra) -> None:
    """Train a RAVE on the corpus. LONG (hours on CPU); resumable — rerun
    with --ckpt to continue. Checkpoints land in runs/<name>/."""
    args = ["train", "--db_path", str(db_dir), "--name", name,
            "--channels", str(channels), "--batch", str(batch),
            "--gpu", str(gpu), "--workers", "0"]
    for k, v in (extra or {}).items():
        args += [f"--{k}", str(v)]
    rave(*args)


def generate(model_dir, input_audio, out_path, stream: bool = True) -> str:
    """Reconstruct/resynthesize `input_audio` through the trained model —
    the latent round-trip that gives RAVE its sound."""
    model = Path(model_dir)
    # a run directory holds version_*/checkpoints; find the newest ckpt
    ckpts = sorted(model.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {model}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["generate", "--model", str(ckpts[-1]),
            "--input", str(input_audio), "--out_path", str(out)]
    if stream:
        args.append("--stream")
    rave(*args)
    produced = out.with_suffix(".wav") if out.suffix != ".wav" else out
    if not produced.exists():
        # generate may name outputs after the input; find the freshest wav
        cands = sorted(out.parent.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if cands:
            shutil.move(str(cands[-1]), produced)
    return str(produced)