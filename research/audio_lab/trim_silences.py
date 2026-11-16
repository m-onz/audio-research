#!/usr/bin/env python3
"""
trim_silences.py — trim (or split out) silences from .wav files.

Usage:
    python3 trim_silences.py INPUT [INPUT ...] -o OUT_DIR [options]
    INPUT can be files or directories (every .wav inside is used).

What it does:
    Audio is analysed in short frames; frames below --threshold-db are silence.
    Silences shorter than --min-silence are kept (they're part of the material);
    longer silences are cut. Tiny islands shorter than --min-segment are dropped.
    Each kept region is padded by --padding on both sides so consonants/atmosphere
    survive. By default each file is written back out as one continuous wav with
    the silences spliced out; with --split every kept region becomes its own file.

Examples:
    python3 trim_silences.py sounds.wav sounds2.wav -o trimmed/
    python3 trim_silences.py trimmed/ -o segments/ --split            # one file per region
    python3 trim_silences.py sounds.wav -o out/ --dry-run             # just report
    python3 trim_silences.py sounds.wav -o out/ --threshold-db -50    # noisier source

(For one-off jobs, sox can also do this:
    sox in.wav out.wav silence -l 1 0.1 1% 1 0.35 1%
 or ffmpeg:
    ffmpeg -i in.wav -af silenceremove=stop_periods=-1:stop_duration=0.35:stop_threshold=-45dB out.wav
 but this script gives you better control over gap-merging and per-segment output.)
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- wav IO

def read_wav(path):
    """Read a wav -> (sr, float32 array shaped (n, ch))."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return int(sr), np.ascontiguousarray(data, dtype=np.float32)
    except ImportError:
        pass
    return _read_wav_stdlib(path)


def _read_wav_stdlib(path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        u = np.frombuffer(raw, np.uint8).reshape(-1, ch, 3).astype(np.uint32)
        v = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16)
        v = v.astype(np.int32)
        v = np.where(v & 0x800000, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    return sr, x.reshape(-1, ch)


def write_wav(path, sr, data):
    """Write 16-bit PCM wav, data float (n, ch) in [-1, 1]."""
    data = np.asarray(data, np.float32)
    if data.ndim == 1:
        data = data[:, None]
    y16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(y16.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y16.tobytes())


# --------------------------------------------------------------------------- analysis helpers

def frame_rms_db(x, hop):
    """Mean |sample| in dB per hop-sized frame, mono-mixed."""
    mono = np.abs(x).mean(axis=1)
    nf = len(mono) // hop
    if nf == 0:
        return np.array([])
    e = mono[: nf * hop].reshape(nf, hop).mean(axis=1)
    return 20.0 * np.log10(e + EPS)


def runs(mask):
    """[(start, end), ...] runs where mask is True, in frame indices."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts, ends))


def merge_overlapping(regions):
    regions = sorted(regions)
    out = []
    for a, b in regions:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def fade_edges(region, sr, fade):
    y = region.copy()
    f = min(len(y) // 2, max(1, int(sr * fade)))
    y[:f] *= np.linspace(0.0, 1.0, f)[:, None]
    y[-f:] *= np.linspace(1.0, 0.0, f)[:, None]
    return y


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Trim or split silences from .wav files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="wav files or directories")
    ap.add_argument("-o", "--out", default="trimmed", help="output directory")
    ap.add_argument("--threshold-db", type=float, default=-45.0,
                    help="frames below this level (dBFS) count as silence")
    ap.add_argument("--min-silence", type=float, default=0.4,
                    help="silences shorter than this many seconds are kept")
    ap.add_argument("--min-segment", type=float, default=0.2,
                    help="islands shorter than this are dropped (coughs, clicks...)")
    ap.add_argument("--padding", type=float, default=0.05,
                    help="seconds of audio to keep on each side of every region")
    ap.add_argument("--split", action="store_true",
                    help="write each kept region as its own file instead of one splice")
    ap.add_argument("--fade", type=float, default=0.01,
                    help="fade length in seconds at every cut point (anti-click)")
    ap.add_argument("--dry-run", action="store_true", help="report, don't write")
    args = ap.parse_args()

    outdir = Path(args.out)
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    files = []
    for a in args.inputs:
        p = Path(a)
        if p.is_dir():
            files += sorted(p.glob("*.wav")) + sorted(p.glob("*.WAV"))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: {a} not found, skipping")
    if not files:
        ap.error("no input files")

    for f in files:
        sr, x = read_wav(f)
        hop = max(64, int(sr * 0.005))                       # 5 ms resolution
        db = frame_rms_db(x, hop)
        if len(db) == 0:
            print(f"{f.name}: too short, skipped")
            continue
        voiced = db > args.threshold_db

        # keep short gaps (they're part of the material)
        for s, e in runs(~voiced):
            if (e - s) * hop / sr < args.min_silence:
                voiced[s:e] = True
        # drop tiny islands
        keep = []
        for s, e in runs(voiced):
            if (e - s) * hop / sr >= args.min_segment:
                keep.append((s, e))
        # expand by padding, merge overlaps
        pad = int(args.padding * sr)
        regions = merge_overlapping(
            [(max(0, s * hop - pad), min(len(x), e * hop + pad)) for s, e in keep]
        )

        kept = sum(b - a for a, b in regions)
        print(f"\n{f.name}: kept {len(regions)} region(s), "
              f"{kept / sr:.1f}s of {len(x) / sr:.1f}s ({100 * kept / max(1, len(x)):.0f}%)")
        for i, (a, b) in enumerate(regions):
            print(f"   {i + 1:03d}  {a / sr:8.2f}s -> {b / sr:8.2f}s  ({(b - a) / sr:.2f}s)")
        if not regions:
            print("   nothing above threshold - try a lower --threshold-db")
            continue
        if args.dry_run:
            continue

        if args.split:
            for i, (a, b) in enumerate(regions):
                seg = fade_edges(x[a:b], sr, args.fade)
                out = outdir / f"{f.stem}_{i + 1:03d}.wav"
                write_wav(out, sr, seg)
                print(f"   wrote {out}")
        else:
            parts = [fade_edges(x[a:b], sr, args.fade) for a, b in regions]
            out = outdir / f"{f.stem}_trimmed.wav"
            write_wav(out, sr, np.concatenate(parts))
            print(f"   wrote {out}")


if __name__ == "__main__":
    sys.exit(main())