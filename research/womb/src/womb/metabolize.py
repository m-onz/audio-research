"""metabolize: the cross-modal feedback loop.

The metabolism: loom-family sound goes IN, doom renders it INTO video,
sonify HEARS the video back into sound, degrade adds generational loss,
and the loop runs again — each pass a measured drift of the last.
The artifact is the last video; the ledger is the manifest.

    stem_0 --doom--> video_0 --sonify--> stem_1 --degrade--> video_1 --> ...

Presets steer the drift: `metabolism` (gentle, listenable), `collapse`
(each pass loses fidelity on purpose), `hold` (the slowest possible rot).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .degrade import degrade_audio, degrade_video
from .reel import Reel
from .sonify import sonify, write_wav
from .video import video_stats

__all__ = ["metabolize", "PRESETS"]

# how each preset treats every pass (index p, 1-based)
PRESETS = {
    "metabolism": dict(
        sonify_modes=("spectra", "grains", "tones"),  # rotate; loop if needed
        audio_degrades=lambda p: [("mp3_round", {"kbps": max(48, 128 - 24 * p)}),
                                  ("crush", {"bits": max(6, 14 - p)})],
        video_degrades=lambda p: [("palette_crush", {"levels": max(6, 24 - 4 * p)}),
                                  ("mp4_round", {"crf": 20 + 3 * p})],
        doom_preset="ps2",
    ),
    "collapse": dict(
        sonify_modes=("tones", "grains", "spectra"),
        audio_degrades=lambda p: [("mp3_round", {"kbps": max(32, 64 - 16 * p)}),
                                  ("resample_war", {"rate": max(3000, 9000 - 2000 * p)}),
                                  ("bitrot", {"amount": 0.0004 * p})],
        video_degrades=lambda p: [("resample_war", {"scale": max(0.2, 0.7 - 0.15 * p)}),
                                  ("frame_drop", {"rate": 1 + p}),
                                  ("palette_crush", {"levels": max(3, 8 - p)})],
        doom_preset="glitchcore",
    ),
    "hold": dict(
        sonify_modes=("spectra",),
        audio_degrades=lambda p: [("mp3_round", {"kbps": 160})],
        video_degrades=lambda p: [("mp4_round", {"crf": 22})],
        doom_preset="scope_laboratory",
    ),
}


def _telemetry(stem: Path, video: Path) -> dict:
    """Per-pass measurements for the drift ledger (cheap, streaming)."""
    import doom

    f = doom.analyze_audio(str(stem), fps=24.0)
    v = video_stats(video)
    rms = float(np.median(f.rms)) if f.n_frames else 0.0
    cent = float(np.median(f.centroid)) if f.n_frames else 0.0
    return {
        "rms": round(rms, 4),
        "centroid": round(cent, 1),
        "onsets": int(f.onset.sum()),
        "luma": round(float(np.median(v.luma)), 4) if v.n_frames else 0.0,
        "motion": round(float(np.median(v.motion)), 5) if v.n_frames else 0.0,
        "entropy": round(float(np.median(v.entropy)), 3) if v.n_frames else 0.0,
    }


def _drift(seed_stem: Path, pass_stem: Path, seed_video: Path, pass_video: Path) -> dict:
    """Distance of this pass from the seed generation (0 = identical)."""
    def wav_sig(p):
        import subprocess as sp

        out = sp.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "f32le", "-ac", "1",
                      "-ar", "8000", "-t", "30", "pipe:1"], capture_output=True).stdout
        x = np.frombuffer(out, np.float32)
        if len(x) < 4096:
            return np.zeros(64)
        # coarse spectral signature: mean log-magnitude in 64 bands
        n = len(x) // 4096 * 4096
        S = np.abs(np.fft.rfft(x[:n].reshape(-1, 4096) * np.hanning(4096), axis=1)).mean(axis=0)
        edges = np.geomspace(1, len(S), 65).astype(int)
        return np.array([np.log1p(S[edges[k]: edges[k + 1]].mean())
                         if edges[k + 1] > edges[k] else 0.0
                         for k in range(64)])

    a = float(np.abs(wav_sig(seed_stem) - wav_sig(pass_stem)).mean()) if seed_stem.exists() else 0.0

    def vid_sig(p):
        from .video import decode_luma

        fr, _ = decode_luma(p)
        if len(fr) == 0:
            return np.zeros(64)
        idx = np.linspace(0, len(fr) - 1, 64).astype(int)
        return fr[idx].mean(axis=(1, 2))

    v = float(np.abs(vid_sig(seed_video) - vid_sig(pass_video)).mean()) if seed_video.exists() else 0.0
    return {"audio_drift": round(a, 4), "video_drift": round(v, 4)}


def metabolize(seed_audio, out, passes: int = 3, preset: str = "metabolism",
               seed: int = 0, doom_preset: str | None = None,
               length: float | None = None, verbose: bool = False) -> Reel:
    """Run the metabolism. Returns the Reel (final.mp4 written)."""
    import doom

    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    cfg = dict(PRESETS[preset])
    if doom_preset:
        cfg["doom_preset"] = doom_preset
    reel = Reel(out)
    t_start = time.time()

    # ---- pass 0: seed audio -> doom video --------------------------------
    seed_path = reel.add_seed(seed_audio)
    kwargs = dict(audio=str(seed_path), out=str(reel.pass_dir(0) / "video.mp4"),
                  preset=cfg["doom_preset"], seed=seed, verbose=verbose)
    if length:
        kwargs["length"] = length
    doom.render(**kwargs)
    seed_record = _telemetry(seed_path, reel.video(0))
    seed_record.update({"audio_drift": 0.0, "video_drift": 0.0})  # reference pass
    reel.add_pass("seed", stem=seed_path, video=reel.video(0), record=seed_record)
    if verbose:
        print(f"[womb] pass 0 (seed): rendered {reel.video(0).name}")

    # ---- passes 1..N: sonify -> degrade -> doom -> degrade -----------------
    for p in range(1, int(passes) + 1):
        mode = cfg["sonify_modes"][(p - 1) % len(cfg["sonify_modes"])]
        prev_video = reel.video(p - 1)
        prev_stem = reel.stem(p - 1)

        # 1. hear the video
        audio, sr = sonify(prev_video, mode=mode, seed=seed + 100 * p,
                           corpus=prev_stem if mode == "grains" else None)
        # 2. degrade the sound
        a_ops = cfg["audio_degrades"](p)
        audio = degrade_audio(audio, sr, a_ops, seed=seed + p)
        stem = reel.pass_dir(p) / "stem.wav"
        write_wav(stem, audio, sr)

        # 3. doom renders the new generation
        kwargs = dict(audio=str(stem), out=str(reel.pass_dir(p) / "video.mp4"),
                      preset=cfg["doom_preset"], seed=seed + 17 * p, verbose=verbose)
        if length:
            kwargs["length"] = length
        doom.render(**kwargs)

        # 4. degrade the video
        v_ops = cfg["video_degrades"](p)
        final_video = degrade_video(reel.video(p), v_ops, seed=seed + p)
        if Path(final_video) != reel.video(p):
            import shutil

            shutil.move(final_video, reel.video(p))

        # 5. ledger
        record = _telemetry(stem, reel.video(p))
        record.update(_drift(reel.stem(0), stem, reel.video(0), reel.video(p)))
        record.update({"sonify_mode": mode, "audio_degrades": [o for o, _ in a_ops],
                       "video_degrades": [o for o, _ in v_ops]})
        reel.add_pass("loop", stem=stem, video=reel.video(p), record=record)
        if verbose:
            print(f"[womb] pass {p} ({mode}): rms={record['rms']} "
                  f"aDrift={record['audio_drift']} vDrift={record['video_drift']}")

    artifact = reel.finalize()
    reel.manifest["preset"] = preset
    reel.manifest["seed"] = seed
    reel.manifest["elapsed"] = round(time.time() - t_start, 1)
    reel.save()
    if verbose:
        print(f"[womb] artifact: {artifact}")
    return reel