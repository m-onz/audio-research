"""Build MP4 movies from the gallery of discovered effects.

Two modes:

  reel      Each selected archive entry is animated live (the same renderer
            as the viewer) for `--clip-frames` frames, with a caption strip
            showing its id / generation / novelty.  Clips follow the chosen
            order (default chronological = the story of discovery).

  slideshow The existing 32x32 fingerprint PNGs are upscaled and shown one
            per clip; the soundtrack sonifies the fingerprints directly
            (fingerprint-as-spectrogram).

Both modes are fully sonified (see sonify.py):
  python3 -m nparticles.main --movie reel.mp4
  python3 -m nparticles.main --movie catalog.mp4 --mode slideshow
Video frames are piped raw into one ffmpeg process (no temp image files);
audio is synthesised in numpy, written as a WAV, then muxed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evolution import DESC_W_GRID
from .genome import Genome
from .post import (DEFAULT_POST, box_lp, parse_post, paulstretch, paulstretch_vary,
                   run_chain, shimmer, shimmer_tv, tape_phasing, tape_phasing_tv)
from .render import Canvas, draw_step
from .sim import STATS, Sim
from .sonify import (SR, Macro, event_layer, genome_mod_signals,
                     rms_balance, schroeder_reverb, spectrogram_layer,
                     texture_layer, to_wav)

import zlib

_ENTROPY_I = [i for i, (n, _) in enumerate(STATS) if n == "entropy"][0]


def load_gallery(run_dir: str) -> list[dict]:
    """Read every archive entry from a run's gallery directory."""
    gdir = Path(run_dir) / "gallery"
    if not gdir.exists():
        sys.exit(f"no gallery in {run_dir} — run the evolver first")
    entries = []
    for d in gdir.iterdir():
        meta_p = d / "stats.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        desc = np.load(d / "descriptor.npy")
        grids = np.clip(desc[:2048].reshape(2, 32, 32) / DESC_W_GRID, 0, 1)
        entries.append({
            "uid": meta["uid"], "gen": meta["gen"], "novelty": meta["novelty"],
            "stats": np.asarray(meta["stats_raw"], dtype=np.float64),
            "genome": Genome.from_dict(json.loads((d / "genome.json").read_text())),
            "grids": grids.astype(np.float32),
            "fingerprint": str(d / "fingerprint.png"),
            "dir": d,
        })
    return entries


def select_entries(entries: list[dict], order: str, count: int,
                   seed: int = 7) -> list[dict]:
    if order == "novelty":
        es = sorted(entries, key=lambda e: e["novelty"], reverse=True)
    elif order == "random":
        es = list(entries)
        __import__("random").Random(seed).shuffle(es)
    else:                                   # chrono: the story of discovery
        es = sorted(entries, key=lambda e: (e["gen"], e["uid"]))
    if count and count > 0 and count < len(es):
        idx = np.linspace(0, len(es) - 1, count).round().astype(int)
        es = [es[i] for i in idx]
    return es


def make_label(text: str, w: int, h: int = 44) -> np.ndarray:
    img = Image.new("RGB", (w, h), (6, 7, 10))
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(20)
    except TypeError:
        font = ImageFont.load_default()
    dr.text((10, int((h - dr.textbbox((0, 0), text, font=font)[3]) / 2)),
            text, fill=(185, 197, 224), font=font)
    return np.asarray(img, dtype=np.float32)


def _ffmpeg_video(path: Path, size: int, fps: int):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size}x{size}", "-r", str(fps), "-i", "pipe:",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def build_movie(out_mp4: str, run_dir: str, mode="reel", order="chrono",
                count=0, clip_frames=0, size=640, fps=30, seed=3,
                particles=240, reverb=True, post=DEFAULT_POST, fx_mod=1.0,
                quiet=False):
    entries = select_entries(load_gallery(run_dir), order, count, seed)
    if not entries:
        sys.exit("gallery is empty")
    if clip_frames <= 0:
        clip_frames = 150 if mode == "reel" else 66
    chain = parse_post(post)
    fx = dict(chain)                                    # name -> arg
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = out.with_suffix(".video.tmp.mp4")
    tmp_wav = out.with_suffix(".audio.tmp.wav")

    gens = [e["gen"] for e in entries]
    gmin, gmax = min(gens), max(gens)
    labels = {}
    for e in entries:
        g = e["genome"]
        labels[e["uid"]] = make_label(
            f"{e['uid']}  gen {e['gen']}  novelty {e['novelty']:.2f}  "
            f"{g.shape}{' +trail' if g.trail else ''}  net {g.size()[0]}h/{g.size()[1]}c",
            size)

    proc = _ffmpeg_video(tmp_video, size, fps)
    clip_len = int(round(clip_frames / fps * SR))       # samples per clip
    tail = int(0.45 * SR)
    n_total = clip_len * len(entries) + tail
    events_st = np.zeros((n_total, 2), dtype=np.float32)   # frame-synced layer
    texture_mono = np.zeros(n_total, dtype=np.float32)     # bed layer (phasing/ghost source)
    # genome-driven FX automation (raw ANN outputs, written per clip)
    MOD_KEYS = ("rate", "depth", "drive", "shimmer", "ggain", "wobble")
    mod_raw = ({k: np.zeros(n_total, dtype=np.float32) for k in MOD_KEYS}
               if fx_mod > 0 else None)
    stretch_ctrl = np.zeros(n_total, dtype=np.float32) if fx_mod > 0 else None
    label_h = labels[entries[0]["uid"]].shape[0]

    t_start = time.perf_counter()
    for ci, e in enumerate(entries):
        s0 = ci * clip_len
        reg = 0.5 * (e["gen"] - gmin) / max(gmax - gmin, 1)     # audible evolution
        macro = Macro(e["stats"], register_shift=reg)
        label = labels[e["uid"]]
        if mode == "reel":
            sim = Sim(e["genome"], n=particles, ttl=170, seed=seed + ci)
            cv = Canvas(size, size, spr_r=max(10, size // 26), fade=0.90,
                        exposure=1.5)
            frames = []
            for fi in range(clip_frames):
                cv.tick()
                st = sim.step()
                draw_step(cv, st)
                frames.append(st)
                f = cv.frame()
                f[size - label_h:, :] = (f[size - label_h:, :].astype(np.float32)
                                         * 0.35 + label * 0.9).astype(np.uint8)
                proc.stdin.write(f.tobytes())
            events_st[s0:s0 + clip_len + tail] += event_layer(frames, macro)
        else:                                                   # slideshow
            img = Image.open(e["fingerprint"]).convert("RGB").resize(
                (size - 96, size - 96), Image.NEAREST)
            base = np.zeros((size, size, 3), dtype=np.uint8)
            base[:] = (8, 8, 12)
            arr = np.asarray(img)
            off = (size - arr.shape[0]) // 2
            base[off:off + arr.shape[0], off:off + arr.shape[1]] = arr
            base[size - label_h:, :] = label.astype(np.uint8)
            blob = base.tobytes()
            for _ in range(clip_frames):
                proc.stdin.write(blob)
            events_st[s0:s0 + clip_len + tail] += spectrogram_layer(
                e["grids"], (clip_len + tail) / SR, macro)
        tx = texture_layer(e["genome"], (clip_len + tail) / SR, macro, seed=seed + ci)
        texture_mono[s0:s0 + clip_len + tail] += tx
        if mod_raw is not None:
            # second performance: the genome plays the mixing desk.
            # Which ANN output drives which FX is permuted per genome id,
            # so similar-sounding discoveries process themselves differently.
            Ym = genome_mod_signals(e["genome"], (clip_len + tail) / SR, macro,
                                    seed=seed + 1009 + ci)
            perm = np.random.default_rng(
                zlib.crc32(e["uid"].encode()) & 0x7FFFFFFF).permutation(6)
            for i, key in enumerate(MOD_KEYS):
                mod_raw[key][s0:s0 + clip_len] = np.clip(
                    Ym[:clip_len, perm[i]] * fx_mod, -1.0, 1.0)
            ent01 = float(np.clip(e["stats"][_ENTROPY_I], 0, 1))
            st_base = float(np.clip(fx.get("ghost", 6.0) * (0.5 + 1.3 * ent01),
                                    1.0, 14.0))
            stretch_ctrl[s0:s0 + clip_len] = st_base
        if not quiet:
            pct = (ci + 1) / len(entries) * 100
            print(f"\r  clip {ci + 1}/{len(entries)} ({pct:.0f}%) "
                  f"{e['uid']} gen {e['gen']}", end="", flush=True)
    if not quiet:
        print(f"\n  video: {len(entries)} clips x {clip_frames} frames "
              f"@ {fps}fps = {len(entries) * clip_frames / fps:.0f}s")

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        sys.exit("ffmpeg video encode failed")

    # ---------------- post-processing: parallel beds, then serial chain ----
    tv = {}
    if mod_raw is not None:
        # smooth the raw automation across clip boundaries, then map each
        # signal onto its FX parameter range
        w = int(0.06 * SR)
        s01 = {k: np.clip(0.5 + 0.5 * box_lp(a, w), 0, 1) for k, a in mod_raw.items()}
        wob = np.clip(mod_raw["wobble"], -0.9, 0.9)
        if "excite" in fx:
            tv["drive"] = np.clip(fx["excite"] * (0.4 + 1.0 * s01["drive"]),
                                   0.9, 8.0)
        if "flanger" in fx:
            tv["rate"] = np.clip(fx["flanger"] * (0.15 + 1.6 * s01["rate"]),
                                  0.01, 0.6)
            tv["depth"] = 0.2 + 13.0 * s01["depth"] ** 2          # ms
        ggain_p = 0.10 + 0.60 * s01["ggain"]
        shim_p = 0.60 * s01["shimmer"]
    else:
        wob, ggain_p, shim_p = None, None, None

    eps = fx.get("phasing", 0.0)
    if eps > 0:
        tex_st = (tape_phasing_tv(texture_mono, eps, wob) if wob is not None
                  else tape_phasing(texture_mono, eps))
    else:
        tex_st = np.stack([texture_mono, texture_mono], axis=1)
    if "ghost" in fx and fx["ghost"] > 0:
        if stretch_ctrl is not None:
            # fill the final tail with the last stretch value
            stretch_ctrl[clip_len * len(entries):] = stretch_ctrl[clip_len * len(entries) - 1]
            gh = paulstretch_vary(texture_mono, stretch_ctrl)[:n_total]
        else:
            gh = paulstretch(texture_mono, fx["ghost"])[:n_total]
        fade = min(int(3 * SR), n_total)
        gh[-fade:] *= np.linspace(1, 0, fade).astype(np.float32)
        gh_st = np.stack([gh, gh], axis=1) * 0.5
        if ggain_p is not None:
            gh_st *= ggain_p[:, None]                      # genome plays ghost gain
        if "shimmer" in fx:
            gh_st = (shimmer_tv(gh_st, shim_p) if shim_p is not None
                     else shimmer(gh_st, mix=fx["shimmer"]))
        tex_st = tex_st + gh_st
    audio = events_st + tex_st
    audio = run_chain(audio, chain, tv=tv)
    if reverb:
        audio = schroeder_reverb(audio, wet=0.30)
    audio = rms_balance(audio)
    to_wav(str(tmp_wav), audio)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_video),
                    "-i", str(tmp_wav), "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", str(out)], check=True)
    tmp_video.unlink()
    tmp_wav.unlink()
    if not quiet:
        print(f"  audio: {audio.shape[0] / SR:.0f}s, "
              f"rendered in {time.perf_counter() - t_start:.0f}s")
        print(f"wrote {out}")
    return str(out)