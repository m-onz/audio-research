#!/usr/bin/env python3
"""
binary_lab.py — sonify binaries, timbre-stamp them, let the data conduct.

The pipeline (see TECHNIQUES.md for the full rationale):

  1. SONIFY   a binary (the lab's own bytecode, a system executable, any file)
     is interpreted as PCM through multiple encodings: direct byte-cast,
     bit-plane slices, G.711 mu-law decode, differential decode, nibble-split,
     16-bit endian pairs. Corrupted variants too: bit-rotation, XOR
     stream-cipher, block byte-swaps, packet zero-runs.
  2. STAMP    the raw streams are pushed through learned corpus timbre models:
     an NMF spectral dictionary fitted to the corpus (structure wears the
     corpus's timbres), LPC resonance models driven by the whitened binary
     (the binary "speaks" through corpus formants), and spectral convolution
     stamps. Plus a wavetable scanner that wraps stamped windows as
     single-cycle oscillators scanning through the binary at entropy-derived
     pitches.
  3. CONDUCT  the piece structure is READ FROM THE DATA, not templated:
     windowed byte entropy becomes the energy arc; entropy mean-shifts become
     section boundaries; entropy variance becomes the chaos dial; zero-byte
     runs become silence; a byte of the binary becomes the Wolfram CA rule
     that gates every hit (pass / mute / reverse / corrupt).
  4. INTERLEAVE the stamped stream-chunks and corpus chunks (dark/mid/bright)
     are sequenced together by the alien-lisp pattern algebra.

Usage:
    python3 binary_lab.py -o audio_lab/binary_pieces              # default binaries
    python3 binary_lab.py -o out --pieces 8 --seed 3
    python3 binary_lab.py -o out /bin/ls audio_lab/granular_lab.py some.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.linalg import solve_toeplitz
from scipy.signal import stft, istft, lfilter

import granular_lab as gl
import alien_rhythms as ar
import rhythm_lab as rl
import montage_lab as mg

SR = 44100
EPS = 1e-10


# ================================================================== sonification

def corrupt_variants(u8, rng):
    """A family of corrupted readings of the same binary."""
    v = {"raw": u8}
    u16 = u8.astype(np.uint16)
    v["rot"] = (((u16 << 3) | (u16 >> 5)) & 0xFF).astype(np.uint8)
    v["xor"] = np.frombuffer(u8.tobytes(), np.uint8) ^ \
        np.frombuffer(rng.bytes(len(u8)), np.uint8)
    sw = u8.copy()
    for _ in range(24):
        a = int(rng.integers(0, max(1, len(sw) - 2048)))
        b = int(rng.integers(0, max(1, len(sw) - 2048)))
        m = int(rng.integers(64, 2048))
        sa, sb = sw[a:a + m].copy(), sw[b:b + m].copy()
        sw[a:a + m], sw[b:b + m] = sb, sa
    v["swap"] = sw
    z = u8.copy()
    for _ in range(14):
        a = int(rng.integers(0, max(1, len(z) - 800)))
        z[a:a + int(rng.integers(100, 800))] = 0
    v["zeroed"] = z
    return v


def _norm(x):
    x = x - np.mean(x)
    s = np.std(x)
    if s > EPS:
        x = x / s * 0.25
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def sonify(u8):
    """Every encoding is a different way of listening to the same bytes."""
    out = {}
    out["int8"] = _norm(u8.astype(np.float32) - 128.0)
    for shift in (0, 4):
        out[f"bits{shift}"] = _norm(((u8 >> shift) & 3).astype(np.float32) - 1.5)
    d = np.diff(u8.astype(np.int16))
    out["delta"] = _norm(np.concatenate([[0], d]))
    u = (~u8.astype(np.uint16)) & 0xFF
    t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
    t = np.where(u & 0x80, 0x84 - t, t - 0x84)
    out["mulaw"] = _norm(t.astype(np.float32))
    out["nib_lo"] = _norm((u8 & 15).astype(np.float32) - 8)
    out["nib_hi"] = _norm((u8 >> 4).astype(np.float32) - 8)
    if len(u8) >= 4:
        n2 = len(u8) // 2
        b = u8[: n2 * 2]
        out["i16le"] = _norm(np.frombuffer(b.tobytes(), "<i2").astype(np.float32))
        out["i16be"] = _norm(np.frombuffer(b.tobytes(), ">i2").astype(np.float32))
    return {k: y for k, y in out.items() if len(y) > SR // 10}


def read_bytes(u8, n, rng):
    """Tile the binary to n bytes, entering at random offsets each loop."""
    if len(u8) >= n:
        return u8[:n]
    out = np.empty(n, np.uint8)
    pos = 0
    while pos < n:
        st = int(rng.integers(0, len(u8)))
        m = min(len(u8) - st, n - pos)
        out[pos:pos + m] = u8[st:st + m]
        pos += m
    return out


# ================================================================== structure

def entropy_curve(u8, win=512):
    n = max(1, len(u8) // win)
    h = np.empty(n)
    for i in range(n):
        c = np.bincount(u8[i * win:(i + 1) * win], minlength=256) / win
        p = c[c > 0]
        h[i] = -(p * np.log2(p)).sum()
    return h / 8.0


def derive_sections(u8, h, target_s, rng, win=512):
    """The binary IS the score: sections from entropy mean-shifts."""
    h_s = np.convolve(h, np.ones(9) / 9.0, "same")
    n_sec = int(np.clip(len(h) // 40, 5, 9))
    seg = max(1, len(h_s) // max(1, n_sec * 4))
    loc = np.convolve(h_s, np.ones(seg) / seg, "same")
    d = np.abs(np.diff(loc))
    sep = max(1, len(h) // n_sec)
    idx = []
    for i in np.argsort(d)[::-1]:
        j = int(i)
        if j < 3 or j > len(h) - 3:
            continue
        if all(abs(j - b) > sep for b in idx):
            idx.append(j)
        if len(idx) >= n_sec - 1:
            break
    bounds = sorted([0] + idx + [len(h)])
    secs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        seg_u = u8[a * win: b * win]
        if len(seg_u) == 0:
            continue
        secs.append(dict(
            frac=(b - a) / len(h),
            energy=float(np.clip(np.mean(h[a:b]), 0.05, 1.0)),
            chaos=float(np.clip(np.std(h[a:b]) * 3.0, 0.0, 1.0)),
            zeros=float(np.mean(seg_u == 0)),
            rule=int(seg_u[len(seg_u) // 2]),
        ))
    if not secs:
        secs = [dict(frac=1.0, energy=0.5, chaos=0.5, zeros=0.05, rule=90)]
    return secs


# ================================================================== timbre stamps

def fit_nmf_dictionary(files, rng, k=32, iters=50, windows=6, win_s=10.0):
    """Learn a spectral dictionary from corpus STFTs (standard ML, no LLM)."""
    mags = []
    for f in files[:windows * 2]:
        try:
            info = sf.info(str(f))
            dur = info.frames / info.samplerate
            start = 0.0 if dur <= win_s else float(rng.uniform(0, dur - win_s))
            with sf.SoundFile(str(f)) as fh:
                fh.seek(int(start * info.samplerate))
                y = fh.read(int(min(win_s, dur) * info.samplerate),
                            dtype="float32", always_2d=True)
        except Exception:
            continue
        if len(y) < SR:
            continue
        _, _, Z = stft(y.mean(axis=1), SR, nperseg=2048, noverlap=1536)
        mags.append(np.abs(Z).astype(np.float32))
        if len(mags) >= windows:
            break
    if not mags:
        return None
    V = np.concatenate(mags, axis=1)
    V = V / (V.mean() + EPS)
    W = (rng.random((V.shape[0], k)) * 0.1 + 0.01).astype(np.float32)
    H = (rng.random((k, V.shape[1])) * 0.1 + 0.01).astype(np.float32)
    for _ in range(iters):
        H *= (W.T @ V) / (W.T @ W @ H + EPS)
        W *= (V @ H.T) / (W @ H @ H.T + EPS)
        nrm = np.linalg.norm(W, axis=0, keepdims=True) + EPS
        W /= nrm
        H *= nrm.T
    return W


def nmf_stamp(x, W, rng, iters=40):
    """Project the sonified stream onto the corpus dictionary; keep its phase."""
    _, _, Z = stft(x, SR, nperseg=2048, noverlap=1536)
    V = np.abs(Z).astype(np.float32)
    Hs = (rng.random((W.shape[1], V.shape[1])) * 0.1 + 0.01).astype(np.float32)
    for _ in range(iters):
        Hs *= (W.T @ V) / (W.T @ W @ Hs + EPS)
    V2 = (W @ Hs).astype(np.float32)
    V2 *= (V.mean() + EPS) / (V2.mean() + EPS)
    _, y = istft(V2 * np.exp(1j * np.angle(Z)), SR, nperseg=2048, noverlap=1536)
    return np.clip(y, -1, 1).astype(np.float32)


def conv_stamp(x, ref):
    """Spectral convolution: binary phase, corpus magnitude envelope."""
    _, _, Zx = stft(x, SR, nperseg=2048, noverlap=1536)
    _, _, Zr = stft(ref, SR, nperseg=2048, noverlap=1536)
    env = np.abs(Zr).mean(axis=1, keepdims=True).astype(np.float32)
    V2 = np.abs(Zx) * (env / (env.mean() + EPS))
    _, y = istft(V2 * np.exp(1j * np.angle(Zx)), SR, nperseg=2048, noverlap=1536)
    return np.clip(y, -1, 1).astype(np.float32)


def _lpc_frame(y, order):
    r = y * np.hanning(len(y))
    ac = np.correlate(r, r, "full")[len(r) - 1:]
    if ac[0] < EPS:
        return None
    ac = ac / ac[0]
    try:
        a = solve_toeplitz(ac[:order], ac[1:order + 1])
    except Exception:
        return None
    poly = np.concatenate([[1.0], -a])
    roots = np.roots(poly)
    mx = np.max(np.abs(roots)) if len(roots) else 0.0
    if mx > 0.985:
        a = a * (0.985 / mx)
    return a.astype(np.float64)


def lpc_stamp(x, ref, rng, order=28):
    """The binary drives corpus resonance models (it 'speaks' the corpus)."""
    xw = np.diff(x, prepend=x[0])
    xw = xw / (np.std(xw) + EPS)
    frames = []
    hop, fl = 1024, 2048
    for i in range(0, max(1, len(ref) - fl), hop):
        a = _lpc_frame(ref[i:i + fl], order)
        if a is not None:
            frames.append(a)
    if not frames:
        return x * 0.5
    A = np.stack(frames)                                    # frames x order
    pos = np.linspace(0, len(A) - 1, max(2, int(len(x) / hop) + 1))
    i0 = np.clip(pos.astype(int), 0, len(A) - 1)
    interp = A[i0] * (1 - (pos - i0)[:, None]) + \
        A[np.clip(i0 + 1, 0, len(A) - 1)] * (pos - i0)[:, None]
    y = np.zeros(len(x))
    zi = np.zeros(order)
    for bi in range(len(interp)):
        a_poly = np.concatenate([[1.0], -interp[bi]])
        s = bi * hop
        e = min(len(x), s + hop)
        if s >= len(x):
            break
        yb, zi = lfilter([1.0], a_poly, xw[s:e], zi=zi)
        y[s:e] = yb
    y = y / (np.std(y) + EPS) * 0.25
    return np.clip(y, -1, 1).astype(np.float32)


def wavetable_voice(stream, dur, f0_hz, rng, table=1024, scan=97.0):
    """Wrap stamped binary windows as single-cycle oscillators, scanning
    through the stream at entropy-derived pitches."""
    n = max(table * 2, int(dur * SR))
    t = np.arange(n) / SR
    f0_t = np.interp(t, np.linspace(0, t[-1], len(f0_hz)), f0_hz)
    vib = 1.0 + 0.006 * np.sin(2 * np.pi * rng.uniform(3, 8) * t)
    phase = np.cumsum(f0_t * vib) / SR
    pos = (phase % 1.0) * table
    scan_pos = (t * scan) % max(1, len(stream) - table - 2)
    p = scan_pos + pos
    i0 = p.astype(np.int64)
    fr = (p - i0).astype(np.float32)
    y = stream[i0] * (1 - fr) + stream[i0 + 1] * fr
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t + rng.uniform(0, 6))
    return (y * env * 0.5).astype(np.float32)


# ================================================================== CA gating

def ca_stream(rule, width, gens, rng):
    """Wolfram elementary CA; the rule number is read from the binary."""
    r = rng.integers(0, 2, width).astype(bool)
    lut = np.unpackbits(np.array([rule & 0xFF], np.uint8), bitorder="little").astype(bool)
    rows = np.empty((gens, width), bool)
    for g in range(gens):
        rows[g] = r
        idx = (np.roll(r, 1).astype(np.uint8) << 2) | \
              (r.astype(np.uint8) << 1) | np.roll(r, -1).astype(np.uint8)
        r = lut[idx]
    return rows


# ================================================================== piece

def _chunks_of(y, rng, lo=0.08, hi=0.7):
    out, pos = [], 0
    while pos < len(y) - 2048:
        m = int(rng.uniform(lo, hi) * SR)
        seg = y[pos:pos + m]
        pos += m
        if len(seg) < 1024:
            continue
        f = min(len(seg) // 4, 256)
        seg = seg.copy()
        seg[:f] *= np.linspace(0, 1, f)
        seg[-f:] *= np.linspace(1, 0, f)
        ft = gl.compute_features(np.ascontiguousarray(seg[:, None]) if seg.ndim == 1
                                 else seg, SR)
        out.append((seg, ft))
    return out


def render_piece(rng, bin_path, corpus_files, W_dict, target_s, args):
    u8 = load_binary(bin_path)
    h = entropy_curve(u8)
    secs = derive_sections(u8, h, target_s, rng)
    variants = corrupt_variants(u8, rng)

    # ---- build stream voices: (variant, encoding) x stamps
    enc_names = list(sonify(u8[: min(len(u8), 65536)]).keys())
    voices = []
    voice_log = []
    n_streams = int(np.clip(3 + len(u8) // 300000, 3, 5))
    stamps = ["nmf", "conv", "lpc", "raw"]
    for si in range(n_streams):
        vname = list(variants)[int(rng.integers(len(variants)))]
        ename = enc_names[int(rng.integers(len(enc_names)))]
        stamp = stamps[si % len(stamps)]
        speed = float(2 ** rng.uniform(-2.0, 2.0))
        n_bytes = int(min(len(u8) * 8, rng.uniform(15.0, 30.0) * SR * speed))
        src = read_bytes(variants[vname], n_bytes, rng)
        stream = sonify(src).get(ename)
        if stream is None:
            continue
        if speed != 1.0:
            stream = gl.resample_grain(_stereo(stream), speed, "hermite")[:, 0]
        ref_path = corpus_files[int(rng.integers(len(corpus_files)))]
        try:
            sr_r, ref = gl.read_wav(ref_path)
            ref = ref.mean(axis=1) if ref.ndim > 1 else ref
        except Exception:
            ref = stream
        if stamp == "nmf" and W_dict is not None:
            y = nmf_stamp(stream, W_dict, rng)
        elif stamp == "conv":
            y = conv_stamp(stream, ref[: min(len(ref), 20 * SR)])
        elif stamp == "lpc":
            y = lpc_stamp(stream, ref[: min(len(ref), 12 * SR)], 28, rng)
        else:
            y = np.tanh(stream * 1.5) * 0.8                      # raw, driven
        y = np.clip(y / (np.std(y) + EPS) * 0.22, -1, 1).astype(np.float32)
        chs = _chunks_of(y, rng)
        if chs:
            voices.append(dict(name=f"{vname}/{ename}/{stamp}", kind="stream",
                               chunks=chs, gain=0.9, pan=rng.uniform(-0.4, 0.4)))
            voice_log.append(dict(voice=si, variant=vname, encoding=ename,
                                 stamp=stamp, speed=round(speed, 2),
                                 chunks=len(chs)))

    # ---- wavetable voice (the binary sings)
    if voices:
        base_stream = voices[0]["chunks"][0][0]                  # reuse chunk 0
        src = read_bytes(variants["raw"], int(8 * SR), rng)
        wt_all = sonify(src)
        wt_src = wt_all.get("i16le")
        if wt_src is None:
            wt_src = wt_all["int8"]
        f0_curve = []
        for s in secs:
            semi = int(s["energy"] * 24) - 12                    # entropy -> pitch
            f0_curve.extend([55.0 * 2 ** (semi / 12.0)] * 8)
        wt = wavetable_voice(wt_src, target_s * 0.8,
                             np.array(f0_curve, np.float32), rng)
        wt_chs = _chunks_of(wt, rng, lo=0.3, hi=2.0)
        if wt_chs:
            voices.append(dict(name="wavetable/raw/scan", kind="wt",
                               chunks=wt_chs, gain=0.5, pan=0.0))
            voice_log.append(dict(voice=len(voices) - 1, variant="raw",
                                  encoding="wavetable", stamp="scan",
                                  speed=1, chunks=len(wt_chs)))

    # ---- corpus interleave voices (dark / mid / bright)
    sub = [corpus_files[i] for i in rng.choice(len(corpus_files),
                                               size=min(8, len(corpus_files)),
                                               replace=False)]
    sr_c, chunks_c = rl.extract_pool(sub, rng, pool_cap=900)
    pools_c = rl.assign_roles(chunks_c)
    for label, keys in (("dark", ("kick", "tom")), ("mid", ("snare", "clap")),
                        ("bright", ("hat", "stab"))):
        pool = [c for k in keys for c in pools_c.get(k, [])]
        if not pool:
            continue
        voices.append(dict(name=f"corpus/{label}", kind="corpus", chunks=pool,
                           gain=0.55, pan=rng.uniform(-0.6, 0.6)))
        voice_log.append(dict(voice=len(voices) - 1, variant="corpus",
                              encoding=label, stamp="-", speed=1,
                              chunks=len(pool)))

    # ---- CA controller (rule byte from the middle of the binary)
    rule = int(u8[len(u8) // 2])
    ca = ca_stream(rule, 128, 64, rng)
    hit_i = 0

    # ---- render: sections -> slots -> pattern-gated hits
    events = []
    t_cursor = 0.0
    for sec in secs:
        dur = sec["frac"] * target_s
        slot_len = float(np.exp(np.interp(sec["chaos"], [0, 1],
                                          [np.log(1.4), np.log(0.08)])))
        n_slots = max(4, int(dur / slot_len))
        silence_p = 0.10 + 0.55 * sec["zeros"]
        for v in voices:
            prof = dict(k=(2, max(3, int(4 + 9 * sec["energy"]))), val="mid",
                        poly=0.3)
            try:
                pat = ar.normalize_velocities(ar.ev(ar.gen_pattern(rng, prof), rng))
            except Exception:
                continue
            L = len(pat)
            if L == 0:
                continue
            for i in range(n_slots):
                val = pat[i % L]
                if val is None or val <= 0.05 or rng.random() < silence_p:
                    continue
                hit_i += 1
                cell = ca[(hit_i // 128) % 64, hit_i % 128]
                t = t_cursor + i * slot_len
                if t >= target_s:
                    break
                if not cell:                                    # CA corruption
                    r = rng.random()
                    if r < 0.55:
                        continue                               # mute
                    mod_speed = 1.5 if r < 0.8 else 0.6
                else:
                    mod_speed = 1.0
                pick = v["chunks"][int(rng.integers(len(v["chunks"])))]
                if v["kind"] == "corpus":
                    seg = pick.data[: int(rl.ROLE_CFG["snare"]["max_dur"] * SR)]
                    amp = min(12.0, 0.10 / max(1e-4, pick.rms))
                else:
                    seg, ft = pick
                    amp = 1.0
                if mod_speed != 1.0:
                    if seg.ndim == 1:
                        seg = gl.resample_grain(_stereo(seg), mod_speed, "hermite")[:, 0]
                    else:
                        seg = gl.resample_grain(seg, mod_speed, "hermite")
                y = seg * (amp * val * v["gain"] * (0.3 + 0.7 * sec["energy"]) *
                           10 ** (rng.uniform(-2, 2) / 20))
                pan = np.clip(v["pan"] + rng.uniform(-0.25, 0.25), -1, 1)
                th = (pan + 1) * np.pi / 4
                if y.ndim == 1:
                    y = np.stack([y * np.cos(th), y * np.sin(th)], axis=1)
                else:
                    y = y * np.array([np.cos(th), np.sin(th)], np.float32)
                events.append((max(0, int(t * SR)),
                               np.ascontiguousarray(y, np.float32)))
        t_cursor += dur

    meta = dict(binary=str(bin_path), size=len(u8),
                entropy_mean=round(float(np.mean(h)), 3),
                entropy_std=round(float(np.std(h)), 3),
                ca_rule=rule,
                sections=[dict(frac=round(s["frac"], 3),
                               energy=round(s["energy"], 3),
                               chaos=round(s["chaos"], 3),
                               zeros=round(s["zeros"], 3)) for s in secs],
                voices=voice_log)
    return events, meta


def load_binary(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8)


def _stereo(x):
    x = np.asarray(x, np.float32)
    return x[:, None] if x.ndim == 1 else x


# ================================================================== CLI

def default_binaries(base):
    out = []
    lab = Path(base)
    out += sorted(str(p) for p in lab.glob("*.py"))
    out += sorted(str(p) for p in (lab / "__pycache__").glob("*.pyc"))
    for sub in sorted(lab.iterdir()):
        if sub.is_dir() and (sub / "manifest.json").exists():
            out.append(str(sub / "manifest.json"))
    out += ["/bin/ls"]
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Sonify binaries, timbre-stamp them with the corpus, "
                    "let the data conduct the structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("files", nargs="*", help="binaries to sonify (default: the lab itself)")
    ap.add_argument("--base", default="audio_lab")
    ap.add_argument("-o", "--out", default="binary_pieces")
    ap.add_argument("--pieces", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=float, default=185.0)
    ap.add_argument("--max-len", type=float, default=235.0)
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("--seed must be >= 0")

    binaries = args.files or default_binaries(args.base)
    corpus_files = [Path(f) for f in mg.discover(args.base, Path(args.out).name)
                    if Path(f).parent.name not in ("final_pieces", "sounds",
                                                    Path(args.out).name)]
    if not corpus_files:
        ap.error("no corpus found")
    rng0 = np.random.default_rng(args.seed)
    print("fitting corpus NMF dictionary (32 components) ...")
    W_dict = fit_nmf_dictionary(corpus_files, rng0)
    print(f"binaries: {len(binaries)} | corpus: {len(corpus_files)} files")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    mpath = outdir / "manifest.json"
    manifest = {}
    if mpath.exists():
        try:
            with open(mpath) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    for i in range(args.pieces):
        pseed = args.seed + 7919 * i + 61
        prng = np.random.default_rng(pseed)
        bpath = Path(binaries[i % len(binaries)])
        if not bpath.exists():
            print(f"  skipping missing {bpath}")
            continue
        target_s = float(prng.uniform(args.min_len, args.max_len))
        events, meta = render_piece(prng, bpath, corpus_files, W_dict,
                                    target_s, args)
        if not events:
            print(f"  bin{i:02d}: no events, skipped")
            continue
        L = max(t + len(y) for t, y in events)
        audio = np.zeros((int(target_s * SR) + SR, 2), np.float32)
        for t, y in events:
            end = min(len(audio), t + len(y))
            if t < len(audio):
                audio[t:end] += y[: end - t]
        audio = audio[: L]
        f = int(0.02 * SR)
        audio[:f] *= np.linspace(0, 1, f)[:, None]
        audio[-f:] *= np.linspace(1, 0, f)[:, None]
        pk = float(np.abs(audio).max())
        if pk > 1e-4:
            audio *= 0.97 / pk
        stem = bpath.name.replace(".", "_")[:40]
        wav = outdir / f"bin{i:02d}_{stem}_{pseed}.wav"
        gl.write_wav(wav, SR, audio)
        print(f"  bin{i:02d} {bpath.name:32s} ({len(u8_of(bpath))//1024}KB, "
              f"entropy {meta['entropy_mean']:.2f}, rule {meta['ca_rule']:3d}, "
              f"{len(meta['sections'])} sections, "
              f"{len(meta['voices'])} voices) -> {wav.name}")
        meta["seed"] = pseed
        manifest[str(wav)] = meta

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"done -> {outdir}/")


def u8_of(p):
    with open(p, "rb") as f:
        return np.frombuffer(f.read(), np.uint8)


if __name__ == "__main__":
    sys.exit(main())