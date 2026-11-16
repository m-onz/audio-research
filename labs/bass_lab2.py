"""bass_lab2 — original, generative BASS one-shots.

Four new synthesis primitives, built as small composable modules, plus a
generative PATCHER that algorithmically wires them into novel chains:

  SOURCES  sub_osc      harmonically-rich sub oscillator
           psola        PSOLA-lite pitch-FLATTEN of the donor to a sub note
                        (keeps the train's grain/formant, makes it pitched)
           grain        granular spray of the donor (texture, unpitched)
           transient    the donor's attack used as a physical EXCITER strike
           noise        filtered noise burst exciter

  CORES    modal        bank of tuned resonant modes struck by the source
                        (harmonic / inharmonic-bar / stretched / rail ratios)
                        -> "struck-metal" bass  [technique #1]
           karplus      Karplus-Strong filtered delay loop tuned to the note
           comb         oneshot_lab.spectral.comb_bank (existing)
           duffing      chaotic nonlinear self-oscillator forced by the source
                        -> growling, never-repeating bass  [technique #3]
           none         pass the source through (for already-pitched sources)

  SHAPES   bode         single-sideband (Bode) frequency shift -> inharmonic
                        / barber-pole / dragged-down sub  [technique #2]
           wavefold     envelope-driven wavefolding (harmonics bloom with the
                        donor's motion)
           difftone     ring-mod difference-tone bass from the donor's highs
           drive        tanh saturation
           sweep        acid resonant low-pass sweep (from bass_lab)

The PATCHER picks SOURCE -> (CORE) -> 0..2 SHAPES -> FINALIZE, with light typing
(exciter sources always get a resonator so the result is pitched) and a FINALIZE
stage that guarantees sub weight, tames fizz, envelopes and normalizes. Every
patch + its params is logged to _manifest, so novel combinations are reproducible.

    python3 bass_lab2.py                      # pick 40 donors, 4 patches each
    python3 bass_lab2.py --pick 24 --variants 6 --sources processed
    python3 bass_lab2.py --patch modal        # force a module into every patch
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import hilbert, lfilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # oneshot_lab + echo live at the repo root

from oneshot_lab import core, spectral                     # noqa: E402
import bass_lab as bl                                       # noqa: E402
import hifi                                                  # noqa: E402

SR = 44100
OUT = HERE / "m-onz_bass2"


# ================================================================= helpers

def _norm(y, p=0.99):
    y = np.nan_to_num(np.asarray(y, np.float32))
    m = float(np.max(np.abs(y)))
    return y * (p / m) if m > 1e-9 else y


def _env(m, sr, win_s=0.02):
    return core.rms_env(np.asarray(m, np.float32), sr, win_s)


def _rbj_bandpass(f, Q, sr):
    f = float(np.clip(f, 20.0, sr * 0.45))
    w0 = 2 * np.pi * f / sr
    a = np.sin(w0) / (2.0 * max(0.5, Q))
    cw = np.cos(w0)
    a0 = 1 + a
    return (np.array([a, 0.0, -a]) / a0,
            np.array([1.0, -2 * cw / a0, (1 - a) / a0]))


# ================================================================= SOURCES

def src_sub_osc(donor, base_f, sr, dur, rng):
    n = int(dur * sr)
    t = np.arange(n) / sr
    vib = 1.0 + 0.004 * np.sin(2 * np.pi * rng.uniform(3, 6) * t)
    f = base_f * vib
    ph01 = (np.cumsum(f) / sr) % 1.0
    dt = f / sr
    ph = 2 * np.pi * ph01
    kind = rng.choice(["saw", "fm", "square"])
    if kind == "saw":
        y = hifi.blsaw(ph01, dt)                             # band-limited
    elif kind == "square":
        y = hifi.blsquare(ph01, dt)                          # band-limited
    else:
        ratio = float(rng.choice([1.0, 2.0, 1.5, 3.0]))
        idx = rng.uniform(2, 6) * np.exp(-3 * t / max(dur, 1e-3))
        y = np.sin(ph + idx * np.sin(ratio * ph))
    y += 0.6 * np.sin(2 * np.pi * np.cumsum(base_f * vib) / sr)   # sub layer
    return _norm(y)


def src_psola(donor, base_f, sr, dur, rng):
    """PSOLA-lite: lay donor grains at the target period -> pitched to base_f
    while preserving the donor's grain texture; read-pointer traverses the
    whole donor so its evolution is kept."""
    m = np.asarray(donor, np.float32)
    Lin, N = len(m), int(dur * sr)
    P = max(8, int(sr / base_f))
    gl = 2 * P
    win = np.hanning(gl).astype(np.float32)
    out = np.zeros(N + gl, np.float32)
    ne = max(1, N // P)
    for e in range(ne):
        frac = e / max(1, ne - 1)
        c = int(frac * (Lin - 1))
        a, b = max(0, c - P), min(Lin, c + P)
        g = np.zeros(gl, np.float32)
        g[: b - a] = m[a:b]
        out[e * P: e * P + gl] += g * win
    return _norm(out[:N])


def src_grain(donor, base_f, sr, dur, rng):
    m = np.asarray(donor, np.float32)
    Lin, N = len(m), int(dur * sr)
    out = np.zeros(N, np.float32)
    gl = int(rng.uniform(0.02, 0.07) * sr)
    win = np.hanning(gl).astype(np.float32)
    hop = max(1, int(gl * rng.uniform(0.3, 0.6)))
    for t0 in range(0, N - gl, hop):
        s = int(rng.integers(0, max(1, Lin - gl)))
        out[t0: t0 + gl] += m[s: s + gl] * win
    return _norm(out)


def src_transient(donor, base_f, sr, dur, rng):
    """The donor's loudest attack, windowed — a physical exciter 'mallet'."""
    m = np.asarray(donor, np.float32)
    e = _env(np.abs(m), sr, 0.005)
    c = int(np.argmax(e))
    L = int(rng.uniform(0.04, 0.11) * sr)
    a = max(0, c - int(0.004 * sr))
    seg = m[a: a + L].copy()
    if len(seg) < 8:
        seg = m[: max(8, L)].copy()
    seg = seg * np.hanning(len(seg)).astype(np.float32)
    N = int(dur * sr)
    out = np.zeros(N, np.float32)
    out[: len(seg)] = seg
    if rng.random() < 0.5:                       # re-strike train (rattle)
        P = max(len(seg), int(sr / base_f) * int(rng.integers(2, 6)))
        g = 0.7
        for t0 in range(P, N - len(seg), P):
            out[t0: t0 + len(seg)] += seg * g
            g *= 0.6
    return _norm(out)


def src_noise(donor, base_f, sr, dur, rng):
    N = int(dur * sr)
    x = rng.standard_normal(N).astype(np.float32)
    b, a = _rbj_bandpass(base_f * rng.uniform(3, 10), rng.uniform(0.7, 2.0), sr)
    x = lfilter(b, a, x).astype(np.float32)
    x *= np.exp(-rng.uniform(6, 20) * np.linspace(0, 1, N)).astype(np.float32)
    return _norm(x)


SOURCES = {"sub_osc": src_sub_osc, "psola": src_psola, "grain": src_grain,
           "transient": src_transient, "noise": src_noise}
EXCITERS = {"grain", "transient", "noise"}        # need a resonator to pitch


# ================================================================= CORES

_MODE_RATIOS = {
    "harmonic": [1, 2, 3, 4, 5, 6, 7, 8],
    "bar":      [1, 2.76, 5.40, 8.93, 13.34, 18.64],       # free-free bar
    "stretched": None,                                      # computed
    "rail":     None,                                       # random inharmonic
}


def core_modal(x, base_f, sr, rng):
    which = rng.choice(list(_MODE_RATIOS))
    if which == "stretched":
        B = rng.uniform(0.0008, 0.004)
        ratios = [k * np.sqrt(1 + B * k * k) for k in range(1, 9)]
    elif which == "rail":
        ratios = sorted(float(r) for r in (1.0, *rng.uniform(1.6, 16.0, 6)))
    else:
        ratios = _MODE_RATIOS[which]
    Qb = rng.uniform(60, 500)                     # higher -> longer ring
    out = np.zeros_like(x)
    for k, r in enumerate(ratios):
        f = base_f * r * rng.uniform(0.997, 1.003)
        if f > sr * 0.45:
            break
        b, a = _rbj_bandpass(f, Qb / (1 + 0.25 * k), sr)
        out += (1.0 / (k + 1)) * lfilter(b, a, x).astype(np.float32)
    return _norm(out)


def core_karplus(x, base_f, sr, rng):
    N = max(2, int(sr / base_f))
    fb = rng.uniform(0.985, 0.999)
    y = np.asarray(x, np.float64).copy()
    if len(y) <= N + 1:
        return _norm(y)
    for n in range(N + 1, len(y)):               # loop LP = KS decay
        y[n] += fb * 0.5 * (y[n - N] + y[n - N - 1])
    return _norm(y)


def core_comb(x, base_f, sr, rng):
    x2 = np.asarray(x, np.float32).reshape(-1, 1)       # comb_bank wants (n, ch)
    y = spectral.comb_bank(x2, sr, rng,
                           n_combs=int(rng.integers(4, 8)),
                           feedback=rng.uniform(0.6, 0.85),
                           f0=base_f, mix=rng.uniform(0.5, 0.85))
    return _norm(core.mono(y) if y.ndim > 1 else y)


def core_duffing(x, base_f, sr, rng):
    """Forced Duffing oscillator integrated at audio rate — the source drives
    a nonlinear system that self-oscillates/growls in the bass register."""
    w0 = 2 * np.pi * base_f
    chaos = rng.random() < 0.5
    a = -(w0 * w0) if chaos else (w0 * w0)
    bcoef = (w0 * w0) * rng.uniform(0.3, 1.2)
    delta = rng.uniform(2.0, 30.0)
    gamma = rng.uniform(0.6, 2.0) * (w0 * w0)
    dt = 1.0 / sr
    xx = vv = 0.0
    xin = np.asarray(x, np.float64)
    out = np.zeros(len(xin))
    for n in range(len(xin)):
        acc = -delta * vv - a * xx - bcoef * xx * xx * xx + gamma * xin[n]
        vv += acc * dt
        xx += vv * dt
        if xx > 3.0: xx, vv = 3.0, 0.0
        elif xx < -3.0: xx, vv = -3.0, 0.0
        out[n] = xx
    return _norm(out)


CORES = {"modal": core_modal, "karplus": core_karplus, "comb": core_comb,
         "duffing": core_duffing, "none": lambda x, *a, **k: x}


# ================================================================= SHAPES

def shape_bode(x, donor, base_f, sr, rng):
    n = len(x)
    an = hilbert(np.asarray(x, np.float64))
    t = np.arange(n) / sr
    df0 = rng.uniform(-90, 60)
    if rng.random() < 0.4:                        # barber-pole glide
        df = df0 + rng.uniform(-40, 40) * t / max(t[-1], 1e-6)
    else:
        df = df0
    y = np.real(an * np.exp(1j * 2 * np.pi * df * t))
    return _norm(y.astype(np.float32))


def shape_wavefold(x, donor, base_f, sr, rng):
    env = _env(donor, sr, 0.02)
    env = np.interp(np.linspace(0, 1, len(x)),
                    np.linspace(0, 1, len(env)), env)
    env = env / (env.max() + 1e-9)
    depth = rng.uniform(1.0, 4.0)
    y = np.sin((1.0 + depth * env) * np.asarray(x, np.float32) * (np.pi / 2))
    return _norm(y)


def shape_difftone(x, donor, base_f, sr, rng):
    spec = np.abs(np.fft.rfft(np.asarray(donor, np.float32)
                              * np.hanning(len(donor)))) + 1e-9
    fr = np.fft.rfftfreq(len(donor), 1.0 / sr)
    cent = float((fr * spec).sum() / spec.sum())
    fmod = float(np.clip(cent - base_f, 80.0, 4000.0))
    t = np.arange(len(x)) / sr
    y = np.asarray(x, np.float32) * np.sin(2 * np.pi * fmod * t).astype(np.float32)
    return _norm(y)


def shape_drive(x, donor, base_f, sr, rng):
    return bl._drive(np.asarray(x, np.float32), float(rng.uniform(0.3, 0.8)))


def shape_sweep(x, donor, base_f, sr, rng):
    n = len(x)
    mode = rng.choice(["down", "wobble", "follow", "up"])
    cut = bl.cutoff_envelope(mode, n, sr, base_f, _env(donor, sr, 0.02), rng)
    return _norm(bl.resonant_sweep(np.asarray(x, np.float32), sr, cut,
                                   float(rng.uniform(3.5, 10.0))))


SHAPES = {"bode": shape_bode, "wavefold": shape_wavefold,
          "difftone": shape_difftone, "drive": shape_drive, "sweep": shape_sweep}


# ================================================================= FINALIZE

def finalize(y, donor, base_f, sr, rng):
    y = np.nan_to_num(np.asarray(y, np.float32))
    if y.ndim > 1:
        y = core.mono(y)
    n = len(y)
    # tame fizz: static low-pass well above the bass
    b, a = bl._rbj_lowpass(base_f * rng.uniform(40, 90), 0.7, sr)
    y = lfilter(b, a, y).astype(np.float32)
    env_mode = "sustain" if rng.random() < 0.4 else "pluck"
    aenv = bl.amp_envelope(env_mode, n, sr, rng)
    y = _norm(y) * aenv
    t = np.arange(n) / sr
    sub = np.sin(2 * np.pi * base_f * t).astype(np.float32) * aenv
    y = _norm(y) * 0.8 + 0.5 * sub
    y = hifi.dc_block(y, sr)
    # stereo: tiny inter-channel delay + detune for width
    d = int(rng.uniform(0.002, 0.009) * sr)
    R = np.concatenate([np.zeros(d, np.float32), y])[: n]
    st = bl._declick(np.stack([y, R * 0.95], axis=1), sr)
    return hifi.master(st, sr, drive=float(rng.uniform(1.3, 2.0)))


# ================================================================= PATCHER

def build_patch(rng, force=None):
    """Algorithmically wire a novel chain with light typing."""
    src = str(rng.choice(list(SOURCES)))
    if src in EXCITERS:
        core_name = str(rng.choice([c for c in CORES if c != "none"]))
    else:
        core_name = str(rng.choice(list(CORES)))
    n_shape = int(rng.integers(0, 3))
    shapes = list(rng.choice(list(SHAPES), size=n_shape, replace=False)) \
        if n_shape else []
    # honour a forced module: make sure it appears somewhere
    if force:
        if force in SOURCES:
            src = force
            core_name = core_name if src not in EXCITERS or core_name != "none" \
                else str(rng.choice([c for c in CORES if c != "none"]))
        elif force in CORES:
            core_name = force
        elif force in SHAPES and force not in shapes:
            shapes = ([force] + shapes)[:3]
    return dict(source=src, core=core_name, shapes=shapes)


def render_patch(patch, donor, base_f, sr, dur, rng):
    y = SOURCES[patch["source"]](donor, base_f, sr, dur, rng)
    y = CORES[patch["core"]](y, base_f, sr, rng)
    for s in patch["shapes"]:
        y = SHAPES[s](y, donor, base_f, sr, rng)
    return finalize(y, donor, base_f, sr, rng)


def make(donor_path, variant, seed, force=None, uid=""):
    rng = np.random.default_rng(seed)
    x, sr = sf.read(str(donor_path), always_2d=True, dtype="float32")
    mod = x.mean(axis=1)
    if sr != SR:
        mod = core.resample(mod, sr / float(SR)); sr = SR
    f0, conf = core.detect_pitch(mod, sr)
    base_f = bl.pick_root(rng, f0, conf, lo=36.0, hi=120.0)
    dur = float(np.clip(len(mod) / sr, 0.9, 2.2))
    patch = build_patch(rng, force=force)
    y = render_patch(patch, mod, base_f, sr, dur, rng)
    tag = f"{patch['source']}-{patch['core']}"
    if patch["shapes"]:
        tag += "-" + "+".join(patch["shapes"])
    name = f"{uid}{Path(donor_path).stem[:36]}__{tag}__v{variant}.wav"
    OUT.mkdir(parents=True, exist_ok=True)
    hifi.write(OUT / name, y, sr)
    return dict(out=str(OUT / name), donor=str(donor_path),
                base_f=round(base_f, 2), dur_s=round(len(y) / sr, 2), **patch)


# ================================================================= main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="both",
                    choices=["raw", "processed", "both"])
    ap.add_argument("--pick", type=int, default=40)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--max-scan", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--patch", default=None,
                    help="force a module name into every patch "
                         "(e.g. modal, duffing, bode, psola)")
    args = ap.parse_args(argv)

    srcs = ["raw", "processed"] if args.sources == "both" else [args.sources]
    picks = bl.listen_and_pick(srcs, args.pick, args.max_scan, args.seed)
    print(f"\n{len(picks)} bass donors chosen; rendering "
          f"{len(picks) * args.variants} generative bass patches ...")

    OUT.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    total = len(picks) * args.variants
    for pi, r in enumerate(picks):
        for v in range(args.variants):
            try:
                recs.append(make(r["path"], v, args.seed * 100003 + pi * 37 + v,
                                 force=args.patch, uid=f"{pi:03d}_"))
            except Exception as e:
                print(f"  ! {Path(r['path']).name} v{v}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
        done = (pi + 1) * args.variants
        if done % 40 == 0 or pi == len(picks) - 1:
            print(f"  {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    from collections import Counter
    print("\npatch distribution:")
    for key in ("source", "core"):
        print(f"  {key}: {dict(Counter(r[key] for r in recs))}")
    print(f"  shape-usage: {dict(Counter(s for r in recs for s in r['shapes']))}")
    with open(OUT / "_manifest.json", "w") as fh:
        json.dump(recs, fh, indent=1)
    print(f"\ndone: {len(recs)} generative bass one-shots -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
