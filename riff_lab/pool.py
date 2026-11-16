"""pool: the source-sound archive — scan your ./sounds corpus, analyze, and
classify every one-shot into bass / noise / fill buckets (by its spectral
features, NOT by folder name), cached to riffs/_pool.json.

Point it at your own corpus of one-shots / short samples by dropping wavs
anywhere under ./sounds (it is scanned recursively). Override the location
with the AUDIO_RESEARCH_SOUNDS environment variable."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import core

CACHE = Path("riffs/_pool.json")

# the source corpus: every .wav under here is a candidate one-shot. Classified
# into roles by analysis, so no particular folder layout is required.
SOUNDS = Path(os.environ.get("AUDIO_RESEARCH_SOUNDS", "sounds"))
# cap the scan so a huge corpus still builds quickly; raise it (or set the env
# var to 0) to mine deeper.
SCAN_CAP = int(os.environ.get("AUDIO_RESEARCH_SOUNDS_CAP", "4000"))


def _hnr_kurt(m, sr):
    """Machine-listening bass quality: harmonic-to-noise ratio and
    spectral kurtosis, measured over active STFT frames.

    HNR: energy in extracted spectral peaks vs total energy — a growly
    but HARMONIC bass scores high; a noise-wall scores low.
    Kurtosis: spiky harmonic spectra are heavy-tailed (high); noisy
    spectra are flat (low)."""
    n = len(m)
    win, hop = (4096, 1024) if n >= 8192 else (1024, 512)  # resolve sub f0
    if n < win * 2:
        return 0.0, 0.0
    e = np.sqrt(np.convolve(m ** 2, np.ones(512) / 512, "same"))
    act = e > 0.15 * e.max()
    frames = [i for i in range(0, n - win, hop) if act[i + win // 2]]
    if not frames:
        frames = list(range(0, n - win, hop))
    frames = frames[: 64]
    hnr_acc, kur_acc = [], []
    for i in frames:
        S = np.abs(np.fft.rfft(m[i: i + win] * np.hanning(win)))
        if S.max() < 1e-6:
            continue
        tot = float((S ** 2).sum()) + 1e-12
        pk = (S[1:-1] > S[:-2]) & (S[1:-1] >= S[2:])
        idx = np.flatnonzero(pk) + 1
        if len(idx) == 0:
            hnr_acc.append(0.0)
        else:
            mask = np.zeros_like(S, bool)
            for j in idx:
                mask[max(0, j - 3): j + 4] = True
            hnr_acc.append(float((S[mask] ** 2).sum()) / tot)
        kur_acc.append(float(((S - S.mean()) ** 4).mean() /
                              (S.var() ** 2 + 1e-12)))
    if not hnr_acc:
        return 0.0, 0.0
    return float(np.mean(hnr_acc)), float(np.mean(kur_acc))


def _decay_shape(m, sr):
    """Punch vs sustain: time for the envelope to fall 20dB under peak,
    and how loud the tail stays relative to the attack."""
    w = max(2, int(0.02 * sr))
    nf = len(m) // w
    if nf < 3:
        return 9.9, 1.0
    e = np.sqrt(np.mean(m[: nf * w].reshape(nf, w) ** 2, axis=1) + 1e-12)
    pk_i = int(np.argmax(e))
    pk = float(e[pk_i])
    below = np.flatnonzero(e[pk_i:] < pk * 10 ** (-20 / 20))
    decay_s = float(below[0] * w / sr) if len(below) else \
        float((nf - pk_i) * w / sr)
    first_q = float(np.mean(e[pk_i: pk_i + max(1, nf // 4)]))
    tail_q = float(np.mean(e[max(pk_i, nf - max(1, nf // 3)):]))
    return decay_s, float(tail_q / (first_q + 1e-9))


def _boom_e(m, sr):
    """Boom-band concentration: energy 120-400Hz / total. The classic
    one-note resonance zone — a boomy, muddy 'bass' concentrates most of
    its energy there while a clean sub-bass spreads f0 + harmonics."""
    w = max(512, int(0.05 * sr))
    e = np.convolve(m ** 2, np.ones(w) / w, "same")
    i0 = max(0, int(np.argmax(e)) - int(0.2 * sr))
    seg = m[i0: i0 + int(0.5 * sr)]
    if len(seg) < 512:
        seg = m[: int(0.5 * sr)]
    if len(seg) < 512:
        return 0.0
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
    tot = float(S.sum()) + 1e-12
    return float(S[(fr >= 120) & (fr < 400)].sum() / tot)
def _attack_ms(m, sr):
    """10->90% rise time of the envelope — the punch measure. Rise time
    is immune to filter group delay (which smeared the old -50dB-to-peak
    measure on sub-bass), while slow warp swells still score high."""
    w = max(2, int(0.002 * sr))
    e = np.sqrt(np.convolve(m ** 2, np.ones(w) / w, "same") + 1e-12)
    pk = float(e.max())
    if pk < 1e-9:
        return 999.0
    pk_i = int(np.argmax(e))
    lo, hi = 0.10 * pk, 0.90 * pk
    below = np.flatnonzero(e[: pk_i + 1] <= lo)
    start = below[-1] if len(below) else 0
    above = np.flatnonzero(e[start:] >= hi)
    end = start + int(above[0]) if len(above) else pk_i
    return float((end - start) / sr * 1000.0)


def _harm_conc(m, sr, f0):
    """Harmonic concentration: fraction of spectral-peak energy sitting
    near integer multiples of the detected f0, over active frames.
    Kills warp-smear mud (ghost/wandering peaks score low)."""
    if f0 <= 0:
        return 0.0
    win, hop = (4096, 1024) if f0 < 150 else (1024, 512)   # resolve sub f0
    n = len(m)
    if n < win * 2:
        return 0.0
    e = np.sqrt(np.convolve(m ** 2, np.ones(512) / 512, "same"))
    frames = [i for i in range(0, n - win, hop)
              if e[i + win // 2] > 0.15 * e.max()][: 48]
    accs = []
    for i in frames:
        S = np.abs(np.fft.rfft(m[i: i + win] * np.hanning(win)))
        if S.max() < 1e-6:
            continue
        pk = (S[1:-1] > S[:-2]) & (S[1:-1] >= S[2:])
        idx = np.flatnonzero(pk) + 1
        if len(idx) == 0:
            continue
        amps = S[idx]
        top = idx[np.argsort(amps)[::-1][: 12]]
        tot_e = float((S[top] ** 2).sum()) + 1e-12
        fr = top * sr / win
        near = np.zeros(len(top), bool)
        bin_w = sr / win
        for k in range(1, 17):
            tol = max(12.0, bin_w, 0.04 * k * f0)   # FFT-resolution aware
            near |= np.abs(fr - k * f0) <= tol
        accs.append(float((S[top[near]] ** 2).sum()) / tot_e)
    return float(np.mean(accs)) if accs else 0.0


def _bands_rolloff(m, sr):
    """tomb-ears band ratios + 85% rolloff on the loudest 1s."""
    e = np.sqrt(np.convolve(m ** 2, np.ones(1024) / 1024, "same"))
    i0 = max(0, int(np.argmax(e)) - int(0.25 * sr))
    seg = m[i0: i0 + int(1.0 * sr)]
    if len(seg) < 512:
        return 0.0, 0.0, 0.0
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
    tot = float(S.sum()) + 1e-12
    bass_e = float(S[fr < 250].sum() / tot)
    air_e = float(S[fr >= 2000].sum() / tot)
    i85 = int(np.searchsorted(np.cumsum(S), 0.85 * float(S.sum())))
    rolloff = float(fr[min(i85, len(fr) - 1)])
    return bass_e, air_e, rolloff


def _analyze_one(path):
    try:
        x, sr = core.read_audio(path, max_in_s=6.0)
        a = core.analyze(x, sr)
        m = core.mono(x)
        f0, conf = core.detect_pitch(m, sr)
        hnr, kurt = _hnr_kurt(m, sr)
        decay_s, sustain = _decay_shape(m, sr)
        atk = _attack_ms(m, sr)
        harm_c = _harm_conc(m, sr, f0)
        bass_e, air_e, rolloff = _bands_rolloff(m, sr)
        boom = _boom_e(m, sr)
        return dict(path=str(path), dur=round(a["dur_s"], 2),
                    rms=round(a["rms"], 4), centroid=round(a["centroid"], 0),
                    flatness=round(a["flatness"], 4), crest=round(a["crest"], 2),
                    transient=round(a["transient"], 2),
                    f0=round(f0, 1), f0_conf=round(conf, 2),
                    hnr=round(hnr, 3), kurt=round(kurt, 1),
                    decay_s=round(decay_s, 2), sustain=round(sustain, 2),
                    atk_ms=round(atk, 1), harm_c=round(harm_c, 3),
                    bass_e=round(bass_e, 3), air_e=round(air_e, 3),
                    boom_e=round(boom, 3), rolloff=round(rolloff, 0))
    except Exception:
        return None


def _gather():
    rng = np.random.default_rng(7)
    if not SOUNDS.is_dir():
        raise SystemExit(
            f"no sound corpus at {SOUNDS!s} — drop some .wav one-shots under "
            f"./sounds (see sounds/README.md) or set AUDIO_RESEARCH_SOUNDS.")
    got = sorted(str(f) for f in SOUNDS.rglob("*.[wW][aA][vV]"))
    if SCAN_CAP and len(got) > SCAN_CAP:
        idx = rng.permutation(len(got))[: SCAN_CAP]
        got = [got[i] for i in idx]
    return got


def build(rebuild=False, workers=8):
    """The classified pool: {'bass': [...], 'noise': [...], 'fill': [...],
    'all': [...]} — built once, cached."""
    if CACHE.exists() and not rebuild:
        return json.load(open(CACHE))
    files = _gather()
    print(f"pool: analyzing {len(files)} sources...")
    recs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_analyze_one, files, chunksize=16):
            if r is not None:
                recs.append(r)
    pool = _classify(recs)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(pool, fh)
    print(f"pool: {len(recs)} analyzed -> bass {len(pool['bass'])}, "
          f"noise {len(pool['noise'])}, fill {len(pool['fill'])}")
    return pool


def _classify(recs):
    """Bucket every sound by its measured character. A sound may appear in
    more than one bucket; selection prefers the earliest hits."""
    bass, noise, fill = [], [], []
    for r in recs:
        pitched = r["f0_conf"] >= 0.30 and r["f0"] > 0
        if pitched and r["f0"] < 500 and r["centroid"] < 4500 \
                and 0.12 <= r["dur"] <= 8:
            bass.append(r)
        if (pitched and r["centroid"] >= 2500) or \
                (r["centroid"] >= 3500 and r["dur"] <= 1.2):
            fill.append(r)
        if r["flatness"] >= 0.22 or \
                (r["centroid"] >= 5000 and r["f0_conf"] < 0.35):
            noise.append(r)
    order = lambda xs: sorted(xs, key=lambda r: -r["rms"])
    return {"bass": order(bass), "noise": order(noise), "fill": order(fill),
            "all": order(recs)}


def pick(pool, bucket, n, rng, avoid=None):
    """Pick n distinct sounds from a bucket, rms-weighted toward the top of
    the bucket but always seeded-random."""
    xs = [r for r in pool[bucket]
          if not avoid or r["path"] not in avoid]
    if not xs:
        xs = list(pool["all"])
    n = min(n, len(xs))
    # weighted sample: top-40% of the bucket, seeded
    top = xs[: max(n * 2, len(xs) // 3)]
    idx = rng.permutation(len(top))[: n] if len(top) >= n else \
        np.arange(len(xs))[: n]
    return [top[i] for i in idx]