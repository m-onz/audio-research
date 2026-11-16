"""ears: the listening layer — restructured around the bomb insight.

Two different things were tangled before:

  DEFECTS   clicks, pops, leading silence, DC, digital clipping, NaN —
            objective technical failures. ALWAYS rejected. No dial.
  CHARACTER  noisiness, fuzz, mud, distortion, brightness, smear —
            TASTE. A distorted reese is interesting; a smeared pad can
            be interesting. Character must be MEASURED, not banned.

So:

  strictness (0..1, evolvable per genome + global CLI flag) is bomb's
  judge dial — at 0 only defects are rejected (open-ended), at 1 the
  character gates close to the tightest settings. Thresholds lerp.

  taste (riffs/_taste.json) is bomb's corpus-envelope calibration: you
  drop sounds you like into riffs/_taste/good/ and ones you reject into
  riffs/_taste/bad/, run `python -m riff_lab judge --fit`, and the
  per-metric thresholds that separate your good from your bad become
  the gates — the filters learn your ears instead of mine.
"""
import json
from pathlib import Path

import numpy as np

from . import core

TASTE_PATH = Path("riffs/_taste.json")

# character metrics measured per sound (names matter: taste fitting and
# the novelty descriptor use them)
CHAR_METRICS = ("hnr", "kurt", "harm_c", "boom_e", "bass_e", "centroid",
                "flatness", "atk_ms", "decay_s", "sustain")

# character preference per voice: metric -> {"min": (open, tight)} or
# {"max": (open, tight)} — the threshold LERPs from open (strictness 0)
# to tight (strictness 1). Taste overrides replace the pair.
_CHAR_RANGES = {
    "bass": {
        "harm_c":    {"min": (0.00, 0.35)},
        "hnr":       {"min": (0.25, 0.75)},
        "boom_e":    {"max": (0.95, 0.60)},
        "atk_ms":    {"max": (400.0, 100.0)},
        "centroid":  {"max": (4500, 2200)},
    },
    "noise": {
        "atk_ms":    {"max": (200.0, 30.0)},
        "decay_s":   {"max": (1.30, 0.50)},
        "sustain":   {"max": (1.00, 0.45)},
    },
    "fill": {
        "atk_ms":    {"max": (250.0, 45.0)},
        "decay_s":   {"max": (1.60, 0.80)},
        "flatness":  {"max": (0.85, 0.30)},
    },
}


def _lerp(a, b, s):
    return a + (b - a) * float(np.clip(s, 0.0, 1.0))


def measure(x, sr, f0_hint=0.0):
    """Full character vector of a sound (defect checks share the pass)."""
    from .pool import (_attack_ms, _bands_rolloff, _boom_e, _decay_shape,
                       _harm_conc, _hnr_kurt)
    m = core.mono(x)
    hnr, kurt = _hnr_kurt(m, sr)
    harm_c = _harm_conc(m, sr, f0_hint) if f0_hint > 0 else 0.0
    bass_e, air_e, rolloff = _bands_rolloff(m, sr)
    boom = _boom_e(m, sr)
    atk = _attack_ms(m, sr)
    decay_s, sustain = _decay_shape(m, sr)
    S = np.abs(np.fft.rfft(m[: min(len(m), sr)] *
                           np.hanning(min(len(m), sr))))
    fr = np.fft.rfftfreq(len(m[: min(len(m), sr)]), 1.0 / sr)
    cent = float(np.sum(fr * S) / (np.sum(S) + 1e-9))
    flat = float(np.exp(np.mean(np.log(S + 1e-9))) / (np.mean(S) + 1e-9))
    return dict(hnr=hnr, kurt=kurt, harm_c=harm_c, boom_e=boom,
                bass_e=bass_e, centroid=cent, flatness=flat,
                atk_ms=atk, decay_s=decay_s, sustain=sustain)


# ----------------------------------------------------------------- defects

def defects(x, sr):
    """Objective technical failures. These are not taste — they are
    broken audio. Returns a list of defect names (empty = sound)."""
    out = []
    y = np.asarray(x, np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] < 256 or not np.all(np.isfinite(y)):
        return ["degenerate"]
    m = np.abs(y).mean(axis=1)
    pk = float(m.max())
    if pk < 1e-4:
        return ["silent"]
    # leading silence (1% of peak; the click-free fade ramp is not silence)
    above = np.flatnonzero(m > 0.01 * pk)
    lead_ms = (above[0] / sr * 1000.0) if len(above) else 9999.0
    if lead_ms > 15.0:
        out.append("lead_in")
    # clicks: an ISOLATED step discontinuity at the head — a real click
    # is a spike that dwarfs its immediate neighbors; a fast ATTACK has
    # large first-differences all around it (oscillation). DNB5-style
    # punchy synth transients are character, not defects.
    d1 = np.abs(np.diff(y, axis=0)).max(axis=1)
    head = int(0.012 * sr)
    early = float(d1[: head].max())
    body = float(np.median(d1[int(0.02 * sr):])) if y.shape[0] > int(0.02 * sr) else 1e-9
    if early > 0.25 and early > 25.0 * max(body, 1e-9):
        i = int(np.argmax(d1[: head]))
        lo = max(0, i - 60)
        hi = min(len(d1), i + 60)
        neigh = np.delete(d1[lo: hi], np.s_[max(0, i - lo - 4): i - lo + 5])
        neigh_med = float(np.median(neigh)) if len(neigh) else 0.0
        if early > 6.0 * max(neigh_med, 1e-9):
            out.append("click")
    # hard digital clipping (flat-topped waveform runs)
    ym = y.mean(axis=1)
    pkf = float(np.abs(ym).max())
    if pkf > 0.995:
        clipped = np.abs(np.abs(ym) - pkf) < 0.002
        if clipped.sum() > int(0.01 * sr):
            out.append("clipped")
    # DC offset
    dc = float(np.mean(ym))
    if abs(dc) > 0.05 * pk:
        out.append("dc")
    return out


# ----------------------------------------------------------------- judging

def character_ok(c, voice, strictness, taste=None):
    """Soft character gates. strictness 0 = wide open (only defects
    reject), 1 = tightest. taste (from _taste.json) overrides the
    learned-from-you thresholds for any metric it covers."""
    ranges = _CHAR_RANGES.get(voice, {})
    for met, bounds in ranges.items():
        if met not in c:
            continue
        v = c[met]
        lo_v = hi_v = None
        if "min" in bounds:
            lo_v = _lerp(bounds["min"][0], bounds["min"][1], strictness)
        if "max" in bounds:
            hi_v = _lerp(bounds["max"][0], bounds["max"][1], strictness)
        if taste and voice in taste.get("voices", {}):
            t = taste["voices"][voice].get(met)
            if t and len(t) == 2:
                lo_v, hi_v = t[0], t[1]
        if lo_v is not None and v < lo_v:
            return False
        if hi_v is not None and v > hi_v:
            return False
    return True


def passes(x, sr, voice, strictness=0.35, f0_hint=0.0, taste=None):
    """The single judgement: hard defects always reject; character gates
    scale with strictness (+ taste overrides)."""
    if defects(x, sr):
        return False
    if strictness <= 0.01:
        return True
    c = measure(x, sr, f0_hint)
    return character_ok(c, voice, strictness, taste)


def load_taste():
    if TASTE_PATH.exists():
        try:
            return json.load(open(TASTE_PATH))
        except Exception:
            return None
    return None


# ----------------------------------------------------------------- taste
# fitting (bomb's corpus-envelope calibration, one-class flavored)

def fit_taste(strictness=0.5):
    """Learn thresholds from your folders: riffs/_taste/good/ (sounds you
    kept) and riffs/_taste/bad/ (sounds you rejected). For every metric,
    find the [lo, hi] window that covers ~85% of good while excluding
    as much bad as possible — the separating envelope."""
    good_files = sorted(Path("riffs/_taste/good").glob("*.wav"))
    bad_files = sorted(Path("riffs/_taste/bad").glob("*.wav"))
    if len(good_files) < 4:
        print("need >= 4 sounds in riffs/_taste/good/ to fit "
              f"(have {len(good_files)})")
        return None
    import soundfile as sf

    def vecs(files):
        out = []
        for f in files:
            try:
                y, sr = sf.read(f, always_2d=True, dtype="float32")
                c = measure(y, sr)
                out.append(c)
            except Exception:
                pass
        return out

    good, bad = vecs(good_files), vecs(bad_files)
    voices = {}
    # sounds are not labeled by voice here — fit a single envelope and
    # apply it to all voices as a baseline taste
    env = {}
    for met in CHAR_METRICS:
        g = np.array([c[met] for c in good])
        b = np.array([c[met] for c in bad]) if bad else np.array([])
        lo, hi = float(np.percentile(g, 8)), float(np.percentile(g, 92))
        if len(b):
            # widen only if bad examples sit inside the envelope
            b_in = float(((b >= lo) & (b <= hi)).mean())
            if b_in > 0.25 and len(g) > 7:
                lo = float(np.percentile(g, 2))
                hi = float(np.percentile(g, 98))
        env[met] = [round(lo, 4), round(hi, 4)]
    for voice in ("bass", "noise", "fill"):
        voices[voice] = dict(env)
    taste = dict(n_good=len(good), n_bad=len(bad),
                 strictness_hint=strictness, voices=voices)
    TASTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TASTE_PATH, "w") as fh:
        json.dump(taste, fh, indent=1)
    print(f"taste fitted from {len(good)} good / {len(bad)} bad -> "
          f"{TASTE_PATH}")
    for met in CHAR_METRICS:
        lo, hi = env[met]
        print(f"  {met:<10} [{lo:.3g}, {hi:.3g}]")
    return taste