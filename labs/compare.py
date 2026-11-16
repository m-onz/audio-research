"""compare — does the Pd offline render match the python preview?

    python3 compare.py <python_preview.wav> <pd_render.wav>

The two renderers differ (python synthesises the sub and applies per-hit
stochastic processing + a master bus; Pd plays the samples through its own
voices and sub synth), so a sample-exact match is neither expected nor the
point. What we validate is that the PATCH PLAYS THE SAME IDEA: same rhythmic
structure (onsets land together) and same broad timbre (aligned spectra
correlate). The tool auto-aligns (Pd has record-start latency), then reports:

  • onset-envelope correlation   — are the hits in the same places?  (structure)
  • mel-spectral correlation     — does it sound broadly the same?   (timbre)
  • onset count ratio            — same number of events?

and an overall PASS/WARN/FAIL.
"""
import sys

import numpy as np
import soundfile as sf

SR = 44100
HOP = 256


def _mono(path):
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    m = x.mean(axis=1)
    if sr != SR:
        import scipy.signal as ss
        m = ss.resample_poly(m, SR, sr)
    return m


def _env(m):
    n = len(m) // HOP
    e = np.sqrt((m[: n * HOP].reshape(n, HOP) ** 2).mean(axis=1) + 1e-12)
    return e


def _best_lag(a, b, max_lag):
    """Lag (in env frames) that best aligns b to a, via normalised xcorr."""
    a0 = (a - a.mean()) / (a.std() + 1e-9)
    b0 = (b - b.mean()) / (b.std() + 1e-9)
    xc = np.correlate(a0, b0, "full") / max(len(a0), len(b0))
    center = len(b0) - 1
    lo, hi = center - max_lag, center + max_lag
    seg = xc[max(0, lo): hi + 1]
    return (max(0, lo) + int(np.argmax(seg))) - center


def onset_times(m, thr=1.0):
    """Onset times (s) via half-wave-rectified spectral flux peak-picking."""
    win = 1024
    S = np.array([np.abs(np.fft.rfft(m[i:i + win] * np.hanning(win)))
                  for i in range(0, len(m) - win, HOP)])
    if len(S) < 3:
        return np.zeros(0)
    flux = np.sqrt((np.clip(np.diff(S, axis=0), 0, None) ** 2).sum(axis=1))
    flux /= (flux.max() + 1e-9)
    t = flux.mean() + thr * flux.std()
    gap = max(1, int(0.03 * SR / HOP))
    on = []
    for i in range(1, len(flux) - 1):
        if flux[i] > t and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            if on and i - on[-1] < gap:
                continue
            on.append(i)
    return np.array(on) * HOP / SR


def onset_f1(ta, tb, tol=0.05):
    """Precision/recall/F1 of onset-time agreement within `tol` seconds."""
    if len(ta) == 0 or len(tb) == 0:
        return 0.0, 0.0, 0.0
    used = np.zeros(len(tb), bool)
    hit = 0
    for t in ta:
        d = np.abs(tb - t)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True
            hit += 1
    prec = hit / len(tb)
    rec = hit / len(ta)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1


def _onset_lag(ta, tb, dur, max_lag=1.5, res=0.005):
    """Best shift (s) to add to tb so its onset train aligns with ta's."""
    n = int(dur / res) + 1
    ia = np.zeros(n); ib = np.zeros(n)
    ia[np.clip((ta / res).astype(int), 0, n - 1)] = 1
    ib[np.clip((tb / res).astype(int), 0, n - 1)] = 1
    xc = np.correlate(ia, ib, "full")
    center = len(ib) - 1
    k = int(max_lag / res)
    seg = xc[center - k: center + k + 1]
    return (int(np.argmax(seg)) - k) * res


def _align(ea, eb):
    lag = _best_lag(ea, eb, max_lag=int(1.5 * SR / HOP))
    if lag > 0:
        eb2, ea2 = eb, ea[lag:]
    else:
        eb2, ea2 = eb[-lag:], ea
    n = min(len(ea2), len(eb2))
    return ea2[:n], eb2[:n], lag


def _mel_corr(a, b, lag):
    import librosa
    s = int(abs(lag) * HOP)
    if lag > 0:
        a = a[s:]
    else:
        b = b[s:]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < SR // 2:
        return 0.0
    Ma = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=SR, n_mels=40))
    Mb = librosa.power_to_db(librosa.feature.melspectrogram(y=b, sr=SR, n_mels=40))
    k = min(Ma.shape[1], Mb.shape[1])
    va, vb = Ma[:, :k].ravel(), Mb[:, :k].ravel()
    va, vb = va - va.mean(), vb - vb.mean()
    return float((va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) < 2:
        print(__doc__); return 1
    a, b = _mono(argv[0]), _mono(argv[1])
    ta, tb = onset_times(a), onset_times(b)
    if len(ta) == 0 or len(tb) == 0:
        print("  one render has no onsets — FAIL"); return 2
    # coarse-align on the first onset (removes Pd's folder-load lead-in), then
    # refine within a small window; a step is 80 ms so match within one step
    off = tb[0] - ta[0]
    ta0, tb0 = ta - ta[0], tb - tb[0]
    lag = _onset_lag(ta0, tb0, max(ta0[-1], tb0[-1]) + 0.1, max_lag=0.25)
    prec, rec, f1 = onset_f1(ta0, tb0 + lag, tol=0.08)
    lag_frames = int(round((off - lag) * SR / HOP))
    mel = _mel_corr(a, b, lag_frames)
    ioi_a = float(np.median(np.diff(ta))) if len(ta) > 2 else 0.0
    ioi_b = float(np.median(np.diff(tb))) if len(tb) > 2 else 0.0
    # same grid even if one render is ~2x denser (python adds the sine-sub per
    # bass hit + per-hit processing); accept integer-ratio onset densities
    r = (ioi_a / ioi_b) if ioi_b > 0 else 0.0
    tempo_ok = any(abs(r - k) < 0.2 or abs(r - 1.0 / k) < 0.1 for k in (1, 2, 3))
    dur_ok = abs(len(a) - len(b)) / SR < 0.3

    print(f"  python : {len(a)/SR:.2f}s   pd : {len(b)/SR:.2f}s   "
          f"(onset-aligned lag {lag:+.2f}s{'  OK' if abs(lag) <= 0.15 else '  !!'})")
    print(f"  tempo (median onset gap) : py {ioi_a*1000:.0f}ms / pd {ioi_b*1000:.0f}ms"
          f"   {'MATCH' if tempo_ok else 'MISMATCH'}")
    print(f"  rhythm recall            : {rec:.2f}  (python hits present in Pd)"
          f"   [precision {prec:.2f}]")
    print(f"  mel-spectral correlation : {mel:+.3f}   (timbre)")
    score = 0.45 * rec + 0.3 * max(0, mel) + 0.15 * tempo_ok + 0.10 * dur_ok
    verdict = "PASS" if (rec >= 0.6 and abs(lag) <= 0.15 and tempo_ok and mel >= 0.2) \
        else "WARN" if (rec >= 0.4 and tempo_ok) else "FAIL"
    print(f"  --> {verdict}   (score {score:.2f})   "
          f"[same idea/tempo/timing; not sample-identical by design]")
    return 0 if verdict != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
