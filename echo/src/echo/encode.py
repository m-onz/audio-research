"""encode: the ear of the archive — audio to a dense temporal vector.

The question the archive answers is 'have we heard something like this
before?', and the encoder decides what 'like' means. Design:

    whole clip -> frames (46ms / 11.6ms hop)
    per frame:  24 MFCCs (64 mel bands, HTK)          <- timbre, perceptual
                10 scalars: rms, crest, zcr, centroid, bandwidth,
                flatness, rolloff, flux, side, corr   <- the proven axes
                12 chroma (pitch-class)               <- harmonic content
    CHUNKEED over 8 temporal segments (order-preserving:
        'click then drone' must NOT equal 'drone then click')
    + global blocks: 64-band spectral silhouette (womb's proven
        signature), dynamics (modulation spectrum, onsets, drift),
        spatial (stereo life).                        <- 556 dims total

Every dimension is z-normalized against a corpus fit (bank.EchoMetric),
and blocks carry weights, so MFCCs don't drown chroma. Pure numpy,
deterministic, ~10 ms per 3 s clip.
"""

from __future__ import annotations

import numpy as np

from .melbank import get_mel_dct

__all__ = ["encode", "encode_blocks", "LAYOUT", "DIM", "BLOCK_NAMES",
           "N_CHUNK", "N_MFCC", "N_MELS", "FMIN", "FMAX"]

# ------------------------------------------------------------------ constants
N_CHUNK = 8          # temporal segments — the time axis of the vector
N_MELS = 64
N_MFCC = 24
FMIN, FMAX = 40.0, 16000.0
SR = 44100
WIN0, HOP0 = 2048, 512        # at 44.1 kHz: 46 ms window, 11.6 ms hop
_N_SCALARS = 10               # rms crest zcr centroid bandwidth flatness rolloff flux side corr
_N_DYNAMICS = 13              # 8 modulation bands + onset drift cv asym fluxratio
_N_SPATIAL = 3                # side_mean corr_mean side_std

def layout_for(n_chunks: int = N_CHUNK):
    """The vector anatomy for a given chunk count — `layout_for(1)` is
    the no-time-axis ablation (echo_flat); block dims shrink with the
    chunks, names and weights stay."""
    return [("mfcc_mean", n_chunks * N_MFCC, 1.00),   # timbre trajectory
            ("mfcc_std", n_chunks * N_MFCC, 0.35),   # intra-chunk texture
            ("scal_mean", n_chunks * _N_SCALARS, 0.80),  # scalar trajectory
            ("silhouette", 64, 0.80),                 # womb's proven block
            ("dynamics", _N_DYNAMICS, 0.90),          # envelope weather
            ("chroma", 12, 0.40),                    # pitch-class content
            ("spatial", _N_SPATIAL, 0.30)]            # stereo life


# (name, dim, weight) in concatenation order — the vector's anatomy
LAYOUT = layout_for(N_CHUNK)
LAYOUT_FLAT = layout_for(1)
BLOCK_NAMES = [name for name, _, _ in LAYOUT]
BLOCK_DIMS = {name: dim for name, dim, _ in LAYOUT}
DIM = sum(dim for _, dim, _ in LAYOUT)
DIM_FLAT = sum(dim for _, dim, _ in LAYOUT_FLAT)
assert DIM == 556, DIM

_EPS = 1e-9
_MAX_FRAMES_PER_PASS = 4096        # long files: never materialize the
                                   # whole (T, win) matrix — ~130 MB/pass


# ------------------------------------------------------------------ framing

def _frame_win_hop(sr: int) -> tuple[int, int]:
    """46 ms / 11.6 ms at any sample rate (power-of-two window)."""
    win = 1 << max(7, int(round(np.log2(0.0464 * sr))))
    return win, max(32, win // 4)


def _sanitize(audio) -> np.ndarray:
    """Anything -> (n, 2) float64, finite, not clipping."""
    x = np.asarray(audio, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.ndim == 2 and x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.ndim == 2 and x.shape[1] > 2:
        x = x[:, :2]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    return x


# ------------------------------------------------------------------ per-frame features

def _frame_features(x: np.ndarray, sr: int, win: int, hop: int):
    """All per-frame measurements. Returns dict of (T, ·) arrays plus the
    steady-state silhouette. Long clips are processed in frame blocks so
    memory stays flat no matter the file length (the archive must index
    20-minute takes as happily as 3-second evals)."""
    mid = x.mean(axis=1)
    T = max(1, 1 + max(0, len(mid) - win) // hop)
    n_fft = win
    hann = np.hanning(win)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb, dct = get_mel_dct(sr, n_fft, N_MELS, N_MFCC, FMIN, FMAX)

    # per-frame outputs (small: T x few)
    out = {k: np.zeros(T) for k in
           ("rms", "crest", "zcr", "centroid", "bandwidth", "flatness",
            "rolloff", "flux", "side", "corr")}
    mfcc = np.zeros((T, N_MFCC))
    chroma = np.zeros((T, 12))
    band = (freqs >= FMIN) & (freqs <= FMAX)
    mask55 = freqs >= 55.0
    fm = freqs[mask55]
    pc_of_bin = (np.round(69.0 + 12.0 * np.log2(fm / 440.0)).astype(int) % 12)

    # silhouette accumulator: mean magnitude over the steady-state middle
    lo_f, hi_f = T // 4, max(T // 4 + 1, (3 * T) // 4)
    sil_sum = np.zeros(len(freqs))
    sil_cnt = 0
    prev_logmag = None

    for start in range(0, T, _MAX_FRAMES_PER_PASS):
        stop = min(T, start + _MAX_FRAMES_PER_PASS)
        s0, s1 = start * hop, (stop - 1) * hop + win
        seg = mid[s0:s1]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        fr = np.lib.stride_tricks.sliding_window_view(seg, win)[::hop]
        if len(fr) > stop - start:
            fr = fr[: stop - start]
        n = len(fr)

        mag = np.abs(np.fft.rfft(fr * hann, axis=1)) + _EPS
        logmag = np.log(mag)

        # mel / MFCC
        logmel = np.log((mag ** 2) @ fb.T + 1e-10)
        mfcc[start: stop] = logmel @ dct.T

        # energy / shape
        rms = np.sqrt((fr ** 2).mean(axis=1)) + _EPS
        peak = np.abs(fr).max(axis=1)
        out["rms"][start: stop] = rms
        out["crest"][start: stop] = peak / rms
        out["zcr"][start: stop] = np.abs(
            np.diff(np.signbit(fr), axis=1)).mean(axis=1)

        # spectral scalars
        S = mag.sum(axis=1)
        cent = (mag @ freqs) / S
        out["centroid"][start: stop] = cent
        out["bandwidth"][start: stop] = np.sqrt(np.maximum(
            (mag @ freqs ** 2) / S - cent ** 2, 0.0))
        mb = mag[:, band]
        out["flatness"][start: stop] = np.exp(
            np.log(mb + _EPS).mean(axis=1)) / (mb.mean(axis=1) + _EPS)
        cum = np.cumsum(mag, axis=1)
        roll = np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)
        out["rolloff"][start: stop] = freqs[np.minimum(roll, len(freqs) - 1)]

        # flux (block-continuous: the first frame of every block after
        # the first leans on the previous block's last frame)
        if prev_logmag is None:
            d = np.maximum(logmag[1:] - logmag[:-1], 0.0).mean(axis=1)
            flux = np.concatenate([[0.0], d])[:n]
        else:
            ref = np.vstack([prev_logmag, logmag[:-1]])
            flux = np.maximum(logmag - ref, 0.0).mean(axis=1)
        out["flux"][start: stop] = flux
        prev_logmag = logmag[-1]

        # chroma
        mm = mag[:, mask55]
        for p in range(12):
            sel = pc_of_bin == p
            if sel.any():
                chroma[start: stop, p] = mm[:, sel].sum(axis=1)

        # silhouette accumulation over this block's share of the middle
        b0, b1 = max(start, lo_f), min(stop, hi_f)
        if b1 > b0:
            sil_sum += mag[b0 - start: b1 - start].sum(axis=0)
            sil_cnt += b1 - b0

    chroma = chroma / (chroma.sum(axis=1, keepdims=True) + _EPS)

    # stereo life
    if x.shape[1] == 2 and len(x) >= win:
        side = np.zeros(T)
        corr = np.ones(T)
        for start in range(0, T, _MAX_FRAMES_PER_PASS):
            stop = min(T, start + _MAX_FRAMES_PER_PASS)
            s0, s1 = start * hop, (stop - 1) * hop + win
            segL = x[s0:s1, 0]
            segR = x[s0:s1, 1]
            if len(segL) < win:
                segL = np.pad(segL, (0, win - len(segL)))
                segR = np.pad(segR, (0, win - len(segR)))
            fL = np.lib.stride_tricks.sliding_window_view(segL, win)[::hop]
            fR = np.lib.stride_tricks.sliding_window_view(segR, win)[::hop]
            fL, fR = fL[: stop - start], fR[: stop - start]
            side[start: stop] = np.sqrt(
                ((fL - fR) ** 2).mean(axis=1)) / (out["rms"][start: stop] + _EPS)
            mL, mR = fL.mean(axis=1), fR.mean(axis=1)
            cov = (fL * fR).mean(axis=1) - mL * mR
            sL = fL.std(axis=1) + _EPS
            sR = fR.std(axis=1) + _EPS
            corr[start: stop] = np.clip(cov / (sL * sR), -1.0, 1.0)
    else:
        side = np.zeros(T)
        corr = np.ones(T)

    sil = sil_sum / max(sil_cnt, 1)
    return {"mfcc": mfcc, "rms": out["rms"], "crest": out["crest"],
            "zcr": out["zcr"], "centroid": out["centroid"],
            "bandwidth": out["bandwidth"], "flatness": out["flatness"],
            "rolloff": out["rolloff"], "flux": out["flux"],
            "side": side, "corr": corr, "chroma": chroma,
            "silhouette_spec": sil}


# ------------------------------------------------------------------ global blocks

def _silhouette_from_spec(sil_spec: np.ndarray) -> np.ndarray:
    """womb's proven 64-band log-magnitude signature, from a mean
    magnitude spectrum (steady-state middle 50% of frames), kept as a
    block so the archive sees what the old drift floor saw — but now
    inside a bigger, fairer distance."""
    S = sil_spec
    edges = np.geomspace(1, len(S), 65).astype(int)
    sig = np.array([np.log1p(max(S[edges[i]: edges[i + 1]].mean(), 0.0))
                    if edges[i + 1] > edges[i] else 0.0 for i in range(64)])
    nrm = np.linalg.norm(sig) + _EPS
    return sig / nrm


def _dynamics(feat: dict, sr: int, hop: int) -> np.ndarray:
    """Envelope weather: 8 modulation bands (0.5–16 Hz) + onsets,
    centroid drift, rms cv, rise/fall asymmetry, flux ratio. Rhythm
    without committing to beats; movement the crackle gates never saw."""
    env = feat["rms"]                                    # (T,) frame energy
    T = len(env)
    # modulation spectrum of the (demeaned) envelope — blockwise so a
    # 20-minute take doesn't materialize one giant FFT
    mod = np.zeros(8)
    block = 512
    if T >= 8:
        e = env - env.mean()
        mfr_full = np.fft.rfftfreq(block, hop / sr)
        edges = np.geomspace(0.5, 16.0, 9)
        P = np.zeros(block // 2 + 1)
        w = np.hanning(block)
        n_blocks = 0
        for i in range(0, T - (T % block), block):
            P += np.abs(np.fft.rfft(e[i: i + block] * w)) ** 2
            n_blocks += 1
        if n_blocks:
            P /= n_blocks
            for i in range(8):
                sel = (mfr_full >= edges[i]) & (mfr_full < edges[i + 1])
                mod[i] = np.log1p(P[sel].mean()) if sel.any() else 0.0
    # onsets: median-relative rising edges per second (womb's rule)
    med = float(np.median(env)) + _EPS
    above = env > 1.6 * med
    dur_s = T * hop / sr + _EPS
    onsets = float(np.count_nonzero(above[1:] & ~above[:-1]) / dur_s)
    cdrift = float(feat["centroid"].std() / 2000.0)
    rms_cv = float(feat["rms"].std() / (feat["rms"].mean() + _EPS))
    denv = np.diff(env)
    pos = float(denv[denv > 0].mean()) if np.any(denv > 0) else 0.0
    neg = float(-denv[denv < 0].mean()) if np.any(denv < 0) else 0.0
    asym = pos / (pos + neg + _EPS)
    flux_ratio = float(feat["flux"].max() / (feat["flux"].mean() + _EPS))
    return np.concatenate([mod, [onsets, cdrift, rms_cv, asym, flux_ratio]])


def _chunk_stats(arr: np.ndarray, n_chunks: int) -> tuple[np.ndarray, np.ndarray]:
    """Order-preserving aggregation: (n_chunks, ·) means and stds. Short
    clips tile their frames so the vector is always the same anatomy."""
    T = arr.shape[0]
    if T < n_chunks:                                     # tile tiny clips
        reps = int(np.ceil(n_chunks / max(1, T)))
        arr = np.concatenate([arr] * reps, axis=0)
        T = arr.shape[0]
    bounds = np.linspace(0, T, n_chunks + 1).astype(int)
    means = np.stack([arr[bounds[i]: bounds[i + 1]].mean(axis=0)
                     if bounds[i + 1] > bounds[i] else np.zeros(arr.shape[1])
                     for i in range(n_chunks)])
    stds = np.stack([arr[bounds[i]: bounds[i + 1]].std(axis=0)
                     if bounds[i + 1] > bounds[i] else np.zeros(arr.shape[1])
                     for i in range(n_chunks)])
    return means, stds


# ------------------------------------------------------------------ the encoder

_SCAL_ORDER = ("rms", "crest", "zcr", "centroid", "bandwidth", "flatness",
               "rolloff", "flux", "side", "corr")


def encode_blocks(audio, sr: int = SR, n_chunks: int = N_CHUNK) -> dict:
    """Audio -> {block_name: (dim,) float64}. See LAYOUT for the anatomy."""
    x = _sanitize(audio)
    win, hop = _frame_win_hop(sr)
    feat = _frame_features(x, sr, win, hop)

    mfcc_mean, mfcc_std = _chunk_stats(feat["mfcc"], n_chunks)
    scal = np.stack([feat[k] for k in _SCAL_ORDER], axis=1)   # (T, 10)
    scal_mean, _ = _chunk_stats(scal, n_chunks)

    return {
        "mfcc_mean": mfcc_mean.reshape(-1),
        "mfcc_std": mfcc_std.reshape(-1),
        "scal_mean": scal_mean.reshape(-1),
        "silhouette": _silhouette_from_spec(feat["silhouette_spec"]),
        "dynamics": _dynamics(feat, sr, hop),
        "chroma": feat["chroma"].mean(axis=0),
        "spatial": np.array([feat["side"].mean(), feat["corr"].mean(),
                             feat["side"].std()]),
    }


def encode(audio, sr: int = SR, n_chunks: int = N_CHUNK) -> np.ndarray:
    """Audio (mono or stereo, any length) -> (556,) float64 feature
    vector. The whole 'machine listening' of the archive, one call."""
    blocks = encode_blocks(audio, sr, n_chunks)
    vec = np.concatenate([blocks[name] for name, _, _ in LAYOUT])
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec