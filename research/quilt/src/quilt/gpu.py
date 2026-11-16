"""gpu: the ear on the Metal — echo's encoder, batched, in torch.

This machine: Apple M4, MPS backend, torch.fft works on MPS, and the
distance work quilt does is 45x faster there than on one CPU thread
(measured: (256x556)@(556x500k) GEMM = 5.9 ms MPS vs 268 ms CPU).
Two accelerations live here:

1. **GpuEar** — echo's 556-D encoder (encode_blocks) ported 1:1 to
   batched torch, same framing, same filterbank (echo's own cached
   mel + DCT matrices), same block order, same epsilons — running in
   float32 on MPS over a *batch of same-length chunks*. The numpy
   encoder stays the reference; the GPU ear is a different precision
   era of the same anatomy, recorded in the bank's provenance
   ("echo-gpu-f32") so eras never silently mix. Indexing a few hundred
   thousand pieces is where it pays.

2. **The atlas's query path** (ChunkAtlas) — distance-to-everything
   on the GPU: one GEMV per walker step, one GEMM for median-NN, PCA
   covariance as one big GEMM. The walkers call the same methods;
   the metal is under them.

Everything degrades to the numpy path when torch or MPS is missing:
`available()` answers, callers obey. Determinism: same device, same
batching, same result — the tests pin encoder equivalence (against
numpy within float32 tolerance) and run-to-run determinism.
"""

from __future__ import annotations

import numpy as np

from echo.encode import (LAYOUT, N_CHUNK, N_MFCC, N_MELS, FMIN, FMAX,
                         _frame_win_hop)
from echo.melbank import get_mel_dct
from echo.probes import SR

__all__ = ["mps_available", "GpuEar", "resolve_device"]

_EPS = 1e-9
_SCAL_ORDER = ("rms", "crest", "zcr", "centroid", "bandwidth", "flatness",
               "rolloff", "flux", "side", "corr")


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def mps_available() -> bool:
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_device(device: str | None) -> str:
    """'auto' -> 'mps' if the metal is there, else 'cpu'."""
    if device in (None, "auto"):
        return "mps" if mps_available() else "cpu"
    if device == "mps" and not mps_available():
        return "cpu"
    return device or "cpu"


# ------------------------------------------------------------------ the ear

class GpuEar:
    """echo's encode_blocks, batched over same-length audio, on torch.

    One era, honestly labeled: float32 on the device. The anatomy,
    block order, framing, epsilons and normalization are echo's —
    only the arithmetic engine differs. `encode_batch` takes a list
    of (n, 2) float arrays that all share the same n (the index
    buckets by exact length — onset pieces are 512-sample quantized,
    fixed pieces are exactly 2.000 s, so the buckets run deep).
    """

    def __init__(self, sr: int = SR, device: str = "auto"):
        torch = _torch()
        if torch is None:
            raise RuntimeError("torch is not installed — GpuEar needs it")
        self.device_name = resolve_device(device)
        self.dev = torch.device(self.device_name)
        self.sr = int(sr)
        self.win, self.hop = _frame_win_hop(sr)
        fb, dct = get_mel_dct(sr, self.win, N_MELS, N_MFCC, FMIN, FMAX)

        d = self.dev
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=d)
        self.fb_t = t(fb.T)                      # (1025, 64)  mel @
        self.dct_t = t(dct.T)                    # (64, 24)    DCT @
        self.hann_t = torch.hann_window(self.win, periodic=False,
                                         device=d)     # == np.hanning
        freqs = np.fft.rfftfreq(self.win, 1.0 / sr)
        self.freqs_t = t(freqs)
        self.freq2_t = t(freqs ** 2)
        band = (freqs >= FMIN) & (freqs <= FMAX)
        self.band_t = torch.as_tensor(band, device=d)            # bool
        mask55 = freqs >= 55.0
        self.mask55_t = torch.as_tensor(mask55, device=d)       # bool
        pc = (np.round(69.0 + 12.0 * np.log2(
            freqs[mask55] / 440.0)).astype(int) % 12)
        pc_onehot = np.zeros((int(mask55.sum()), 12), np.float32)
        pc_onehot[np.arange(len(pc)), pc] = 1.0
        self.pc_t = t(pc_onehot)                  # (nm, 12) chroma @
        # silhouette: 64 log-spaced bands over the spectrum, as one matmul
        edges = np.geomspace(1, len(freqs), 65).astype(int)
        sil_w = np.zeros((len(freqs), 64), np.float32)
        sil_c = np.zeros(64, np.float32)
        for i in range(64):
            a, b = edges[i], edges[i + 1]
            if b > a:
                sil_w[a:b, i] = 1.0
                sil_c[i] = b - a
        self.silw_t, self.silc_t = t(sil_w), t(sil_c)
        # modulation spectrum bands over rfftfreq(512, hop/sr)
        self.mod_block = 512
        mfr = np.fft.rfftfreq(self.mod_block, self.hop / sr)
        medges = np.geomspace(0.5, 16.0, 9)
        mod_w = np.zeros((len(mfr), 8), np.float32)
        mod_c = np.zeros(8, np.float32)
        for i in range(8):
            sel = (mfr >= medges[i]) & (mfr < medges[i + 1])
            if sel.any():
                mod_w[sel, i] = 1.0
                mod_c[i] = int(sel.sum())
        self.modw_t, self.modc_t = t(mod_w), t(modc := mod_c)
        self.hann512_t = torch.hann_window(self.mod_block, periodic=False,
                                            device=d)

    # ------------------------------------------------------------- encode
    def encode_batch(self, audios: list) -> np.ndarray:
        """List of (n, 2) floats (all the same n) -> (B, 556) float64.
        Same contract as echo.encode per item; batched for the metal."""
        import torch
        if not audios:
            return np.zeros((0, sum(d for _, d, _ in LAYOUT)))
        n = len(audios[0])
        assert all(len(a) == n for a in audios), "bucket by exact length"
        # sanitize exactly like echo (f64 on CPU): finite, clipped, peak
        clean = []
        for a in audios:
            x = np.asarray(a, np.float64)
            if x.ndim == 1:
                x = np.stack([x, x], axis=1)
            elif x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            elif x.shape[1] > 2:
                x = x[:, :2]
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            peak = float(np.abs(x).max()) if x.size else 0.0
            if peak > 1.0:
                x = x / peak
            clean.append(x)
        X = torch.as_tensor(np.stack(clean), dtype=torch.float32,
                             device=self.dev)      # (B, n, 2)
        B = X.shape[0]
        T = max(1, 1 + max(0, n - self.win) // self.hop)
        mid = X.mean(dim=2)                        # (B, n)
        if n < self.win:
            mid = torch.nn.functional.pad(mid, (0, self.win - n))
            T = 1
        fr = mid.unfold(1, self.win, self.hop)[:, :T]        # (B, T, win)

        mag = torch.abs(torch.fft.rfft(fr * self.hann_t, dim=-1)) + _EPS
        logmag = torch.log(mag)
        logmel = torch.log(mag ** 2 @ self.fb_t + 1e-10)
        mfcc = logmel @ self.dct_t                          # (B, T, 24)

        rms = torch.sqrt((fr ** 2).mean(dim=-1)) + _EPS
        crest = fr.abs().amax(dim=-1) / rms
        zcr = fr.signbit().diff(dim=-1).abs().float().mean(dim=-1)
        S = mag.sum(dim=-1)
        cent = (mag @ self.freqs_t) / S
        bw = torch.sqrt(torch.clamp(
            (mag @ self.freq2_t) / S - cent ** 2, min=0.0))
        mb = mag[:, :, self.band_t]
        flat = torch.exp(torch.log(mb + _EPS).mean(dim=-1)) \
            / (mb.mean(dim=-1) + _EPS)
        cum = torch.cumsum(mag, dim=-1)
        roll = torch.argmax((cum >= 0.85 * cum[..., -1:]).float(), dim=-1)
        rolloff = self.freqs_t[torch.clamp(roll, max=mag.shape[-1] - 1)]
        d = torch.clamp(logmag[:, 1:] - logmag[:, :-1], min=0.0).mean(dim=-1)
        flux = torch.cat([torch.zeros(B, 1, device=self.dev), d], dim=1)

        chroma = (mb := mag[:, :, self.mask55_t]) @ self.pc_t
        chroma = chroma / (chroma.sum(dim=-1, keepdim=True) + _EPS)

        lo_f, hi_f = T // 4, max(T // 4 + 1, (3 * T) // 4)
        sil_spec = mag[:, lo_f:hi_f].mean(dim=1)             # (B, 1025)
        sil = torch.log1p(torch.clamp(
            sil_spec @ self.silw_t / self.silc_t, min=0.0))
        sil = sil / (sil.norm(dim=-1, keepdim=True) + _EPS)

        if X.shape[2] == 2 and n >= self.win:
            fL = X[:, :, 0].unfold(1, self.win, self.hop)[:, :T]
            fR = X[:, :, 1].unfold(1, self.win, self.hop)[:, :T]
            side = torch.sqrt(((fL - fR) ** 2).mean(dim=-1)) / (rms + _EPS)
            mL, mR = fL.mean(dim=-1), fR.mean(dim=-1)
            cov = (fL * fR).mean(dim=-1) - mL * mR
            sL = fL.std(dim=-1, unbiased=False) + _EPS
            sR = fR.std(dim=-1, unbiased=False) + _EPS
            corr = torch.clamp(cov / (sL * sR), -1.0, 1.0)
        else:
            side = torch.zeros(B, T, device=self.dev)
            corr = torch.ones(B, T, device=self.dev)

        scal = torch.stack([rms, crest, zcr, cent, bw, flat, rolloff,
                            flux, side, corr], dim=-1)        # (B, T, 10)

        mfcc_m, mfcc_s = self._chunk_stats(mfcc, n_chunks=N_CHUNK)
        scal_m, _ = self._chunk_stats(scal, n_chunks=N_CHUNK)
        dyn = self._dynamics(rms, cent, flux)
        chroma_mean = chroma.mean(dim=1)
        spatial = torch.stack([side.mean(dim=-1), corr.mean(dim=-1),
                               side.std(dim=-1, unbiased=False)], dim=-1)

        vec = torch.cat([mfcc_m.flatten(1), mfcc_s.flatten(1),
                         scal_m.flatten(1), sil, dyn, chroma_mean,
                         spatial], dim=-1)
        vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        assert vec.shape[1] == sum(d for _, d, _ in LAYOUT), vec.shape
        return vec.cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------- blocks
    def _chunk_stats(self, arr, n_chunks: int):
        """(B, T, D) -> ((B, n_chunks, D) means, same stds), with the
        tiling trick for T < n_chunks — echo's semantics, batched."""
        import torch
        T = arr.shape[1]
        if T < n_chunks:
            reps = int(np.ceil(n_chunks / max(1, T)))
            arr = arr.repeat(1, reps, 1)
            T = arr.shape[1]
        bounds = np.linspace(0, T, n_chunks + 1).astype(int)
        means, stds = [], []
        for i in range(n_chunks):
            seg = arr[:, bounds[i]:bounds[i + 1]]
            if bounds[i + 1] > bounds[i]:
                means.append(seg.mean(dim=1))
                stds.append(seg.std(dim=1, unbiased=False))
            else:
                z = torch.zeros_like(arr[:, 0])
                means.append(z)
                stds.append(z)
        return torch.stack(means, dim=1), torch.stack(stds, dim=1)

    def _dynamics(self, rms, cent, flux):
        """Envelope weather, batched: 8 modulation bands + onsets,
        centroid drift, rms cv, rise/fall asym, flux ratio."""
        import torch
        B, T = rms.shape
        dev = self.dev
        env = rms
        mod = torch.zeros(B, 8, device=dev)
        if T >= 8:
            e = env - env.mean(dim=-1, keepdim=True)
            if T >= self.mod_block:
                nb = T // self.mod_block
                eb = e[:, : nb * self.mod_block].reshape(
                    B, nb, self.mod_block)
                P = (torch.abs(torch.fft.rfft(
                    eb * self.hann512_t, dim=-1)) ** 2).mean(dim=1)
                bandmean = P @ self.modw_t / self.modc_t
                mod = torch.log1p(bandmean)
        med = torch.quantile(env, 0.5, dim=-1, keepdim=True) + _EPS
        above = env > 1.6 * med
        dur = T * self.hop / self.sr + _EPS
        rising = (above[:, 1:] & ~above[:, :-1]).float().sum(dim=-1)
        onsets = rising / dur
        cdrift = cent.std(dim=-1, unbiased=False) / 2000.0
        rms_cv = env.std(dim=-1, unbiased=False) \
            / (env.mean(dim=-1) + _EPS)
        denv = env.diff(dim=-1)
        pos = self._masked_mean(denv, denv > 0)
        neg = self._masked_mean(-denv, denv < 0)
        asym = pos / (pos + neg + _EPS)
        fratio = flux.amax(dim=-1) / (flux.mean(dim=-1) + _EPS)
        return torch.stack([*mod.unbind(dim=-1), onsets, cdrift, rms_cv,
                            asym, fratio], dim=-1)

    @staticmethod
    def _masked_mean(x, mask):
        import torch
        cnt = mask.float().sum(dim=-1)
        s = (x * mask.float()).sum(dim=-1)
        return torch.where(cnt > 0, s / cnt.clamp(min=1.0),
                           torch.zeros_like(s))