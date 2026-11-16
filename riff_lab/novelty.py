"""novelty: open-ended search driven by THREE signals, all relative to
the persistent archive of everything heard before:

    novelty(x) = Wa * audio_sparseness   (echo 556-d, fitted-metric, lensed)
               + Wp * pattern_sparseness (per-voice groove structure)
               + Wc * complexity          (the complexification ratchet)

- AUDIO similarity: echo's 556-d encoding of the rendered preview (incl.
  drums), compared in a FITTED EchoMetric space (per-dim z-norm + block
  weights) so no single Hz-scaled axis dominates, averaged over rotating
  lenses (timbre / time / pitch).
- PATTERN similarity: a structural descriptor of the alien expressions
  (per-voice density, syncopation, IOI variance, position entropy) so two
  riffs with different GROOVES but similar timbres still read as far apart.
- COMPLEXITY: total expression node-count — a gentle, saturating pressure
  so lineages ratchet toward more intricate motifs (complexification),
  not just new timbres.

Selection = rank by this blended novelty. Behaviour is measured against
the archive UNION the sibling batch. MCNS viability floor = ears.defects
only. No quality/taste objective ever touches selection.
"""
import json
import sys
from pathlib import Path

import numpy as np

from . import patterns as P

# echo's src dir is normally put on sys.path by riff_lab/__init__.py (robust to
# CWD); this fallback keeps novelty.py importable on its own from the repo root.
_ECHO_SRC = Path(__file__).resolve().parent.parent / "echo" / "src"
if _ECHO_SRC.is_dir() and str(_ECHO_SRC) not in sys.path:
    sys.path.insert(0, str(_ECHO_SRC))

ARCHIVE = Path("riffs/_novelty_archive.json")
K = 15
N_CHUNKS = 8

# the three novelty weights — the main tuning dials
W_AUDIO = 0.50
W_PATTERN = 0.35
W_COMPLEX = 0.15
COMPLEX_SAT = 120.0          # node-count at which complexity novelty saturates
# complexity is measured on the EVOLVABLE exprs (melodic motifs + drum fill),
# NOT the constant archetype backbones — so the ratchet has a live signal.
_EVOLVABLE = ("bass_expr", "noise_expr", "fill_expr", "drum_fill")

LENSES = {
    "timbre": ("mfcc_mean", "mfcc_std"),
    "time": ("dynamics", "scal_mean"),
    "pitch": ("chroma", "silhouette"),
}

_VOICES = ("bass_expr", "noise_expr", "fill_expr")
_DRUMS = ("kick", "snare", "hihat", "openhh", "perc")
_DESC_RNG = 12345            # fixed seed -> deterministic structural descriptor


# ----------------------------------------------------------------- encoding

def encode_audio(preview, sr=44100):
    from echo.encode import encode
    return np.asarray(encode(preview, sr, n_chunks=N_CHUNKS), np.float64)


def _voice_stats(expr, steps=32):
    """density, offbeat fraction, IOI variance, position entropy."""
    if expr is None:
        return [0.0, 0.0, 0.0, 0.0]
    stream = P.eval_stream(expr, np.random.default_rng(_DESC_RNG), steps)
    hits = [i for i, v in enumerate(stream) if P.is_hit(v)]
    if not hits:
        return [0.0, 0.0, 0.0, 0.0]
    density = len(hits) / steps
    offbeat = sum(1 for h in hits if (h % 16) % 4 != 0) / len(hits)
    iois = np.diff(sorted(hits))
    ioi_var = float(np.std(iois)) / 16.0 if len(iois) else 0.0
    pos = np.bincount([h % 16 for h in hits], minlength=16).astype(float)
    p = pos / pos.sum()
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum()) / 4.0   # /log2(16)
    return [density, offbeat, min(ioi_var, 1.0), ent]


def genome_complexity(g):
    """Complexity of the EVOLVABLE expressions (motifs + drum fill)."""
    c = 0
    for key in _EVOLVABLE:
        if g.get(key) is not None:
            c += P.complexity(g[key])
    return c


def pattern_descriptor(g):
    """Structural fingerprint of a genome's grooves (melodic voices + the
    assembled drum break) + its complexity — pattern-novelty behaviour and
    the complexification axis."""
    out = []
    for key in _VOICES:                         # melodic groove stats
        out += _voice_stats(g.get(key))
    de = g.get("drum_exprs", {})                # the assembled break groove
    for v in _DRUMS:
        out += _voice_stats(de.get(v))
    total_c = genome_complexity(g)              # evolvable-only complexity
    out += [g.get("bars", 4) / 16.0, g.get("swing", 0.0) * 5.0,
            min(total_c / COMPLEX_SAT, 1.0)]
    return np.asarray(out, np.float64), int(total_c)


# ----------------------------------------------------------------- distance

def _metric(vecs):
    from echo.bank import EchoMetric
    from echo.encode import layout_for
    m = EchoMetric(layout=layout_for(N_CHUNKS))
    if len(vecs) >= 2:
        m.fit(np.stack(vecs))
    return m


def _lens_cols(layout, blocks):
    cols, pos = [], 0
    for name, d, _ in layout:
        if blocks is None or name in blocks:
            cols.extend(range(pos, pos + d))
        pos += d
    return np.asarray(cols, int)


def _audio_sparseness(child_vecs, neighbor_vecs, k=K):
    """Lens-averaged metric-space sparseness for each child."""
    from echo.encode import layout_for
    layout = layout_for(N_CHUNKS)
    allv = list(neighbor_vecs) + list(child_vecs)
    metric = _metric(allv)
    P_ = np.stack([metric.prepare(v) for v in allv])
    n_nb, nc = len(neighbor_vecs), len(child_vecs)
    rho = np.zeros(nc)
    for lens, blocks in LENSES.items():
        cols = _lens_cols(layout, blocks)
        Pl = P_[:, cols]
        norm = np.sqrt((Pl * Pl).sum(axis=1))
        scale = norm[n_nb:].mean() + 1e-9
        for i in range(nc):
            me = n_nb + i
            others = np.delete(np.arange(len(allv)), me)
            d = np.sqrt(((Pl[others] - Pl[me]) ** 2).sum(axis=1))
            kk = min(k, len(d))
            rho[i] += (np.sort(d)[:kk].mean() / scale) if kk else 0.0
    return rho / len(LENSES)


def _desc_sparseness(child_descs, neighbor_descs, k=K):
    """z-normed euclidean sparseness over the structural descriptors."""
    allv = list(neighbor_descs) + list(child_descs)
    M = np.stack(allv)
    sd = M.std(axis=0); sd[sd < 1e-6] = 1.0
    Mz = (M - M.mean(axis=0)) / sd
    n_nb, nc = len(neighbor_descs), len(child_descs)
    rho = np.zeros(nc)
    for i in range(nc):
        me = n_nb + i
        others = np.delete(np.arange(len(allv)), me)
        d = np.sqrt(((Mz[others] - Mz[me]) ** 2).sum(axis=1))
        kk = min(k, len(d))
        rho[i] = np.sort(d)[:kk].mean() if kk else 0.0
    return rho


def _norm1(x):
    x = np.asarray(x, np.float64)
    m = x.mean()
    return x / m if m > 1e-9 else x


# ----------------------------------------------------------------- archive

class NoveltyArchive:
    """Permanent record of where search has been (sounds AND patterns).
    Admission thresholded; rho_min adapts (stall -5% / flood +20%)."""

    def __init__(self):
        self.entries = []
        self.rho_min = None
        if ARCHIVE.exists():
            try:
                state = json.load(open(ARCHIVE))
                self.entries = state.get("entries", [])
                self.rho_min = state.get("rho_min")
            except Exception:
                self.entries = []
        self._adds = 0
        self._rng = np.random.default_rng(1234)

    def avecs(self):
        return [np.asarray(e["avec"], np.float64) for e in self.entries if e.get("avec")]

    def descs(self):
        return [np.asarray(e["desc"], np.float64) for e in self.entries if e.get("desc")]

    def consider(self, riff_id, desc, avec, novelty):
        if avec is None:
            return False
        if self.rho_min is None:
            self.rho_min = max(novelty * 0.6, 1e-6)
        admit = novelty >= self.rho_min or self._rng.random() < 0.02
        if admit:
            self.entries.append(dict(
                id=riff_id,
                desc=list(map(float, desc)) if desc is not None else None,
                avec=list(map(float, avec))))
            self._adds += 1
        return admit

    def note_generation(self):
        if self.rho_min is not None:
            if self._adds == 0:
                self.rho_min *= 0.95
            elif self._adds > 4:
                self.rho_min *= 1.20
        self._adds = 0
        self.save()

    def save(self):
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with open(ARCHIVE, "w") as fh:
            json.dump(dict(entries=self.entries, rho_min=self.rho_min), fh)

    def __len__(self):
        return len(self.entries)


def select_by_novelty(children, n, archive):
    """Blended novelty: audio + pattern + complexity, vs archive u batch.
    Each child must carry 'avec' (echo vector) and 'desc' (pattern
    descriptor) and 'complexity' (node count)."""
    arch_a = archive.avecs()
    arch_d = archive.descs()
    child_a = [c["avec"] for c in children]
    child_d = [c["desc"] for c in children]
    audio = _audio_sparseness(child_a, arch_a)
    desc = _desc_sparseness(child_d, arch_d)
    comp = np.array([min(c["complexity"] / COMPLEX_SAT, 1.0) for c in children])
    nov = (W_AUDIO * _norm1(audio) + W_PATTERN * _norm1(desc)
           + W_COMPLEX * comp / max(comp.mean(), 1e-9))
    order = np.argsort(-nov)
    for i, c in enumerate(children):
        c["novelty"] = float(nov[i])
        c["_nov_parts"] = (float(audio[i]), float(desc[i]), int(c["complexity"]))
    return [children[i] for i in order[:n]]
