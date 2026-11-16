#!/usr/bin/env python3
"""
grammar_lab.py — corpus grammar generators (port of alien-python 08/10/07).

The corpus becomes a language:
  1. every curated sound -> STFT log-magnitude frames
  2. k-means (256 tokens) -> every frame becomes a token   (audio -> text)
  3. an order-3 n-gram LM learns which tokens follow which — the corpus's
     own spectral grammar
  4. GENERATE:
       - temperature dreams   (low = the corpus speaking properly, high = babble)
       - prompt continuation  (a real corpus phrase's tokens, then the LM
                              finishes it — the corpus continues itself)
       - state-Markov with real frames (order-1 transitions, decoded from
         actual member frames rather than centroids — granular-vocoder grit)
  5. decode tokens -> frames -> GRIFFIN-LIM phase reconstruction -> audio

Usage:
    python3 grammar_lab.py                       # -> audio_lab/grammar_out/
    python3 grammar_lab.py --phrases 18 --tokens 256
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from sklearn.cluster import MiniBatchKMeans

SR = 44100
N_FFT = 2048
HOP = 512
EPS = 1e-10


# ------------------------------------------------------------------ frames

def corpus_frames(files, cap_s=30.0):
    mags, owners, seqs = [], [], []
    for f in files:
        try:
            y, sr = sf.read(str(f), always_2d=True)
        except Exception:
            continue
        y = y.mean(axis=1)[: int(cap_s * sr)]
        if len(y) < N_FFT * 2:
            continue
        _, _, Z = stft(y, sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
        V = np.log1p(np.abs(Z).T).astype(np.float32)     # frames x freq
        mags.append(V)
        owners.append(len(V))
        seqs.append(V)
    return mags, owners


# ------------------------------------------------------------------ griffin-lim

def griffin_lim(mag_frames, n_iter=36, rng=None):
    """mag_frames: (frames, freqs) linear magnitudes -> audio."""
    M = mag_frames.T                                   # freqs x frames
    if rng is None:
        rng = np.random.default_rng(0)
    angles = np.exp(2j * np.pi * rng.random(M.shape))
    y = None
    for _ in range(n_iter):
        _, y = istft(M * angles, SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
        _, _, Z = stft(y, SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
        ang = np.exp(1j * np.angle(Z))
        if ang.shape[1] >= M.shape[1]:
            angles = ang[:, :M.shape[1]]
        else:
            pad = np.repeat(ang[:, -1:], M.shape[1] - ang.shape[1], axis=1)
            angles = np.concatenate([ang, pad], axis=1)
    _, y = istft(M * angles, SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    return y.astype(np.float32)


def frames_to_audio(V_log, rng):
    M = np.expm1(V_log)
    y = griffin_lim(M, rng=rng)
    y = y / (np.std(y) + EPS) * 0.25
    return np.clip(y, -1, 1)


# ------------------------------------------------------------------ n-gram LM

class NgramLM:
    def __init__(self, tokens, order=3):
        self.order = order
        self.vocab = int(max(tokens)) + 1
        self.tables = []
        for o in range(1, order + 1):
            # backoff chain: highest order first
            pass
        self.counts = {}
        ctxs = {}
        seq = tokens
        for i in range(order, len(seq)):
            ctx = tuple(seq[i - order:i])
            nxt = seq[i]
            self.counts.setdefault(ctx, np.zeros(self.vocab, np.int32))
            self.counts[ctx][nxt] += 1
        # also unigram fallback
        uni = np.bincount(seq, minlength=self.vocab).astype(np.int64)
        self.uni = uni + 1

    def sample(self, ctx, rng, temperature=1.0):
        for o in range(self.order, 0, -1):
            c = tuple(ctx[-o:]) if o > 0 else ()
            row = self.counts.get(c)
            if row is not None and row.sum() > 0:
                p = row.astype(np.float64) ** (1.0 / max(0.05, temperature))
                p /= p.sum()
                return int(rng.choice(self.vocab, p=p))
        p = self.uni.astype(np.float64) ** (1.0 / max(0.05, temperature))
        p /= p.sum()
        return int(rng.choice(self.vocab, p=p))


def build_lm_from_seqs(seqs, order=3):
    lm = NgramLM(np.concatenate(seqs), order)          # simple: pooled corpus
    return lm


def markov_matrix(seqs, k):
    T = np.zeros((k, k))
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            T[a, b] += 1
    T = (T + 0.05) / (T.sum(axis=1, keepdims=True) + 0.05 * k)
    return T


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="audio_lab/curated_corpus")
    ap.add_argument("-o", "--out", default="audio_lab/grammar_out")
    ap.add_argument("--phrases", type=int, default=18)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    rng = np.random.default_rng(2024)
    files = sorted(Path(args.corpus).glob("*.wav"))
    if not files:
        ap.error(f"no wavs in {args.corpus}")
    print(f"tokenizing {len(files)} curated sounds ...")
    mags, owners = corpus_frames(files)
    V = np.concatenate(mags, axis=0)
    print(f"  {V.shape[0]} frames x {V.shape[1]} freqs")
    km = MiniBatchKMeans(n_clusters=args.tokens, batch_size=4096,
                         n_init=3, random_state=7).fit(V)
    cents = km.cluster_centers_.astype(np.float32)
    seqs, ptr = [], 0
    for n in owners:
        seqs.append(km.labels_[ptr: ptr + n])
        ptr += n

    lm = build_lm_from_seqs(seqs, order=3)
    T1 = markov_matrix(seqs, args.tokens)
    members = {t: V[km.labels_ == t] for t in range(args.tokens)
               if (km.labels_ == t).sum() > 0}

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for i in range(args.phrases):
        engine = ["dream", "dream", "cont", "markov"][i % 4]
        temp = [0.7, 1.3][i % 2]
        dur_s = float(rng.uniform(3.0, 9.0))
        n_tok = int(dur_s * SR / HOP)
        if engine == "markov":
            s = int(rng.integers(args.tokens))
            toks = [s]
            for _ in range(n_tok - 1):
                p = T1[toks[-1]] ** (1.0 / temp)
                p /= p.sum()
                toks.append(int(rng.choice(args.tokens, p=p)))
            # decode from REAL member frames (granular-vocoder grit)
            Fr = np.stack([members[t][int(rng.integers(len(members[t])))]
                           if t in members else cents[t] for t in toks])
        else:
            if engine == "cont":
                src = int(rng.integers(len(seqs)))
                n_prompt = int(rng.integers(30, 120))
                ctx = list(seqs[src][:n_prompt])
                toks = list(ctx)
                need = n_tok - len(toks)
                for _ in range(max(0, need)):
                    toks.append(lm.sample(toks, rng, temp))
            else:
                ctx = [int(rng.integers(args.tokens)) for _ in range(3)]
                toks = list(ctx)
                for _ in range(n_tok - 3):
                    toks.append(lm.sample(toks, rng, temp))
            Fr = cents[np.array(toks)]
        y = frames_to_audio(Fr, rng)
        name = f"phrase{idx:03d}_{engine}_t{temp:.1f}.wav"
        sf.write(str(outdir / name), y, SR, subtype="PCM_16")
        rows.append(dict(file=name, engine=engine, temperature=temp,
                         tokens=len(toks), dur_s=round(len(y) / SR, 2)))
        idx += 1
        print(f"  {name} ({len(y)/SR:.1f}s, {engine}, T={temp})")
    with open(outdir / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"done -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())