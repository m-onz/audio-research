"""compose: the piece, end to end — spine, form, organism, corpus, damage.

One call makes a finished composition from the corpus:

    1. SPINE     a corpus piece, measured (onsets -> spike-decay senses)
    2. FORM      a tempo map — sections, slow bits, incredible fast BPM
                 (tectonic / accel / flip / your json)
    3. ORGANISM  a CPPN score genome — champion, evolved specimen, or
                 founder — sampled over the form's non-uniform tick grid
    4. PALETTE   every score voice resolved to real corpus sound, the
                 spine mixed underneath (the fusion recipe: spine +
                 duck + new parts)
    5. POST      the mastering hand — stutters in the fast sections,
                 time-stretches in the slow bits, tape, reverses,
                 euclid gates — random but seeded, ledgered

The artifact is a wav + a manifest naming everything: the spine's
sha, the form, the genome, the voice provenance, the post ledger.
Same seed -> byte-identical piece. You are the judge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .champions import build_champion, random_score_genome
from .form import FORMS, build_form, describe_form
from .genome import Genome
from .mapper import mapped_audio
from .palette import auto_voices
from .post import duck, post_process
from .scaffold import load_spine
from .score import score_from, score_summary, write_score
from echo.probes import SR, write_wav

__all__ = ["compose", "resolve_genome"]


def resolve_genome(champion: str | None = None, genome_path: str | None = None,
                   random_seed: int | None = None) -> Genome:
    """Exactly one way to pick the organism."""
    picks = [x for x in (champion, genome_path, random_seed) if x is not None]
    if len(picks) != 1:
        raise SystemExit("compose wants exactly one of "
                         "--champion, --genome, --random-seed")
    if champion:
        return build_champion(champion)
    if genome_path:
        d = json.loads(Path(genome_path).read_text())
        return Genome.from_dict(d["genome"] if "genome" in d else d)
    return random_score_genome(np.random.default_rng(random_seed))


def compose(spine: str, form: str = "tectonic", length: float = 180.0,
            seed: int = 0, champion: str | None = None,
            genome_path: str | None = None, random_seed: int | None = None,
            mode: str = "strike", threshold: float = 0.12,
            senses=("spikes", "weather"), corpus: str = "corpus",
            voice_overrides: dict | None = None, spine_gain: float = 0.45,
            duck_depth: float = 0.28, post_intensity: float = 1.0,
            loop: bool = True, out: str = "maestro_out/piece",
            verbose: bool = True) -> dict:
    """The whole pipeline -> (piece.wav, piece.score.json, manifest.json)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = build_form(form)
    genome = resolve_genome(champion, genome_path, random_seed)

    # 1+2. the spine, measured once, sampled onto the form's tick grid
    scaf0 = load_spine(spine, bpm=120.0, ticks_per_beat=f.ticks_per_beat)
    tick_times = f.tick_times(length)
    scaf = scaf0.sample(tick_times, loop=loop)

    # 3. the organism over the formed grid
    score = score_from(genome, scaf, senses=senses, threshold=threshold,
                       mode=mode, form=f.to_dict())
    write_score(score, out / "score.json")

    # 4. voices -> corpus sounds; the spine rides underneath
    voices = sorted(score["matrix"].keys())
    voice_map, provenance = auto_voices(voices, corpus_dir=corpus, seed=seed,
                                        overrides=voice_overrides)
    voice_map["spine_gain"] = spine_gain
    mix = mapped_audio(score, voice_map, tail_s=0.5, spine_loop=loop)

    # groove glue: the low voices duck the mix (the fusion recipe)
    if duck_depth > 0:
        low_times = [e["t"] for e in score["events"]
                     if e["voice"] in ("low", "pulse", "bass", "sub", "kick")
                     or "kick" in provenance.get(e["voice"], {}).get("family", "")]
        if low_times:
            mix = duck(mix, SR, low_times, depth=duck_depth)

    # 5. the mastering hand
    bounds = f.section_bounds(length)
    piece, post_ledger = post_process(mix, SR, bounds, seed=seed,
                                      intensity=post_intensity)

    wav_path = out / "piece.wav"
    write_wav(wav_path, piece.astype(np.float32), SR)

    manifest = {
        "maestro": "composition", "version": 1, "seed": seed,
        "spine": {"path": scaf.path, "sha256": scaf.sha256,
                  "spine_length_s": scaf.spine_length_s,
                  "loop": loop},
        "form": f.to_dict(),
        "genome": {"source": champion or genome_path or f"random:{random_seed}",
                   "genome": genome.to_dict()},
        "mode": mode, "threshold": threshold, "senses": list(senses),
        "voices": {v: {"file": d["file"], "gain": d.get("gain", 1.0),
                       "pan": d.get("pan", 0.0),
                       "family": provenance.get(v, {}).get("family", "")}
                   for v, d in voice_map.items() if v != "spine_gain"},
        "spine_gain": spine_gain, "duck_depth": duck_depth,
        "post": post_ledger,
        "score_summary": score_summary(score),
        "piece": {"path": str(wav_path), "sr": SR,
                  "length_s": round(float(piece.shape[0] / SR), 2)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    if verbose:
        s = score_summary(score)
        print(f"[maestro] form   : {form if isinstance(form, str) else 'custom'}")
        print(describe_form(f))
        print(f"[maestro] spine  : {spine} (loop {loop})")
        print(f"[maestro] score  : {s['n_events']} events,"
              f" {len(s['voices'])} voices"
              + (f" over grid {s['length_s']:.0f}s" if s['length_s'] else ""))
        for v, d in s["voices"].items():
            print(f"  {v:10s} {d['events']:5d} events  {d['density']:5.1f}/s"
                  f"  <- {Path(provenance.get(v, {}).get('file', '?')).name}")
        kinds = {}
        for e in post_ledger:
            kinds[e["type"]] = kinds.get(e["type"], 0) + 1
        print(f"[maestro] post   : " + ", ".join(f"{k}×{v}"
                                                 for k, v in sorted(kinds.items())))
        print(f"[maestro] piece  : {wav_path}"
              f" ({piece.shape[0] / SR:.1f}s)")
        print(f"[maestro] proven : {out / 'manifest.json'}")
    return manifest