"""Pipeline: the high-level `compose()` API.

One call carries a piece from corpus to finished .wav + manifest:

    import loom
    loom.compose(sources="corpus/", length=180, preset="drift", seed=3)

Steps: open sources -> chunk (strategy) -> catalog (analyze, feature ranks)
-> processing passes (re-processing, optional derivation) -> select ->
arrange (strategy) -> Score -> render -> .wav + manifest.json.

Every stochastic step draws from one seeded RNG tree, so a given seed
reproduces the piece exactly. The manifest records the fully-resolved
configuration, the pass lineage, and the output's checksum.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import __version__
from .arrangers import make_arranger
from .catalog import Catalog
from .chunkers import make_chunker
from .curves import Curve
from .processors import Processor, build_chain
from .rng import Seeded
from .render import render
from .score import Score
from .sources import Sources, open_sources
from .store import open_store

__all__ = ["compose", "ComposeResult", "run_passes", "_UNSET"]

_UNSET = object()  # sentinel: distinguishes "default" from "explicitly passed"


def _jsonable(obj):
    """Make a config value JSON-safe for the manifest."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Curve):
        return obj.spec
    if isinstance(obj, Processor):
        return f"<{type(obj).__name__} instance>"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


@dataclass
class ComposeResult:
    out: Path | None
    score: Score
    catalog: Catalog
    manifest: dict
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.manifest
        lines = [
            f"loom {__version__} | seed={m['seed']} | {m['length']}s @{m['sr']}Hz",
            f"sources: {', '.join(m['sources'])}",
            f"chunks: {m['catalog']['n_chunks']} | tags: {m['catalog'].get('tags') or '-'}",
            f"events: {m['score']['events']} using {m['score']['chunks_used']} chunks",
        ]
        if self.out is not None:
            lines.append(f"out: {self.out} ({self.stats.get('seconds', '?')}s, "
                         f"peak {self.stats.get('peak', 0):.3f}, {self.stats.get('bits')} bit)")
        return "\n".join(lines)

    def close(self) -> None:
        """Release derived-chunk storage (the catalog/score become unusable)."""
        if self.catalog.store is not None:
            self.catalog.store.close()


def compose(
    sources,
    length: float = _UNSET,
    out: str | Path | None = None,
    preset: str | None = None,
    seed: int = _UNSET,
    sr: int | None = _UNSET,
    chunker: str = _UNSET,
    chunker_params: dict | None = _UNSET,
    select: str = _UNSET,
    passes: Sequence[dict] | None = _UNSET,
    arranger: str = _UNSET,
    arranger_params: dict | None = _UNSET,
    processors=_UNSET,
    normalize: bool = _UNSET,
    bits: int = _UNSET,
    soft_clip: bool = _UNSET,
    store: str = _UNSET,
    verbose: bool = False,
    dry_run: bool = False,
    **overrides,
) -> ComposeResult:
    """Compose a piece. See README for the full parameter guide.

    `sources`: a .wav path, a folder of .wavs, or a list of either.
    `preset`: named starting point (see loom/presets.py); explicit kwargs win.
    `passes`: list of re-processing passes, each
        {"select": "rms>0.7", "processors": [...], "tag": "dark",
         "replace": False}
    `processors`: render-time chain applied per event (spec: see
        processors.build_chain).
    """
    from .presets import BASE_DEFAULTS, PRESETS, apply_preset

    if overrides:
        raise TypeError(f"unknown compose() argument(s): {sorted(overrides)}")
    explicit = {
        k: v
        for k, v in dict(
            length=length,
            seed=seed,
            sr=sr,
            chunker=chunker,
            chunker_params=chunker_params,
            select=select,
            passes=passes,
            arranger=arranger,
            arranger_params=arranger_params,
            processors=processors,
            normalize=normalize,
            bits=bits,
            soft_clip=soft_clip,
            store=store,
        ).items()
        if v is not _UNSET
    }
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
        config = apply_preset(preset, explicit)
    else:
        config = dict(BASE_DEFAULTS)
        config.update(explicit)
    for k in config:
        if k not in BASE_DEFAULTS:
            raise TypeError(f"unknown compose() argument {k!r}")

    length = float(config["length"])
    seed = int(config["seed"])
    if length <= 0:
        raise ValueError("length must be positive seconds")

    if verbose:
        t_start = time.time()
    srcs: Sources = open_sources(sources, sr=config["sr"])
    rng = Seeded(seed)

    # 1. chunk ---------------------------------------------------------------
    chunker_fn = make_chunker(config["chunker"])
    refs = chunker_fn(srcs, rng.spawn("chunk"), **dict(config["chunker_params"] or {}))
    if not refs:
        raise RuntimeError("chunker produced no chunks")
    if verbose:
        print(f"[loom] {len(srcs)} source(s), {len(refs)} chunks")

    # 2. catalog (analyze) -----------------------------------------------------
    store_obj = open_store(config["store"])
    try:
        catalog = Catalog(srcs, store_obj)
        catalog.add_refs(refs)
        if verbose:
            print(f"[loom] catalog: {catalog.n} analyzed chunks")

        # 3. re-processing passes ----------------------------------------------------
        if config["passes"]:
            catalog = run_passes(catalog, list(config["passes"]), rng.spawn("passes"), verbose=verbose)

        # 4. select + arrange --------------------------------------------------------
        ids = catalog.query(config["select"])
        if not ids:
            raise RuntimeError(f"selection {config['select']!r} matched no chunks")
        chain = build_chain(config["processors"])
        arranger_fn = make_arranger(config["arranger"])
        events = arranger_fn(
            catalog,
            ids,
            length,
            srcs.sr,
            rng.spawn("arrange"),
            dict(config["arranger_params"] or {}),
            chain,
        )
        events = [e for e in events if e.t < length]
        if not events:
            raise RuntimeError("arranger produced no events")
        score = Score(sr=srcs.sr, length=length, seed=seed, events=events)
        if verbose:
            print(f"[loom] score: {len(events)} events, coverage {score.summary()['coverage']}x")
    except Exception:
        store_obj.close()  # never leak a temp store on failure
        raise

    # 5. manifest skeleton --------------------------------------------------------
    manifest = {
        "loom_version": __version__,
        "seed": seed,
        "length": length,
        "sr": srcs.sr,
        "sources": srcs.names,
        "config": {k: _jsonable(v) for k, v in config.items() if k != "sr"},
        "catalog": catalog.summary(),
        "score": score.summary(),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if dry_run:
        if verbose:
            print("[loom] dry run: no render")
        return ComposeResult(None, score, catalog, manifest)

    # 6. render -----------------------------------------------------------------
    if out is None:
        name = f"{preset or config['arranger']}_{seed}.wav"
        out = Path("loom_out") / name
    stats = render(
        score,
        catalog,
        out,
        bits=int(config["bits"]),
        normalize=bool(config["normalize"]),
        soft_clip=bool(config["soft_clip"]),
        verbose=verbose,
    )
    manifest["render"] = {k: v for k, v in stats.items() if k != "path"}

    # 7. write manifest next to the wav ---------------------------------------------
    out_path = Path(stats["path"])
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest["sha256"] = digest
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    result = ComposeResult(out_path, score, catalog, manifest, stats)
    if verbose:
        print(f"[loom] wrote {out_path} in {time.time() - t_start:.1f}s")
    for s in srcs.items:
        s.close()
    return result


def run_passes(catalog: Catalog, passes: list[dict], rng: Seeded, verbose: bool = False) -> Catalog:
    """Apply re-processing passes in order.

    Each pass: {"select": cond, "processors": chain-spec, "tag": name,
    "replace": bool}. Matching chunks are processed; by default the result is
    *derived* (appended, keeping the parent), with replace=True the chunk is
    transformed in place (id preserved). Passes compose: later passes can
    select on earlier passes' tags (``"tag=dark"``).
    """
    for pi, spec in enumerate(passes):
        spec = dict(spec or {})
        cond = str(spec.get("select", "all"))
        ids = catalog.query(cond)
        if not ids:
            if verbose:
                print(f"[loom] pass {pi}: {cond!r} matched nothing, skipped")
            continue
        chain = build_chain(spec.get("processors"))
        tag = spec.get("tag")
        replace = bool(spec.get("replace", False))
        lineage = tuple(step["name"] for step in chain.describe()) if chain.steps else ()
        n_done = 0
        for j, cid in enumerate(ids):
            chunk = catalog.chunks[cid]
            audio = chunk.materialize(catalog.sources, catalog.store)
            if audio.shape[0] == 0:
                continue
            audio = chain.run(audio, catalog.sources.sr, j / max(1, len(ids)), rng.spawn(f"p{pi}c{cid}"))
            if audio.shape[0] == 0:
                continue
            catalog.add_derived(chunk, audio, tag, lineage, replace=replace)
            n_done += 1
        if verbose:
            kind = "replaced" if replace else "derived"
            print(f"[loom] pass {pi}: {cond!r} -> {n_done} {kind} chunks [{lineage}] tag={tag}")
    return catalog