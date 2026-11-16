"""ga: population management — generations of riff folders, breeding from
parents you liked, mutation of a single riff."""
import json
from pathlib import Path

import numpy as np

from . import build, genome, novelty, patterns as P, pool

RIFFS = Path("riffs")
MIN_BASS_HITS = 3          # MCNS viability: a DnB riff must have a bassline


def _viable_arrangement(g):
    """Minimal criterion on the arrangement (not a quality judgement): the
    bass must actually play, or it is not a DnB riff. Pure novelty would
    otherwise happily select silent-bass outliers as 'different'."""
    s = P.eval_stream(g["bass_expr"], np.random.default_rng(99), 32)
    return sum(1 for v in s if P.is_hit(v)) >= MIN_BASS_HITS


def next_gen_dir():
    RIFFS.mkdir(exist_ok=True)
    gens = sorted(int(p.name.split("_")[1]) for p in RIFFS.glob("gen_*"))
    return RIFFS / f"gen_{(gens[-1] + 1) if gens else 1:03d}"


def _prepare_child(g, pl, strictness):
    """Evaluate an individual: prepare the riff (defects are the MCNS
    minimal criterion — a defective child is inviable), echo-encode the
    rendered preview (the behavioural characterization), then DISCARD the
    audio. Only the 556-float vector + genome are carried, so the batch
    never holds rendered audio — memory stays flat at any --n. Winners are
    deterministically re-prepared for writing (same seed, same audio)."""
    if not _viable_arrangement(g):          # no bassline -> inviable (MCNS)
        return None
    try:
        prep = build.prepare_riff(g, pl, strictness=strictness)
    except Exception:
        return None
    if prep is None:
        return None
    avec = novelty.encode_audio(prep["preview"])
    desc, comp = novelty.pattern_descriptor(g)
    return dict(genome=g, avec=avec, desc=desc, complexity=comp)


def _reproduce(population, rng, n_children, seeds, i0, bars, genre, bpm):
    """Canonical reproduction: the most novel of the population
    reproduce via mutation/crossover, with fresh-random immigration
    (the papers' search never collapses because novelty is relative,
    but immigration keeps the gene pool honest)."""
    children = []
    # low immigration (15%) so lineages PERSIST and complexify across gens
    # instead of half-resetting to simple random genomes every generation.
    n_immig = max(1, int(0.15 * n_children)) if population else n_children
    for i in range(n_children):
        if not population or i < n_immig:
            g = genome.random_genome(seeds + (i0 + i) * 101,
                                     bars=bars, genre=genre, bpm=bpm)
            g["strictness"] = 0.0
            children.append(g)
            continue
        # tournament of two (the GP paper's low-pressure selection)
        a, b = rng.choice(len(population), size=2, replace=False)
        p1 = population[a] if population[a]["novelty"] >= population[b]["novelty"] \
            else population[b]
        c, d = rng.choice(len(population), size=2, replace=False)
        p2 = population[c] if population[c]["novelty"] >= population[d]["novelty"] \
            else population[d]
        r = rng.random()
        if r < 0.45 and p1 is not p2:
            g = genome.crossover(p1["genome"], p2["genome"], rng)
        else:
            rate = float(rng.choice(
                [rng.uniform(0.05, 0.15),
                 rng.uniform(0.2, 0.4),
                 rng.uniform(0.4, 0.6)]))
            g = genome.mutate(p1["genome"], rng, rate=rate)
        g["seed"] = int(rng.integers(2 ** 31))
        children.append(g)
    return children


def gen_random(n=6, bars=None, genre=None, bpm=None, seed0=None,
               candidates_x=3, use_novelty=True, gens=1,
               strictness=None):
    """CANONICAL novelty search (Lehman & Stanley):

      - behavior = echo-encoded rendered preview
      - novelty = sparseness (mean k-NN distance, k=15) against the
        ARCHIVE UNION POPULATION UNION BATCH
      - selection = rank by pure novelty, top n built
      - the most novel reproduce (tournament of two, low pressure)
      - archive admission = dynamic threshold (stall -5% / flood +20%)
        + 1% stochastic adds
      - minimal criteria (MCNS) = ears.defects only — defective children
        are inviable; NO other objective, character gate, or taste
        filter touches selection

    A persistent population carries across --gens; each generation the
    population reproduces, children are evaluated, the most novel are
    built and join the population (least-novel pruned at the cap)."""
    # pure novelty: strictness 0 = defects-only minimal criterion
    strictness = 0.0 if strictness is None else float(strictness)
    pl = pool.build()
    arch = novelty.NoveltyArchive()
    rng = np.random.default_rng(int(seed0 or
                                    np.random.default_rng().integers(2 ** 31)))
    seeds = int(rng.integers(2 ** 31))
    population = []          # dicts: genome, avec, novelty (in-memory)
    POP_CAP = 48
    made_all = []
    for gen_i in range(max(1, gens)):
        gdir = next_gen_dir()
        # ---- reproduce (or fresh seeds when the population is empty)
        n_children = n * max(1, candidates_x)
        genomes = _reproduce(population, rng, n_children, seeds,
                              gen_i * 1000, bars, genre, bpm)
        # ---- evaluate: prepare + echo-encode (behavior = the audio)
        children = []
        for g in genomes:
            ch = _prepare_child(g, pl, strictness)
            if ch is not None:
                children.append(ch)
        if not children:
            continue
        # ---- pure novelty ranking (archive u population u batch)
        winners = novelty.select_by_novelty(children, n, arch)
        # ---- build, admit to archive, print. Each write is fault-tolerant
        # (one bad render must never kill a long --gens run), and prep (the
        # rendered audio) is dropped the instant it is written so memory
        # stays flat no matter how large --n / --candidates is.
        made = []
        for i, w in enumerate(winners):
            out = gdir / f"riff_{i + 1:03d}"
            try:
                prep = build.prepare_riff(w["genome"], pl, strictness=strictness)
                build.write_riff(prep, out)
            except Exception as e:
                print(f"  ! skipped {out}: {type(e).__name__}: {e}")
                continue
            arch.consider(str(out), w["desc"], w["avec"], novelty=w["novelty"])
            made.append(out)
            g = w["genome"]
            print(f"  {out}  ({g['genre']} {g['bpm']}bpm {g['bars']} bars "
                  f"src={g['source_mode']['bass']}/{g['source_mode']['noise']}/"
                  f"{g['source_mode']['fill']})  novelty={w['novelty']:.2f}")
        # ---- population update: winners + top runners-up join; prune.
        for w in winners:
            population.append(w)
        for w in children[len(winners): len(winners) + n]:
            population.append(w)                    # unbuilt novelty seeds
        population.sort(key=lambda c: -c["novelty"])
        population = population[: POP_CAP]
        children = None                              # free the batch
        arch.note_generation()
        # diversity metrics per generation
        novs = [c["novelty"] for c in winners] or [0.0]
        from collections import Counter
        srcs = Counter(w["genome"]["source_mode"]["bass"] for w in winners)
        comps = [w.get("complexity", 0) for w in winners]
        mc = float(np.mean(comps)) if comps else 0.0
        print(f"generation -> {gdir}  ({len(made)} riffs, archive "
              f"{len(arch)}, rho_min {arch.rho_min:.2f})")
        print(f"  diversity: novelty {min(novs):.2f}-{max(novs):.2f} | "
              f"complexity mean {mc:.0f} (max {max(comps) if comps else 0}) | "
              f"bass-src={dict(srcs)}")
        made_all += made
    return made_all


def breed(parent_dirs, n=6, mutation_rate=0.12):
    """parents: riff folders (in any order) — crossover + mutate."""
    pl = pool.build()
    parents = []
    for p in parent_dirs:
        with open(Path(p) / "_genome.json") as fh:
            parents.append(json.load(fh)["genome"])
    rng = np.random.default_rng(int(rng_int()))
    gdir = next_gen_dir()
    made = []
    for i in range(n):
        a = parents[int(rng.integers(len(parents)))]
        b = parents[int(rng.integers(len(parents)))]
        if len(parents) >= 2 and a is b:
            b = parents[(parents.index(a) + 1) % len(parents)]
        c = genome.crossover(a, b, rng)
        c = genome.mutate(c, rng, rate=mutation_rate)
        c["seed"] = int(rng.integers(2 ** 31))
        out = gdir / f"riff_{i + 1:03d}"
        made.append(build.build_riff(c, pl, out))
        print(f"  {out}  ({c['genre']} {c['bpm']}bpm {c['bars']} bars "
              f"<- {Path(parent_dirs[0]).name}...)")  # bred child
    print(f"generation -> {gdir}  ({n} children of "
          f"{len(parent_dirs)} parents)")
    return gdir, made


def mutate_one(parent_dir, n=4, mutation_rate=0.2):
    pl = pool.build()
    with open(Path(parent_dir) / "_genome.json") as fh:
        parent = json.load(fh)["genome"]
    rng = np.random.default_rng(int(rng_int()))
    gdir = next_gen_dir()
    made = []
    for i in range(n):
        m = genome.mutate(parent, rng, rate=mutation_rate)
        out = gdir / f"riff_{i + 1:03d}"
        made.append(build.build_riff(m, pl, out))
        print(f"  {out}")
    print(f"generation -> {gdir}  ({n} mutants of {parent_dir})")
    return gdir, made


def rng_int():
    return np.random.default_rng().integers(2 ** 31)