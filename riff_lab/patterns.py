"""patterns: the alien Lisp pattern language + a generative grammar.

Ported and adapted from old/experiments/audio_lab/alien_rhythms.py — a
Python port of the Pure-Data `alien` pattern language. This replaces
riff_lab's hand-coded 16-step style templates with evolvable pattern
PROGRAMS: expressions evaluated every bar, so stochastic operators
(prob / drunk / rand / choose / shuffle) make a line breathe and never
quite repeat. Non-16 pattern lengths phase against the bar as polymeter.

A genome stores one expression per voice (an AST = nested lists). The
expression is the genetic material: mutation is AST surgery, crossover
grafts subtrees. The SAME expression exports straight into [alien] in
the patch (to_src), so what you evolve is what the patch plays.

Two value conventions:
  gate  — a 0/1 (or velocity) stream: non-rest = a hit. Used for noise
          and fill, where the folder index is chosen downstream.
  index — integer values 1..N (and beyond; they wrap modulo the folder,
          exactly as playdir~ does). Used for bass + its sub line, so
          the melodic contour lives in the pattern itself.
"""
import copy

import numpy as np

REST = None
STEP = 16


# ----------------------------------------------------------------- parse / print

def _atom(t):
    if t in ("-", ".", "_"):
        return REST
    try:
        return float(t) if ("." in t or "e" in t or "E" in t) else int(t)
    except ValueError:
        return t


def parse_pattern(src):
    toks = src.replace("(", " ( ").replace(")", " ) ").split()

    def read():
        if not toks:
            raise ValueError("unexpected end of pattern")
        t = toks.pop(0)
        if t == "(":
            out = []
            while toks and toks[0] != ")":
                out.append(read())
            if not toks:
                raise ValueError("missing )")
            toks.pop(0)
            return out
        if t == ")":
            raise ValueError("unexpected )")
        return _atom(t)

    ast = read()
    if toks:
        raise ValueError("trailing tokens after pattern")
    return ast


def to_src(a):
    if a is None:
        return "-"
    if isinstance(a, list):
        return "(" + " ".join(to_src(x) for x in a) + ")"
    if isinstance(a, float):
        return f"{a:g}"
    return str(a)


# ----------------------------------------------------------------- operators

def _as_seq(x):
    return x if isinstance(x, list) else [x]


def _mapv(s, f):
    return [None if v is None else f(v) for v in _as_seq(s)]


def _op_seq(a, rng):
    out = []
    for x in a:
        out.extend(_as_seq(x))
    return out


def _op_euclid(a, rng):
    hits, steps = a[0], int(a[1])
    rot = int(a[2]) if len(a) > 2 else 0
    val = a[3] if len(a) > 3 else 1.0
    if steps <= 0:
        return []
    hits_seq = _as_seq(hits) if isinstance(hits, list) else None
    k = len(hits_seq) if hits_seq else int(hits)
    if k <= 0:
        return [REST] * steps
    pat = [0] * steps
    if k >= steps:
        pat = [1] * steps
    else:
        bucket = 0
        for i in range(steps):
            bucket += k
            if bucket >= steps:
                bucket -= steps
                pat[i] = 1
    out, j = [], 0
    for i in range(steps):
        if pat[(i + rot) % steps]:
            out.append(hits_seq[j % len(hits_seq)] if hits_seq else val)
            j += 1
        else:
            out.append(REST)
    return out


def _fold1(v, lo, hi):
    if hi <= lo:
        return v
    span = hi - lo
    v = (v - lo) % (2 * span)
    return lo + (v if v <= span else 2 * span - v)


def _rotseq(s, n):
    if not s:
        return s
    n %= len(s)
    return s[-n:] + s[:-n]


def _interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def _drunk(rng, a):
    count, step, v = int(a[0]), abs(float(a[1])), float(a[2])
    lo = float(a[3]) if len(a) > 3 else None
    hi = float(a[4]) if len(a) > 4 else None
    out = []
    for _ in range(max(0, count)):
        v += rng.uniform(-step, step)
        if lo is not None and hi is not None and hi > lo:
            v = _fold1(v, lo, hi)
        out.append(v)
    return out


def _arp(notes, direction, length):
    notes = [n for n in notes if n is not None]
    if not notes or length <= 0:
        return []
    if direction == 1:
        seq = list(reversed(notes))
    elif direction == 2:
        seq = notes + list(reversed(notes[1:-1])) if len(notes) > 2 else notes
    else:
        seq = notes
    return [seq[i % len(seq)] for i in range(length)]


def _grow(s):
    out = []
    for i in range(1, len(s) + 1):
        out += s[:i] + [REST] * (len(s) - i)
    return out


def _mask(src, gate):
    out, j = [], 0
    for g in gate:
        if g is None:
            out.append(REST)
        else:
            out.append(src[j % len(src)] if src else REST)
            j += 1
    return out


OPS = {
    "seq": _op_seq,
    "rep": lambda a, rng: _as_seq(a[0]) * max(0, int(a[1])),
    "add": lambda a, rng: _mapv(a[0], lambda v: v + a[1]),
    "sub": lambda a, rng: _mapv(a[0], lambda v: v - a[1]),
    "mul": lambda a, rng: _mapv(a[0], lambda v: v * a[1]),
    "mod": lambda a, rng: _mapv(a[0], lambda v: v % a[1] if a[1] else v),
    "clamp": lambda a, rng: _mapv(a[0], lambda v: min(max(v, a[1]), a[2])),
    "fold": lambda a, rng: _mapv(a[0], lambda v: _fold1(v, a[1], a[2])),
    "euclid": _op_euclid,
    "subdiv": lambda a, rng: [v for v in _as_seq(a[0]) for _ in range(max(1, int(a[1])))],
    "reverse": lambda a, rng: list(reversed(_as_seq(a[0]))),
    "rotate": lambda a, rng: _rotseq(_as_seq(a[0]), int(a[1])),
    "interleave": lambda a, rng: _interleave(_as_seq(a[0]), _as_seq(a[1])),
    "shuffle": lambda a, rng: list(rng.permutation(np.asarray(_as_seq(a[0]), object))),
    "mirror": lambda a, rng: _as_seq(a[0]) + list(reversed(_as_seq(a[0])[:-1])),
    "take": lambda a, rng: _as_seq(a[0])[: max(0, int(a[1]))],
    "drop": lambda a, rng: _as_seq(a[0])[max(0, int(a[1])):],
    "every": lambda a, rng: _as_seq(a[0])[:: max(1, int(a[1]))],
    "range": lambda a, rng: list(range(int(a[0]), int(a[1]) + 1,
                                       int(a[2]) if len(a) > 2 else 1)),
    "ramp": lambda a, rng: [a[0] + (a[1] - a[0]) * i / max(1, int(a[2]) - 1)
                           for i in range(max(1, int(a[2])))],
    "choose": lambda a, rng: _as_seq(a[int(rng.integers(len(a)))]),
    "rand": lambda a, rng: [int(rng.integers(int(a[1] if len(a) > 1 else 0),
                                            int(a[2] if len(a) > 2 else 8) + 1))
                            for _ in range(max(0, int(a[0])))],
    "prob": lambda a, rng: [v if (v is not None and rng.random() * 100 < a[1])
                            else REST for v in _as_seq(a[0])],
    "drunk": lambda a, rng: _drunk(rng, a),
    "arp": lambda a, rng: _arp(_as_seq(a[0]), int(a[1]), int(a[2])),
    "cycle": lambda a, rng: (_as_seq(a[0]) * (int(a[1]) // max(1, len(_as_seq(a[0]))) + 1))[: int(a[1])],
    "grow": lambda a, rng: _grow(_as_seq(a[0])),
    "gate": lambda a, rng: [v if i % max(1, int(a[1])) == 0 else REST
                            for i, v in enumerate(_as_seq(a[0]))],
    "speed": lambda a, rng: [x for v in _as_seq(a[0])
                             for x in [v] + [REST] * (max(1, int(a[1])) - 1)],
    "mask": lambda a, rng: _mask(_as_seq(a[0]), _as_seq(a[1])),
    "delay": lambda a, rng: [REST] * max(0, int(a[1])) + _as_seq(a[0]),
}


def ev(node, rng):
    """Evaluate a pattern AST -> list of values (and REST=None)."""
    if not isinstance(node, list):
        return node
    op = node[0]
    if not isinstance(op, str):
        raise ValueError(f"bad operator {op!r}")
    f = OPS.get(op)
    if f is None:
        raise ValueError(f"unknown operator {op!r}")
    args = [ev(a, rng) for a in node[1:]]
    return f(args, rng)


# ----------------------------------------------------------------- the grammar

def gen_value_expr(rng, span):
    """A melodic-ish integer contour in 1..span (bass index stream)."""
    r = rng.random()
    if r < 0.33:
        return ["arp", ["range", 1, int(span)],
                int(rng.integers(0, 3)), int(rng.integers(4, 12))]
    if r < 0.6:
        return ["drunk", int(rng.integers(4, 10)), 1.0,
                float(rng.integers(1, span + 1)), 1.0, float(span)]
    if r < 0.82:
        return ["seq"] + [int(rng.integers(1, span + 1))
                          for _ in range(int(rng.integers(2, 5)))]
    return ["rand", int(rng.integers(3, 8)), 1, int(span)]


def gen_gate(rng, steps=16, k_range=(3, 8), poly=0.25):
    """A euclidean gate, sometimes at a polymeter length (phases vs bar)."""
    if rng.random() < poly:
        steps = int(rng.choice(np.array([7, 9, 11, 12, 14, 20, 24])))
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    rot = int(rng.integers(0, steps))
    return ["euclid", k, steps, rot]


def _wrap(rng, pat):
    """One modifier wrap (the grammar's intricacy move)."""
    w = rng.random()
    if w < 0.18:
        return ["rotate", pat, int(rng.integers(1, 8))]
    if w < 0.36:
        return ["prob", pat, int(rng.integers(45, 95))]
    if w < 0.52:
        return ["speed", pat, int(rng.integers(1, 3))]   # ratchet / stutter feel
    if w < 0.66:
        return ["gate", pat, int(rng.integers(2, 4))]
    if w < 0.80:
        return ["subdiv", pat, 2] if rng.random() < 0.5 else ["mirror", pat]
    if w < 0.92:
        return ["interleave", pat, gen_gate(rng)]
    return ["cycle", pat, 16]


def gen_voice(rng, voice, span):
    """Generate one voice's pattern expression.

    bass: masked INDEX stream (melodic motif, values = folder indices).
    noise: sparser gate, hocket-ish, more wraps (intricate interjections).
    fill:  dense euclid, laddering values, lands the phrase.
    """
    if voice == "bass":
        gate = gen_gate(rng, k_range=(3, 7), poly=0.2)
        pat = ["mask", gen_value_expr(rng, span), gate]
        n_wrap = int(rng.integers(0, 3))
    elif voice == "noise":
        gate = gen_gate(rng, k_range=(2, 6), poly=0.4)
        pat = gate
        if rng.random() < 0.6:
            pat = ["mask", gen_value_expr(rng, span), pat]
        n_wrap = int(rng.integers(1, 4))
    else:  # fill
        gate = gen_gate(rng, k_range=(4, 10), poly=0.3)
        pat = ["mask", ["ramp", 1, float(span), int(rng.integers(4, 12))], gate]
        n_wrap = int(rng.integers(0, 2))
    for _ in range(n_wrap):
        pat = _wrap(rng, pat)
    return pat


# ----------------------------------------------------------------- drum archetypes
# Classic constrained DnB breaks (16-step backbones): kick + snare + hats +
# openhh + perc. The backbone is the recognizable groove; the FILL (below)
# is the evolvable, complexifying end-of-phrase element.

def _P(src):
    return parse_pattern(src)


ARCHETYPES = {
    # two-step: kick on 1 and the "& of 3", snare on 2 & 4, 8th hats
    "classic": {
        "kick":   _P("(seq 1 - - - - - - - - - 1 - - - - -)"),
        "snare":  _P("(seq - - - - 1 - - - - - - - 1 - - -)"),
        "hihat":  _P("(euclid 8 16 0)"),
        "openhh": _P("(seq - - - - - - - - 1 - - - - - - -)"),
        "perc":   _P("(prob (euclid 3 16 5) 45)"),
    },
    # jungle: amen-flavoured — busier kick, ghost snares, dense 16th hats
    "jungle": {
        "kick":   _P("(seq 1 - - - - - 1 - - - 1 - - - - -)"),
        "snare":  _P("(seq - - - - 1 - - - - - 2 - 1 - 2 -)"),
        "hihat":  _P("(euclid 13 16 0)"),
        "openhh": _P("(seq - - - - - - 1 - - - - - - - 1 -)"),
        "perc":   _P("(prob (euclid 5 16 3) 60)"),
    },
    # techstep: mechanical — kick 1 & 3, rigid backbeat, straight 16th hats
    "techstep": {
        "kick":   _P("(seq 1 - - - - - - - 1 - - - - - - -)"),
        "snare":  _P("(seq - - - - 1 - - - - - - - 1 - - -)"),
        "hihat":  _P("(euclid 12 16 0)"),
        "openhh": _P("(seq - - - - - - - - - - - - 1 - - -)"),
        "perc":   _P("(prob (euclid 2 16 7) 40)"),
    },
    # halftime: kick 1, snare on 3 only, sparse hats — heavy and slow-feeling
    "halftime": {
        "kick":   _P("(seq 1 - - - - - - - - - - - - - 1 -)"),
        "snare":  _P("(seq - - - - - - - - 1 - - - - - - -)"),
        "hihat":  _P("(euclid 6 16 0)"),
        "openhh": _P("(seq - - - - - - - - - - - - - - - -)"),
        "perc":   _P("(prob (euclid 3 16 9) 40)"),
    },
}
STYLES = tuple(ARCHETYPES)
FILL_VOICES = ("snare", "perc")          # which voices take the end fill


def gen_fill(rng, span=1):
    """An end-of-phrase fill: a dense, ratcheting 16-step figure. This is
    the EVOLVABLE part — mutation complexifies it (rolls, edits, stutters)."""
    k = int(rng.integers(3, 6))
    pat = ["subdiv", ["euclid", k, 8, int(rng.integers(8))], 2]   # 16th roll
    if rng.random() < 0.5:
        pat = ["prob", pat, int(rng.integers(60, 95))]
    if span > 1:
        pat = ["mask", ["ramp", 1, float(span), int(rng.integers(4, 10))], pat]
    if rng.random() < 0.4:
        pat = _wrap(rng, pat)
    return pat


def drum_full(backbone, lastbar, bars):
    """Assemble a voice's phrase: the backbone repeated, with `lastbar`
    (the fill, or the backbone again) on the final bar."""
    if bars <= 1:
        return copy.deepcopy(lastbar)
    return ["seq", ["rep", copy.deepcopy(backbone), int(bars - 1)],
            copy.deepcopy(lastbar)]


# ----------------------------------------------------------------- drums (legacy free gen)

def gen_drum(rng, voice, span=1):
    """DnB-sensible but evolvable drum patterns (16-step alien gates).
    Values index the drum folder (wrap modulo). Kick sparse, snare on the
    backbeat, hats rolling, openhh offbeat, perc euclidean ghosts."""
    def idx(expr):                               # mask a gate with an index
        return expr if span <= 1 else ["mask", ["rand", 16, 1, int(span)], expr]
    if voice == "kick":
        k = int(rng.choice([2, 2, 3, 3, 4]))
        pat = ["euclid", k, 16, 0]               # lands a hit on step 0
        if rng.random() < 0.4:
            pat = ["prob", pat, int(rng.integers(70, 100))]
    elif voice == "snare":
        pat = ["rotate", ["euclid", 2, 16, 0], 4]   # steps 4 & 12 (2 & 4)
        if rng.random() < 0.45:                       # ghost snares
            pat = ["interleave", pat, ["prob", ["euclid", 3, 16,
                   int(rng.integers(16))], int(rng.integers(20, 45))]]
    elif voice == "hihat":
        k = int(rng.integers(8, 15))
        pat = ["euclid", k, 16, int(rng.integers(16))]
        if rng.random() < 0.4:
            pat = ["speed", ["euclid", int(rng.integers(4, 8)), 16, 0],
                   int(rng.choice([1, 2]))]
    elif voice == "openhh":
        pat = ["rotate", ["euclid", int(rng.integers(2, 4)), 16, 0], 2]
        pat = ["prob", pat, int(rng.integers(55, 90))]
    else:  # perc / ghost
        k = int(rng.integers(3, 7))
        pat = ["prob", ["euclid", k, 16, int(rng.integers(16))],
               int(rng.integers(45, 85))]
    pat = idx(pat)
    if rng.random() < 0.3:
        pat = _wrap(rng, pat)
    return pat


# ----------------------------------------------------------------- mutation

def _tweak_number(ast, rng):
    positions = []

    def walk(n, path):
        if isinstance(n, list):
            for i, c in enumerate(n):
                walk(c, path + [i])
        elif isinstance(n, (int, float)) and not isinstance(n, bool):
            positions.append(path)

    walk(ast, [])
    if not positions:
        return ast
    path = positions[int(rng.integers(len(positions)))]
    cur = ast
    for i in path[:-1]:
        cur = cur[i]
    old = cur[path[-1]]
    if isinstance(old, int):
        cur[path[-1]] = max(0, old + int(rng.integers(-2, 3)))
    else:
        cur[path[-1]] = float(np.clip(old * rng.uniform(0.6, 1.5), 0.05, 32.0))
    return ast


MAX_NODES = 28          # complexity cap — nest deep, but not to bloat
_UNWRAP = ("rotate", "prob", "speed", "gate", "subdiv", "mirror",
           "cycle", "interleave", "delay", "every")


def complexity(ast):
    """Node count of an expression — the complexification measure."""
    if not isinstance(ast, list):
        return 1
    return 1 + sum(complexity(a) for a in ast)


def _cap(ast, max_nodes=MAX_NODES):
    """Keep an expression intricate but playable: while oversized, drop the
    outermost op and keep its main operand (ast[1]) — nesting shrinks, the
    motif survives."""
    guard = 0
    while complexity(ast) > max_nodes and guard < 24:
        if isinstance(ast, list) and len(ast) >= 2 and isinstance(ast[0], str):
            ast = ast[1]
        else:
            break
        guard += 1
    return ast


def mutate_expr(ast, rng, voice="noise", span=4):
    """AST surgery biased toward GROWTH (complexification): mostly wrap /
    deepen / tweak; shrink only when oversized; rarely reset. Capped at
    MAX_NODES so it nests deep without bloating into unusable soup."""
    ast = copy.deepcopy(ast)
    size = complexity(ast)
    r = rng.random()
    if size > MAX_NODES:                            # oversized: simplify
        if isinstance(ast, list) and ast and ast[0] in _UNWRAP and r < 0.6:
            return ast[1]
        return _cap(_tweak_number(ast, rng))
    if r < 0.42:
        out = _wrap(rng, ast)                       # grow (nest one op)
    elif r < 0.62:
        out = _wrap(rng, _wrap(rng, ast))           # grow deeper
    elif r < 0.82:
        out = _wrap(rng, _tweak_number(ast, rng))   # grow + retune
    elif r < 0.92:
        out = _tweak_number(ast, rng)               # neutral drift
    elif r < 0.97 and isinstance(ast, list) and ast and ast[0] in _UNWRAP:
        out = ast[1]                                # rare prune
    else:
        out = gen_voice(rng, voice, span)           # rare fresh idea
    return _cap(out)


def crossover_expr(a, b, rng):
    """Graft: wrap one parent's body with the other's outermost op, or
    interleave them — subtree recombination at the expression level."""
    r = rng.random()
    if r < 0.4:
        out = ["interleave", copy.deepcopy(a), copy.deepcopy(b)]
    elif r < 0.7 and isinstance(b, list) and len(b) >= 2:
        out = copy.deepcopy(b)
        out[1] = copy.deepcopy(a)                   # a wears b's outer op
    else:
        out = copy.deepcopy(a if rng.random() < 0.5 else b)
    return _cap(out)


# ----------------------------------------------------------------- evaluation

def eval_bar(ast, rng, steps=STEP):
    """Evaluate an expression to exactly `steps` values (pad/truncate the
    program's natural length; non-16 lengths are the polymeter phasing,
    which collect_events handles by cycling the full stream)."""
    try:
        out = ev(copy.deepcopy(ast), rng)
    except Exception:
        return [REST] * steps
    out = _as_seq(out)
    if len(out) < steps:
        out = (out * (steps // max(1, len(out)) + 1))[:steps]
    return out[:steps]


def eval_stream(ast, rng, n_steps):
    """Evaluate to a flat stream of exactly n_steps values, re-evaluating
    the program as many times as needed (stochastic ops breathe across
    repeats). This is the polymeter-native read: the program's own length
    phases against the bar grid."""
    out = []
    guard = 0
    while len(out) < n_steps and guard < 4096:
        chunk = _as_seq(ev(copy.deepcopy(ast), rng))
        if not chunk:
            chunk = [REST]
        out.extend(chunk)
        guard += len(chunk)
    return out[:n_steps]


def is_hit(v, thr=0.05):
    return v is not None and (not isinstance(v, str)) and float(v) > thr
