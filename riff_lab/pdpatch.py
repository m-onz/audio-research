"""pdpatch: write a Pure-Data patch per riff — the patterns as global
messages, ready to import into the live dnb-1 performance patch.

The patch holds one 'SEND ALL' message box (every voice's alien
expression as a `; <voice> <expr>` global send) wired to a loadbang +
a clickable bang, so opening the file (or clicking the bang) broadcasts
all patterns to the running patch at once. Individual per-voice message
boxes are laid out below for toggling single voices.

Voice names match the patch's alien voices (kick/snare/hihat/openhh/
perc + bass/noise/fill/sub), so the sends land directly.
"""


def _esc(s):
    """Escape a string for a Pd message-box record."""
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,") \
            .replace("$", "\\$")


def build_pd(title, clauses):
    """clauses: list of 'voice (expr)' strings. Returns .pd file text."""
    L = ["#N canvas 180 140 820 560 12;"]
    n = [0]

    def add(line):
        L.append(line); n[0] += 1
        return n[0] - 1

    tc = add(f"#X text 18 14 {_esc(title)};")
    add("#X text 18 34 tempo 80 ms/step. opening this (or clicking the bang) "
        "broadcasts every pattern to the running dnb-1 patch as global "
        "messages to each alien voice.;")
    lb = add("#X obj 26 74 loadbang;")
    bng = add("#X obj 26 98 bng 30 250 50 0 empty empty SEND-ALL 20 9 0 12 "
              "#e0e0e0 #000000 #000000;")
    # the combined global send: '\; v1 expr \; v2 expr ...'
    allmsg = " ".join("\\; " + _esc(c) for c in clauses)
    msg_all = add(f"#X msg 26 150 {allmsg};")
    add("#X connect {} 0 {} 0;".format(lb, bng))
    add("#X connect {} 0 {} 0;".format(bng, msg_all))
    # individual per-voice sends, clickable
    y = 210
    add(f"#X text 18 {y - 24} per-voice (click to send just one):;")
    for c in clauses:
        add(f"#X msg 26 {y} \\; {_esc(c)};")
        y += 26
    return "\n".join(L) + "\n"
