"""chromaria: Soros' experiment run inside womb.

Same seed, three strictness regimes (0.1 loose / 0.5 calibrated / 1.0
strict). Hypothesis (soros_alife16): mid-strictness yields orderly
divergence — admissions keep flowing, occupied cells and descriptor
spread rise; both extremes stagnate (loose: junk floods in, drifts
toward the silent basin, spread collapses; strict: almost nothing is
born, cells stay at the founder baseline).
"""
import sys, time, json
sys.path.insert(0, "src")
from womb.breed import seed_run, report_run

REGIMES = [(0.1, "loose"), (0.5, "calibrated"), (1.0, "strict")]
KW = dict(pop=20, ears=8, gens=12, pairs=2, seconds=3.0,
          champion_seeds=2, seed=7)

rows = []
for strictness, label in REGIMES:
    t0 = time.time()
    out = f"womb_out/chromaria_{label}"
    r = seed_run(out=out, strictness=strictness, verbose=True, **KW)
    log = r["log"]
    first, last = log[0], log[-1]
    spread0 = first["spread"]
    spreadN = last["spread"]
    rows.append({
        "regime": label, "strictness": strictness,
        "specimens": r["specimens"],
        "admissions": sum(g["proc_admissions"] for g in log),
        "cells_start": [first["cells"]["texture"], first["cells"]["time"], first["cells"]["space"]],
        "cells_final": [last["cells"]["texture"], last["cells"]["time"], last["cells"]["space"]],
        "spread_start": sum(spread0.values()) / len(spread0),
        "spread_final": sum(spreadN.values()) / len(spreadN),
        "mean_rms_start": first["mean_rms"], "mean_rms_final": last["mean_rms"],
        "elapsed": round(time.time() - t0, 1),
    })
    print()

print("=" * 88)
print(f"{'regime':>11} {'strict':>7} {'specs':>6} {'adm':>5} {'cells0':>9} {'cellsN':>9} "
      f"{'spread0':>8} {'spreadN':>8} {'rms0':>7} {'rmsN':>7}")
for row in rows:
    print(f"{row['regime']:>11} {row['strictness']:>7} {row['specimens']:>6} {row['admissions']:>5} "
          f"{'/'.join(map(str, row['cells_start'])):>9} {'/'.join(map(str, row['cells_final'])):>9} "
          f"{row['spread_start']:>8.3f} {row['spread_final']:>8.3f} "
          f"{row['mean_rms_start'] or 0:>7} {row['mean_rms_final'] or 0:>7}")
print("=" * 88)
print("hypothesis: calibrated dominates admissions + cells growth + spread;")
print("            loose collapses toward the silent basin; strict barely births.")
json.dump(rows, open("womb_out/chromaria_summary.json", "w"), indent=1)