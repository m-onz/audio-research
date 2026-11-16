"""clickstorm check: avalanche statistics + event rate at SOC."""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from womb.cryptid import build_champion, simulate

g = build_champion("clickstorm")
t0 = time.time()
L, R, rate, nodes = simulate(g, 30.0)
print(f"30s simulated in {time.time()-t0:.1f}s (sim rate {rate})")

# event stats from the raw ear: click impulses
mag = np.abs(L) + np.abs(R)
clicks = mag > 0.05
n_events = int(np.count_nonzero(clicks))  # may merge simultaneous
print(f"click samples > 0.05: {n_events} ({n_events/30:.0f}/s incl. bed)")
sizes = []
for amp in mag[clicks]:
    sizes.append(amp)
sizes = np.array(sizes)
if len(sizes):
    print(f"amp distribution: p10={np.percentile(sizes,10):.2f} p50={np.percentile(sizes,50):.2f} "
          f"p90={np.percentile(sizes,90):.2f} p99={np.percentile(sizes,99):.2f} max={sizes.max():.2f}")
    big = np.count_nonzero(sizes > 0.55)
    print(f"fraction big (>0.55): {100*big/len(sizes):.1f}%  (power-law-ish if small but nonzero)")
act = nodes[2].act
print(f"final activity EMA: {act:.3f}")
# bed level over time
bed = L[(np.abs(L) < 0.05)]
print(f"quiet-sample |L| mean (bed): {np.abs(bed).mean():.4f}")