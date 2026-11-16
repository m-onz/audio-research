"""chimera check: does the ring keep a sustained coherence split?"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from womb.cryptid import make_law

RATE = 4410

def check(radius, k, alpha, spread, freq=110.0, secs=40.0, seed=101, n=128, windows=16):
    law = make_law("osc_pop", dict(n=n, radius=radius, k=k, alpha=alpha, freq=freq,
                                   spread=spread, windows=windows, kernel="ring",
                                   ic="half", seed=seed), seed, RATE)
    dt = 1.0 / RATE
    steps = int(secs * RATE)
    rec = []
    for t in range(steps):
        law.step(t, dt, [])
        if t % RATE == 0:  # once per second
            rec.append((law.win["coh"].max(), law.win["coh"].min(),
                        float(np.argmax(law.win["coh"]))))
    rec = np.array(rec)
    mx, mn, am = rec[:, 0], rec[:, 1], rec[:, 2]
    split_frac = float(np.mean((mx > 0.75) & (mn < 0.45)))
    # wander: std of the bump position over the last 3/4
    wander = float(np.std(am[len(am)//4:]))
    return split_frac, wander, float(mx.mean()), float(mn.mean())

if __name__ == "__main__":
    print(f"{'radius':>7} {'k':>6} {'alpha':>6} {'spread':>7} | {'split%':>7} {'wander':>7} {'meanmax':>8} {'meanmin':>8}")
    for radius in (22, 34, 48):
        for k in (0.05, 0.1, 0.2, 0.4, 0.8):
            for alpha in (1.38, 1.46, 1.52):
                for spread in (0.0, 1.0, 3.0):
                    t0 = time.time()
                    sf, wa, mm, mn = check(radius, k, alpha, spread)
                    if sf > 0.3:
                        print(f"{radius:>7} {k:>6} {alpha:>6} {spread:>7} | {100*sf:>6.0f}% {wa:>7.2f} {mm:>8.3f} {mn:>8.3f}  ({time.time()-t0:.0f}s)")