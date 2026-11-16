"""jerk search v4: the full sparse Sprott jerk family, Lyapunov as judge.

    x''' = a x'' + b x' + c x + d x^3 + e x^2 + f x x' + g x'^2

Random coefficients, each nonlinearity active with p=0.4 (Sprott's
systems are sparse). Judge: bounded + aperiodic, then Lyapunov
divergence > 50x. Then audio-rate strangeness + modulation survival.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np

def sim(co, x0, secs, rate):
    a, b, c, d, e, f, g = co
    dt = 1.0 / rate
    n = int(secs * rate)
    x, v, ac = x0, 0.0, 0.0
    out = np.empty(n)
    for i in range(n):
        k1a = a*ac + b*v + c*x + d*x*x*x + e*x*x + f*x*v + g*v*v
        xp, vp, ap = x + v*dt, v + ac*dt, ac + k1a*dt
        k2a = a*ap + b*vp + c*xp + d*xp*xp*xp + e*xp*xp + f*xp*vp + g*vp*vp
        x = x + 0.5*dt*(v + vp)
        v = v + 0.5*dt*(ac + ap)
        ac = ac + 0.5*dt*(k1a + k2a)
        if not (-80 < x < 80 and abs(v) < 900 and abs(ac) < 15000):
            return None
        out[i] = x
    return out

def alive(x):
    if x is None:
        return False
    w = x[len(x)//5:]
    rms = float(np.sqrt((w*w).mean()))
    zc = int(np.count_nonzero(np.diff(np.sign(w)) > 0))
    return 0.02 < rms < 0.9 and zc >= 20

def acpeak(x):
    w = x[len(x)//10:]
    w = w - w.mean()
    ac = np.correlate(w[:len(w)//2], w[:len(w)//2], "full")[len(w)//2:]
    ac = ac / (ac[0] + 1e-12)
    neg = np.nonzero(ac < 0)[0]
    lag0 = int(neg[0]) if len(neg) else 20
    return float(ac[max(lag0, 20):].max())

def lyap(co, x0, secs=80.0, rate=60.0, eps=1e-6):
    x1 = sim(co, x0, secs, rate)
    x2 = sim(co, x0 + eps, secs, rate)
    if x1 is None or x2 is None:
        return 0.0
    return float(np.abs(x1 - x2)[int(rate):].max() / eps)

def audio_render(co, x0, speed=110.0, sim_rate=11025, secs=3.0, ode_rate=200.0):
    a, b, c, d, e, f, g = co
    dt_audio = 1.0 / sim_rate
    dt_ode = 1.0 / ode_rate
    ksub = max(1, int(round(speed * dt_audio / dt_ode)))
    n = int(secs * sim_rate)
    x, v, ac = x0, 0.0, 0.0
    out = np.empty(n)
    for i in range(n):
        for _ in range(ksub):
            k1a = a*ac + b*v + c*x + d*x*x*x + e*x*x + f*x*v + g*v*v
            xp, vp, ap = x + v*dt_ode, v + ac*dt_ode, ac + k1a*dt_ode
            k2a = a*ap + b*vp + c*xp + d*xp*xp*xp + e*xp*xp + f*xp*vp + g*vp*vp
            x = x + 0.5*dt_ode*(v + vp)
            v = v + 0.5*dt_ode*(ac + ap)
            ac = ac + 0.5*dt_ode*(k1a + k2a)
            if not (-80 < x < 80 and abs(v) < 900 and abs(ac) < 15000):
                return None
        out[i] = x
    return out

def metrics(x, rate=11025):
    if x is None:
        return None
    w = x[len(x)//10:]
    rms = float(np.sqrt((w*w).mean()))
    nfft = 4096
    m = len(w)//nfft*nfft
    if m < nfft:
        return None
    seg = w[:m].reshape(-1, nfft)
    S = np.abs(np.fft.rfft(seg*np.hanning(nfft), axis=1)).mean(0) + 1e-12
    flat = float(np.exp(np.log(S).mean()) / S.mean())
    fr = np.fft.rfftfreq(nfft, 1/rate)
    cent = float((fr*S).sum()/S.sum())
    return dict(rms=rms, flat=flat, cent=cent)

if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    t0 = time.time()
    N = 8000
    hits = []
    for i in range(N):
        co = [float(rng.uniform(-2.5, -0.1))]          # a: damping
        co.append(float(rng.uniform(-3.0, 3.0)))      # b
        co.append(float(rng.uniform(-3.0, 3.0)))      # c
        for _ in range(4):                            # d,e,f,g sparse
            co.append(float(rng.uniform(-3.0, 3.0)) if rng.random() < 0.4 else 0.0)
        x0 = float(rng.choice([0.4, 1.5, -1.5]))
        x = sim(co, x0, 150.0, 60.0)
        if not alive(x):
            continue
        if acpeak(x) > 0.85:
            continue
        dv = lyap(co, x0)
        if dv > 50:
            hits.append((co, x0, dv))
    print(f"pass 1-2: {len(hits)}/{N} chaotic (lyap>50) in {time.time()-t0:.0f}s")
    for co, x0, dv in hits[:20]:
        act = [i for i, v in enumerate(co[3:]) if v != 0]
        print(f"   div={dv:8.0f}  a={co[0]:+.2f} b={co[1]:+.2f} c={co[2]:+.2f} "
              f"nl[{','.join('defg'[i]+'=' + f'{co[3+i]:+.2f}' for i in act)}] x0={x0:+.1f}")

    print("pass 3: audio strangeness + modulation survival ...")
    final = []
    for co, x0, dv in hits:
        ms = []
        for cc in (co[2], co[2]*1.22, co[2]*0.78):
            c2 = list(co); c2[2] = cc
            ms.append(metrics(audio_render(tuple(c2), x0)))
        if any(m is None for m in ms):
            continue
        m = ms[0]
        score = (0.4*max(0.0, 1.0 - abs(np.log10(m['flat']+1e-9) - np.log10(0.03))/1.5)
                 + 0.3*min(m['cent']/2000.0, 1.0) + 0.3*max(0.0, 1.0 - m['rms']*1.5))
        final.append((score, co, x0, m))
    final.sort(key=lambda r: -r[0])
    print(f"  {len(final)} through modulation; top 8 ({time.time()-t0:.0f}s):")
    for s, co, x0, m in final[:8]:
        act = [i for i, v in enumerate(co[3:]) if v != 0]
        print(f"  {s:.3f}  a={co[0]:+.2f} b={co[1]:+.2f} c={co[2]:+.2f} "
              f"nl[{','.join('defg'[i]+'='+f'{co[3+i]:+.2f}' for i in act)}] x0={x0:+.1f}"
              f"  rms={m['rms']:.3f} flat={m['flat']:.4f} cent={m['cent']:.0f}")