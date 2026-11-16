"""tune bomb founders: hand-designed layered networks, iterated against
the corpus MC until they pass. The analysis said the material is dense
harmonic-noise that keeps moving; single-law voices are too pure."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from bomb import make_judge, SR
from womb.breed import render_pair

env = json.loads(open("corpus_envelope.json").read())
judge = make_judge(env, 0.5)

def founder_choir(freq=330.0, spread=4.0, pulse_gain=0.35, noise_gain=0.10, osc_g=0.5):
    proc = {"nodes": [
        {"law": "osc_pop", "params": dict(n=96, windows=12, radius=24, k=0.9,
                                         alpha=1.2, freq=freq, spread=spread,
                                         kernel="global", ic="uniform")},
        {"law": "excitable", "params": dict(a=0.7, b=0.8, eps=0.05, z0=0.45)},
        {"law": "map", "params": dict(r=3.97, period_s=2.7)},
    ], "edges": [
        {"src": 2, "ch": 0, "dst": 1, "gain": 1.0},
    ], "sim_rate": 4410, "seed": 5}
    ear = [
        {"op": "tap", "node": 0, "ch": 0, "gain": osc_g, "pan": -0.5},
        {"op": "tap", "node": 0, "ch": 1, "gain": 0.6 * osc_g, "pan": 0.6},
        {"op": "tap", "node": 1, "ch": 0, "gain": pulse_gain, "pan": 0.2},
        {"op": "events", "node": 1, "gain": noise_gain, "pan_spread": 0.8},
        {"op": "embed", "node": 0, "ch": 0, "delay": 300, "gain": 0.25 * osc_g, "pan": -0.1},
    ]
    return proc, ear

def founder_floor(freq=550.0, interval=0.004, osc_g=0.4, sand_g=0.30):
    """sandpile avalanche bed (dense-driven = smooth noise floor) + tonal ring above"""
    proc = {"nodes": [
        {"law": "sand", "params": dict(w=32, h=32, interval_s=interval, jitter=0.9,
                                       warmup=1200, act_tau=0.05)},
        {"law": "osc_pop", "params": dict(n=64, windows=8, radius=18, k=1.1,
                                         alpha=1.3, freq=freq, spread=3.0,
                                         kernel="ring", ic="uniform")},
        {"law": "map", "params": dict(r=3.93, period_s=3.5)},
    ], "edges": [], "sim_rate": 11025, "seed": 9}
    ear = [
        {"op": "tap", "node": 0, "ch": 0, "gain": sand_g, "pan": -0.3},
        {"op": "events", "node": 0, "gain": 0.22, "pan_spread": 0.9},
        {"op": "tap", "node": 1, "ch": 0, "gain": osc_g, "pan": 0.45},
        {"op": "tap", "node": 1, "ch": 1, "gain": 0.5 * osc_g, "pan": -0.65},
        {"op": "embed", "node": 1, "ch": 0, "delay": 220, "gain": 0.3 * osc_g, "pan": 0.1},
    ]
    return proc, ear

def founder_buzz(bias=2.0, osc_g=0.45, buzz_g=0.22):
    """fast fire pulse train (bright buzz) + tonal ring + clamp shadow"""
    proc = {"nodes": [
        {"law": "fire", "params": dict(bias=bias, theta=1.0, refr_s=0.004)},
        {"law": "clamp", "params": dict(gain=1.4, steps=6)},
        {"law": "osc_pop", "params": dict(n=48, windows=8, radius=14, k=1.2,
                                         alpha=1.25, freq=740.0, spread=5.0,
                                         kernel="global", ic="uniform")},
    ], "edges": [
        {"src": 0, "ch": 0, "dst": 1, "gain": 1.0},
    ], "sim_rate": 11025, "seed": 3}
    ear = [
        {"op": "tap", "node": 0, "ch": 0, "gain": buzz_g, "pan": -0.4},
        {"op": "tap", "node": 1, "ch": 0, "gain": buzz_g * 0.8, "pan": 0.5},
        {"op": "tap", "node": 2, "ch": 0, "gain": osc_g, "pan": 0.15},
        {"op": "tap", "node": 2, "ch": 1, "gain": 0.4 * osc_g, "pan": -0.7},
        {"op": "embed", "node": 2, "ch": 0, "delay": 150, "gain": 0.25 * osc_g, "pan": 0.35},
    ]
    return proc, ear

def report(name, proc, ear):
    try:
        audio = render_pair(proc, ear, 3.0)
        j = judge(audio, SR)
        worst = sorted(j["inside"].items(), key=lambda kv: kv[1])[:3]
        ok = "PASS" if j["viable"] else "fail"
        print(f"{name}: {ok} kinship={j['kinship']} rms={j['rms']} worst={[(k, v) for k, v in worst]}")
        return j
    except Exception as e:
        print(f"{name}: render error {e}")
        return None

if __name__ == "__main__":
    report("choir v1", *founder_choir())
    report("floor v1", *founder_floor())
    report("buzz  v1", *founder_buzz())
    # iterate: gains and brightness
    for osc_g in (0.35, 0.5):
        for pg in (0.2, 0.5):
            report(f"choir osc={osc_g} pulse={pg}", *founder_choir(osc_g=osc_g, pulse_gain=pg))
    for osc_g in (0.3, 0.55):
        for sg in (0.15, 0.4):
            report(f"floor osc={osc_g} sand={sg}", *founder_floor(osc_g=osc_g, sand_g=sg))
    for bg in (0.12, 0.35):
        for osc_g in (0.3, 0.6):
            report(f"buzz  b={bg} osc={osc_g}", *founder_buzz(buzz_g=bg, osc_g=osc_g))