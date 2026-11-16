"""process: the dynamical unit laws — cryptid's organisms.

Each law is a numpy function: (state, params, dt, inputs) -> new_state.
No oscillators, no notes, no filters, no envelopes — just laws.
The sound IS the state; the ear decides which state to observe.

All laws take float32 state vectors and return float32. NaN/Inf = death
(the stability requirement IS part of the minimal criterion).
"""

from __future__ import annotations

import numpy as np

__all__ = ["LAWS", "LAW_NAMES", "step_unit", "n_state_vars"]

# each law: (name, n_state_vars, param_names, param_defaults, step_fn)
# step_fn(state: (n_vars,), params: dict, dt: float, inputs: (n_inputs,)) -> (n_vars,)
LAWS = {}
LAW_NAMES = []


def register_law(name, n_vars, param_defaults):
    def wrap(fn):
        LAWS[name] = dict(n_vars=n_vars, params=param_defaults, step=fn)
        LAW_NAMES.append(name)
        return fn
    return wrap


# ------------------------------------------------------------------ jerk (Sprott chaos)

@register_law("jerk", 3, dict(a=-0.5, b=0.4, c=-1.1, d=100.0))
def _jerk(state, p, dt, inp):
    x, v, acc = state
    force = p["a"] * acc + p["b"] * v + p["c"] * x + p["d"] * x**3
    if inp is not None and len(inp) > 0:
        force += 0.1 * float(np.sum(inp))
    new_acc = acc + force * dt
    new_v = v + new_acc * dt
    new_x = x + new_v * dt
    return np.array([new_x, new_v, new_acc], dtype=np.float32)


# ------------------------------------------------------------------ excitable (FitzHugh-Nagumo cousin)

@register_law("excitable", 2, dict(a=0.02, b=0.2, c=-0.5, thresh=0.5))
def _excitable(state, p, dt, inp):
    v, w = state
    inp_sum = float(np.sum(inp)) if inp is not None and len(inp) > 0 else 0.0
    dv = (v - v**3 / 3 - w + inp_sum * 0.3) * 100 * dt
    dw = (p["a"] + p["b"] * v + p["c"] * w) * dt
    return np.array([v + dv, w + dw], dtype=np.float32)


# ------------------------------------------------------------------ fire (integrate-and-fire)

@register_law("fire", 1, dict(thresh=0.7, leak=0.02, reset=0.0, gain=0.3))
def _fire(state, p, dt, inp):
    v = state[0]
    inp_sum = float(np.sum(inp)) if inp is not None and len(inp) > 0 else 0.0
    v += (inp_sum * p["gain"] - v * p["leak"]) * dt
    if v > p["thresh"]:
        v = p["reset"]
    return np.array([v], dtype=np.float32)


# ------------------------------------------------------------------ osc (Kuramoto phase)

@register_law("osc", 1, dict(intrinsic=6.28, coupling_gain=0.5))
def _osc(state, p, dt, inp):
    phase = state[0]
    inp_sum = float(np.sum(inp)) if inp is not None and len(inp) > 0 else 0.0
    dphase = (p["intrinsic"] + p["coupling_gain"] * inp_sum) * dt
    return np.array([(phase + dphase) % (2 * np.pi)], dtype=np.float32)


# ------------------------------------------------------------------ sand (BTW sandpile)

@register_law("sand", 1, dict(critical=4, dissipation=0.8))
def _sand(state, p, dt, inp):
    h = state[0]
    inp_sum = float(np.sum(inp)) if inp is not None and len(inp) > 0 else 0.0
    h += inp_sum * dt + np.random.random() * 0.01
    if h > p["critical"]:
        h = h * p["dissipation"] - p["critical"] * 0.5
    return np.array([h], dtype=np.float32)


# ------------------------------------------------------------------ map (logistic/tent)

@register_law("map", 1, dict(r=3.7, rate=1.0))
def _map(state, p, dt, inp):
    v = state[0]
    inp_sum = float(np.sum(inp)) if inp is not None and len(inp) > 0 else 0.0
    if np.random.random() < p["rate"] * dt:
        v = p["r"] * v * (1 - v) + 0.01 * inp_sum
    return np.array([np.clip(v, 0, 1)], dtype=np.float32)


# ------------------------------------------------------------------ bool (NK-style regulatory)

@register_law("bool", 1, dict(thresh=0.5, decay=0.95))
def _bool(state, p, dt, inp):
    v = state[0]
    inp_sum = float(np.sum(inp)) if inp is not None and len(inp) > 0 else 0.0
    if inp_sum > p["thresh"]:
        v = 1.0
    else:
        v *= p["decay"]
    return np.array([v], dtype=np.float32)


# ------------------------------------------------------------------ delay (state feedback)

@register_law("delay", 1, dict(gain=0.6, mix=0.5))
def _delay(state, p, dt, inp):
    v = state[0]
    inp_sum = float(np.sum(inp)) if inp is not None and len(inp) > 0 else 0.0
    # delayed feedback: v feeds back into itself plus input
    new_v = v * p["mix"] + inp_sum * p["gain"] * (1 - p["mix"])
    return np.array([new_v], dtype=np.float32)


# ------------------------------------------------------------------ clamp (nonlinear quantizer)

@register_law("clamp", 1, dict(levels=8, gain=1.5))
def _clamp(state, p, dt, inp):
    v = state[0]
    inp_sum = float(np.sum(inp)) if inp is not None and len(inp) > 0 else 0.0
    raw = (v + inp_sum * p["gain"]) * p["levels"]
    return np.array([np.clip(np.round(raw) / p["levels"], -1, 1)], dtype=np.float32)


# ------------------------------------------------------------------ bytefeed (file bytes as dynamics)

@register_law("bytefeed", 1, dict(data=None, rate=0.5))
def _bytefeed(state, p, dt, inp):
    v = state[0]
    data = p.get("data")
    if data is None:
        return np.array([v], dtype=np.float32)
    if np.random.random() < p["rate"] * dt:
        idx = int(abs(v)) % len(data)
        v = (data[idx] - 128) / 128.0
    return np.array([v], dtype=np.float32)


def step_unit(name: str, state: np.ndarray, params: dict, dt: float,
              inputs: np.ndarray | None) -> np.ndarray:
    """Advance one unit one step. Returns new state or kills on NaN."""
    law = LAWS[name]
    new = law["step"](state, params, dt, inputs)
    if not np.isfinite(new).all():
        return None  # death: stability is part of the MC
    return new


def n_state_vars(name: str) -> int:
    return LAWS[name]["n_vars"]


def default_params(name: str) -> dict:
    return dict(LAWS[name]["params"])