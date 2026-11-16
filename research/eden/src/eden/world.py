"""world: the organism — a 3D torus of CPPN voices, heard through a
reactive listener.

eden v2: every agent's voice is a prism CPPN read as an evolving
spectrum (the spectra traversal: CPPN(time, log-freq, neighbor
distance) -> magnitude field -> phase-accumulated frames). The
composition functions subsume the old technique palette — sin∘lin is
a tone, sin∘sin is FM, gauss is a formant window, step is click
weather — and the voice REACTS continuously to its neighbors through
the d input. The alphabet grows: add_node/add_conn/add_output mutate
the voice graphs themselves (complexification is intrinsic).

The listener is a maestro-style reactive navigator: a second CPPN
whose brain is a function of the world's global weather (loudness,
brightness, change) and a linearly-decaying spike train of collision
events (the NEAT Drummer encoding, applied to movement). Its outputs
steer a 3D heading; what it passes is what we hear.

3D localization in pure numpy: distance falloff + air absorption
(one-pole, cutoff falling with distance), constant-power azimuth
panning, interaural time difference (ITD gather, <=0.6 ms), and an
elevation tilt (one-pole, cutoff falling with |elevation|). No game
engine, no new dependencies — the cues that matter, offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from prism.genome import Genome, random_genome as random_cppn

__all__ = ["WorldGenome", "AgentSpec", "ListenerSpec", "render",
           "SR", "SECONDS", "BLOCK", "MAX_AGENTS", "REACT_EPOCH",
           "listener_brain"]

SR = 44100
SECONDS = 8.0
BLOCK = 512                       # control rate 86.1 Hz (world-scale
                                  # motion needs meters, not milliseconds)
MAX_AGENTS = 6
REACT_EPOCH = 100                 # collision spike decay (blocks)

# spectra-traversal sizes (per-agent voices)
HOP, WIN, FBANDS = 1024, 2048, 96
FMIN = 40.0
REACT = ("T_JUMP", "BURST", "ENTRAIN", "FLEE", "ECHO")
MOTIONS = ("DRIFT", "ORBIT")


@dataclass
class AgentSpec:
    voice: Genome = None           # the CPPN voice (prism genome)
    amp: float = 0.6
    pan_bias: float = 0.0
    f_center: float = 600.0         # register: the voice's band center
    span_oct: float = 5.0           # bandwidth in octaves: drone .. hiss
    sharp: float = 1.6             # spectral exponent: buzz .. formants
    # motion genes (3D)
    motion: str = "DRIFT"
    pos: tuple = (0.5, 0.5, 0.5)
    speed: float = 0.06
    wander: float = 0.6
    home: tuple = (0.5, 0.5, 0.5)
    orbit_r: float = 0.15
    orbit_rate: float = 0.7
    orbit_axis: tuple = (0.3, 0.3, 0.9)   # orbit plane normal-ish
    # interaction genes
    coupling: float = 0.0
    reaction: str = "T_JUMP"
    react_mag: float = 0.3
    contact_r: float = 0.10
    cooldown_s: float = 0.8
    wobble: float = 0.15           # collision t_offset jump magnitude

    def __post_init__(self):
        if self.voice is None:
            self.voice = random_cppn(np.random.default_rng(0))


@dataclass
class ListenerSpec:
    brain: Genome = None           # the reactive navigator (prism genome)
    pos: tuple = (0.3, 0.3, 0.5)
    speed: float = 0.10            # world units per second (base)
    hearing_r: float = 0.30
    seek: float = 0.5              # unused in v2 brain mode (kept for labels)

    def __post_init__(self):
        if self.brain is None:
            self.brain = listener_brain(np.random.default_rng(1))


@dataclass
class WorldGenome:
    agents: list = field(default_factory=list)
    listener: ListenerSpec = field(default_factory=ListenerSpec)
    seed: int = 0

    # ------------------------------------------------------------ labels
    def archetype(self) -> str:
        """Genome-derived taxonomy: voice-graph composition x society."""
        n = len(self.agents)
        fns = []
        for a in self.agents:
            fns += [nd.fn for nd in a.voice.nodes.values()
                    if nd.kind == "hid"]
        if not fns:
            fns = ["lin"]
        periodic = sum(f in ("sin", "cos") for f in fns) / len(fns)
        hard = sum(f in ("step", "sq", "abs") for f in fns) / len(fns)
        soft = sum(f in ("gauss", "sig", "tanh") for f in fns) / len(fns)
        if periodic >= 0.5:
            timbre = "tonal"
        elif hard >= 0.4:
            timbre = "machinery"
        elif soft >= 0.4:
            timbre = "glass"
        else:
            timbre = "mixed"
        social = "solo" if n <= 1 else ("duet" if n == 2 else
                                         ("trio" if n == 3 else "swarm"))
        return f"{social}_{timbre}"

    def tonality(self) -> float:
        """The greedy objective (kept for symmetry): periodic purity."""
        t = 0.0
        for a in self.agents:
            fns = [nd.fn for nd in a.voice.nodes.values() if nd.kind == "hid"]
            if fns:
                per = sum(f in ("sin", "cos") for f in fns) / len(fns)
                t += a.amp * per
        return float(t)


ARCHETYPES = ("solo_tonal", "solo_machinery", "solo_glass", "solo_mixed",
              "duet_tonal", "duet_machinery", "duet_glass", "duet_mixed",
              "trio_tonal", "trio_machinery", "trio_glass", "trio_mixed",
              "swarm_tonal", "swarm_machinery", "swarm_glass", "swarm_mixed")


# ------------------------------------------------------------------ helpers

def listener_brain(rng) -> Genome:
    """A small reactive navigator: weather + collision spikes -> turn,
    speed, lift. Grown by the same CPPN mutations afterwards."""
    g = random_cppn(rng, n_hidden=2)
    for name in ("turn", "speed", "lift"):
        if name not in g.outputs:
            out = g.add_output(name, fn="tanh" if name != "turn" else "sin")
            src = g.order[int(rng.integers(0, len(g.order) - 1))]
            g.conn(src, out, 0.6 * float(rng.standard_normal()) or 0.5)
    return g


def _wrap3(p):
    return p % 1.0


def _delta(a, b):
    """Wrap-around 3D difference a - b."""
    d = np.asarray(a) - np.asarray(b)
    return d - np.round(d)


def _dist3(a, b):
    return float(np.linalg.norm(_delta(a, b)))


def _norm_range(x, lo=1.0, hi=99.0):
    """prism's percentile normalizer, local copy (keep eden decoupled
    from prism's private helpers)."""
    flat = x if x.size <= 100_000 else x.reshape(-1)[:: x.size // 50_000 + 1]
    a, b = np.percentile(flat, [lo, hi])
    if b - a < 1e-9:
        return np.full_like(x, 0.5, dtype=np.float32)
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(np.float32)


def _spikes(events_t, nblocks, block_s, decay_blocks=REACT_EPOCH):
    """Linearly-decaying collision spikes (the NEAT Drummer frame): a
    spike decays to zero until the NEXT event, floor-guarded."""
    v = np.zeros(nblocks)
    for t in events_t:
        i = int(t / block_s)
        v[i] = 1.0
    out = np.zeros(nblocks)
    last = 0.0
    for i in range(nblocks):
        if v[i] > 0.0:
            last = 1.0
        else:
            last = max(0.0, last - 1.0 / decay_blocks)
        out[i] = last
    return out


# ------------------------------------------------------------------ the episode

def render(g: WorldGenome, seconds: float = SECONDS, sr: int = SR):
    """One episode: simulate the agents, feed the listener's brain,
    steer it through 3D, and return what it heard. Deterministic from
    the genome seed. Returns (stereo (n, 2), telemetry dict)."""
    rng = np.random.default_rng(g.seed & 0x7FFFFFFF)
    nblocks = int(seconds * sr) // BLOCK
    N = nblocks * BLOCK
    block_s = BLOCK / sr
    if nblocks < 8:
        return np.zeros((int(seconds * sr), 2), np.float32), \
            {"collisions": 0, "agents": len(g.agents)}

    # ---------------- phase 1: agent dynamics (listener-free)
    agents = []
    for a in g.agents:
        axis = np.asarray(a.orbit_axis, float)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        u = np.array([axis[1], -axis[0], 0.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0])
        u = u / np.linalg.norm(u)
        w = np.cross(axis, u)
        agents.append({
            "spec": a, "pos": np.asarray(a.pos, float) % 1.0,
            "vel": rng.normal(0, 1, 3), "heading": rng.normal(0, 1, 3),
            "orbit_a": rng.uniform(0, 2 * np.pi),
            "u": u, "w": w,
            "t_off": 0.0, "boost": 0.0, "cool": {},
            "pos_b": np.zeros((nblocks, 3)), "nn_b": np.ones(nblocks),
            "amp_b": np.zeros(nblocks), "bright_b": np.zeros(nblocks),
        })
    collisions = 0
    events_t = []

    nA = len(agents)
    contact_rs = np.array([ag["spec"].contact_r for ag in agents])
    cooldowns = np.array([ag["spec"].cooldown_s for ag in agents])
    for b in range(nblocks):
        t_s = b * block_s
        # motion
        for ag in agents:
            a = ag["spec"]
            if a.motion == "ORBIT":
                ag["orbit_a"] += a.orbit_rate * block_s
                ag["pos"] = _wrap3(np.asarray(a.home)
                                   + a.orbit_r * (np.cos(ag["orbit_a"]) * ag["u"]
                                                  + np.sin(ag["orbit_a"]) * ag["w"]))
            else:
                ag["heading"] = ag["heading"] + \
                    rng.normal(0, a.wander, 3) * 0.15
                n = np.linalg.norm(ag["heading"]) + 1e-9
                speed = a.speed * (1.0 + 0.3 * rng.normal())
                ag["vel"] = ag["heading"] / n * speed
                ag["pos"] = _wrap3(ag["pos"] + ag["vel"] * block_s)
        if nA > 1:
            # pairwise torus distances, vectorized (the hot loop)
            P = np.stack([ag["pos"] for ag in agents])          # (A, 3)
            rel = P[:, None, :] - P[None, :, :]
            rel -= np.round(rel)
            D = np.linalg.norm(rel, axis=2)                      # (A, A)
            np.fill_diagonal(D, np.inf)
            # coupling forces (attract/repel)
            for i, ag in enumerate(agents):
                a = ag["spec"]
                if a.motion != "DRIFT" or abs(a.coupling) < 0.02:
                    continue
                mask = (D[i] < 0.3) & (D[i] > 1e-6)
                if mask.any():
                    f = a.coupling * block_s * 0.02
                    dirs = -rel[i][mask] / D[i][mask][:, None]
                    ag["vel"] = ag["vel"] + f * dirs.sum(axis=0)
                    ag["pos"] = _wrap3(ag["pos"] + ag["vel"] * block_s)
            # nearest-neighbor distance (the voices' d input)
            for i, ag in enumerate(agents):
                ag["nn_b"][b] = float(D[i].min())
            # collisions -> reactions
            for i in range(nA):
                for j in range(i + 1, nA):
                    if D[i, j] < min(contact_rs[i], contact_rs[j]) \
                            and t_s - agents[i]["cool"].get(j, -99) > \
                            max(cooldowns[i], cooldowns[j]) \
                            and t_s - agents[j]["cool"].get(i, -99) > \
                            max(cooldowns[i], cooldowns[j]):
                        agents[i]["cool"][j] = t_s
                        agents[j]["cool"][i] = t_s
                        collisions += 1
                        events_t.append(t_s)
                        _react(agents[i], agents[j], rng)
                        _react(agents[j], agents[i], rng)
        for ag in agents:
            ag["boost"] *= 0.985
            ag["pos_b"][b] = ag["pos"]
            ag["amp_b"][b] = ag["spec"].amp * (1.0 + ag["boost"])

    # ---------------- phase 2: voices (spectra traversal, batch)
    # Each voice reads its CPPN over (time, log-freq, neighbor distance)
    # into a magnitude field. THREE diversity genes shape the readout:
    #   f_center  the register — where the voice's band sits
    #   span_oct the bandwidth — a narrow band is a drone, a wide one
    #            a hiss (with a full 9-octave span every voice was
    #            broadband and every world measured the same)
    #   sharp    the spectral exponent (flat buzz .. peaky formants)
    # and normalization is per WORLD, not per voice: quiet voices stay
    # quiet, so episodes keep inter-voice timbre contrast (per-voice
    # percentile normalization was measured to collapse all worlds
    # into the same dense mid-bright texture — cells saturated at 9).
    nframes = max(4, N // HOP)
    frame_t = (np.arange(nframes) * HOP / sr).astype(np.float32) / seconds
    lin_f = np.fft.rfftfreq(WIN, 1.0 / sr)

    raw_fields = []
    for ag in agents:
        a = ag["spec"]
        nn_frame = np.interp((np.arange(nframes) + 0.5) * HOP / sr,
                             np.arange(nblocks) * block_s,
                             np.clip(ag["nn_b"], 0.0, 1.0))
        center = float(np.clip(a.f_center, 60.0, 8000.0))
        span = float(np.clip(a.span_oct, 1.0, 9.0))
        f_lo = center * 2.0 ** (-span / 2.0)
        f_hi = min(center * 2.0 ** (span / 2.0), 0.48 * sr)
        if f_hi < 2.0 * f_lo:
            f_lo = f_hi / 2.0
        f_axis = np.geomspace(f_lo, f_hi, FBANDS)
        y_axis = (np.log(f_axis / f_lo) / np.log(f_hi / f_lo)
                  * 2.0 - 1.0).astype(np.float32)
        ag["f_axis"] = f_axis
        ag["y_axis"] = y_axis
        X = frame_t[:, None] + np.float32(ag["t_off"])
        Y = y_axis[None, :]
        D = nn_frame.astype(np.float32)[:, None]
        outs = a.voice.evaluate({"x": X, "t": X, "y": Y, "d": D, "b": 1.0})
        mag = None
        for name in ("amp", "wave", "field"):
            if name in outs:
                mag = np.asarray(outs[name], np.float32)
                break
        if mag is None:
            mag = np.asarray(next(iter(outs.values())), np.float32)
        raw_fields.append(mag)
    # per-WORLD percentile range (keeps inter-voice contrast)
    allv = np.concatenate([f.reshape(-1) for f in raw_fields]) \
        if raw_fields else np.zeros(1)
    p1, p99 = np.percentile(allv, [1.0, 99.0])
    span = max(p99 - p1, 1e-6)

    voices = []
    for ag, raw in zip(agents, raw_fields):
        a = ag["spec"]
        f_axis = ag["f_axis"]
        pos = np.clip(np.interp(lin_f, f_axis,
                                np.linspace(0.0, 1.0, FBANDS))
                      * (FBANDS - 1), 0.0, FBANDS - 1)
        k0 = np.floor(pos).astype(np.int64)
        k1 = np.minimum(k0 + 1, FBANDS - 1)
        alf = (pos - k0).astype(np.float32)
        W = np.zeros((lin_f.size, FBANDS), np.float32)
        rows = np.arange(lin_f.size)
        np.add.at(W, (rows, k0), 1.0 - alf)
        np.add.at(W, (rows, k1), alf)
        # interp CLAMPS outside the band — every bin beyond f_axis would
        # inherit the edge band's magnitude (96% of a 150 Hz drone's
        # energy ended up above 2 kHz before this line existed)
        W[~((lin_f >= f_axis[0]) & (lin_f <= f_axis[-1]))] = 0.0
        mag = np.clip((raw - p1) / span, 0.0, 1.0) ** float(
            np.clip(a.sharp, 0.6, 4.0))
        M = mag @ W.T                                   # (frames, bins)
        M[:, lin_f < f_axis[0]] = 0.0
        M[:, lin_f > 0.48 * sr] = 0.0
        # phase-accumulated frames -> mono waveform. Initial phases
        # are RANDOM per bin: with hop = win/2 the per-bin advance is
        # exactly pi*k, so a zero init would align every even frame's
        # phases into one giant impulse per frame (a click train, not
        # the intended spectrum). Random init keeps frames dense and
        # lets the CPPN's spectral envelope speak.
        advance = lin_f * (2.0 * np.pi * HOP / sr)
        jitter = rng.standard_normal((nframes, lin_f.size)) * 0.08
        tbuf = np.zeros((nframes - 1) * HOP + WIN)
        hann = np.hanning(WIN)
        ph_acc = rng.uniform(0, 2 * np.pi, lin_f.size)
        for i in range(nframes):
            ph_acc = ph_acc + advance + jitter[i]
            tbuf[i * HOP: i * HOP + WIN] += \
                np.fft.irfft(M[i] * np.exp(1j * ph_acc), WIN) * hann
        voices.append(tbuf[:N].astype(np.float32))
        # the voice's own brightness (for the heard-weather feeds)
        ag["bright_b"] = np.interp(
            (np.arange(nblocks) + 0.5) * block_s,
            (np.arange(nframes) + 0.5) * HOP / sr,
            (mag * ag["y_axis"][None, :]).sum(axis=1)
            / (mag.sum(axis=1) + 1e-9))

    # ---------------- phase 3: the listener's brain (one batch pass)
    if agents:
        rms_w = np.sum([ag["amp_b"] for ag in agents], axis=0)
        bright_w = (np.sum([ag["amp_b"] * ag["bright_b"] for ag in agents],
                           axis=0) / (rms_w + 1e-9))
    else:
        rms_w = np.zeros(nblocks)
        bright_w = np.zeros(nblocks)
    flux_w = np.abs(np.diff(rms_w, prepend=rms_w[0]))
    feeds = {
        "t": (np.arange(nblocks) * block_s / seconds).astype(np.float32),
        "b": 1.0,
        "live_rms": _norm_range(rms_w).astype(np.float32),
        "live_centroid": (bright_w * 0.5 + 0.5).astype(np.float32),
        "live_flux": _norm_range(flux_w).astype(np.float32),
        "d": _spikes(events_t, nblocks, block_s).astype(np.float32),
        "x": np.zeros(nblocks, np.float32),
        "y": np.zeros(nblocks, np.float32),
    }
    br = g.listener.brain.evaluate(feeds)
    turn = np.asarray(br.get("turn", np.zeros(nblocks)))
    speed = np.asarray(br.get("speed", np.ones(nblocks)))
    lift = np.asarray(br.get("lift", np.zeros(nblocks)))
    # integrate the 3D trajectory (normalized outputs -> steering)
    lpos = np.asarray(g.listener.pos, float) % 1.0
    yaw = rng.uniform(0, 2 * np.pi)
    pitch = rng.uniform(-0.4, 0.4)
    lpos_b = np.zeros((nblocks, 3))
    for b in range(nblocks):
        yaw = yaw + float(turn[b]) * 0.15
        pitch = float(np.clip(pitch + float(lift[b]) * 0.08, -1.2, 1.2))
        v = np.array([np.cos(pitch) * np.cos(yaw),
                      np.cos(pitch) * np.sin(yaw),
                      np.sin(pitch)])
        lpos = _wrap3(lpos + v * float(g.listener.speed * block_s
                                       * (0.5 + np.clip(speed[b], -1, 1))))
        lpos_b[b] = lpos

    # ---------------- phase 4: spatialize + mix (what the ear receives)
    Lmix = np.zeros(N)
    Rmix = np.zeros(N)
    heard = 0.0
    centers = (np.arange(nblocks) + 0.5) * BLOCK
    tsamp = np.arange(N)
    for ag, wv in zip(agents, voices):
        a = ag["spec"]
        # vectorized listener frame: wrap-difference -> (d, az, el)
        rel = ag["pos_b"] - lpos_b
        rel -= np.round(rel)
        d_b = np.linalg.norm(rel, axis=1)
        az_b = np.arctan2(rel[:, 1], rel[:, 0])
        el_b = np.arctan2(rel[:, 2], np.hypot(rel[:, 0], rel[:, 1]))
        gain_b = ag["amp_b"] / (1.0 + (d_b / max(g.listener.hearing_r, 1e-3)) ** 2)
        heard += float(gain_b.max())
        pan_b = np.clip(np.sin(az_b) + a.pan_bias, -1.0, 1.0)
        itd_b = np.round(25.0 * np.sin(az_b)).astype(int)   # <=0.57 ms
        # per-sample smooth arrays
        gain = np.interp(tsamp, centers, gain_b)
        pan = np.interp(tsamp, centers, pan_b)
        theta = (np.clip(pan, -1, 1) + 1.0) * np.pi / 4.0
        # ITD: integer-delay gather — one channel reads early, one late
        wl_d = _gather(wv * gain * np.cos(theta), np.repeat(itd_b, BLOCK), +1)
        wr_d = _gather(wv * gain * np.sin(theta), np.repeat(-itd_b, BLOCK), +1)
        # air absorption + elevation tilt: one combined one-pole per
        # frame-rate segment (coefficients move slowly)
        cut = np.interp((np.arange(nframes) + 0.5) * HOP / sr,
                        (np.arange(nblocks) + 0.5) * block_s,
                        14000.0 - 12000.0 * np.clip(d_b, 0, 1) ** 2
                        - 3000.0 * np.abs(np.sin(el_b)))
        seg = np.zeros(0)
        zi = 0.0
        zi_l = np.zeros(1)
        zi_r = np.zeros(1)
        for i in range(nframes):
            s0, s1 = i * HOP, min((i + 1) * HOP, N)
            if s1 <= s0:
                break
            fc = float(np.clip(cut[i], 300.0, 16000.0))
            alpha = float(np.exp(-2 * np.pi * fc * HOP / sr))
            b0 = 1.0 - alpha
            yl, zi_l = lfilter([b0], [1.0, -alpha], wl_d[s0:s1], zi=zi_l)
            yr, zi_r = lfilter([b0], [1.0, -alpha], wr_d[s0:s1], zi=zi_r)
            Lmix[s0:s1] += yl
            Rmix[s0:s1] += yr

    want = int(seconds * sr)
    stereo = np.stack([Lmix, Rmix], axis=1)[:want]
    if len(stereo) < want:
        stereo = np.pad(stereo, ((0, want - len(stereo)), (0, 0)))
    stereo = stereo.astype(np.float32)
    if not np.isfinite(stereo).all():
        stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.0, neginf=0.0)
    # the house safety stage (prism's rule): RMS-normalize so sparse
    # worlds are comparable in loudness, fade the edges, peak-limit.
    mono_ref = stereo.mean(axis=1).astype(np.float64)
    rms = float(np.sqrt((mono_ref ** 2).mean()))
    if rms > 1e-6:
        stereo = stereo * (0.18 / rms)
        nf = int(0.04 * sr)
        if len(stereo) > 2 * nf:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, nf))
            stereo[:nf] *= ramp[:, None]
            stereo[-nf:] *= ramp[::-1][:, None]
        peak = float(np.abs(stereo).max())
        if peak > 0.95:
            stereo = stereo * (0.95 / peak)
    stereo = np.clip(stereo, -0.99, 0.99).astype(np.float32)
    fns = []
    for a in g.agents:
        fns += [nd.fn for nd in a.voice.nodes.values() if nd.kind == "hid"]
    tele = {"collisions": collisions, "agents": len(agents),
            "events": len(events_t),
            "path_len": float(np.linalg.norm(
                np.diff(lpos_b, axis=0), axis=1).sum()),
            "voice_nodes": int(np.mean(
                [len(a.voice.nodes) for a in g.agents])) if g.agents else 0,
            "heard_max": float(heard) if agents else 0.0,
            "fns": sorted(set(fns))}
    return stereo, tele


def _gather(x, delay, sign):
    """Integer-delay gather: out[n] = x[n - sign*delay[n]]."""
    n = len(x)
    idx = np.arange(n) - sign * delay
    idx = np.clip(idx, 0, n - 1)
    return x[idx]


def _react(ag, other, rng):
    """Fire ag's reaction (runtime state only — the genome is sacred)."""
    a = ag["spec"]
    mag = a.react_mag
    if a.reaction == "T_JUMP":
        ag["t_off"] += rng.uniform(-mag, mag)          # read elsewhere in the field
    elif a.reaction == "BURST":
        ag["boost"] += 1.5 * mag
    elif a.reaction == "ENTRAIN":
        ag["t_off"] = 0.7 * ag["t_off"] + 0.3 * other["t_off"]
        ag["boost"] = 0.7 * ag["boost"] + 0.3 * other["boost"]
    elif a.reaction == "FLEE":
        away = _delta(ag["pos"], other["pos"])
        n = np.linalg.norm(away) + 1e-9
        ag["vel"] = ag["vel"] + away / n * 0.04 * mag
        ag["heading"] = ag["heading"] + away / n * 0.04 * mag
    elif a.reaction == "ECHO":
        pass                                            # the spike IS the echo
    if a.wobble > 0:
        ag["t_off"] += float(rng.normal(0, a.wobble))  # touches retune the read