"""world: the organism — agents on a torus, heard through a listener.

Genome anatomy:

    WorldGenome
      agents (1..6), each an AgentSpec:
        technique      TONE | FM | NOISE | CLICKS | GRAIN
        voice genes    per technique (freqs, ratios, index, rates...)
        motion genes   DRIFT (speed, turn, wander) or ORBIT (home, radius, rate)
        coupling       attract/repel strength toward nearby agents
        reaction       what a collision fires: FREQ_JUMP | BURST | ENTRAIN |
                       FLEE | ECHO — and collisions wobble the agent's own
                       voice genes (interactions change them)
        contact_r      collision radius; cooldown
      listener (ListenerSpec):
        wander_speed, turn, hearing radius, seek (attraction to loudness)
      seed              deterministic episode

Dynamics run at the control rate (one block = 256 samples, ~172 Hz);
audio is assembled per agent from per-block snapshots with smooth
parameter interpolation (glides, not steps). The listener hears:
gain = distance falloff, pan = left-right bearing. Everything moves,
everything is heard only in passing — the phenotype is sound over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["WorldGenome", "AgentSpec", "ListenerSpec", "render",
           "TECHNIQUES", "REACTIONS", "SR", "SECONDS", "MAX_AGENTS",
           "BLOCK"]

SR = 44100
SECONDS = 8.0
BLOCK = 256                       # control block (samples) — 172.3 Hz
MAX_AGENTS = 6

TECHNIQUES = ("TONE", "FM", "NOISE", "CLICKS", "GRAIN")
REACTIONS = ("FREQ_JUMP", "BURST", "ENTRAIN", "FLEE", "ECHO")
MOTIONS = ("DRIFT", "ORBIT")


@dataclass
class AgentSpec:
    technique: str = "TONE"
    # voice genes (union across techniques; unused ones ignored)
    freq: float = 440.0            # TONE base / FM fundamental / GRAIN pitch
    amp: float = 0.6
    c_ratio: float = 1.0          # FM carrier ratio
    m_ratio: float = 2.01          # FM modulator ratio
    index: float = 3.0             # FM depth
    center: float = 1500.0         # NOISE band center
    width: float = 0.4             # NOISE band width (octaves)
    rate: float = 6.0              # CLICKS/GRAIN events per second
    jitter: float = 0.5            # event-time jitter
    grain_ms: float = 40.0         # GRAIN length
    # motion genes
    motion: str = "DRIFT"
    pos: tuple = (0.5, 0.5)
    vel: tuple = (0.02, 0.0)
    speed: float = 0.08            # world units per second
    turn: float = 1.0              # radians per second
    wander: float = 0.5
    home: tuple = (0.5, 0.5)
    orbit_r: float = 0.15
    orbit_rate: float = 0.7        # radians per second
    # interaction genes
    coupling: float = 0.0          # +attract / -repel toward neighbors
    reaction: str = "FREQ_JUMP"
    react_mag: float = 0.3
    contact_r: float = 0.08
    cooldown_s: float = 0.8
    gene_wobble: float = 0.05      # collisions mutate voice genes by this
    pan_bias: float = 0.0


@dataclass
class ListenerSpec:
    pos: tuple = (0.3, 0.3)
    vel: tuple = (0.0, 0.0)
    speed: float = 0.10            # world units per second
    turn: float = 0.8
    wander: float = 0.7
    hearing_r: float = 0.30        # falloff radius (world units)
    seek: float = 0.5              # attraction to loudness gradients


@dataclass
class WorldGenome:
    agents: list = field(default_factory=list)
    listener: ListenerSpec = field(default_factory=ListenerSpec)
    seed: int = 0

    # ------------------------------------------------------------ labels
    def archetype(self) -> str:
        """A coarse genome-derived taxonomy (fixed cascade, no audio)."""
        n = len(self.agents)
        techs = [a.technique for a in self.agents]
        if n <= 1:
            return "solo"
        kinds = set(techs)
        if len(kinds) == 1:
            return f"colony_{techs[0].lower()}"
        clicks = techs.count("CLICKS") + techs.count("GRAIN")
        if clicks >= 2 and len(kinds) >= 2:
            return "rhythmic_swarm"
        if "FM" in kinds and "NOISE" in kinds:
            return "field_mod"
        if self.listener.seek > 0.6 and any(a.coupling < -0.2 for a in self.agents):
            return "chase"
        if len(kinds) >= 3:
            return "mixed_bag"
        return "duet" if n == 2 else "ensemble"

    def tonality(self) -> float:
        """The greedy objective (kept for symmetry; not used in the
        default arms): 'sweetness' = tonal energy over everything."""
        t = 0.0
        for a in self.agents:
            if a.technique in ("TONE", "FM"):
                t += a.amp * 2
            elif a.technique == "GRAIN":
                t += a.amp * 0.5
            else:
                t -= a.amp
        return float(t - 2.0 * self.listener.seek)


ARCHETYPES = ("solo", "duet", "ensemble", "colony_tone", "colony_fm",
              "colony_noise", "colony_clicks", "colony_grain",
              "rhythmic_swarm", "field_mod", "chase", "mixed_bag")


# ------------------------------------------------------------------ helpers

def _wrap(x):
    return x % 1.0


def _dist(a, b):
    """Torus distance (wrap-around)."""
    dx = abs(a[0] - b[0]) % 1.0
    dy = abs(a[1] - b[1]) % 1.0
    dx = min(dx, 1.0 - dx)
    dy = min(dy, 1.0 - dy)
    return float(np.hypot(dx, dy))


def _bearing(listener, agent):
    """Agent's position in the listener's frame: pan in [-1, 1]."""
    dx = (agent[0] - listener[0]) % 1.0
    if dx > 0.5:
        dx -= 1.0
    return float(np.clip(dx * 2.0, -1.0, 1.0))


def _falloff(d, r):
    return 1.0 / (1.0 + (max(d, 1e-6) / max(r, 1e-3)) ** 2)


# ------------------------------------------------------------------ the episode

def render(g: WorldGenome, seconds: float = SECONDS, sr: int = SR):
    """Run one episode. Returns (stereo audio, telemetry dict).

    Deterministic from the genome seed: the same world sounds the same,
    every time. Collisions mutate the agents' RUNTIME voice genes (the
    genome itself is never touched — simulate on copies).
    """
    rng = np.random.default_rng(g.seed & 0x7FFFFFFF)
    nblocks = int(seconds * sr) // BLOCK
    if nblocks < 4:
        return np.zeros((int(seconds * sr), 2), np.float32), {"collisions": 0}

    # ---- runtime state (copies: episodes never touch the genome)
    agents = []
    for a in g.agents:
        agents.append({
            "spec": a, "freq": a.freq, "center": a.center, "index": a.index,
            "boost": 0.0, "cool": {}, "phase": 0.0, "mphase": 0.0,
            "pos": np.array(a.pos, float), "vel": np.array(a.vel, float),
            "orbit_a": rng.uniform(0, 2 * np.pi), "events": [],
            "gain_b": np.zeros(nblocks), "pan_b": np.zeros(nblocks),
            "freq_b": np.zeros(nblocks), "amp_b": np.zeros(nblocks),
            "center_b": np.zeros(nblocks), "index_b": np.zeros(nblocks),
        })
    L = {"pos": np.array(g.listener.pos, float),
         "vel": np.array(g.listener.vel, float), "heading": rng.uniform(0, 2 * np.pi)}
    collisions = 0

    # ---- control loop: move, collide, react, listen
    for b in range(nblocks):
        t_s = b * BLOCK / sr
        # --- agent motion
        for ag in agents:
            a = ag["spec"]
            if a.motion == "ORBIT":
                ag["orbit_a"] += a.orbit_rate * BLOCK / sr
                target = np.array([a.home[0] + a.orbit_r * np.cos(ag["orbit_a"]),
                                   a.home[1] + a.orbit_r * np.sin(ag["orbit_a"])])
                ag["pos"] = _wrap_array(target)
            else:
                heading = np.arctan2(ag["vel"][1], ag["vel"][0]) \
                    + rng.normal(0, a.wander * 1.5) * BLOCK / sr
                speed = a.speed * (1.0 + 0.3 * rng.normal())
                ag["vel"] = np.array([np.cos(heading), np.sin(heading)]) * speed
                # coupling: attract/repel toward/away neighbors
                for other in agents:
                    if other is ag:
                        continue
                    d = _dist(ag["pos"], other["pos"])
                    if d < 0.25 and d > 1e-6:
                        f = a.coupling * BLOCK / sr * 0.02
                        ag["vel"] = ag["vel"] + f * (
                            other["pos"] - ag["pos"]) / d
                ag["pos"] = _wrap_array(ag["pos"] + ag["vel"] * BLOCK / sr)

        # --- collisions -> reactions (pairwise, with cooldowns)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ai, aj = agents[i], agents[j]
                d = _dist(ai["pos"], aj["pos"])
                r = min(ai["spec"].contact_r, aj["spec"].contact_r)
                cool = max(ai["spec"].cooldown_s, aj["spec"].cooldown_s)
                if d < r and t_s - ai["cool"].get(j, -99) > cool \
                        and t_s - aj["cool"].get(i, -99) > cool:
                    ai["cool"][j] = t_s
                    aj["cool"][i] = t_s
                    collisions += 1
                    here_pos = b * BLOCK + BLOCK // 2
                    _react(ai, aj, rng, here_pos)
                    _react(aj, ai, rng, here_pos)

        # --- listener motion: wander + seek the loudness gradient
        L["heading"] += rng.normal(0, g.listener.wander * 1.5) * BLOCK / sr
        step = g.listener.speed * BLOCK / sr
        L["vel"] = np.array([np.cos(L["heading"]), np.sin(L["heading"])]) * step
        if g.listener.seek > 0.01 and agents:
            # finite differences on the loudness field: the listener is
            # drawn toward whatever is sounding right now (a heritable
            # appetite for action — narrative arcs for free). Normalized
            # gradient: seek scales how hard it chases, up to ~2x speed.
            def field(p):
                return sum(a["spec"].amp * _falloff(_dist(p, a["pos"]),
                                                   g.listener.hearing_r)
                           for a in agents)
            e = 0.05
            here = field(L["pos"])
            gx = field(_wrap_array(L["pos"] + np.array([e, 0.0]))) - here
            gy = field(_wrap_array(L["pos"] + np.array([0.0, e]))) - here
            grad = np.array([gx, gy])
            gn = float(np.linalg.norm(grad))
            if gn > 1e-9:
                L["vel"] = L["vel"] + g.listener.seek * step * (grad / gn) * 2.0
        L["pos"] = _wrap_array(L["pos"] + L["vel"])

        # --- what the listener hears this block
        for ag in agents:
            d = _dist(L["pos"], ag["pos"])
            a = ag["spec"]
            gain = a.amp * _falloff(d, g.listener.hearing_r) \
                * (1.0 + ag["boost"])
            ag["boost"] *= 0.985                       # burst decay
            ag["gain_b"][b] = gain
            ag["pan_b"][b] = _bearing(L["pos"], ag["pos"])
            ag["freq_b"][b] = ag["freq"]
            ag["amp_b"][b] = a.amp * (1.0 + ag["boost"])
            ag["center_b"][b] = ag["center"]
            ag["index_b"][b] = ag["index"]
        # event emitters: Poisson clicks/grains for this block
        for ag in agents:
            a = ag["spec"]
            if a.technique in ("CLICKS", "GRAIN"):
                lam = a.rate * BLOCK / sr
                k = int(rng.poisson(lam)) if lam > 0 else 0
                for _ in range(k):
                    pos = b * BLOCK + rng.uniform(0, BLOCK)
                    ag["events"].append((float(pos), ag["freq"]))

    # ---- audio assembly (per agent, from block snapshots)
    N = nblocks * BLOCK
    t_samples = np.arange(N)
    centers_b = (np.arange(nblocks) + 0.5) * BLOCK
    Lmix = np.zeros(N)
    Rmix = np.zeros(N)

    def smooth(block_vals):
        return np.interp(t_samples, centers_b, block_vals)

    for ag in agents:
        a = ag["spec"]
        gain = smooth(ag["gain_b"])
        if gain.max() < 1e-4:
            continue
        pan = np.clip(smooth(ag["pan_b"]) + a.pan_bias, -1.0, 1.0)
        w = np.zeros(N)
        if a.technique == "TONE":
            f = smooth(ag["freq_b"])
            ph = 2 * np.pi * np.cumsum(f) / sr
            w = np.sin(ph)
        elif a.technique == "FM":
            f0 = smooth(ag["freq_b"])
            fm = smooth(ag["freq_b"]) * a.m_ratio
            dev = np.clip(smooth(ag["index_b"]) * fm, 0.0, 6000.0)
            mph = 2 * np.pi * np.cumsum(fm) / sr
            cph = 2 * np.pi * np.cumsum(f0 * a.c_ratio + dev * np.sin(mph)) / sr
            w = np.sin(cph)
        elif a.technique == "NOISE":
            w = _band_noise(rng, N, smooth(ag["center_b"]), a.width, sr)
        elif a.technique in ("CLICKS", "GRAIN"):
            klen = max(64, int(a.grain_ms * sr / 1000)) if a.technique == "GRAIN" \
                else 128
            for pos, freq in ag["events"]:
                p0 = int(pos)
                if p0 + klen >= N:
                    continue
                if a.technique == "GRAIN":
                    seg = np.sin(2 * np.pi * freq * np.arange(klen) / sr) \
                        * np.hanning(klen)
                else:
                    seg = np.exp(-np.linspace(0, 6, klen)) * np.exp(
                        rng.normal(0, 0.3)) * np.sign(rng.normal())
                w[p0: p0 + klen] += seg
        w = w * gain
        Lmix += w * np.cos((pan + 1) * np.pi / 4)
        Rmix += w * np.sin((pan + 1) * np.pi / 4)

    want = int(seconds * sr)
    stereo = np.stack([Lmix, Rmix], axis=1)[:want]
    if len(stereo) < want:
        stereo = np.pad(stereo, ((0, want - len(stereo)), (0, 0)))
    stereo = stereo.astype(np.float32)
    peak = float(np.abs(stereo).max())
    if peak > 1e-9:
        stereo = stereo / peak * 0.95
    if not np.isfinite(stereo).all():
        stereo = np.zeros_like(stereo)
    tele = {"collisions": collisions,
            "agents": len(agents),
            "techniques": [a["spec"].technique for a in agents],
            "events": sum(len(ag["events"]) for ag in agents),
            "mean_heard": float(np.mean(
                [ag["gain_b"].max() for ag in agents])) if agents else 0.0}
    return stereo, tele


def _react(ag, other, rng, here_pos=None):
    """Fire ag's reaction against other (runtime gene mutation —
    interactions change the agents, the user's own law)."""
    a = ag["spec"]
    mag = a.react_mag
    if a.reaction == "FREQ_JUMP":
        ag["freq"] = float(np.clip(ag["freq"] * 2 ** rng.uniform(-mag, mag),
                                   30.0, 6000.0))
    elif a.reaction == "BURST":
        ag["boost"] += 1.5 * mag
    elif a.reaction == "ENTRAIN":
        ag["freq"] = float(0.7 * ag["freq"] + 0.3 * other["freq"])
        ag["index"] = float(0.7 * ag["index"] + 0.3 * other["index"])
    elif a.reaction == "FLEE":
        away = ag["pos"] - other["pos"]
        away = away - np.round(away)                    # torus direction
        n = np.linalg.norm(away) + 1e-9
        ag["vel"] = ag["vel"] + away / n * 0.04 * mag
    elif a.reaction == "ECHO" and here_pos is not None:
        if a.technique in ("CLICKS", "GRAIN"):
            ag["events"].append((float(here_pos), ag["freq"]))
    # collisions wobble the voice genes: evolution inside the episode
    if a.gene_wobble > 0:
        ag["freq"] = float(np.clip(ag["freq"] * 10 ** rng.normal(0, a.gene_wobble),
                                    30.0, 6000.0))
        ag["index"] = float(np.clip(ag["index"] * 10 ** rng.normal(0, a.gene_wobble),
                                    0.0, 25.0))
        ag["center"] = float(np.clip(ag["center"] * 10 ** rng.normal(0, a.gene_wobble),
                                     100.0, 8000.0))


def _wrap_array(p):
    return p % 1.0


def _band_noise(rng, N, center_hz, width_oct, sr):
    """Time-varying bandpassed noise: STFT (512/256 = control rate),
    gaussian band mask per block, overlap-add. center_hz is a
    per-sample array (smoothed), width_oct a scalar."""
    frames = N // BLOCK
    if frames < 2:
        return rng.normal(0, 0.3, N)
    w = np.hanning(512)
    nz = rng.normal(0, 1, (frames, 512))
    S = np.fft.rfft(nz * w[None, :], axis=1)
    fr = np.fft.rfftfreq(512, 1 / sr)                 # (257,) 1-D
    # per-block mask from the block's center
    cb = center_hz[BLOCK // 2::BLOCK][:frames]
    mask = np.zeros_like(S)
    for i, c in enumerate(cb):
        lo = np.log2(np.maximum(fr, 20.0) / max(c, 20.0))
        mask[i] = np.exp(-0.5 * (lo / max(width_oct, 0.05)) ** 2)
    frames_ = np.fft.irfft(S * mask, 512, axis=1) * w[None, :]
    # overlap-add
    out = np.zeros(N + 512)
    for i in range(frames):
        p = i * BLOCK
        if p + 512 > len(out):
            break
        out[p: p + 512] += frames_[i]
    # ~unity gain for hann analysis+synthesis at 50% overlap
    return out[:N] * 1.0