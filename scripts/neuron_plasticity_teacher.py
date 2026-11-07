import argparse
import math

import torch

from scripts.neuron_circuit_teacher import SPIKE_AT, TAU_SYN, initial_state, make_circuit, stack_circuits

TAU_TRACE = 20.0
A_PLUS = 0.2
A_MINUS = 0.2
G_MIN, G_MAX = 0.2, 3.0
WEIGHT_SCALE = 0.5
PULSE_AMP = 30.0
PULSE_LEN = 3
BACKGROUND_NOISE = 0.3
EPISODE_STEPS = 720
PROBE_TICKS = (40, 485, 685)
INDUCTION_START = 100
PAIR_INTERVAL = 40
PAIRS = 8
PROBE_WINDOW = 12
CONDITIONS = {"forward": 5, "reversed": -5, "none": 0}


def make_plastic_circuit(n, seed):
    circ = make_circuit(n, seed)
    circ["W"] = circ["W"] * WEIGHT_SCALE
    return circ


def plastic_mask(circ):
    return (circ["W"] != 0) & circ["exc"][:, None, :]


@torch.no_grad()
def simulate_plastic(circ, ext, state0, keep_g=True):
    """state0 (B, N, 5) = (v, u, s, spike, trace). Excitatory-presynaptic
    synapses carry a multiplier g in [G_MIN, G_MAX] updated each tick by pair
    STDP on the per-cell traces: post spike potentiates by A_PLUS * pre trace,
    pre spike depresses by A_MINUS * post trace. Returns states
    (B, T + 1, N, 5) and g history (B, T + 1, N, N) in float16 (or None)."""
    decay_s = math.exp(-1.0 / TAU_SYN)
    decay_x = math.exp(-1.0 / TAU_TRACE)
    weights, a, b, c, d = circ["W"], circ["a"], circ["b"], circ["c"], circ["d"]
    mask = plastic_mask(circ).to(weights.dtype)
    g = torch.ones_like(weights)
    v, u, s, x = state0[..., 0], state0[..., 1], state0[..., 2], state0[..., 4]
    out, gs = [state0], [g.half()] if keep_g else None
    for t in range(ext.shape[1]):
        current = ext[:, t] + torch.bmm(weights * g, s.unsqueeze(-1)).squeeze(-1)
        for _ in range(2):
            v = v + 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + current)
        v = v.clamp(min=-150.0)
        u = u + a * (b * v - u)
        spk = (v >= SPIKE_AT).to(v.dtype)
        v = torch.where(spk > 0, c, v)
        u = u + d * spk
        dg = A_PLUS * spk.unsqueeze(2) * x.unsqueeze(1) - A_MINUS * x.unsqueeze(2) * spk.unsqueeze(1)
        g = (g + dg * mask).clamp(G_MIN, G_MAX)
        s = s * decay_s + spk
        x = x * decay_x + spk
        out.append(torch.stack([v, u, s, spk, x], dim=-1))
        if keep_g:
            gs.append(g.half())
    return torch.stack(out, dim=1), (torch.stack(gs, dim=1) if keep_g else None)


def pick_synapse(circ, gen):
    mask = plastic_mask({k: v[None] for k, v in circ.items()})[0]
    post, pre = mask.nonzero(as_tuple=True)
    k = int(torch.randint(0, len(post), (1,), generator=gen))
    return int(pre[k]), int(post[k])


def build_ext(n, pre, post, delta, seed, steps=EPISODE_STEPS, pairs=PAIRS, interval=PAIR_INTERVAL, start=INDUCTION_START,
              probes=PROBE_TICKS, extra_rate=0.0):
    """Quiet background plus pulses forcing single spikes: a probe pulse on
    `pre` at each probe tick, and `pairs` induction pairs (pre at t, post at
    t + delta; delta = 0 means no induction). Pulse amplitude is jittered."""
    g = torch.Generator().manual_seed(seed)
    ext = BACKGROUND_NOISE * torch.randn(steps, n, generator=g)

    def pulse(cell, t):
        if 0 <= t < steps - PULSE_LEN:
            ext[t:t + PULSE_LEN, cell] += PULSE_AMP * (1.0 + 0.1 * (torch.rand(1, generator=g) - 0.5) * 2)

    for tp in probes:
        pulse(pre, tp)
    if delta != 0:
        for k in range(pairs):
            t = start + k * interval
            pulse(pre, t)
            pulse(post, t + delta)
    if extra_rate > 0:
        hits = torch.rand(steps, n, generator=g) < extra_rate
        ext = ext + PULSE_AMP * hits.float() * (1.0 + 0.1 * (torch.rand(steps, n, generator=g) - 0.5))
    return ext


def run_episodes(circuits, exts, device, seed0=0):
    circ = {k: v.to(device) for k, v in stack_circuits(circuits).items()}
    state0 = torch.stack([_state0(c, seed0 + i) for i, c in enumerate(circuits)]).to(device)
    return circ, simulate_plastic(circ, torch.stack(exts).to(device), state0)


def _state0(c, seed):
    base = initial_state({"b": c["b"]}, seed)
    return torch.cat([base, torch.zeros_like(base[..., :1])], dim=-1)


def probe_response(v, spk, post, tick):
    """Peak post-cell voltage in the PROBE_WINDOW ticks after the probe pulse
    (a spike counts as SPIKE_AT) minus its mean voltage in the 5 ticks before.
    v and spk are (..., T + 1) for the post cell; index t is the state after t
    ticks, the pulse acts in ticks tick..tick+PULSE_LEN-1."""
    viz = torch.where(spk > 0.5, torch.full_like(v, SPIKE_AT), v)
    peak = viz[..., tick + 1: tick + 1 + PROBE_WINDOW].amax(dim=-1)
    base = v[..., tick - 4: tick + 1].mean(dim=-1)
    return peak - base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--circuits", type=int, default=6)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    for cond, delta in CONDITIONS.items():
        rs, rates, pre_spk = [[], [], []], [], []
        circs, exts, picks = [], [], []
        for k in range(args.circuits):
            c = make_plastic_circuit(args.n, 4738 + k)
            g = torch.Generator().manual_seed(k)
            pre, post = pick_synapse(c, g)
            circs.append(c)
            picks.append((pre, post))
            exts.append(build_ext(args.n, pre, post, delta, k))
        circ, (states, gh) = run_episodes(circs, exts, args.device)
        for k, (pre, post) in enumerate(picks):
            v, spk = states[k, :, post, 0], states[k, :, post, 3]
            for i, tp in enumerate(PROBE_TICKS):
                rs[i].append(float(probe_response(v, spk, post, tp)))
            rates.append(float(states[k, :, :, 3].mean() * 1000))
            pre_spk.append(float(gh[k, -1, post, pre]))
        print(cond, "probe response", [round(sum(r) / len(r), 2) for r in rs], "g final", round(sum(pre_spk) / len(pre_spk), 2),
              "mean rate Hz", round(sum(rates) / len(rates), 2))


if __name__ == "__main__":
    main()
