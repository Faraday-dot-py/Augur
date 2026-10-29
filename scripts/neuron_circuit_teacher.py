import argparse
import math

import torch

TAU_SYN = 5.0
SPIKE_AT = 30.0
EXC_WEIGHT = 3.0
INH_WEIGHT = 6.0
DRIVE_BASE_MAX = 4.0
DRIVE_LEVEL_MAX = 6.0
DRIVE_NOISE = 1.0


def make_circuit(n, seed):
    g = torch.Generator().manual_seed(seed)
    n_exc = round(0.8 * n)
    exc = torch.randperm(n, generator=g) < n_exc
    p = min(0.5, 10.0 / n)
    mask = torch.rand(n, n, generator=g) < p
    mask.fill_diagonal_(False)
    mag = 0.5 + 0.5 * torch.rand(n, n, generator=g)
    sign_mag = torch.where(exc, torch.tensor(EXC_WEIGHT), torch.tensor(-INH_WEIGHT))
    weights = mask * mag * sign_mag[None, :]
    r = torch.rand(n, generator=g)
    a = torch.where(exc, torch.full((n,), 0.02), 0.02 + 0.08 * r)
    b = torch.where(exc, torch.full((n,), 0.2), 0.25 - 0.05 * r)
    c = torch.where(exc, -65.0 + 15.0 * r ** 2, torch.full((n,), -65.0))
    d = torch.where(exc, 8.0 - 6.0 * r ** 2, torch.full((n,), 2.0))
    return {"W": weights, "exc": exc, "a": a, "b": b, "c": c, "d": d}


def make_drive(n, steps, seed):
    g = torch.Generator().manual_seed(seed)
    seg = int(torch.randint(40, 121, (1,), generator=g))
    n_seg = math.ceil(steps / seg)
    base = DRIVE_BASE_MAX * torch.rand(n, generator=g)
    levels = DRIVE_LEVEL_MAX * torch.rand(n_seg, n, generator=g)
    levels = levels + 5.0 * torch.rand(n_seg, 1, generator=g) - 2.0
    drive = base[None, :] + levels.repeat_interleave(seg, dim=0)[:steps]
    return drive + DRIVE_NOISE * torch.randn(steps, n, generator=g)


def stack_circuits(circuits):
    return {k: torch.stack([c[k] for c in circuits]) for k in circuits[0]}


def initial_state(circ, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = circ["b"]
    v = -65.0 + 5.0 * torch.rand(b.shape, generator=g).to(b.device)
    return torch.stack([v, b * v, torch.zeros_like(v), torch.zeros_like(v)], dim=-1)


@torch.no_grad()
def simulate(circ, ext, state0):
    """circ tensors have a leading batch dim (B, N[, N]); ext is (B, T, N);
    state0 is (B, N, 4) = (v, u, s, spike). Returns (B, T + 1, N, 4). The
    recorded v is the post-reset value, so a spiking cell shows its reset
    voltage with the spike flag set."""
    decay = math.exp(-1.0 / TAU_SYN)
    weights, a, b, c, d = circ["W"], circ["a"], circ["b"], circ["c"], circ["d"]
    v, u, s = state0[..., 0], state0[..., 1], state0[..., 2]
    out = [state0]
    for t in range(ext.shape[1]):
        current = ext[:, t] + torch.bmm(weights, s.unsqueeze(-1)).squeeze(-1)
        for _ in range(2):
            v = v + 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + current)
        v = v.clamp(min=-150.0)
        u = u + a * (b * v - u)
        spk = (v >= SPIKE_AT).to(v.dtype)
        v = torch.where(spk > 0, c, v)
        u = u + d * spk
        s = s * decay + spk
        out.append(torch.stack([v, u, s, spk], dim=-1))
    return torch.stack(out, dim=1)


def generate(n, circuit_seeds, drive_seeds, steps, burn, device):
    circs = [make_circuit(n, s) for s in circuit_seeds]
    circ = {k: v.to(device) for k, v in stack_circuits(circs).items()}
    ext = torch.stack([make_drive(n, steps + burn, s) for s in drive_seeds]).to(device)
    state0 = torch.stack([initial_state({"b": c["b"]}, s) for c, s in zip(circs, drive_seeds)]).to(device)
    states = simulate(circ, ext, state0)
    return circ, ext[:, burn:], states[:, burn:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--circuits", type=int, default=4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    seeds = list(range(4738, 4738 + args.circuits))
    circ, ext, states = generate(args.n, seeds, seeds, args.steps, 100, args.device)
    rate = states[:, 1:, :, 3].mean(dim=(1,)) * 1000.0
    v = states[..., 0]
    print(f"n={args.n} rate Hz mean {rate.mean():.1f} min {rate.min():.1f} max {rate.max():.1f}")
    print(f"v range {v.min():.1f} {v.max():.1f} finite {bool(torch.isfinite(states).all())}")
    print(f"exc rate {rate[circ['exc']].mean():.1f} inh rate {rate[~circ['exc']].mean():.1f}")


if __name__ == "__main__":
    main()
