import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

G = 8
N = 48
RC = 1.5
K = 1.0
DT = 0.1
NBITS = 8


def pair_force(pos):
    d = pos[:, :, None, :] - pos[:, None, :, :]
    r = d.norm(dim=-1).clamp_min(1e-6)
    mag = K * (1 - r / RC).clamp_min(0) ** 2
    mag = mag * (1 - torch.eye(pos.shape[1], device=pos.device))[None]
    return (mag[..., None] * d / r[..., None]).sum(2)


def cell_ids(pos):
    c = pos.floor().clamp(0, G - 1).long()
    return c[..., 0] * G + c[..., 1]


def cell_mean(pos, val):
    B = pos.shape[0]
    idx = cell_ids(pos)
    s = torch.zeros(B, G * G, val.shape[-1]).scatter_add_(1, idx[..., None].expand(-1, -1, val.shape[-1]), val)
    n = torch.zeros(B, G * G).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float))
    return s / n.clamp_min(1)[..., None], n


def features(pos):
    off = pos - pos.floor() - 0.5
    mo, n = cell_mean(pos, off)
    f = torch.cat([n[..., None], mo], -1)
    return f.reshape(-1, G, G, 3).permute(0, 3, 1, 2), n


def centered_windows(grid):
    B = grid.shape[0]
    p = F.pad(grid, (G // 2, G // 2, G // 2, G // 2))
    w = p.unfold(2, G, 1).unfold(3, G, 1)[:, :, :G, :G]
    return w.permute(0, 2, 3, 1, 4, 5).reshape(B, G * G, 3 * G * G)


class CellMLP(nn.Module):
    def __init__(self, mode, out, hidden=128):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(G * G, 16) if mode == "global" else None
        din = 3 * G * G + (16 if mode == "global" else 0)
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, grid):
        B = grid.shape[0]
        if self.mode == "global":
            x = grid.reshape(B, 1, -1).expand(-1, G * G, -1)
            x = torch.cat([x, self.emb.weight[None].expand(B, -1, -1)], -1)
        else:
            x = centered_windows(grid)
        return self.net(x)


def encode(dp, scale):
    q = (dp / scale).round().clamp(-128, 127).long() & 0xFF
    return ((q[..., None] >> torch.arange(NBITS)) & 1).float()


def decode(bits, scale):
    q = (bits.round().long() << torch.arange(NBITS)).sum(-1)
    q = torch.where(q >= 128, q - 256, q)
    return q.float() * scale


def sample(B, gen):
    return torch.rand(B, N, 2, generator=gen) * G


def run_arm(name, mode, kind, scale, steps, bs, gen, dev_seed):
    torch.manual_seed(dev_seed)
    out = 2 * NBITS if kind == "bits" else 2
    m = CellMLP(mode, out)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for s in range(steps):
        pos = sample(bs, gen)
        grid, n = features(pos)
        tgt, _ = cell_mean(pos, pair_force(pos) * DT)
        mask = (n > 0).float()
        y = m(grid)
        if kind == "bits":
            t = encode(tgt, scale).reshape(bs, G * G, -1)
            loss = (F.binary_cross_entropy_with_logits(y, t, reduction="none").mean(-1) * mask).sum() / mask.sum()
        else:
            loss = (((y - tgt / scale) ** 2).mean(-1) * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 500 == 0:
            print(name, s, round(loss.item(), 4), flush=True)
    return m


def predict_dp(m, kind, pos, scale):
    grid, n = features(pos)
    y = m(grid)
    if kind == "bits":
        b = (y > 0).float().reshape(*y.shape[:2], 2, NBITS)
        return decode(b, scale), y
    return (y * scale), y


def apply_to_particles(pos, cell_dp):
    idx = cell_ids(pos)
    return torch.gather(cell_dp, 1, idx[..., None].expand(-1, -1, 2)) / DT


def rollout(pos0, vel0, force_fn, steps):
    pos, vel = pos0.clone(), vel0.clone()
    traj = []
    for _ in range(steps):
        vel = vel + force_fn(pos) * DT
        pos = pos + vel * DT
        low, high = pos < 0, pos > G
        pos = torch.where(low, -pos, pos)
        pos = torch.where(high, 2 * G - pos, pos)
        vel = torch.where(low | high, -vel, vel)
        traj.append(pos.clone())
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", default="results/bit_force_smoke.json")
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(a.seed)
    cal = sample(2048, gen)
    cal_dp, cn = cell_mean(cal, pair_force(cal) * DT)
    scale = (cal_dp.abs().flatten().quantile(0.999) / 127).item()
    print("scale", scale, "max |dp| cell", cal_dp.abs().max().item())

    arms = [("bits-global", "global", "bits"), ("float-global", "global", "float"),
            ("bits-centered", "centered", "bits"), ("float-centered", "centered", "float")]
    models = {n: run_arm(n, md, k, scale, a.steps, a.bs, gen, a.seed) for n, md, k in arms}

    tg = torch.Generator().manual_seed(a.seed + 1)
    pos = sample(512, tg)
    Ftrue = pair_force(pos)
    tgt_cell, n = cell_mean(pos, Ftrue * DT)
    mask = n > 0
    res = {"scale": scale}
    oracle_p = apply_to_particles(pos, tgt_cell)
    rel = lambda p: ((p - Ftrue).norm(dim=-1).sum() / Ftrue.norm(dim=-1).sum()).item()
    res["particle_force_rel_err"] = {"zero": 1.0, "oracle_cell_mean": rel(oracle_p)}
    res["cell_dp_rel_err"] = {}
    res["bit_acc"] = {}
    fns = {"true": pair_force, "oracle_cell_mean": lambda p: apply_to_particles(p, cell_mean(p, pair_force(p) * DT)[0])}
    for name, mode, kind in arms:
        m = models[name]
        with torch.no_grad():
            dp, y = predict_dp(m, kind, pos, scale)
            res["particle_force_rel_err"][name] = rel(apply_to_particles(pos, dp))
            res["cell_dp_rel_err"][name] = ((dp - tgt_cell).norm(dim=-1)[mask].sum() / tgt_cell.norm(dim=-1)[mask].sum()).item()
            if kind == "bits":
                t = encode(tgt_cell, scale).reshape(512, G * G, -1)
                acc = ((y > 0).float() == t).float()[mask].mean(0).reshape(2, NBITS)
                res["bit_acc"][name] = {"per_bit_lsb_to_msb_x": [round(v, 4) for v in acc[0].tolist()], "mean": acc.mean().item()}
        fns[name] = (lambda mm, kk: lambda p: apply_to_particles(p, predict_dp(mm, kk, p, scale)[0]))(m, kind)

    vel0 = torch.randn(256, N, 2, generator=tg) * 0.3
    pos0 = sample(256, tg)
    with torch.no_grad():
        trajs = {k: rollout(pos0, vel0, f, 20) for k, f in fns.items()}
    res["rollout_err"] = {}
    for k in fns:
        if k == "true":
            continue
        res["rollout_err"][k] = {f"err@{t}": (trajs[k][t - 1] - trajs["true"][t - 1]).norm(dim=-1).mean().item() for t in (5, 10, 20)}
    res["rollout_err"]["zero_force"] = {f"err@{t}": (rollout(pos0, vel0, lambda p: torch.zeros_like(p), 20)[t - 1] - trajs["true"][t - 1]).norm(dim=-1).mean().item() for t in (5, 10, 20)}
    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
