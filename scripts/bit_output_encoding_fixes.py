import argparse
import json
import os
import signal
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from bit_force_smoke import G, N, DT, CellMLP, pair_force, cell_mean, features, sample, apply_to_particles, rollout

MU = 255.0
COND_TOP = 4


class Codec:
    def __init__(self, nbits, scale, gray=False, mulaw=False, weighted=False):
        self.nbits, self.scale, self.gray, self.mulaw = nbits, scale, gray, mulaw
        self.xmax = 127 * scale
        w = 2.0 ** torch.arange(nbits).float() if weighted else torch.ones(nbits)
        self.w = w / w.mean()

    def to_q(self, dp):
        if self.mulaw:
            y = torch.sign(dp) * torch.log1p(MU * dp.abs().clamp_max(self.xmax) / self.xmax) / torch.log1p(torch.tensor(MU))
            return (y * 127).round().clamp(-128, 127).long()
        half = 2 ** (self.nbits - 1)
        return (dp / self.scale).round().clamp(-half, half - 1).long()

    def from_q(self, q):
        if self.mulaw:
            y = q.float() / 127
            return torch.sign(y) * self.xmax * torch.expm1(y.abs() * torch.log1p(torch.tensor(MU))) / MU
        return q.float() * self.scale

    def encode(self, dp):
        u = self.to_q(dp) & (2 ** self.nbits - 1)
        if self.gray:
            u = u ^ (u >> 1)
        return ((u[..., None] >> torch.arange(self.nbits)) & 1).float()

    def bits_to_q(self, bits):
        u = (bits.round().long() << torch.arange(self.nbits)).sum(-1)
        if self.gray:
            for s in (1, 2, 4):
                u = u ^ (u >> s)
        half = 2 ** (self.nbits - 1)
        return torch.where(u >= half, u - 2 * half, u)

    def decode(self, bits):
        return self.from_q(self.bits_to_q(bits))

    def soft_decode(self, logits):
        p = torch.sigmoid(logits)
        wt = 2.0 ** torch.arange(self.nbits).float()
        wt[-1] = -wt[-1]
        return (p * wt).sum(-1) * self.scale


class CondCellMLP(CellMLP):
    def __init__(self, mode, nbits, hidden=128):
        super().__init__(mode, hidden, hidden)
        self.net = nn.Sequential(*self.net, nn.ReLU())
        self.nbits = nbits
        self.nlo = nbits - COND_TOP
        self.head_hi = nn.Linear(hidden, 2 * COND_TOP)
        self.head_lo = nn.Sequential(nn.Linear(hidden + 2 * COND_TOP, 64), nn.ReLU(), nn.Linear(64, 2 * self.nlo))

    def forward(self, grid, tgt=None):
        h = super().forward(grid)
        hi = self.head_hi(h).reshape(*h.shape[:-1], 2, COND_TOP)
        if tgt is None:
            given = (hi > 0).float()
        else:
            given = tgt.reshape(*tgt.shape[:-1], 2, self.nbits)[..., self.nlo:]
        lo = self.head_lo(torch.cat([h, 2 * given.reshape(*h.shape[:-1], -1) - 1], -1)).reshape(*h.shape[:-1], 2, self.nlo)
        return torch.cat([lo, hi], -1).reshape(*h.shape[:-1], 2 * self.nbits)


def build(mode, kind, codec):
    if kind == "float":
        return CellMLP(mode, 2)
    if kind == "conditional":
        return CondCellMLP(mode, codec.nbits)
    return CellMLP(mode, 2 * codec.nbits)


def train(name, mode, kind, codec, scale, steps, bs, seed, dev):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed + 100)
    m = build(mode, kind, codec)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    w = codec.w.repeat(2) if codec is not None else None
    for s in range(steps):
        pos = sample(bs, gen)
        grid, n = features(pos)
        tgt, _ = cell_mean(pos, pair_force(pos) * DT)
        mask = (n > 0).float()
        if kind == "float":
            y = m(grid)
            loss = (((y - tgt / scale) ** 2).mean(-1) * mask).sum() / mask.sum()
        else:
            t = codec.encode(tgt).reshape(bs, G * G, -1)
            y = m(grid, t) if kind == "conditional" else m(grid)
            bce = F.binary_cross_entropy_with_logits(y, t, reduction="none")
            loss = ((bce * w).mean(-1) * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 500 == 0:
            print(name, mode, seed, s, round(loss.item(), 4), flush=True)
    return m


def predict(m, kind, codec, scale, pos, soft=False):
    grid, n = features(pos)
    y = m(grid)
    if kind == "float":
        return y * scale, y
    yb = y.reshape(*y.shape[:2], 2, codec.nbits)
    if soft:
        return codec.soft_decode(yb), y
    return codec.decode((yb > 0).float()), y


def make_eval(seed, dev):
    tg = torch.Generator(device=dev).manual_seed(seed + 1)
    pos = sample(512, tg)
    Ftrue = pair_force(pos)
    tgt_cell, n = cell_mean(pos, Ftrue * DT)
    vel0 = torch.randn(256, N, 2, generator=tg, device=dev) * 0.3
    pos0 = sample(256, tg)
    e = dict(pos=pos, Ftrue=Ftrue, tgt=tgt_cell, mask=n > 0, vel0=vel0, pos0=pos0)
    e["true"] = rollout(pos0, vel0, pair_force, 20)
    return e


def rel(p, Ftrue):
    return ((p - Ftrue).norm(dim=-1).sum() / Ftrue.norm(dim=-1).sum()).item()


def rollout_err(e, fn):
    with torch.no_grad():
        tr = rollout(e["pos0"], e["vel0"], fn, 20)
    return {f"err@{t}": (tr[t - 1] - e["true"][t - 1]).norm(dim=-1).mean().item() for t in (5, 10, 20)}


def references(e):
    oracle = lambda p: apply_to_particles(p, cell_mean(p, pair_force(p) * DT)[0])
    return {"particle_force_rel_err": {"zero": 1.0, "oracle_cell_mean": rel(oracle(e["pos"]), e["Ftrue"])},
            "rollout_err": {"zero_force": rollout_err(e, lambda p: torch.zeros_like(p)), "oracle_cell_mean": rollout_err(e, oracle)}}


def evaluate(m, kind, codec, scale, e, soft=False):
    with torch.no_grad():
        dp, y = predict(m, kind, codec, scale, e["pos"], soft)
        mask = e["mask"]
        out = {"particle_force_rel_err": rel(apply_to_particles(e["pos"], dp), e["Ftrue"]),
               "cell_dp_rel_err": ((dp - e["tgt"]).norm(dim=-1)[mask].sum() / e["tgt"].norm(dim=-1)[mask].sum()).item()}
        if kind != "float":
            t = codec.encode(e["tgt"]).reshape(512, G * G, -1)
            acc = ((y > 0).float() == t).float()[mask].mean(0).reshape(2, codec.nbits)
            out["bit_acc_own_encoding_lsb_to_msb"] = {"x": [round(v, 4) for v in acc[0].tolist()], "y": [round(v, 4) for v in acc[1].tolist()]}
            fine = Codec(8, scale)
            tq = fine.to_q(e["tgt"])
            pq = fine.to_q(dp)
            tb = ((tq & 255)[..., None] >> torch.arange(8)) & 1
            pb = ((pq & 255)[..., None] >> torch.arange(8)) & 1
            out["bit_acc_decoded_plain_binary_lsb_to_msb"] = [round(v, 4) for v in (tb == pb).float()[mask].mean((0, 1)).tolist()]
    pf = lambda p: apply_to_particles(p, predict(m, kind, codec, scale, p, soft)[0])
    out["rollout_err"] = rollout_err(e, pf)
    return out


def arm_specs(scale):
    return {
        "plain-bits": ("bits", Codec(8, scale)),
        "significance-weighted-bce": ("bits", Codec(8, scale, weighted=True)),
        "drop-low-4-bits": ("bits", Codec(4, scale * 16)),
        "drop-low-5-bits": ("bits", Codec(3, scale * 32)),
        "gray-code": ("bits", Codec(8, scale, gray=True)),
        "mu-law-companded": ("bits", Codec(8, scale, mulaw=True)),
        "coarse-to-fine-conditioned": ("conditional", Codec(8, scale)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[4738, 4739, 4740])
    ap.add_argument("--out", default="results/bit_output_encoding_fixes/results.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--improve-margin", type=float, default=0.02)
    ap.add_argument("--conditional-trigger-ratio", type=float, default=1.03)
    a = ap.parse_args()
    dev = a.device
    torch.set_default_device(dev)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ckdir = os.path.join(os.path.dirname(a.out), "ckpt")
    os.makedirs(ckdir, exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    def save():
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, a.out)

    def run(seed, arm, mode, kind, codec, scale, e):
        key = f"{seed}/{arm}/{mode}"
        if key in res["runs"]:
            return
        t0 = time.time()
        m = train(arm, mode, kind, codec, scale, a.steps, a.bs, seed, dev)
        torch.save(m.state_dict(), os.path.join(ckdir, f"{seed}_{arm}_{mode}.pt"))
        res["runs"][key] = evaluate(m, kind, codec, scale, e)
        if arm == "plain-bits":
            res["runs"][f"{seed}/soft-decode/{mode}"] = evaluate(m, kind, codec, scale, e, soft=True)
        res["runs"][key]["train_seconds"] = round(time.time() - t0, 1)
        save()
        print("DONE", key, res["runs"][key]["particle_force_rel_err"], flush=True)

    res.setdefault("runs", {})
    res.setdefault("meta", {"steps": a.steps, "bs": a.bs, "device": dev, "improve_margin": a.improve_margin, "conditional_trigger_ratio": a.conditional_trigger_ratio})
    res.setdefault("refs", {})
    res.setdefault("scale", {})
    res.setdefault("decisions", {})
    for seed in a.seeds:
        gen = torch.Generator(device=dev).manual_seed(seed)
        cal = sample(2048, gen)
        cal_dp, _ = cell_mean(cal, pair_force(cal) * DT)
        scale = (cal_dp.abs().flatten().quantile(0.999) / 127).item()
        res["scale"][str(seed)] = scale
        e = make_eval(seed, dev)
        res["refs"][str(seed)] = references(e)
        save()
        specs = arm_specs(scale)
        for mode in ("centered", "global"):
            run(seed, "float", mode, "float", None, scale, e)
            run(seed, "plain-bits", mode, *specs["plain-bits"], scale, e)
        fr = lambda arm, mode: res["runs"][f"{seed}/{arm}/{mode}"]["particle_force_rel_err"]
        ratio = lambda arm, mode: fr(arm, mode) / fr("float", mode)
        phase1 = [k for k in specs if k not in ("plain-bits", "coarse-to-fine-conditioned")]
        for arm in phase1:
            if seed != a.seeds[0] and not res["decisions"][arm]:
                continue
            for mode in ("centered", "global"):
                run(seed, arm, mode, *specs[arm], scale, e)
            if seed == a.seeds[0]:
                res["decisions"][arm] = any(ratio(arm, md) < ratio("plain-bits", md) - a.improve_margin for md in ("centered", "global"))
        if seed == a.seeds[0]:
            closed = any(ratio(arm, "centered") <= a.conditional_trigger_ratio for arm in phase1 + ["soft-decode"])
            res["decisions"]["coarse-to-fine-conditioned"] = not closed
        if res["decisions"]["coarse-to-fine-conditioned"]:
            for mode in ("centered", "global"):
                run(seed, "coarse-to-fine-conditioned", mode, *specs["coarse-to-fine-conditioned"], scale, e)
        save()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
