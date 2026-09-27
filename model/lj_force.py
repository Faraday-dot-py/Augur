import torch
import torch.nn as nn

from model.token_graph import build_radius_graph_periodic


class LJForceDynamics(nn.Module):
    """Learned radial pair force on a periodic box, integrated with velocity Verlet in `substeps` steps per
    frame. The force on i from j is f(d) along the minimum-image line, f a learned function of distance
    only inside `cutoff`, so momentum is conserved exactly. f(d) = net(log d) * d^-7 puts the LJ tail scale
    into the parametrisation. Positions are kept unwrapped; only minimum-image differences are used."""

    def __init__(self, dt=0.005, substeps=20, cutoff=2.5, width=64):
        super().__init__()
        self.dt = dt
        self.substeps = substeps
        self.cutoff = cutoff
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 1))
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)
        self.table = None

    def pair_force(self, d):
        return self.net(torch.log(d)[:, None])[:, 0] * d ** -7

    def accel(self, pos, box, chunk=1024, stats=False):
        """pos (B, N, 2), box (B, 2). Returns acc (B, N, 2) and, if stats, (pe, virial) summed per system."""
        b, n, _ = pos.shape
        acc = torch.zeros(b * n, 2, dtype=pos.dtype, device=pos.device)
        pe = pos.new_zeros(b)
        vir = pos.new_zeros(b)
        box_e = box[:, None, None, :]
        for s in range(0, n, chunk):
            rel = pos[:, s:s + chunk, None, :] - pos[:, None, :, :]
            rel = rel - box_e * torch.round(rel / box_e)
            r2 = (rel ** 2).sum(-1)
            mask = r2 < self.cutoff ** 2
            ii = torch.arange(s, min(s + chunk, n), device=pos.device)
            mask[:, torch.arange(len(ii)), ii] = False
            bi, ri, _ = mask.nonzero(as_tuple=True)
            d = torch.sqrt(r2[mask] + 1e-12)
            f = self.pair_force(d)
            acc = acc.index_add(0, bi * n + s + ri, (f / d)[:, None] * rel[mask])
            if stats:
                pe = pe.index_add(0, bi, 0.5 * self.potential(d))
                vir = vir.index_add(0, bi, 0.5 * f * d)
        acc = acc.view(b, n, 2)
        return (acc, pe, vir) if stats else acc

    @torch.no_grad()
    def build_table(self, dmin=0.4, points=8000):
        d = torch.linspace(dmin, self.cutoff, points, dtype=torch.float64, device=self.net[0].weight.device)
        f = self.pair_force(d)
        seg = 0.5 * (f[1:] + f[:-1]) * (d[1:] - d[:-1])
        u = torch.cat([seg.flip(0).cumsum(0).flip(0), seg.new_zeros(1)])
        self.table = (dmin, (self.cutoff - dmin) / (points - 1), u)

    def potential(self, d):
        dmin, h, u = self.table
        x = ((d - dmin) / h).clamp(0, len(u) - 1.001)
        i = x.long()
        w = x - i
        return u[i] * (1 - w) + u[i + 1] * w

    def force_fn(self, pos, box):
        acc, pe, vir = self.accel(pos[None], box[None], stats=True)
        return acc[0], pe[0], vir[0]

    def step_frame(self, pos, vel, box):
        h = self.dt
        a = self.accel(pos, box)
        for _ in range(self.substeps):
            vel = vel + 0.5 * h * a
            pos = pos + h * vel
            a = self.accel(pos, box)
            vel = vel + 0.5 * h * a
        return pos, vel

    def rollout(self, pos, vel, box, frames):
        ps, vs = [], []
        for _ in range(frames):
            pos, vel = self.step_frame(pos, vel, box)
            ps.append(pos)
            vs.append(vel)
        return torch.stack(ps), torch.stack(vs)

    def accel_single(self, pos, box, stats=False):
        """O(N) periodic cell-list accel for one (unbatched) system: pos (N, 2), box (2,). Same
        physics as `accel` (verified to match on small N in scripts/lj_scaling.py); used for
        rollouts too large for the dense chunked all-pairs path in `accel`/`step_frame`."""
        edges = build_radius_graph_periodic(pos.detach(), box, self.cutoff)
        acc = torch.zeros_like(pos)
        pe = pos.new_zeros(())
        vir = pos.new_zeros(())
        if edges.shape[1] == 0:
            return (acc, pe, vir) if stats else acc
        src, dst = edges[0], edges[1]
        rel = pos[dst] - pos[src]
        rel = rel - box * torch.round(rel / box)
        d = torch.sqrt((rel ** 2).sum(-1) + 1e-12)
        f = self.pair_force(d)
        acc = acc.index_add(0, dst, (f / d)[:, None] * rel)
        if stats:
            pe = 0.5 * self.potential(d).sum()
            vir = 0.5 * (f * d).sum()
        return (acc, pe, vir) if stats else acc

    def force_fn_single(self, pos, box):
        return self.accel_single(pos, box, stats=True)

    def step_frame_single(self, pos, vel, box):
        h = self.dt
        a = self.accel_single(pos, box)
        for _ in range(self.substeps):
            vel = vel + 0.5 * h * a
            pos = pos + h * vel
            a = self.accel_single(pos, box)
            vel = vel + 0.5 * h * a
        return pos, vel
