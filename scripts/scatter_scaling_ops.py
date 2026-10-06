"""Opt-in helpers for scaling tests of ScatterField (no edits to scatter_field.py).

chunked_pp(model): replace pp_acc by a query-chunked exact kNN (same selection and numerics as the stock all-pairs version, O(chunk*N) memory instead of O(N^2)).
retarget(model, grid, extent): change the world scale (cell count / domain extent) of a trained model in place; the weights are grid-size independent.
exact_acc / exact_energy: chunked fp64 softened all-pairs gravity (G=1, unit or given masses) for N too large for a dense N x N matrix.
"""
import types

import torch

CHUNK_ELEMS = 2.0e8


def _pp_acc_chunked(self, pos, mass, mask, knn=16):
    B, N = mask.shape
    k = min(knn, N - 1)
    if k <= 0:
        return torch.zeros_like(pos)
    mj_all = mass * mask
    c = max(1, int(CHUNK_ELEMS / (B * N)))
    cols = torch.arange(N, device=pos.device)
    outs = []
    for s in range(0, N, c):
        e = min(N, s + c)
        d_all = pos[:, None, :, :] - pos[:, s:e, None, :]
        r2 = (d_all ** 2).sum(-1)
        bad = ((1 - mask)[:, None, :] > 0) | (cols[None, :] == torch.arange(s, e, device=pos.device)[:, None])[None]
        r2 = r2.masked_fill(bad, 1e12)
        r2_sel, idx = torch.topk(r2, k, dim=-1, largest=False)
        d = torch.gather(d_all, 2, idx[..., None].expand(B, e - s, k, 2))
        mj = torch.gather(mj_all[:, None, :].expand(B, e - s, N), 2, idx)
        ok = (r2_sel < 1e11).to(pos.dtype) * mask[:, s:e, None]
        r = torch.sqrt(r2_sel.clamp(max=1e11) + 1e-8)
        g = self.ppmlp(torch.stack([r, torch.log(r + 0.05)], -1))[..., 0]
        win = (1 - (r / self.pp).clamp(max=1.0) ** 2) ** 2
        w = g * win * mj * ok
        outs.append((w[..., None] * d / r[..., None]).sum(2))
    return torch.cat(outs, 1)


def chunked_pp(model):
    model.pp_acc = types.MethodType(_pp_acc_chunked, model)
    return model


def no_pp(model):
    model.pp_acc = types.MethodType(lambda self, pos, mass, mask, knn=16: torch.zeros_like(pos), model)
    return model


def retarget(model, grid, extent):
    model.grid, model.extent, model.h = grid, float(extent), float(extent) / grid
    return model


@torch.no_grad()
def exact_acc(pos, eps, mass=None, elems=3.0e8):
    """pos (N,2) float64 (cuda). Softened all-pairs acceleration, G=1, chunked over targets."""
    N = pos.shape[0]
    m = torch.ones(N, dtype=pos.dtype, device=pos.device) if mass is None else mass
    c = max(1, int(elems / N))
    out = torch.empty_like(pos)
    for s in range(0, N, c):
        e = min(N, s + c)
        d = pos[None, :, :] - pos[s:e, None, :]
        r2 = (d ** 2).sum(-1) + eps ** 2
        inv = r2 ** -1.5
        idx = torch.arange(s, e, device=pos.device)
        inv[idx - s, idx] = 0.0
        out[s:e] = (d * (inv * m[None, :])[..., None]).sum(1)
    return out


@torch.no_grad()
def exact_rollout(pos, vel, steps, dt, substeps, eps):
    h = dt / substeps
    ps, vs = [pos], [vel]
    a = exact_acc(pos, eps)
    for _ in range(steps):
        for _ in range(substeps):
            vel = vel + 0.5 * h * a
            pos = pos + h * vel
            a = exact_acc(pos, eps)
            vel = vel + 0.5 * h * a
        ps.append(pos)
        vs.append(vel)
    return torch.stack(ps), torch.stack(vs)


@torch.no_grad()
def exact_energy(pos, vel, eps, elems=3.0e8):
    """Total energy for unit masses, G=1, softened (sum over pairs i<j)."""
    N = pos.shape[0]
    c = max(1, int(elems / N))
    pot = torch.zeros((), dtype=pos.dtype, device=pos.device)
    for s in range(0, N, c):
        e = min(N, s + c)
        d = pos[None, :, :] - pos[s:e, None, :]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -0.5
        idx = torch.arange(s, e, device=pos.device)
        inv[idx - s, idx] = 0.0
        pot = pot + inv.sum()
    return float(0.5 * (vel ** 2).sum() - 0.5 * pot)
