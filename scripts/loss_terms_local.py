"""Soft loss terms for window rollouts (docs/debugging/loss-analysis-2026-10-26.md, designs 1 and 2). Training-time only, computed from truth states.

work_bias_step: per scene, particles are split by truth local density at the window start into lowest 50% / middle 40% / top 10% (neighbours within
radius 3). Per bin, the SIGNED mean over particles of (model - truth) kinetic-energy change since the window start is squared and averaged over
non-empty bins, so a one-sided bias cannot cancel against per-particle noise the way it does inside a per-particle MSE.
coherent_step: squared error of the mean position (and 0.1 x velocity) error over each particle's 16 nearest truth neighbours (self included)."""
import torch

RADIUS = 3.0
BIN_EDGES = (0.5, 0.9)
KNN = 16


@torch.no_grad()
def density_bins(pos, mask, radius=RADIUS):
    """pos (B, N, 2), mask (B, N) -> long (B, N) in {0, 1, 2}: lowest 50% / middle 40% / top 10% of neighbour count within radius, per scene."""
    d2 = ((pos[:, :, None] - pos[:, None]) ** 2).sum(-1)
    cnt = ((d2 < radius ** 2) & (mask[:, None] > 0)).sum(-1).float() - 1.0
    cnt = torch.where(mask > 0, cnt, torch.full_like(cnt, -1.0))
    out = torch.zeros_like(mask, dtype=torch.long)
    for b in range(pos.shape[0]):
        ok = mask[b] > 0
        n = int(ok.sum())
        if n == 0:
            continue
        rank = torch.zeros(pos.shape[1], device=pos.device)
        order = cnt[b][ok].argsort(stable=True)
        r = torch.empty(n, device=pos.device)
        r[order] = torch.arange(n, device=pos.device, dtype=torch.float32)
        rank[ok] = (r + 0.5) / n
        out[b] = (rank >= BIN_EDGES[0]).long() + (rank >= BIN_EDGES[1]).long()
    return out


def work_bias_step(vel_m, vel_t, vel0, mass, w, bins):
    """Mean over scenes and non-empty bins of (bin mean of dKE_model - dKE_truth)^2. vel0 is the truth velocity at the window start; w (B, N) masks valid particles."""
    dke = 0.5 * mass * ((vel_m ** 2).sum(-1) - (vel0 ** 2).sum(-1))
    dke_t = 0.5 * mass * ((vel_t ** 2).sum(-1) - (vel0 ** 2).sum(-1))
    e = (dke - dke_t) * w
    tot = e.new_zeros(())
    cnt = e.new_zeros(())
    for k in range(3):
        sel = ((bins == k) & (w > 0)).to(e.dtype)
        n = sel.sum(-1)
        bm = (e * sel).sum(-1) / n.clamp_min(1.0)
        has = (n > 0).to(e.dtype)
        tot = tot + (bm ** 2 * has).sum()
        cnt = cnt + has.sum()
    return tot / cnt.clamp_min(1.0)


@torch.no_grad()
def knn_index(pos_t, w, k=KNN):
    """(B, N, k) indices of the k nearest valid truth particles (self included); invalid candidates get infinite distance. Second return flags real neighbours."""
    d2 = ((pos_t[:, :, None] - pos_t[:, None]) ** 2).sum(-1)
    d2 = d2.masked_fill(w[:, None] <= 0, float("inf"))
    k = min(k, pos_t.shape[1])
    val, idx = d2.topk(k, dim=-1, largest=False)
    return idx, torch.isfinite(val).to(pos_t.dtype)


def knn_mean(err, idx, ok):
    """err (B, N, C); mean over each particle's neighbours -> (B, N, C)."""
    B, N, k = idx.shape
    g = torch.gather(err, 1, idx.reshape(B, N * k, 1).expand(-1, -1, err.shape[-1])).reshape(B, N, k, -1)
    return (g * ok[..., None]).sum(2) / ok.sum(2).clamp_min(1.0)[..., None]


def coherent_step(pos_m, pos_t, vel_m, vel_t, w, k=KNN, vel_weight=0.1):
    """Per-component mean over valid particles of |neighbourhood-mean error|^2 for position plus vel_weight x velocity."""
    idx, ok = knn_index(pos_t, w, k)
    cp = knn_mean((pos_m - pos_t) * w[..., None], idx, ok)
    cv = knn_mean((vel_m - vel_t) * w[..., None], idx, ok)
    den = w.sum() * 2
    return (((cp ** 2).sum(-1) * w).sum() + vel_weight * ((cv ** 2).sum(-1) * w).sum()) / den
