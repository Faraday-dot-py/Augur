"""Distribution metric for rollouts past the chaos horizon (experiment sampled-force-tier-0-scored-on-distributions-past-the-chaos-horizon).

Exact reference = an ensemble of K float64 runs (kick-drift-kick, 4 substeps per tick, eps 0.5, G=1) whose initial positions are the scene plus rel * L * gaussian
(L = rms radius of the scene, as in chaos_floor.py). Per scene and tick the model state is reduced to two radial profiles about the centre of mass: body count per
radial bin and rms speed per radial bin (speed relative to the centre-of-mass velocity). A bin is scored when the ensemble-mean count is at least 2; it passes when the
model value lies within the ensemble mean +- 2 standard deviations (ddof 1). The leave-one-out coverage of the exact members is the reference's own noise floor.
"""
import numpy as np
import torch

EDGES = np.concatenate([np.geomspace(0.5, 64.0, 13), [np.inf]])
NB = len(EDGES) - 1
TOL = 1e-12


def exact_batch(pos, vel, steps, device, dt=0.1, substeps=4, eps=0.5, g=1.0, keep=()):
    """pos, vel (B,N,2) float64 tensors, unit masses -> {tick: (pos, vel)} for the ticks in keep (0 and steps included)."""
    h = dt / substeps
    pos, vel = pos.to(device), vel.to(device)
    n = pos.shape[1]
    eye = torch.eye(n, dtype=torch.bool, device=device)

    def acc(p):
        d = p[:, None, :, :] - p[:, :, None, :]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
        inv = inv.masked_fill(eye, 0.0)
        return g * (d * inv[..., None]).sum(2)

    out = {0: (pos.cpu(), vel.cpu())}
    a = acc(pos)
    for t in range(1, steps + 1):
        for _ in range(substeps):
            vel = vel + 0.5 * h * a
            pos = pos + h * vel
            a = acc(pos)
            vel = vel + 0.5 * h * a
        if t in keep or t == steps:
            out[t] = (pos.cpu(), vel.cpu())
    return out


def perturb(pos, rel, k, rng):
    """pos (N,2) float64 numpy -> (k, N, 2) perturbed copies."""
    com = pos.mean(0)
    L = float(np.sqrt(((pos - com) ** 2).sum(-1).mean()))
    return pos[None] + rel * L * rng.standard_normal((k,) + pos.shape)


def profiles(pos, vel):
    """pos, vel (N,2) float64 numpy -> (count (NB,), rms speed (NB,), nan where the bin is empty)."""
    r = np.linalg.norm(pos - pos.mean(0), axis=-1)
    v = np.linalg.norm(vel - vel.mean(0), axis=-1)
    idx = np.searchsorted(EDGES, r, side="right") - 1
    idx = np.clip(idx, 0, NB - 1)
    cnt = np.bincount(idx, minlength=NB).astype(np.float64)
    sq = np.bincount(idx, weights=v ** 2, minlength=NB)
    rms = np.where(cnt > 0, np.sqrt(sq / np.maximum(cnt, 1)), np.nan)
    return cnt, rms


def inside_band(x, ens):
    """x (NB,), ens (K, NB) with nan -> bool (NB,): |x - mean| <= 2 std (ddof 1), nan model value fails."""
    mu = np.nanmean(ens, axis=0)
    sd = np.nanstd(ens, axis=0, ddof=1)
    return np.abs(x - mu) <= 2.0 * sd + TOL


def score_scene(model_prof, ens_prof):
    """model_prof (cnt, rms); ens_prof (cnt (K,NB), rms (K,NB)). Returns per-metric dict of scored/inside counts for the model and for leave-one-out exact members."""
    out = {}
    k = ens_prof[0].shape[0]
    scored = ens_prof[0].mean(0) >= 2.0
    for name, x, ens in (("density", model_prof[0], ens_prof[0]), ("speed", model_prof[1], ens_prof[1])):
        ins = inside_band(x, ens) & scored
        loo = 0
        for j in range(k):
            rest = np.delete(ens, j, axis=0)
            loo += int((inside_band(ens[j], rest) & scored).sum())
        out[name] = dict(scored=int(scored.sum()), inside=int(ins.sum()), loo_scored=int(scored.sum()) * k, loo_inside=loo)
    return out


def pool(scene_scores):
    """list of score_scene dicts -> pooled fractions per metric."""
    res = {}
    for name in ("density", "speed"):
        s = sum(x[name]["scored"] for x in scene_scores)
        i = sum(x[name]["inside"] for x in scene_scores)
        ls = sum(x[name]["loo_scored"] for x in scene_scores)
        li = sum(x[name]["loo_inside"] for x in scene_scores)
        res[name] = dict(frac=i / max(s, 1), scored_bins=s, inside_bins=i, self_frac=li / max(ls, 1))
    return res
