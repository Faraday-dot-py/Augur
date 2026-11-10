"""Random law zoo and law encoder (experiment law-inferred-from-motion-on-random-law-zoo).

A law is the tuple of scatter_far_general.LAWS: (a1, p1, a2, p2, inv_lam, eps, decaying), force magnitude
(a1 r / (r^2 + eps^2)^((p1+1)/2) + a2 r / (r^2 + eps^2)^((p2+1)/2)) exp(-r inv_lam). The zoo samples smooth radial laws of this family. A sampled law is excluded from
training when its force curve (log10 |F| on PROBE_R, mean removed) is within SHAPE_TOL rms of pow1.5 or yuk32, the two held-out laws; held-out laws are drawn inside that region.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from scripts import scatter_far_general as sfar
from scripts import scatter_multi as sm

EPS = sfar.EPS
PROBE_R = np.geomspace(2.0, 32.0, 16)
EXPONENT_R = np.geomspace(8.0, 32.0, 8)
SHAPE_TOL = 0.10
HELD_REFS = ("pow1.5", "yuk32")
CODE_DIM = 5


def _curve(law, r):
    return sfar.force_t(torch.tensor(r), law).abs().clamp_min(1e-12).log10().numpy()


def shape_curve(law):
    c = _curve(law, PROBE_R)
    return c - c.mean()


def shape_distance(law_a, law_b):
    return float(np.sqrt(((shape_curve(law_a) - shape_curve(law_b)) ** 2).mean()))


def held_out_distance(law):
    return min(shape_distance(law, sfar.LAWS[n]) for n in HELD_REFS)


def effective_exponent(law):
    """Negative slope of log F against log r over EXPONENT_R (2 for gravity, the power p of a pure power law away from the softening)."""
    x = np.log(EXPONENT_R)
    y = _curve(law, EXPONENT_R) * math.log(10.0)
    return float(-np.polyfit(x, y, 1)[0])


def sample_law(rng):
    p1 = float(rng.uniform(0.5, 4.0))
    a1 = float(rng.choice([-1.0, 1.0], p=[0.25, 0.75]))
    il = 0.0 if rng.random() < 0.5 else float(np.exp(rng.uniform(math.log(1 / 256), math.log(1 / 4))))
    a2, p2 = 0.0, 2.0
    if rng.random() < 0.3:
        p2 = float(rng.uniform(p1 + 0.5, 5.0))
        a2 = -a1 * float(10 ** rng.uniform(-1.0, 0.5))
    return (a1, p1, a2, p2, il, EPS, il > 0.0)


def sample_zoo(n, seed):
    """n training laws (none within SHAPE_TOL of a held-out law) and the number of rejected draws."""
    rng = np.random.default_rng([seed, 1])
    out, rejected = [], 0
    while len(out) < n:
        law = sample_law(rng)
        if held_out_distance(law) < SHAPE_TOL:
            rejected += 1
            continue
        out.append(law)
    return out, rejected


def sample_heldout(n, seed):
    """n held-out laws, half in the pow1.5 neighbourhood (p in [1.4, 1.6]) and half in the yuk32 neighbourhood (lambda in [28, 36])."""
    rng = np.random.default_rng([seed, 2])
    out = []
    for i in range(n):
        if i % 2 == 0:
            law = (1.0, float(rng.uniform(1.4, 1.6)), 0.0, 2.0, 0.0, EPS, False)
        else:
            law = (1.0, 2.0, 0.0, 2.0, 1.0 / float(rng.uniform(28.0, 36.0)), EPS, True)
        assert held_out_distance(law) < SHAPE_TOL
        out.append(law)
    return out


def params_of_law(law, scale):
    """(7, 4) fp64 parameter rows for a single-law scene, as scatter_multi.params_of with slot 0 only."""
    out = np.zeros((7, 4))
    out[:, :] = np.array([0.0, 2.0, 0.0, 2.0, 0.0, EPS, 0.0])[:, None]
    a1, p1, a2, p2, il, eps, _ = law
    out[:, 0] = (a1 * scale, p1, a2 * scale, p2, il, eps, 0.0)
    return out


def gen_zoo_batch(laws, rng, device, steps):
    """laws: list of (law tuple, amplitude scale). Same scenes, truth integrator and acceptance rule as scatter_multi.gen_batch."""
    B = len(laws)
    pos0 = np.zeros((B, sm.N_PAD, 2))
    vel0 = np.zeros((B, sm.N_PAD, 2))
    mask = np.zeros((B, sm.N_PAD))
    par = np.zeros((B, 7, 4))
    nn_ = np.zeros(B, dtype=np.int64)
    for b, (law, scale) in enumerate(laws):
        p, v = sm.init_scene(rng, "uniform" if rng.random() < 0.5 else "cluster")
        n = p.shape[0]
        pos0[b, :n], vel0[b, :n], mask[b, :n], nn_[b] = p, v, 1.0, n
        par[b] = params_of_law(law, scale)
    t = lambda a: torch.tensor(a, device=device, dtype=torch.float64)
    P, V = sm.simulate(t(pos0), t(vel0), t(np.zeros((B, sm.N_PAD))), t(mask), t(par), steps)
    out = (P.abs().amax(-1) > 32.0) & (t(mask)[:, None, :] > 0)
    frac = out.any(1).double().sum(1) / t(mask).sum(1)
    ok = torch.isfinite(P).all(-1).all(-1).all(-1) & torch.isfinite(V).all(-1).all(-1).all(-1) & (frac <= 0.05)
    return {"P": P.float(), "V": V.float(), "mask": t(mask).float(), "n": nn_, "ok": ok, "params": t(par)}


def sample_scale(rng):
    return float(10 ** rng.uniform(-1.0, 1.0))


class LawEncoder(nn.Module):
    """Reads T0 observed ticks (positions and velocities, no theta) and returns a (B, sm.DC) code that takes theta's place in MultiScatterField.set_law.

    Each body i with its knn nearest bodies j gives tokens (log r_ij, a_i . e_ij, log(1 + |a_i|)) with a_i the finite-difference acceleration over a tick and e_ij the unit vector
    to j; tokens are embedded, mean-pooled over bodies, neighbours and ticks, and mapped to CODE_DIM numbers placed in the first slot, the other slots zero (single-law structure).
    """

    def __init__(self, knn=16, t0=4, hidden=64, dt=0.1):
        super().__init__()
        self.knn, self.t0, self.dt = knn, t0, dt
        self.tok = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, CODE_DIM))

    def forward(self, P, V, mask):
        """P, V (B, T, N, 2) with T > t0, mask (B, N) -> (B, sm.DC)."""
        B, _, N, _ = P.shape
        k = min(self.knn, N - 1)
        valid = mask > 0
        feats, weights = [], []
        for t in range(self.t0):
            a = (V[:, t + 1] - V[:, t]) / self.dt
            d = P[:, t, None, :, :] - P[:, t, :, None, :]
            r = d.norm(dim=-1)
            big = torch.finfo(r.dtype).max
            r = r.masked_fill(~valid[:, None, :] | ~valid[:, :, None], big)
            r = r.masked_fill(torch.eye(N, dtype=torch.bool, device=r.device)[None], big)
            rk, jk = r.topk(k, dim=-1, largest=False)
            dk = torch.gather(d, 2, jk[..., None].expand(B, N, k, 2))
            e = dk / rk.clamp_min(1e-6)[..., None]
            radial = (a[:, :, None, :] * e).sum(-1)
            mag = torch.log1p(a.norm(dim=-1))[:, :, None].expand(B, N, k)
            w = (valid[:, :, None] & (rk < big)).to(P.dtype)
            feats.append(torch.stack([torch.log(rk.clamp_min(1e-6)), radial, mag], -1))
            weights.append(w)
        f, w = torch.cat(feats, 1), torch.cat(weights, 1)
        h = self.tok(f.to(self.tok[0].weight.dtype)) * w[..., None].to(self.tok[0].weight.dtype)
        cnt = w.sum((1, 2)).clamp_min(1.0)[:, None].to(h.dtype)
        mean = h.sum((1, 2)) / cnt
        sq = ((h - mean[:, None, None, :] * w[..., None].to(h.dtype)) ** 2).sum((1, 2)) / cnt
        code = self.head(torch.cat([mean, sq.sqrt()], -1))
        return torch.cat([code, code.new_zeros(B, sm.DC - CODE_DIM)], -1)


def exponent_probe(train_codes, train_exp, test_codes, test_exp):
    """Linear (least squares with intercept) map from code to effective exponent fitted on train laws; returns R-squared on the test laws."""
    X = np.concatenate([train_codes, np.ones((len(train_codes), 1))], 1)
    w = np.linalg.lstsq(X, train_exp, rcond=None)[0]
    pred = np.concatenate([test_codes, np.ones((len(test_codes), 1))], 1) @ w
    ss_res = float(((test_exp - pred) ** 2).sum())
    ss_tot = float(((test_exp - test_exp.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)
