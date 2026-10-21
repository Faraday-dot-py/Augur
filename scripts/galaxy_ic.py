"""Disc-galaxy initial conditions, equal-mass bodies, no dark matter: truncated exponential stellar disc + truncated 2D-Plummer bulge. Circular speed from the enclosed mass of both components, v_c^2 = M(<R) R^2 / (R^2 + eps^2)^1.5 (softened G=1, eps .5, same units as the scatter-field models). Disc: Toomre-Q radial dispersion, epicycle azimuthal dispersion. Bulge: isotropic 2D-Jeans dispersion times cdisp, no rotation. All speeds capped at 0.85 of the local escape speed. Chunked and deterministic given (seed, chunk) so any N up to 1e9 can be generated on the GPU; bodies are i.i.d., so the same distribution holds at every N."""
import math

import torch

EPS = 0.5

PRESETS = {
    "g2": dict(mdisc=80.0, mbulge=20.0, rd=4.0, rmax=20.0, ab=1.5, rbmax=8.0, Q=1.5, cdisp=1.0),
    "g1": dict(mdisc=240.0, mbulge=60.0, rd=4.0, rmax=20.0, ab=1.5, rbmax=8.0, Q=1.5, cdisp=1.0),
}


def total_mass(P):
    return P["mdisc"] + P["mbulge"]


def _disc_frac(r, P):
    x = r.clamp(max=P["rmax"]) / P["rd"]
    xm = P["rmax"] / P["rd"]
    return (1 - (1 + x) * torch.exp(-x)) / (1 - (1 + xm) * math.exp(-xm))


def _bulge_frac(r, P):
    r = r.clamp(max=P["rbmax"])
    return (r ** 2 / (r ** 2 + P["ab"] ** 2)) / (P["rbmax"] ** 2 / (P["rbmax"] ** 2 + P["ab"] ** 2))


def menc(r, P):
    return P["mdisc"] * _disc_frac(r, P) + P["mbulge"] * _bulge_frac(r, P)


def omega2(r, P):
    return menc(r, P) / (r ** 2 + EPS ** 2) ** 1.5


def vcirc(r, P):
    return (omega2(r, P) * r ** 2).sqrt()


def kappa(r, P):
    d = 1e-3 * r.clamp(min=0.05)
    dom = (omega2(r + d, P) - omega2((r - d).clamp(min=1e-3), P)) / (2 * d)
    return (4 * omega2(r, P) + r * dom).clamp(min=0.25 * omega2(r, P)).sqrt()


def sigma_disc(r, P):
    sig = P["mdisc"] / (2 * math.pi * P["rd"] ** 2) * torch.exp(-r / P["rd"]) / (1 - (1 + P["rmax"] / P["rd"]) * math.exp(-P["rmax"] / P["rd"]))
    return sig


RT = 40.0
NT = 16001


def tables(P, dev):
    """Radial tables: escape speed and 2D-Jeans isotropic bulge dispersion from the total softened enclosed-mass gravity g(r) = M(<r) r / (r^2 + eps^2)^1.5."""
    r = torch.linspace(0, RT, NT, device=dev, dtype=torch.float64)
    gr = menc(r, P) * r / (r ** 2 + EPS ** 2) ** 1.5
    dr = r[1] - r[0]
    cum = torch.flip(torch.cumsum(torch.flip((gr[1:] + gr[:-1]) * 0.5 * dr, [0]), 0), [0])
    cum = torch.cat([cum, cum.new_zeros(1)])
    phi = -cum - total_mass(P) / (RT ** 2 + EPS ** 2) ** 0.5
    sb = (r ** 2 + P["ab"] ** 2) ** -2 * (r <= P["rbmax"])
    f = sb * gr
    cumf = torch.flip(torch.cumsum(torch.flip((f[1:] + f[:-1]) * 0.5 * dr, [0]), 0), [0])
    cumf = torch.cat([cumf, cumf.new_zeros(1)])
    sig2 = cumf / sb.clamp(min=1e-12)
    return r, (-2 * phi).sqrt().float(), sig2.clamp(min=0).sqrt().float() * P["cdisp"]


def interp(r, grid, tab):
    x = (r / (RT / (NT - 1))).clamp(0, NT - 1.001)
    i = x.floor().long()
    w = x - i
    return tab[i] * (1 - w) + tab[i + 1] * w


def sample_chunk(m, P, g, dev, tab=None):
    """Returns pos (m,2), vel (m,2) float32."""
    if tab is None:
        tab = tables(P, dev)
    fb = P["mbulge"] / total_mass(P)
    isb = torch.rand(m, device=dev, generator=g) < fb
    ur = torch.rand(m, device=dev, generator=g)
    th = torch.rand(m, device=dev, generator=g) * (2 * math.pi)
    z = torch.randn(m, 3, device=dev, generator=g)
    xm = P["rmax"] / P["rd"]
    tab_x = torch.linspace(0, xm, 8193, device=dev, dtype=torch.float64)
    tab_f = (1 - (1 + tab_x) * torch.exp(-tab_x)) / (1 - (1 + xm) * math.exp(-xm))
    idx = torch.searchsorted(tab_f.float(), ur).clamp(1, 8192)
    f0, f1 = tab_f[idx - 1].float(), tab_f[idx].float()
    x0, x1 = tab_x[idx - 1].float(), tab_x[idx].float()
    rdisc = (x0 + (ur - f0) / (f1 - f0).clamp(min=1e-12) * (x1 - x0)) * P["rd"]
    umax = P["rbmax"] ** 2 / (P["rbmax"] ** 2 + P["ab"] ** 2)
    ub = ur * umax
    rb = P["ab"] * (ub / (1 - ub)).sqrt()
    r = torch.where(isb, rb, rdisc)
    vc = vcirc(r, P)
    om = omega2(r, P).sqrt()
    kap = kappa(r, P)
    sr = (P["Q"] * 3.36 * sigma_disc(r, P) / kap).clamp(max=0.4 * vc)
    sp = sr * kap / (2 * om.clamp(min=1e-6))
    er = torch.stack([torch.cos(th), torch.sin(th)], 1)
    et = torch.stack([-torch.sin(th), torch.cos(th)], 1)
    vr_d = sr * z[:, 0]
    vt_d = vc + sp * z[:, 1]
    sb = interp(r, *tab[:1], tab[2])
    vr_b = sb * z[:, 0]
    vt_b = sb * z[:, 1]
    vr = torch.where(isb, vr_b, vr_d)
    vt = torch.where(isb, vt_b, vt_d)
    v = vr[:, None] * er + vt[:, None] * et
    cap = 0.85 * interp(r, tab[0], tab[1])
    v = v * (cap / v.norm(dim=-1).clamp(min=1e-9)).clamp(max=1.0)[:, None]
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return r[:, None] * er, v


def make_galaxy(n, P, seed, chunk, c, dev, pd=torch.float32, md=torch.float32):
    """pos (n,2) in pd, relativistic momentum (n,2) in md with zero net momentum; ICs drawn chunk by chunk on the GPU in two passes (pass 0 for the mean momentum)."""
    pos = torch.empty(n, 2, device=dev, dtype=pd)
    mom = torch.empty(n, 2, device=dev, dtype=md)
    psum = torch.zeros(2, dtype=torch.float64, device=dev)
    tab = tables(P, dev)
    for pas in range(2):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        pmean = (psum / n).float()
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            p, v = sample_chunk(e - s, P, g, dev, tab)
            vmag = v.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
            q = v * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
            if pas == 0:
                psum += q.sum(0, dtype=torch.float64)
                continue
            pos[s:e] = p.to(pd)
            mom[s:e] = (q - pmean).to(md)
    return pos, mom
