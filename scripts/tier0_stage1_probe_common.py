"""Shared closed-loop rollout driver for the stage-1 probes of the frozen tiled tier 0 (halo exchange as trained). Same Verlet loop as
scripts/replay_mechanism_force_error_breakdown.collect in 'own' mode: the model feeds its own states; a hook may replace the force it returns,
an observer sees every call. Energy is compared with the exact trajectory's E(t) at the same tick, never E(0)."""
import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_tiered as ttr

SEEDS = (9100, 9200, 9300)
DT, EPS = ttr.DT, ttr.EPS
ERR_TICKS = (5, 10, 20, 50, 100)
ENERGY_TICKS = (20, 60, 100)


@torch.no_grad()
def rollout(tt, d, hook=None, observe=None):
    """hook(t, pos, vel_in, acc) -> acc used by the integrator (t = 0 is the initial call). observe(t, pos, vel_in, acc_model, acc_used, fld_in, fld_out)
    runs after every call from tick 1. Returns per-scene arrays: err[t] (mean position error per body), energy per tick."""
    m, mk = d["M"].float(), d["mask"].float()
    T = d["P"].shape[1]
    pos, vel = d["P"][:, 0].float(), d["V"][:, 0].float()
    fld = tt.init_state(pos.shape[0])
    a_prev, fld = tt.force(pos, vel, m, mk, fld)
    if hook is not None:
        a_prev = hook(0, pos, vel, a_prev)
    cnt = d["mask"].sum(1)
    err, en = {}, [sf.energy(pos.double(), vel.double(), d["M"], d["mask"], EPS).cpu()]
    for t in range(1, T):
        pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
        vin = vel + DT * a_prev
        a_new, fld_out = tt.force(pn, vin, m, mk, fld)
        a_use = a_new if hook is None else hook(t, pn, vin, a_new)
        if observe is not None:
            observe(t, pn, vin, a_new, a_use, fld, fld_out)
        vel = vel + 0.5 * DT * (a_prev + a_use)
        pos, a_prev, fld = pn, a_use, fld_out
        en.append(sf.energy(pos.double(), vel.double(), d["M"], d["mask"], EPS).cpu())
        if t in ERR_TICKS:
            err[t] = (((pos.double() - d["P"][:, t]).norm(dim=-1) * d["mask"]).sum(1) / cnt).cpu().numpy()
    return {"err": err, "energy": torch.stack(en)}


def truth_energy(d):
    return torch.stack([sf.energy(d["P"][:, t], d["V"][:, t], d["M"], d["mask"], EPS).cpu() for t in range(d["P"].shape[1])])


def energy_vs_truth(en, et):
    """Per-scene |E_model(t) - E_true(t)| / |E_true(t)| and the signed value, at ENERGY_TICKS."""
    out = {}
    for t in ENERGY_TICKS:
        s = (en[t] - et[t]) / et[t].abs().clamp_min(1e-6)
        out[t] = {"abs": s.abs().numpy(), "signed": s.numpy()}
    return out


def tolist(x):
    if isinstance(x, dict):
        return {str(k): tolist(v) for k, v in x.items()}
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x.tolist()
    return x
