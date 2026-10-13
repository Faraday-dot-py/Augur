"""D2b (recurrent-state contraction retry): option-C ContactScatterField with an optional sigmoid forget gate on the recurrent field input, and the
large-signal growth penalty. See docs/debugging/error-correction-D2b-contraction.md."""
import torch
import torch.nn.functional as F

from scripts import scatter_contact as sc
from scripts import scatter_contact_scenes as cs
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf


def build(ckpt, dev, gate=None, bias=-6.0):
    """gate=None: gate iff the checkpoint has fgate weights. Plain C / D5ctrl checkpoints load unchanged (gate off)."""
    sd = torch.load(ckpt, map_location=dev, weights_only=False)["model"] if ckpt and ckpt != "none" else None
    if gate is None:
        gate = sd is not None and "fgate.weight" in sd
    cfg = tsf.EXPS["E"]
    kw = dict(tsf.BASE)
    kw.update(tsf.VARIANTS["ms_kp_pot_v_g128"])
    model = sc.ContactScatterField(extent=cfg["extent"], dt=cs.DT, in_scale=cfg["in_scale"], obstacle="chan", **kw)
    if gate:
        model.add_gate(bias)
    if sd is not None:
        r = model.load_state_dict(sd, strict=False)
        print(f"loaded {ckpt}: gate {gate} missing {list(r.missing_keys)} unexpected {list(r.unexpected_keys)}", flush=True)
    return model.to(dev).eval()


def _f(field):
    return field[0] if isinstance(field, tuple) else field


def rms(f):
    return f.pow(2).mean((1, 2, 3)).add(1e-12).sqrt()


def growth_pen(model, pos, vel, m, mk, field, ext, rho, kappa, amp_p, amp_max, gen):
    """Large-signal contraction loss from a detached state: optionally scale the carried field by lam (log-uniform 3..amp_max, prob amp_p per sample),
    roll `ext` more ticks, penalise per tick e = relu(log(R_out + kappa) - log(rho R_in + kappa))^2 (log form: the first linear-ratio hinge had pen ~1e3 and a diverging fine-tune), R = field rms and max over the grid (mean of both). Returns (penalty, stats)."""
    pos, vel, m = pos.detach(), vel.detach(), m.detach()
    f0, rest = (field[0], field[1:]) if isinstance(field, tuple) else (field, None)
    f0 = f0.detach()
    B = f0.shape[0]
    u = torch.rand(B, generator=gen).to(f0.device)
    lam = torch.where(u < amp_p, torch.exp(torch.log(torch.tensor(3.0)) + torch.rand(B, generator=gen) * (torch.log(torch.tensor(float(amp_max))) - torch.log(torch.tensor(3.0)))).to(f0.device), torch.ones(B, device=f0.device))
    f0 = f0 * lam.view(B, 1, 1, 1)
    field = (f0,) + tuple(x.detach() if x is not None else None for x in rest) if rest is not None else f0
    pen, fin, fmax = 0.0, [], []
    for t in range(ext):
        fi = _f(field)
        pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
        fo = _f(field)
        e = 0.0
        for r_in, r_out in ((rms(fi), rms(fo)), (fi.abs().amax((1, 2, 3)), fo.abs().amax((1, 2, 3)))):
            e = e + 0.5 * F.relu(torch.log(r_out + kappa) - torch.log(rho * r_in + kappa)).pow(2)
        pen = pen + e.mean() / ext
        fmax.append(float(fo.abs().amax()))
    return pen, dict(lam_max=float(lam.max()), fmax_first=fmax[0], fmax_last=fmax[-1])
