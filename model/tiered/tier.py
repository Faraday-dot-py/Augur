import torch
import torch.nn as nn
import torch.nn.functional as F


class TierNet(nn.Module):
    """One upper tier: padded-tile input channels -> per-cell polynomial potential (phi0, g, symmetric H),
    optional soft-penalised residual force, error estimate, latent summary and hidden field. Weights are shared
    by every tile of the tier; cell size h only enters through the output scaling."""

    def __init__(self, cin, h, width=32, layers=6, residual=True, latent_ch=0, err=True, hidden_ch=0, out_scale=1.0, log_rho=False, learned_pool=False):
        super().__init__()
        self.h, self.residual, self.latent_ch, self.err, self.hidden_ch, self.out_scale, self.log_rho = h, residual, latent_ch, err, hidden_ch, out_scale, log_rho
        self.ncoef = 8 if residual else 6
        self.pool = nn.Conv2d(latent_ch + int(err), latent_ch + int(err), 2, 2) if learned_pool and latent_ch + int(err) else None
        self.inp = nn.Conv2d(cin, width, 3, padding=1)
        self.mid = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(layers - 1))
        self.out = nn.Conv2d(width, self.ncoef + latent_ch + int(err) + hidden_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def summary_ch(self):
        return self.latent_ch + int(self.err)

    def forward(self, x):
        if self.log_rho:
            x = torch.cat([torch.log1p(x[:, :1]), x[:, 1:]], 1)
        h = F.gelu(self.inp(x))
        for c in self.mid:
            h = h + F.gelu(c(h))
        y = self.out(h)
        s, hh = self.out_scale, self.h
        scale = torch.tensor([s, s / hh, s / hh, s / hh ** 2, s / hh ** 2, s / hh ** 2, s / hh, s / hh][:self.ncoef], device=y.device, dtype=y.dtype)
        coefs = y[:, :self.ncoef] * scale[None, :, None, None]
        o = self.ncoef
        latent = y[:, o:o + self.latent_ch]
        err = F.softplus(y[:, o + self.latent_ch:o + self.summary_ch]) if self.err else y[:, :0]
        hidden = y[:, o + self.summary_ch:]
        return coefs, torch.cat([latent, err], 1), hidden
