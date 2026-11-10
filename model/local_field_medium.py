import math

import torch
import torch.nn as nn

from scripts import scatter_field as sf

H_CH = 4


class LocalFieldMedium(nn.Module):
    """Curiosity model: grid state (phi, psi, hidden) advanced by a learned local rule (two 3x3 convs, 1x1 head), n_sub substeps per tick, so signal speed is at most 2*h*n_sub/dt. Teacher-forced."""

    def __init__(self, grid=128, extent=64.0, width=32, dt=0.1):
        super().__init__()
        self.helper = sf.ScatterField(grid=grid, extent=extent, dt=dt, width=4, layers=1)
        for p in self.helper.parameters():
            p.requires_grad_(False)
        self.grid, self.extent, self.h, self.dt = grid, extent, extent / grid, dt
        cin = 2 + H_CH + 2
        self.c1 = nn.Conv2d(cin, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.head = nn.Conv2d(width, 2 + H_CH, 1)
        self.act = nn.GELU()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def max_speed(self, nsub):
        return 2 * self.h * nsub / self.dt

    def init_state(self, B, device):
        return torch.zeros(B, 2 + H_CH, self.grid, self.grid, device=device)

    def density(self, pos, mass, mask):
        rho = self.helper.scatter(pos, torch.zeros_like(pos), mass, mask)[:, :1]
        return sf.softclip(rho)

    def substeps(self, S, rho, nsub):
        lg = torch.full_like(rho, math.log(nsub))
        for _ in range(nsub):
            S = S + (self.dt / nsub) * self.head(self.act(self.c2(self.act(self.c1(torch.cat([S, rho, lg], 1))))))
        return S

    def tick(self, S, pos, mass, mask, nsub):
        S = self.substeps(S, self.density(pos, mass, mask), nsub)
        acc = self.helper.gather(self.helper.neg_grad(S[:, :1]), pos) * mask[..., None]
        return S, acc
