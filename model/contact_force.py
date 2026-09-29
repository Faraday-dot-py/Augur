import torch
import torch.nn as nn

from model.token_graph import build_radius_graph_cells


class ContactForceDynamics(nn.Module):
    """Learned pairwise potential V(d, r_i+r_j, s_ij), s_ij a swap-symmetric
    function of (mass_i, mass_j). Force is -dV/dd along the pair line (via
    autograd), so momentum is conserved exactly and a step is symplectic
    (same structure as CentralForceDynamics, generalized per
    docs/superpowers/specs/2026-09-28-cfd-contact-generalization-design.md
    §3). radius=0 for every body reduces this to CentralForceDynamics'
    f(d) exactly (spec §5) since r_sum/m_sum/m_prod/m_diff are then extra,
    learnable-away input channels rather than a structural change.
    kinematic bodies (obstacles) still exert force but never receive a
    position/velocity update (spec §2). Same call convention as
    CentralForceDynamics/TokenFreeDynamics otherwise: returns (dp, dv,
    hidden) for `pos + vel * dt + dp`, `vel + dv`."""

    def __init__(self, dt=0.1, neighbor_radius=100.0, width=64):
        super().__init__()
        self.dt = dt
        self.neighbor_radius = neighbor_radius
        self.hidden_dim = 1
        self.potential = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                        nn.Linear(width, 1))
        nn.init.zeros_(self.potential[-1].weight)
        nn.init.zeros_(self.potential[-1].bias)

    def pair_features(self, d, r_sum, mass_src, mass_dst):
        m_sum = mass_src + mass_dst
        m_prod = mass_src * mass_dst
        m_diff = (mass_src - mass_dst).abs()
        return torch.cat([torch.log(d), r_sum, m_sum, m_prod, m_diff], dim=-1)

    def accel(self, positions, radius, mass):
        edges = build_radius_graph_cells(positions, self.neighbor_radius)
        acc = torch.zeros_like(positions)
        if edges.shape[1] == 0:
            return acc
        src, dst = edges[0], edges[1]
        rel = positions[src] - positions[dst]
        with torch.enable_grad():
            d = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
            d.requires_grad_(True)
            r_sum = (radius[src] + radius[dst]).unsqueeze(-1)
            feat = self.pair_features(d, r_sum, mass[src].unsqueeze(-1), mass[dst].unsqueeze(-1))
            v = self.potential(feat)
            (grad_d,) = torch.autograd.grad(v.sum(), d, create_graph=True)
        force = (-grad_d) * rel / d
        net_force = acc.index_add(0, dst, force)
        return net_force / mass.unsqueeze(-1)

    def forward(self, positions, velocities, hidden, radius, mass, kinematic):
        dt = self.dt
        a0 = self.accel(positions, radius, mass)
        dp = 0.5 * dt * dt * a0
        a1 = self.accel(positions + velocities * dt + dp, radius, mass)
        dv = 0.5 * dt * (a0 + a1)
        kin = kinematic.unsqueeze(-1)
        dp = torch.where(kin, torch.zeros_like(dp), dp)
        dv = torch.where(kin, torch.zeros_like(dv), dv)
        return dp, dv, hidden
