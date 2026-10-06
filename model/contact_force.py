import torch
import torch.nn as nn

from model.token_graph import build_radius_graph_cells


class ContactForceDynamics(nn.Module):
    """Learned pairwise potential V(pen, r_i+r_j, s_ij), pen = r_i+r_j - d
    (negative when apart), s_ij a swap-symmetric function of (mass_i,
    mass_j). V is gated by sigmoid(pen / 0.5), so it goes smoothly to zero
    once bodies are well apart, like augur.py's penalty force (exactly zero
    without overlap). Force is -dV/dd along the pair line (via autograd), so
    momentum is conserved exactly and a step is symplectic -- the same
    design pattern as CentralForceDynamics (learned pairwise function,
    same integrator), per
    docs/superpowers/specs/2026-09-28-cfd-contact-generalization-design.md
    §3, but not a literal superset of it: CentralForceDynamics has no decay
    gate, no mass division and a different graph builder.
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

    def pair_features(self, pen, r_sum, mass_src, mass_dst):
        m_sum = mass_src + mass_dst
        m_prod = mass_src * mass_dst
        m_diff = (mass_src - mass_dst).abs()
        return torch.cat([pen, r_sum, m_sum, m_prod, m_diff], dim=-1)

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
            pen = r_sum - d
            feat = self.pair_features(pen, r_sum, mass[src].unsqueeze(-1), mass[dst].unsqueeze(-1))
            gate = torch.sigmoid(pen / 0.5)
            v = self.potential(feat) * gate
            (grad_d,) = torch.autograd.grad(v.sum(), d, create_graph=True)
        force = grad_d * rel / d
        net_force = acc.index_add(0, dst, force)
        return net_force / mass.unsqueeze(-1)

    def forward(self, positions, velocities, hidden, radius, mass, kinematic):
        dt = self.dt
        kin = kinematic.unsqueeze(-1)
        a0 = self.accel(positions, radius, mass)
        dp = torch.where(kin, torch.zeros_like(a0), 0.5 * dt * dt * a0)
        a1 = self.accel(positions + velocities * dt + dp, radius, mass)
        dv = 0.5 * dt * (a0 + a1)
        dv = torch.where(kin, torch.zeros_like(dv), dv)
        return dp, dv, hidden


class ContactForceDynamicsSymlog(ContactForceDynamics):
    """Same as ContactForceDynamics, but the potential head's final Linear(64,1)
    is replaced with a sign+log-magnitude decomposition:
    V_raw = tanh(sign_head(feat)) * exp(log_scale_head(feat)), still * gate.
    sign_head and log_scale_head are both plain Linear(width,1) on the same
    shared trunk (the two hidden Tanh layers), everywhere-differentiable (no
    hard threshold), so -dV/dd in accel() stays exact. Hypothesis: giving the
    head unconstrained log-scale range (instead of a raw unbounded linear
    scalar) fits the contact potential's dynamic range better, in particular
    at sparse boundary-point spacing (see the v2 wall-density rebound-failure
    finding in docs/debugging/experiment-log.md, 2026-09-29)."""

    def __init__(self, dt=0.1, neighbor_radius=100.0, width=64):
        nn.Module.__init__(self)
        self.dt = dt
        self.neighbor_radius = neighbor_radius
        self.hidden_dim = 1
        self.trunk = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.sign_head = nn.Linear(width, 1)
        self.log_scale_head = nn.Linear(width, 1)
        nn.init.zeros_(self.sign_head.weight)
        nn.init.zeros_(self.sign_head.bias)
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.zeros_(self.log_scale_head.bias)

    def potential(self, feat):
        h = self.trunk(feat)
        sign = torch.tanh(self.sign_head(h))
        scale = torch.exp(self.log_scale_head(h))
        return sign * scale


class ContactForceDynamicsSymlogCapped(ContactForceDynamicsSymlog):
    """Same as ContactForceDynamicsSymlog, but log_scale_head's raw output is
    soft-capped to +/-log_scale_cap via log_scale_cap * tanh(raw / log_scale_cap)
    before exp(), instead of being left unconstrained. Hypothesis (see
    docs/debugging/contact-force-architecture-ideas-untested.md, 'Likely next
    step'): the symlog head's wall-contact fix works by giving it more dynamic
    range, but that same unconstrained range is what let it overfit the
    under-sampled unseen_obstacle_shape/long-horizon cases (err@20/30 0.046/0.044
    -> 0.33/0.33 regression, job 3139). Capping log_scale should still leave
    enough range to fix wall-pinning (which needed ~11-24x, i.e. roughly
    exp(2.5)) while bounding how far the head can extrapolate outside training
    distribution. Everywhere-differentiable (smooth tanh cap, no hard clamp), so
    -dV/dd in accel() stays exact."""

    def __init__(self, dt=0.1, neighbor_radius=100.0, width=64, log_scale_cap=4.0):
        super().__init__(dt=dt, neighbor_radius=neighbor_radius, width=width)
        self.log_scale_cap = log_scale_cap

    def potential(self, feat):
        h = self.trunk(feat)
        sign = torch.tanh(self.sign_head(h))
        raw_log_scale = self.log_scale_head(h)
        log_scale = self.log_scale_cap * torch.tanh(raw_log_scale / self.log_scale_cap)
        scale = torch.exp(log_scale)
        return sign * scale
