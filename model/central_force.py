import torch
import torch.nn as nn

from model.token_graph import build_radius_graph


class CentralForceDynamics(nn.Module):
    """Learned pairwise central force integrated with velocity Verlet.
    The force on dst from src is f(d) along the line between them, with f a
    learned function of distance only, so momentum is conserved exactly and
    a step is symplectic. Same call signature as TokenFreeDynamics: returns
    (dp, dv, hidden) for `pos + vel * dt + dp`, `vel + dv`."""

    def __init__(self, dt=0.1, neighbor_radius=100.0, width=64):
        super().__init__()
        self.dt = dt
        self.neighbor_radius = neighbor_radius
        self.hidden_dim = 1
        self.force = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                   nn.Linear(width, 1))
        nn.init.zeros_(self.force[4].weight)
        nn.init.zeros_(self.force[4].bias)

    def accel(self, positions):
        edges = build_radius_graph(positions, self.neighbor_radius)
        acc = torch.zeros_like(positions)
        if edges.shape[1] == 0:
            return acc
        src, dst = edges[0], edges[1]
        rel = positions[src] - positions[dst]
        d = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
        f = self.force(torch.log(d)) / (d ** 2 + 1.0)
        return acc.index_add(0, dst, f * rel / d)

    def forward(self, positions, velocities, hidden):
        dt = self.dt
        a0 = self.accel(positions)
        dp = 0.5 * dt * dt * a0
        a1 = self.accel(positions + velocities * dt + dp)
        return dp, 0.5 * dt * (a0 + a1), hidden
