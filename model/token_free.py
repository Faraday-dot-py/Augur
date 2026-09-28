import torch
import torch.nn as nn

from model.token_graph import build_radius_graph, build_radius_graph_cells


def wall_features(positions, n, wall_range):
    """(N, 4) proximity to each wall, 0 beyond `wall_range` cells and 1 at
    or past the wall itself. Zero in the interior, so interior translation
    invariance is preserved; the wall itself is the one absolute-position
    fact the dynamics needs once observation correction is gone."""
    x, y = positions[:, 0], positions[:, 1]
    d = torch.stack([x, (n - 1) - x, y, (n - 1) - y], dim=1)
    return (wall_range - d.clamp(0.0, wall_range)) / wall_range


def wall_contact_features(positions, velocities, n, radius, dt):
    """(N, 8): penetration depth into each wall now, then after one
    free-flight step (`pos + vel * dt`), as relu(radius - distance) / radius.
    Contact -- the stiff, near-elastic wall impulse in bounce.py -- starts
    at penetration > 0, and whether a step is an impulse step depends on
    where the ball will be, so the lookahead half tells the network that
    before it happens."""
    def penetration(pos):
        x, y = pos[:, 0], pos[:, 1]
        d = torch.stack([x, (n - 1) - x, y, (n - 1) - y], dim=1)
        return (radius - d).clamp(min=0.0) / radius
    return torch.cat([penetration(positions), penetration(positions + velocities * dt)], dim=1)


def pair_invariants(rel_pos, rel_vel, contact_dist, dt):
    """(E, 5) rotation/reflection-invariant edge features and the two
    directions the pair force may point along. rel_pos/rel_vel are
    dst - src. Returns (features, unit, tangent): d, normal relative speed,
    |tangential relative speed|, penetration now and after one free-flight
    step, both as relu(contact_dist - distance) / contact_dist."""
    d = torch.sqrt((rel_pos ** 2).sum(dim=-1) + 1e-12)
    unit = rel_pos / d.unsqueeze(-1)
    normal_speed = (rel_vel * unit).sum(dim=-1)
    tangent = rel_vel - normal_speed.unsqueeze(-1) * unit
    d_next = torch.sqrt(((rel_pos + rel_vel * dt) ** 2).sum(dim=-1) + 1e-12)
    feats = torch.stack([
        d, normal_speed, torch.sqrt((tangent ** 2).sum(dim=-1) + 1e-12),
        (contact_dist - d).clamp(min=0.0) / contact_dist,
        (contact_dist - d_next).clamp(min=0.0) / contact_dist,
    ], dim=-1)
    return feats, unit, tangent


def _radial_mlp(width=64):
    net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
    nn.init.zeros_(net[4].weight)
    nn.init.zeros_(net[4].bias)
    return net


class TokenFreeDynamics(nn.Module):
    """TokenDynamics (radius-graph attention + GRU + zero-init delta head)
    with wall-proximity node features -- see
    docs/superpowers/specs/2026-09-23-token-free-rollout-design.md."""

    def __init__(self, n, hidden_dim=32, neighbor_radius=4.0, wall_range=3.0, mirror_sym=False,
                 wall_lookahead=False, wall_head=False, radius=0.75, dt=0.15, pair_impulse=False,
                 adaptive_radius="off", conservative_contact=False, contact_substeps=8, contact_residual=False):
        super().__init__()
        # adaptive_radius: "pair" grows the pair_head edge radius to
        # 2*max|v|*dt + 2*radius each step so a fast pair cannot cross contact
        # inside one step with no edge; "all" also grows the attention graph.
        self.adaptive_radius = adaptive_radius
        # conservative_contact: replaces wall_head/pair_head with a learned
        # distance-only pair force and wall force (potential-derived, exact
        # pair momentum), integrated by `contact_substeps` velocity-Verlet
        # sub-steps with a learned uniform acceleration. contact_residual keeps
        # the attention/GRU/delta_head path as an additive residual.
        self.conservative_contact = conservative_contact
        self.contact_substeps = contact_substeps
        self.contact_residual = contact_residual
        self.n = n
        self.radius = radius
        self.dt = dt
        self.wall_lookahead = wall_lookahead
        self.core_dim = hidden_dim
        self.mirror_sym = mirror_sym
        self.hidden_dim = hidden_dim * (2 if mirror_sym else 1)
        self.neighbor_radius = neighbor_radius
        self.cell_graph = False
        self.local_softmax = False
        self.wall_range = wall_range
        wall_dim = 4 + (8 if wall_lookahead else 0)
        node_dim = 2 + wall_dim + self.core_dim
        edge_dim = 2
        self.query = nn.Linear(node_dim, self.core_dim)
        self.key = nn.Linear(node_dim + edge_dim, self.core_dim)
        self.value = nn.Linear(node_dim + edge_dim, self.core_dim)
        self.gru = nn.GRUCell(self.core_dim, self.core_dim)
        self.delta_head = nn.Linear(self.core_dim, 4)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.wall_head = None
        self.pair_head = None
        if conservative_contact:
            if wall_head or pair_impulse:
                raise ValueError("conservative_contact replaces wall_head/pair_head")
            self.force_scale = 100.0
            self.pair_force = _radial_mlp()
            self.wall_force = _radial_mlp()
            self.gravity = nn.Parameter(torch.zeros(2))
        if wall_head and not conservative_contact:
            # Separate two-layer path from (velocity, wall features) to the
            # delta, outside the GRU: the wall impulse is a sharp function
            # of position and velocity that the linear->GRU->linear path
            # only expresses as a smooth brake (see docs/debugging/
            # experiment-log.md, wall-bounce diagnosis).
            self.wall_head = nn.Sequential(nn.Linear(2 + wall_dim, self.core_dim), nn.ReLU(),
                                           nn.Linear(self.core_dim, 4))
            nn.init.zeros_(self.wall_head[2].weight)
            nn.init.zeros_(self.wall_head[2].bias)
        if pair_impulse and not conservative_contact:
            # Explicit pairwise contact impulse, summed (not softmax-
            # normalized) over neighbours: f = alpha * unit + beta * tangent
            # with alpha/beta from an MLP on invariants, applied +f to dst
            # and -f to src by antisymmetry, so pair momentum is conserved by
            # construction. Softmax attention cannot represent an impulse that
            # scales with distance and relative speed (docs/debugging/
            # experiment-log.md, pair-collision diagnosis).
            self.pair_head = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
                                           nn.Linear(32, 4))
            nn.init.zeros_(self.pair_head[4].weight)
            nn.init.zeros_(self.pair_head[4].bias)

    def contact_accel(self, positions):
        n, r = self.n, self.radius
        x, y = positions[:, 0], positions[:, 1]
        d = torch.stack([x, (n - 1) - x, y, (n - 1) - y], dim=1)
        pen = (r - d).clamp(min=0.0) / r
        f = pen * self.wall_force(pen.unsqueeze(-1)).squeeze(-1) * self.force_scale
        acc = torch.stack([f[:, 0] - f[:, 1], f[:, 2] - f[:, 3]], dim=1) + self.gravity * 10.0
        graph = build_radius_graph_cells if self.cell_graph else build_radius_graph
        edges = graph(positions, 2 * r)
        if edges.shape[1] > 0:
            src, dst = edges[0], edges[1]
            rel = positions[dst] - positions[src]
            dist = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
            pen = (2 * r - dist).clamp(min=0.0) / (2 * r)
            fp = pen * self.pair_force(pen) * self.force_scale
            acc = acc.index_add(0, dst, fp * rel / dist)
        return acc

    def _conservative(self, positions, velocities, hidden):
        h = self.dt / self.contact_substeps
        p, v = positions, velocities
        a = self.contact_accel(p)
        for _ in range(self.contact_substeps):
            v_half = v + 0.5 * h * a
            p = p + h * v_half
            a = self.contact_accel(p)
            v = v_half + 0.5 * h * a
        dp, dv = p - (positions + velocities * self.dt), v - velocities
        if self.contact_residual:
            rp, rv, hidden = self._core(positions, velocities, hidden)
            return dp + rp, dv + rv, hidden
        return dp, dv, hidden

    def forward(self, positions, velocities, hidden):
        if self.conservative_contact:
            return self._conservative(positions, velocities, hidden)
        if not self.mirror_sym:
            return self._core(positions, velocities, hidden)
        h, h_m = hidden[:, :self.core_dim], hidden[:, self.core_dim:]
        dp, dv, nh = self._core(positions, velocities, h)
        mirrored_pos = torch.stack([positions[:, 0], (self.n - 1) - positions[:, 1]], dim=1)
        mirrored_vel = velocities * velocities.new_tensor([1.0, -1.0])
        dp_m, dv_m, nh_m = self._core(mirrored_pos, mirrored_vel, h_m)
        flip = dp.new_tensor([1.0, -1.0])
        return (dp + dp_m * flip) / 2, (dv + dv_m * flip) / 2, torch.cat([nh, nh_m], dim=-1)

    def _core(self, positions, velocities, hidden):
        n = positions.shape[0]
        walls = wall_features(positions, self.n, self.wall_range)
        if self.wall_lookahead:
            walls = torch.cat([walls, wall_contact_features(positions, velocities, self.n, self.radius, self.dt)], dim=1)
        node_state = torch.cat([velocities, walls, hidden], dim=-1)
        q = self.query(node_state)

        graph = build_radius_graph_cells if self.cell_graph else build_radius_graph
        edge_index = graph(positions, self.neighbor_radius)
        pair_edges = edge_index
        if self.adaptive_radius != "off" and n > 1:
            wide = max(self.neighbor_radius, 2.0 * float(velocities.norm(dim=1).max()) * self.dt + 2 * self.radius)
            if wide > self.neighbor_radius:
                edge_wide = graph(positions, wide)
                pair_edges = edge_wide
                if self.adaptive_radius == "all":
                    edge_index = edge_wide
        self_loops = torch.arange(n, device=positions.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        attn_out = torch.zeros(n, self.core_dim, device=positions.device, dtype=positions.dtype)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            rel_pos = positions[src] - positions[dst]
            edge_input = torch.cat([node_state[src], rel_pos], dim=-1)
            k = self.key(edge_input)
            v = self.value(edge_input)
            scores = (q[dst] * k).sum(dim=-1) / (self.core_dim ** 0.5)
            if self.local_softmax:
                # Per-destination max instead of the global one: same softmax,
                # but a token's attention no longer depends on how far its
                # scores sit below the scene-wide max (which attenuates it via
                # the denom clamp), so results don't change with scene size.
                top = torch.full((n,), float("-inf"), device=positions.device, dtype=scores.dtype)
                top = top.scatter_reduce(0, dst, scores, "amax", include_self=True)
                weights = torch.exp(scores - top[dst])
            else:
                weights = torch.exp(scores - scores.max())
            denom = torch.zeros(n, device=positions.device, dtype=positions.dtype)
            denom = denom.index_add(0, dst, weights)
            weights = weights / denom[dst].clamp(min=1e-6)
            attn_out = attn_out.index_add(0, dst, weights.unsqueeze(-1) * v)

        new_hidden = self.gru(attn_out, hidden)
        delta = self.delta_head(new_hidden)
        if self.wall_head is not None:
            delta = delta + self.wall_head(torch.cat([velocities, walls], dim=-1))
        if self.pair_head is not None and pair_edges.shape[1] > 0:
            src, dst = pair_edges[0], pair_edges[1]
            feats, unit, tangent = pair_invariants(
                positions[dst] - positions[src], velocities[dst] - velocities[src], 2 * self.radius, self.dt
            )
            coef = self.pair_head(feats)
            f_dp = coef[:, 0:1] * unit + coef[:, 1:2] * tangent
            f_dv = coef[:, 2:3] * unit + coef[:, 3:4] * tangent
            pair = torch.zeros_like(delta).index_add(0, dst, torch.cat([f_dp, f_dv], dim=-1))
            delta = delta + pair
        return delta[:, :2], delta[:, 2:], new_hidden
