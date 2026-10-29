import torch
import torch.nn as nn

V_SHIFT, V_SCALE = 65.0, 20.0
U_SHIFT, U_SCALE = 13.0, 10.0
EXT_SCALE = 5.0
WEIGHT_SCALE = 3.0
SPIKE_POS_WEIGHT = 5.0


def normalize_state(x):
    return torch.stack([(x[..., 0] + V_SHIFT) / V_SCALE, (x[..., 1] + U_SHIFT) / U_SCALE, x[..., 2], x[..., 3]], dim=-1)


def denormalize_voltage(vn):
    return vn * V_SCALE - V_SHIFT


class Graph:
    """Directed synapses of B disjoint circuits of N cells, flattened to
    B*N nodes. src -> dst carries weight w; `dense` (B, N, N) is kept only for
    the linear baseline."""

    def __init__(self, weights):
        b, n, _ = weights.shape
        bi, di, si = (weights != 0).nonzero(as_tuple=True)
        self.src = bi * n + si
        self.dst = bi * n + di
        self.w = weights[bi, di, si]
        self.batch, self.n = b, n
        self.dense = weights


def decode_state(x, out, gate=None):
    """x is (M, 4) normalized; out is (M, 7): voltage/adaptation/synapse deltas
    for the no-spike and spike branches plus the next-spike logit. The branch
    is picked by the next-step spike, hard with a straight-through gradient
    unless `gate` is given (teacher forcing)."""
    dv0, dv1, du0, du1, ds0, ds1, logit = out.unbind(-1)
    if gate is None:
        p = torch.sigmoid(logit)
        hard = (logit > 0).to(p.dtype)
        gate = hard + p - p.detach()
        spike = hard
    else:
        spike = gate
    v = x[:, 0] + (1 - gate) * dv0 + gate * dv1
    u = x[:, 1] + (1 - gate) * du0 + gate * du1
    s = x[:, 2] + (1 - gate) * ds0 + gate * ds1
    return torch.stack([v, u, s, spike], dim=-1), logit


class NeuronGraphDynamics(nn.Module):
    """Per-cell GRU token; the update input is the sum over incoming synapses
    of a message from the presynaptic (state, token) and the signed synapse
    weight. With messages=False the same message network is applied to the
    cell's own (state, token) with weight 0 instead of the incoming synapses,
    so parameters and compute are unchanged."""

    def __init__(self, hidden=32, message_dim=32, messages=True):
        super().__init__()
        self.hidden, self.messages = hidden, messages
        self.encoder = nn.Sequential(nn.Linear(5, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.msg = nn.Sequential(nn.Linear(4 + hidden + 1, message_dim), nn.Tanh(), nn.Linear(message_dim, message_dim))
        self.gru = nn.GRUCell(message_dim + 5, hidden)
        self.head = nn.Linear(hidden, 7)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def init_hidden(self, x, ext):
        return self.encoder(torch.cat([x.reshape(-1, 4), ext.reshape(-1, 1) / EXT_SCALE], dim=-1))

    def step(self, x, ext, h, graph, gate=None):
        b, n, _ = x.shape
        xf = x.reshape(-1, 4)
        node = torch.cat([xf, h], dim=-1)
        if self.messages:
            m = self.msg(torch.cat([node[graph.src], graph.w[:, None] / WEIGHT_SCALE], dim=-1))
            agg = torch.zeros(node.shape[0], m.shape[1], device=x.device, dtype=x.dtype).index_add(0, graph.dst, m)
        else:
            agg = self.msg(torch.cat([node, torch.zeros_like(node[:, :1])], dim=-1))
        h = self.gru(torch.cat([agg, xf, ext.reshape(-1, 1) / EXT_SCALE], dim=-1), h)
        nxt, logit = decode_state(xf, self.head(h), None if gate is None else gate.reshape(-1))
        return nxt.reshape(b, n, 4), logit.reshape(b, n), h


class PerCellMlpDynamics(nn.Module):
    """No token, no messages: next state of each cell from its own current
    state and external input."""

    def __init__(self, width=92):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 7))
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def init_hidden(self, x, ext):
        return x.new_zeros(x.shape[0] * x.shape[1], 1)

    def step(self, x, ext, h, graph, gate=None):
        b, n, _ = x.shape
        xf = x.reshape(-1, 4)
        out = self.net(torch.cat([xf, ext.reshape(-1, 1) / EXT_SCALE], dim=-1))
        nxt, logit = decode_state(xf, out, None if gate is None else gate.reshape(-1))
        return nxt.reshape(b, n, 4), logit.reshape(b, n), h


class PersistenceDynamics:
    def init_hidden(self, x, ext):
        return None

    def step(self, x, ext, h, graph, gate=None):
        return torch.cat([x[..., :3], torch.zeros_like(x[..., 3:])], dim=-1), None, h


def linear_features(x, ext, graph):
    syn_s = torch.bmm(graph.dense, x[..., 2:3]) / WEIGHT_SCALE
    syn_spk = torch.bmm(graph.dense, x[..., 3:4]) / WEIGHT_SCALE
    return torch.cat([x, ext.unsqueeze(-1) / EXT_SCALE, syn_s, syn_spk, torch.ones_like(syn_s)], dim=-1)


class LinearDynamics:
    """Least-squares x(t+1) = A [x, ext, W s, W spike, 1] shared across cells
    and circuits. The synaptic input enters linearly, so it is a linear
    dynamics fit with the connectivity given."""

    def __init__(self):
        self.coef = None

    def fit(self, feats, targets, ridge=1e-3):
        f = feats.reshape(-1, feats.shape[-1]).double()
        t = targets.reshape(-1, targets.shape[-1]).double()
        gram = f.T @ f + ridge * torch.eye(f.shape[1], dtype=f.dtype, device=f.device)
        self.coef = torch.linalg.solve(gram, f.T @ t).float()

    def init_hidden(self, x, ext):
        return None

    def step(self, x, ext, h, graph, gate=None):
        nxt = linear_features(x, ext, graph) @ self.coef
        spike = (nxt[..., 3] > 0.5).to(nxt.dtype)
        return torch.cat([nxt[..., :3], spike.unsqueeze(-1)], dim=-1), nxt[..., 3], h


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
