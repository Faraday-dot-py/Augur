import torch
import torch.nn as nn

from model.neuron_graph import EXT_SCALE, U_SCALE, U_SHIFT, V_SCALE, V_SHIFT, WEIGHT_SCALE

STATE_DIM = 5
PLASTIC_STEP = 0.1


def normalize_plastic_state(x):
    return torch.stack([(x[..., 0] + V_SHIFT) / V_SCALE, (x[..., 1] + U_SHIFT) / U_SCALE, x[..., 2], x[..., 3], x[..., 4]], dim=-1)


def denormalize_plastic_state(x):
    return torch.stack([x[..., 0] * V_SCALE - V_SHIFT, x[..., 1] * U_SCALE - U_SHIFT, x[..., 2], x[..., 3], x[..., 4]], dim=-1)


class PlasticNeuronGraphDynamics(nn.Module):
    """NeuronGraphDynamics with a persistent per-synapse state m_e (effective
    weight = w_e * (1 + m_e)). With plastic=True each tick m_e moves by a
    learned function of purely local quantities (pre and post trace, pre and
    post next-tick spike, w_e, m_e); with plastic=False m is never updated, so
    the circuit has no mutable synaptic state (the update network is kept so
    the parameter count is identical). State per cell: v, u, s, spike, trace."""

    def __init__(self, hidden=32, message_dim=32, plastic=True):
        super().__init__()
        self.hidden, self.plastic = hidden, plastic
        self.encoder = nn.Sequential(nn.Linear(STATE_DIM + 1, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.msg = nn.Sequential(nn.Linear(STATE_DIM + hidden + 3, message_dim), nn.Tanh(), nn.Linear(message_dim, message_dim))
        self.gru = nn.GRUCell(message_dim + STATE_DIM + 1, hidden)
        self.head = nn.Linear(hidden, 9)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.rule = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.rule[4].weight)
        nn.init.zeros_(self.rule[4].bias)

    def init_hidden(self, x, ext):
        return self.encoder(torch.cat([x.reshape(-1, STATE_DIM), ext.reshape(-1, 1) / EXT_SCALE], dim=-1))

    def step(self, x, ext, h, m, graph, gate=None):
        b, n, _ = x.shape
        xf = x.reshape(-1, STATE_DIM)
        node = torch.cat([xf, h], dim=-1)
        w = graph.w[:, None] / WEIGHT_SCALE
        mm = m[:, None]
        msg = self.msg(torch.cat([node[graph.src], w, mm, w * (1 + mm)], dim=-1))
        agg = torch.zeros(node.shape[0], msg.shape[1], device=x.device, dtype=x.dtype).index_add(0, graph.dst, msg)
        h = self.gru(torch.cat([agg, xf, ext.reshape(-1, 1) / EXT_SCALE], dim=-1), h)
        out = self.head(h)
        dv0, dv1, du0, du1, ds0, ds1, dx0, dx1, logit = out.unbind(-1)
        if gate is None:
            p = torch.sigmoid(logit)
            hard = (logit > 0).to(p.dtype)
            gt = hard + p - p.detach()
            spike = hard
        else:
            gt = spike = gate.reshape(-1)
        nxt = torch.stack([
            xf[:, 0] + (1 - gt) * dv0 + gt * dv1, xf[:, 1] + (1 - gt) * du0 + gt * du1,
            xf[:, 2] + (1 - gt) * ds0 + gt * ds1, spike, xf[:, 4] + (1 - gt) * dx0 + gt * dx1], dim=-1)
        if self.plastic:
            feats = torch.stack([xf[graph.src, 4], xf[graph.dst, 4], spike[graph.src], spike[graph.dst], m, graph.w / WEIGHT_SCALE], dim=-1)
            m = m + PLASTIC_STEP * self.rule(feats).squeeze(-1)
        return nxt.reshape(b, n, STATE_DIM), logit.reshape(b, n), h, m
