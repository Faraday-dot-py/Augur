import torch
import torch.nn as nn
import torch.nn.functional as F

from model.neuron_graph import EXT_SCALE, WEIGHT_SCALE, NeuronGraphDynamics, decode_state

PATCH = 16


class SlowPatchTokenDynamics(NeuronGraphDynamics):
    """NeuronGraphDynamics plus a slow per-cell token updated every PATCH ticks
    from a summary of the finished patch (spike count, mean voltage, mean
    synaptic state) by a residual MLP. The slow token is an extra GRU input and
    drives an auxiliary head for the next patch's per-cell spike count (log
    rate). The recurrent state h packs [GRU token | slow token | running sums
    of count, voltage, synaptic state | tick in patch] so the step interface
    and the free-rollout code are unchanged."""

    def __init__(self, hidden=32, message_dim=32, slow_dim=16, messages=True):
        super().__init__(hidden=hidden, message_dim=message_dim, messages=messages)
        self.slow_dim = slow_dim
        self.gru = nn.GRUCell(message_dim + 5 + slow_dim, hidden)
        self.slow_update = nn.Sequential(nn.Linear(slow_dim + 3, slow_dim), nn.Tanh(), nn.Linear(slow_dim, slow_dim))
        self.count_head = nn.Linear(slow_dim + hidden, 1)
        nn.init.zeros_(self.count_head.weight)
        nn.init.zeros_(self.count_head.bias)
        self.count_pred = None

    def init_hidden(self, x, ext):
        base = super().init_hidden(x, ext)
        return torch.cat([base, base.new_zeros(base.shape[0], self.slow_dim + 4)], dim=-1)

    def step(self, x, ext, h, graph, gate=None):
        b, n, _ = x.shape
        xf = x.reshape(-1, 4)
        g, slow, acc, tick = h[:, :self.hidden], h[:, self.hidden:self.hidden + self.slow_dim], h[:, -4:-1], h[:, -1:]
        node = torch.cat([xf, g], dim=-1)
        if self.messages:
            m = self.msg(torch.cat([node[graph.src], graph.w[:, None] / WEIGHT_SCALE], dim=-1))
            agg = torch.zeros(node.shape[0], m.shape[1], device=x.device, dtype=x.dtype).index_add(0, graph.dst, m)
        else:
            agg = self.msg(torch.cat([node, torch.zeros_like(node[:, :1])], dim=-1))
        g = self.gru(torch.cat([agg, xf, ext.reshape(-1, 1) / EXT_SCALE, slow], dim=-1), g)
        nxt, logit = decode_state(xf, self.head(g), None if gate is None else gate.reshape(-1))
        acc = acc + torch.stack([nxt[:, 3], nxt[:, 0], nxt[:, 2]], dim=-1)
        tick = tick + 1.0
        self.count_pred = None
        if int(tick[0, 0].item()) >= PATCH:
            summary = torch.cat([acc[:, :1], acc[:, 1:] / PATCH], dim=-1)
            slow = slow + self.slow_update(torch.cat([slow, summary], dim=-1))
            self.count_pred = self.count_head(torch.cat([slow, g], dim=-1)).squeeze(-1)
            acc, tick = torch.zeros_like(acc), torch.zeros_like(tick)
        return nxt.reshape(b, n, 4), logit.reshape(b, n), torch.cat([g, slow, acc, tick], dim=-1)


def poisson_count_loss(log_rate, count):
    return F.poisson_nll_loss(log_rate, count, log_input=True)
