import math

import torch
import torch.nn as nn

from model.neuron_graph import EXT_SCALE, WEIGHT_SCALE, NeuronGraphDynamics, decode_state

WINDOW = 32


class CausalWindowAttention(nn.Module):
    """Single-head attention of the current tick over the last `window` cached
    ticks of the same cell (including the current one), with a learned bias per
    lag and a validity mask for ticks before the start of the rollout."""

    def __init__(self, dim, window):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.lag_bias = nn.Parameter(torch.zeros(window))
        self.dim = dim
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, query, buf, valid):
        scores = torch.einsum("md,mwd->mw", self.q(query), self.k(buf)) / math.sqrt(self.dim) + self.lag_bias.flip(0)
        weights = torch.softmax(scores.masked_fill(~valid, -1e9), dim=-1)
        return self.o(torch.einsum("mw,mwd->md", weights, self.v(buf)))


class WindowedAttentionDynamics(NeuronGraphDynamics):
    """Replaces the GRU token with two layers, each a causal temporal attention
    over the cell's own last WINDOW ticks followed by a graph-masked message sum
    over incoming synapses (O(cells * WINDOW + edges)). The recurrent state h
    packs the per-layer caches of the last WINDOW layer inputs and the number of
    ticks seen: [cache 0 | cache 1 | count]."""

    def __init__(self, dim=32, message_dim=32, window=WINDOW, messages=True):
        super().__init__(hidden=dim, message_dim=message_dim, messages=messages)
        self.dim, self.window = dim, window
        self.embed = nn.Sequential(nn.Linear(5, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.attn = nn.ModuleList([CausalWindowAttention(dim, window) for _ in range(2)])
        self.msgs = nn.ModuleList([nn.Sequential(nn.Linear(dim + 1, message_dim), nn.Tanh(), nn.Linear(message_dim, dim)) for _ in range(2)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim)) for _ in range(2)])
        del self.gru, self.msg, self.encoder

    def init_hidden(self, x, ext):
        m = x.shape[0] * x.shape[1]
        return x.new_zeros(m, 2 * self.window * self.dim + 1)

    def step(self, x, ext, h, graph, gate=None):
        b, n, _ = x.shape
        xf = x.reshape(-1, 4)
        w, d = self.window, self.dim
        caches = [h[:, :w * d].reshape(-1, w, d), h[:, w * d:2 * w * d].reshape(-1, w, d)]
        count = h[:, -1:]
        z = self.embed(torch.cat([xf, ext.reshape(-1, 1) / EXT_SCALE], dim=-1))
        valid = torch.arange(w, device=x.device)[None, :] >= (w - 1 - count.clamp(max=w - 1))
        new_caches = []
        for layer in range(2):
            buf = torch.cat([caches[layer][:, 1:], z[:, None]], dim=1)
            new_caches.append(buf)
            z = z + self.attn[layer](z, buf, valid)
            if self.messages:
                m = self.msgs[layer](torch.cat([z[graph.src], graph.w[:, None] / WEIGHT_SCALE], dim=-1))
                z = z + torch.zeros_like(z).index_add(0, graph.dst, m)
            else:
                z = z + self.msgs[layer](torch.cat([z, torch.zeros_like(z[:, :1])], dim=-1))
            z = z + self.ffn[layer](z)
        nxt, logit = decode_state(xf, self.head(z), None if gate is None else gate.reshape(-1))
        h = torch.cat([new_caches[0].reshape(-1, w * d), new_caches[1].reshape(-1, w * d), count + 1.0], dim=-1)
        return nxt.reshape(b, n, 4), logit.reshape(b, n), h
