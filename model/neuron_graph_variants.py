import torch
import torch.nn as nn

from model.neuron_graph import EXT_SCALE, WEIGHT_SCALE, NeuronGraphDynamics


class NeuronGraphVariantDynamics(NeuronGraphDynamics):
    """NeuronGraphDynamics with three independent opt-in changes.
    absolute_reset: the spike branch outputs the next voltage itself (the head
    slot that was a voltage delta), not a delta from the previous voltage.
    normalize_messages: the sum over incoming synapses is divided by the cell's
    in-degree (clamped at 1). calibrated_emission: a spike is emitted when the
    logit exceeds a buffer `emit_threshold` that `calibrate` keeps at the
    quantile matching the true spike fraction (EMA), instead of logit > 0.
    All three leave the parameter count unchanged."""

    def __init__(self, hidden=32, message_dim=32, messages=True, absolute_reset=False, normalize_messages=False,
                 calibrated_emission=False):
        super().__init__(hidden=hidden, message_dim=message_dim, messages=messages)
        self.absolute_reset, self.normalize_messages, self.calibrated_emission = absolute_reset, normalize_messages, calibrated_emission
        self.register_buffer("emit_threshold", torch.zeros(()))

    def calibrate(self, logit, target_spike, momentum=0.98):
        rate = target_spike.float().mean().clamp(min=1e-3, max=0.5)
        new = torch.quantile(logit.detach().flatten().float(), float(1.0 - rate))
        self.emit_threshold.mul_(momentum).add_((1.0 - momentum) * new)

    def step(self, x, ext, h, graph, gate=None):
        b, n, _ = x.shape
        xf = x.reshape(-1, 4)
        node = torch.cat([xf, h], dim=-1)
        if self.messages:
            m = self.msg(torch.cat([node[graph.src], graph.w[:, None] / WEIGHT_SCALE], dim=-1))
            agg = torch.zeros(node.shape[0], m.shape[1], device=x.device, dtype=x.dtype).index_add(0, graph.dst, m)
            if self.normalize_messages:
                deg = torch.bincount(graph.dst, minlength=node.shape[0]).clamp(min=1).to(x.dtype)
                agg = agg / deg[:, None]
        else:
            agg = self.msg(torch.cat([node, torch.zeros_like(node[:, :1])], dim=-1))
        h = self.gru(torch.cat([agg, xf, ext.reshape(-1, 1) / EXT_SCALE], dim=-1), h)
        dv0, dv1, du0, du1, ds0, ds1, logit = self.head(h).unbind(-1)
        if gate is None:
            shifted = logit - self.emit_threshold if self.calibrated_emission else logit
            p = torch.sigmoid(shifted)
            hard = (shifted > 0).to(p.dtype)
            gt = hard + p - p.detach()
            spike = hard
        else:
            gt = spike = gate.reshape(-1)
        v_spike = dv1 if self.absolute_reset else xf[:, 0] + dv1
        v = (1 - gt) * (xf[:, 0] + dv0) + gt * v_spike
        u = xf[:, 1] + (1 - gt) * du0 + gt * du1
        s = xf[:, 2] + (1 - gt) * ds0 + gt * ds1
        return torch.stack([v, u, s, spike], dim=-1).reshape(b, n, 4), logit.reshape(b, n), h
