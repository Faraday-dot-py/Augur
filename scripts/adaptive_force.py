"""Force provider for N-body rollouts: momentum-symmetric dual tree with learned node-pair error estimator, end-to-end audit, and a
geometric fallback (scripts/dual_tree.py, dual_estimator.py, kernels.py).

modes: geo (fixed theta), geo_audit (theta steered by the audit), est (estimator, fixed lam, no audit), adaptive (estimator + audit
controller + fallback). Audit = sample rel_l2 of `audit_k` random particles against the exact sum, every `audit_every` calls.
adaptive: lam *= 1.5 if audit error > target else /= 1.1 (>= lam_min); fall back to the geometric rule when the audit error exceeds
`severe` x target, or lam exceeds lam_max, or `fail_limit` consecutive audits fail; in fallback theta *= 0.85 when over target, *= 1.05
when under; every `reprobe_every` calls try the estimator again (lam reset), one failed audit sends it straight back.
"""
import time

import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import kernels


def sync():
    torch.cuda.synchronize()
    return time.time()


class AdaptiveForce:
    def __init__(self, kernel, head=None, mode="adaptive", cap=8, theta_max=1.2, tol=1e-3, target=0.01, audit_every=10, audit_k=1000,
                 lmax=18, lam_max=8.0, lam_min=0.3, fail_limit=4, severe=3.0, geo_theta=0.35, geo_min=0.1, geo_max=0.7, reprobe_every=100,
                 seed=4738, device="cuda"):
        self.kernel, self.head, self.mode, self.cap, self.theta_max, self.tol, self.target = kernel, head, mode, cap, theta_max, tol, target
        self.audit_every, self.audit_k, self.lmax = audit_every, audit_k, lmax
        self.lam_max, self.lam_min, self.fail_limit, self.severe = lam_max, lam_min, fail_limit, severe
        self.geo0, self.geo_min, self.geo_max, self.reprobe_every = geo_theta, geo_min, geo_max, reprobe_every
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.reset()

    def reset(self):
        self.now = "est" if self.mode in ("est", "adaptive") else "geo"
        self.lam, self.theta, self.fails, self.probing, self.since_fb, self.calls, self.prev_a = 1.0, self.geo0, 0, False, 0, 0, None
        self.stats, self.events = [], []

    def _scale(self, tr, fl):
        if self.prev_a is not None:
            return de.node_scale(fl, self.prev_a[tr.order])
        a_c, _, _ = dt.dual_accel(tr, fl, self.theta_max, self.cap)
        return de.node_scale(fl, a_c)

    def __call__(self, pos):
        kernels.current = self.kernel
        t0 = sync()
        tr = ao.Tree(pos, self.lmax)
        fl = dt.Flat(tr)
        if self.now == "est":
            fs = self._scale(tr, fl)
            a_s, m2l, dirn = dt.dual_accel(tr, fl, self.theta_max, self.cap, accept_fn=de.make_accept(fl, self.head, self.lam, self.tol, fs))
        else:
            a_s, m2l, dirn = dt.dual_accel(tr, fl, self.theta, self.cap)
        a = torch.empty_like(a_s)
        a[tr.order] = a_s
        self.prev_a = a
        t1 = sync()
        rec = {"call": self.calls, "mode": self.now, "lam": self.lam, "theta": self.theta, "cost": (m2l + dirn) / pos.shape[0], "time_s": t1 - t0, "audit": None}
        if self.mode in ("geo_audit", "adaptive") and self.calls % self.audit_every == 0:
            err = de.audit_e2e(pos, a, self.audit_k, self.gen)
            rec["audit"], rec["audit_time_s"] = err, sync() - t1
            self._control(err)
        self.calls += 1
        self.stats.append(rec)
        return a

    def _fallback(self, why):
        self.events.append({"call": self.calls, "event": "fallback", "why": why, "lam": self.lam})
        self.now, self.theta, self.fails, self.probing, self.since_fb, self.lam = "geo", self.geo0, 0, False, 0, 1.0

    def _control(self, err):
        over = err > self.target
        if self.now == "est":
            if err > self.severe * self.target:
                return self._fallback(f"severe audit error {err:.4f}")
            if self.probing:
                if over:
                    return self._fallback("probe failed")
                self.probing = False
                self.events.append({"call": self.calls, "event": "probe passed"})
            self.fails = self.fails + 1 if over else 0
            self.lam = self.lam * 1.5 if over else max(self.lam / 1.1, self.lam_min)
            if self.lam > self.lam_max or self.fails >= self.fail_limit:
                self._fallback(f"lam {self.lam:.2f} fails {self.fails}")
        else:
            self.theta = max(self.theta * 0.85, self.geo_min) if over else min(self.theta * 1.05, self.geo_max)
            self.since_fb += 1
            if self.mode == "adaptive" and self.since_fb * self.audit_every >= self.reprobe_every:
                self.now, self.probing, self.lam = "est", True, 1.0
                self.events.append({"call": self.calls, "event": "probe estimator"})
