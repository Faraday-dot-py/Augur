"""Usage: PYTHONPATH=. python scripts/fallback_run.py --name X --ic flyby [--kernel analytic] [--mode adaptive] [--est PATH] [--corrupt none|bias:-2|noise|const:-10] ..."""
import argparse
import json

from scripts import fallback_lib as fl

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--ic", default="flyby")
ap.add_argument("--kernel", default="analytic")
ap.add_argument("--mode", default="adaptive")
ap.add_argument("--est", default="checkpoints/est_analytic.pt")
ap.add_argument("--corrupt", default="none")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--target", type=float, default=0.01)
ap.add_argument("--audit-every", type=int, default=10)
ap.add_argument("--audit-k", type=int, default=1000)
ap.add_argument("--lam-max", type=float, default=8.0)
ap.add_argument("--reprobe-every", type=int, default=100)
ap.add_argument("--severe", type=float, default=3.0)
ap.add_argument("--fail-limit", type=int, default=4)
ap.add_argument("--diag-every", type=int, default=100)
ap.add_argument("--true-every", type=int, default=5)
ap.add_argument("--noise-ks", type=int, nargs="*", default=[])
a = ap.parse_args()
cfg = {"name": a.name, "ic": a.ic, "kernel": a.kernel, "mode": a.mode, "est": a.est, "corrupt": a.corrupt, "n": a.n, "steps": a.steps, "target": a.target,
       "audit_every": a.audit_every, "audit_k": a.audit_k, "lam_max": a.lam_max, "reprobe_every": a.reprobe_every, "severe": a.severe,
       "fail_limit": a.fail_limit, "true_every": a.true_every, "diag_every": a.diag_every, "noise_ks": a.noise_ks}
out = fl.run(cfg)
json.dump(out, open(f"results/fallback_{a.name}.json", "w"))
print("done", a.name, out["wall_s"], "events", len(out["events"]))
