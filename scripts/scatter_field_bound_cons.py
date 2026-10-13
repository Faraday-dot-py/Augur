import numpy as np
import torch


def conservation_rows(model, dev, args):
    """Standing long-horizon rows (1000 ticks): median over scenes of |dE|/(K0+|U0|), |dL|/Lscale, |dP|, bound fraction, virial, outside-grid fraction,
    max |v|/c and fraction with |v|>=c (bh only), model and exact-sim truth (4 substeps), regimes from scatter_conservation."""
    from scripts import scatter_conservation as sc

    out = {}
    for name in ("nr100", "bh300"):
        kind, n, per = sc.REGIMES[name]
        pos0, vel0, seeds = sc.make_scenes(kind, n, per)
        P0 = torch.tensor(pos0, device=dev)
        V0 = torch.tensor(vel0, device=dev)
        rel = kind == "bh"
        r0 = (P0 - P0.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt()
        recs = {}
        for tag in ("model", "true"):
            rec = sc.Recorder(rel, args.eps, model.extent, r0, 1000, len(seeds), dev, args.dt)
            if tag == "model":
                sc.roll_model(model, P0, V0, rel, 1000, rec, None, args.dt)
            else:
                sc.roll_truth(P0, V0, rel, 1000, 4, args.eps, rec, None, args.dt)
            recs[tag] = rec.rows
        res = {}
        for tag, r in recs.items():
            K0, U0 = r["K"][0], r["U"][0]
            lsc = r0 * torch.sqrt(2 * n * K0.clamp_min(1e-12))
            t = 1000
            res[tag] = {"dE": float(((r["E"][t] - r["E"][0]).abs() / (K0 + U0.abs())).median()),
                        "dL": float(((r["L"][t] - r["L"][0]).abs() / lsc).median()), "dP": float((r["P"][t] - r["P"][0]).abs().median()),
                        "bound": float(r["bound"][t].median()), "vir": float(r["vir"][t].median()), "outgrid": float(r["outgrid"][t].median()),
                        "vmax_over_c": float(r["vmax"].amax() / sc.C) if rel else None, "frac_ge_c": float(r["fracge"].amax()) if rel else None,
                        "finite": float(r["finite"].min())}
        res["bound_gap"] = abs(res["model"]["bound"] - res["true"]["bound"])
        res["vir_gap"] = abs(float(np.log(max(res["model"]["vir"], 1e-9) / max(res["true"]["vir"], 1e-9))))
        out[name] = res
        print(f"cons {name} {1000}t dE {res['model']['dE']:.4g} (true {res['true']['dE']:.4g}) dL {res['model']['dL']:.4g} bound {res['model']['bound']:.3f}/{res['true']['bound']:.3f} "
              f"vir {res['model']['vir']:.3g}/{res['true']['vir']:.3g} outgrid {res['model']['outgrid']:.3f}", flush=True)
    return out
