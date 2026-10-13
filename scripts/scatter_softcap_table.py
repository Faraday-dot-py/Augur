"""Print c_id/c_nonconvex/box15 err, tunnel, KE ratio, dE and wall restitution from results/contact/eval_<tag>.json (scatter_contact_scenes.py eval).

  python3 scripts/scatter_softcap_table.py C C_sc100 D5ctrl_sc0 D5ctrl_sc100
"""
import json
import sys

f = lambda x: "-" if x is None else f"{x:.3g}"
for tag in sys.argv[1:]:
    d = json.load(open(f"results/contact/eval_{tag}.json"))
    print("==", tag)
    for fam, r in d["families"].items():
        m = r["mean"]
        print(f"  {fam:11s} err@5/20/100 {f(m['err']['5'])}/{f(m['err']['20'])}/{f(m['err']['100'])} tunnel@20/100 {f(m['tunnel_model']['20'])}/{f(m['tunnel_model']['100'])} "
              f"ke@100 {f(m['ke_ratio']['100'])} dE@20/100 {f(m['dE_model']['20'])}/{f(m['dE_model']['100'])} (truth {f(m['dE_truth']['20'])}/{f(m['dE_truth']['100'])})")
    print("  wall e_eff", {v: f((x or {}).get("model")) for v, x in d["restitution_wall"].items()})
    print("  circle e_eff", {v: f((x or {}).get("model")) for v, x in d["restitution_circle"].items()})
