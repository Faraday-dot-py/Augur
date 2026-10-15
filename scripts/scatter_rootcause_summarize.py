import json
import sys

d = json.load(open(sys.argv[1]))
for fam in ("expB", "cluster"):
    r = d[fam]
    print(f"=== {fam}  N={r['N'][:5]}...  exact_verlet1 TF one-step pos/vel {r['exact_verlet1_tf']}")
    print("-- rollouts (err@5/20/50/100)")
    for k, v in r["rollouts"].items():
        print(f"  {k:34s} " + " ".join(f"{x:9.4g}" for x in v.values()))
    print("-- force error at truth states: rel_rms | share_err2 | var_share | vdep/err | boost/err | fld/err  (per nn bin)")
    for m, f in r["force"].items():
        print(f" [{m}] tf_pos {f['tf_pos_mean']:.2e} tf_vel {f['tf_vel_mean']:.2e}")
        for b, s in f.items():
            if isinstance(s, dict) and s:
                print(f"   {b:22s} frac {s['frac']:.3f} rel {s['rel_rms']:.4f} abs {s['abs_rms']:.4f} ae {s['ae_rms']:.3f} share {s.get('share_err2', 1):.3f} var {s['var_share']:.3f} vdep {s['vdep_over_err']:.3f} boost {s['boost_over_err']:.3f} fld {s['fld_over_err']:.3f}")
print("=== pair probe (rad_mean/rel_rms vs r)")
for m, res in d["pair_probe"].items():
    for vk, rows in res.items():
        print(f" [{m} {vk}] " + " ".join(f"r{x['r']:g}:{x['rad_mean']:+.3f}/{x['rel_rms']:.3f}" for x in rows))
