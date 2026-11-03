import json, sys, math
V2 = {"liquid": (.0018, .0083, .096), "coexist": (.0031, .0124, .12), "quench": (.0015, .0079, .19),
      "solid": (.0004, .0007, .0015), "melt": (.0003, .0007, .0015), "gas": (0, .0001, .003)}
for tag in sys.argv[1:]:
    d = json.load(open(f"results/scatter_lj_{tag}.json"))
    print("==", tag)
    allpass, kill = True, 0
    for s, v2 in V2.items():
        r = d[s]; t, m = r["truth"], r["model"]
        dT = abs(m["T"] - t["T"]) / t["T"]; dpsi = abs(m["psi6"] - t["psi6"]); dP = abs(m["P"] - t["P"]) / max(abs(t["P"]), 1e-9)
        e = [r["err_model"][str(k)] for k in (5, 10, 20)]
        ratio = [x / max(2 * b, .002) for x, b in zip(e, v2)]
        ok = r["finite"] and dT <= .01 and dpsi <= .003 and dP <= .10 and all(x <= 1 for x in ratio)
        k20 = e[2] / max(v2[2], .002)
        if not r["finite"] or dT > .05 or dpsi > .03 or math.isnan(dT):
            kill_flag = True
        else:
            kill_flag = False
        print(f"{s:8s} fin={r['finite']} dT={dT:.3f} dpsi6={dpsi:.4f} dP={dP:.3f} err5/10/20={e[0]:.4f}/{e[1]:.4f}/{e[2]:.4f} (v2 {v2}) k20/v2={k20:.1f} gL2={r['g_rel_l2']:.2f} Ts={t['T']:.3f}/{m['T']:.3f} psi={t['psi6']:.3f}/{m['psi6']:.3f} P={t['P']:.3f}/{m['P']:.3f} pass={ok} killflag={kill_flag}")
        allpass &= ok; kill += kill_flag or 0
        if k20 > 5: print("   traj>5x v2 at frame 20")
    print("ALL PASS" if allpass else "not all pass", "kill-scenes", kill)
