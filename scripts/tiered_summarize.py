"""Print headline numbers from scripts/tiered_eval.py JSON files: python scripts/tiered_summarize.py results/tiered/eval_*.json"""
import json
import sys

KEEP = ("err", "rel", "ratio", "exponent", "z", "jump", "ms_per_step", "peak_mb", "escaped", "fragments", "gap", "shift", "rot",
        "windows", "active", "min_gap", "mean_diff", "mean_signed", "verdict", "n_sweep", "empty_sweep", "neighbours")


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)


def compact(d):
    """One line for a compare() result: err@k, err of constant velocity at the last k, drift model|truth at the last k."""
    ks = sorted(d["err"], key=int)
    s = "err@" + " ".join(f"{k}={d['err'][k]:.4g}" for k in ks)
    s += f" | constvel@{ks[-1]}={d['err_constvel'][ks[-1]]:.4g}"
    for nm in ("dE", "dp", "dL"):
        e = d["drift"][nm]
        k = sorted(e, key=int)[-1]
        s += f" | {nm}@{k} {e[k]['model']:.3g}|{e[k]['truth']:.3g}"
    ch = d["channels"]
    s += f" | pos_xy_rms {ch['pos_x_rms']:.3g},{ch['pos_y_rms']:.3g} vel_xy_rms {ch['vel_x_rms']:.3g},{ch['vel_y_rms']:.3g}"
    if "pos_err_by_mass_tercile" in ch:
        s += f" mass_terc {fmt(ch['pos_err_by_mass_tercile'])}"
    nf = [x for x in d["first_nonfinite"] if x]
    if nf:
        s += f" | NONFINITE at {nf}"
    return s


def walk(o, path, out):
    if isinstance(o, dict):
        if "err" in o and "drift" in o and "channels" in o:
            out.append(f"   {path}: {compact(o)}")
            for k, v in o.items():
                if k not in ("err", "err_constvel", "drift", "channels", "first_nonfinite", "secs", "n"):
                    walk(v, f"{path}/{k}", out)
            return
        for k, v in o.items():
            walk(v, f"{path}/{k}" if path else str(k), out)
    elif isinstance(o, (int, float, list, str)) and not isinstance(o, bool) and any(k in path for k in KEEP) and "edges" not in path:
        out.append(f"   {path}: {fmt(o)}")


def main():
    for f in sys.argv[1:]:
        d = json.load(open(f))
        print(f"== {f}: model {d['model']} laws {d['laws']} reach {d['reach']} seeds {d['seeds']} quick {d['quick']}")
        for g, r in d["groups"].items():
            print(f"-- {g}: {r['status']} ({r['seconds']}s)")
            src = r["summary"] or (r["seeds"][next(iter(r["seeds"]))] if r["seeds"] else {})
            out = []
            walk(src, "", out)
            print("\n".join(out))


if __name__ == "__main__":
    main()
