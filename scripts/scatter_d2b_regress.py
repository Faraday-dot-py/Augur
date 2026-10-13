"""scatter_regress.py harness with the model built as a (gated) option-C ContactScatterField, geometry-free (plain E / C / D2b checkpoints all load).
  PYTHONPATH=. python scripts/scatter_d2_regress.py run --ckpt <pt> --out <json> --opt kcache,fastio [--no-cons]
  PYTHONPATH=. python scripts/scatter_d2_regress.py compare --baseline <json> --candidate <json>"""
from scripts import scatter_d2b as d2
from scripts import scatter_field as sf
from scripts import scatter_regress as sr


def build_model(args, dev):
    model = d2.build(args.ckpt, dev)
    sf.apply_opts(model, args.opt)
    return model


sr.build_model = build_model

if __name__ == "__main__":
    sr.main()
