import sys

import numpy as np

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
rng = np.random.default_rng(4738)


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


for key in sorted(k[:-8] for k in a.files if k.endswith('_decoder')):
    da, db = a[key + '_decoder'], b[key + '_decoder']
    if da.shape[1] != db.shape[1] and da.shape[0] == db.shape[1]:
        da, db = da.T, db.T
    ea, eb = a[key + '_law_eta2'], b[key + '_law_eta2']
    la, lb = np.isfinite(ea), np.isfinite(eb)
    ua, ub = unit(da)[la], unit(db)[lb]
    best = np.abs(ua @ ub.T).max(1)
    rnd = np.abs(unit(rng.standard_normal(ua.shape)) @ unit(rng.standard_normal(ub.shape)).T).max(1)
    top = np.argsort(-np.nan_to_num(ea[la]))[:5]
    print(key, 'live', int(la.sum()), int(lb.sum()), 'median best |cos|', round(float(np.median(best)), 3),
          'random baseline', round(float(np.median(rnd)), 3), 'frac>0.9', round(float((best > 0.9).mean()), 3),
          'top5 law feats', np.round(best[top], 2).tolist())
