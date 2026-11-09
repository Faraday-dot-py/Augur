import sys, numpy as np

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
for key in sorted(k[:-8] for k in a.files if k.endswith('_decoder')):
    da, db = a[key + '_decoder'], b[key + '_decoder']
    da = da / (np.linalg.norm(da, axis=1, keepdims=True) + 1e-12)
    db = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-12)
    if da.shape[1] != db.shape[1] and da.shape[0] == db.shape[1]:
        da, db = da.T, db.T
    ea, eb = a[key + '_law_eta2'], b[key + '_law_eta2']
    live_a = np.isfinite(ea) & (ea > 0)
    live_b = np.isfinite(eb)
    c = np.abs(da @ db.T)
    best = c.max(1)
    top = np.argsort(-np.nan_to_num(ea))[:10]
    print(key, 'feats', da.shape[0], 'median best |cos|', round(float(np.median(best)), 3),
          'frac>0.9', round(float((best > 0.9).mean()), 3),
          'top10 law feats best |cos|', np.round(best[top], 2).tolist())
