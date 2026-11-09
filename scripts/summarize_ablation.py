import sys, json

for p in sys.argv[1:]:
    r = json.load(open(p))
    print(p, r['key'], [(f['feature'], round(f['eta2'], 2), f['dominant_law']) for f in r['top_law_features']])
    for law, arms in r['errors'].items():
        print(' ', law.ljust(7), ' '.join(f"{a[:8]} {v['median']:.3f}/{v['p90']:.3f}" for a, v in arms.items()))
