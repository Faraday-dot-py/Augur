import json, sys

d = json.load(open(sys.argv[1]))
worst = 0.0
for f in ('plummer', 'two_clump', 'exp_disk'):
    for n in (10000, 100000):
        c = d[f'{f}_n{n}']
        l = c['learned_warm']['norm_by_rms']['median']
        b = c['barnes_hut_theta_0.5']['norm_by_rms']['median']
        print(f, n, round(l, 4), round(b, 4), round(l / b, 2))
        worst = max(worst, l / b)
verdict = 'KILL' if worst > 5 else 'PASS' if worst <= 2 else 'INCONCLUSIVE'
print('max ratio', round(worst, 2), verdict)
