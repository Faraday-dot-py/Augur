import json
import sys

d = json.load(open(sys.argv[1]))
for i in (0, 4, 7, 13):
    r = d["scenes"][i]
    print("scene", i)
    for t in list(range(0, 40, 3)):
        x = r[t]
        print(f" t{t:2d} fld={x['fld_max']:.3g} gacc={x['gacc_max']:.3g} a0={x['a0_max']:.3g} cell=({x['cell_x']:.1f},{x['cell_y']:.1f}) nearest_ball={x['cell_nearest_ball']:.1f} sdf={x['cell_sdf']:.1f}")
