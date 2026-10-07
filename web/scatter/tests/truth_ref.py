import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))
import gravity_sim

here = os.path.dirname(__file__)
init = json.load(open(os.path.join(here, "truth_init.json")))
n = init["n"]
pos = np.array(init["pos"]).reshape(n, 2)
vel = np.array(init["vel"]).reshape(n, 2)
ps, vs = gravity_sim.rollout(pos, vel, init["ticks"], dt=init["dt"])
keep = [10, 100, 1000]
out = {
    "pos": {str(t): ps[t].reshape(-1).tolist() for t in keep},
    "E": {str(t): float(gravity_sim.energy(ps[t], vs[t], 0.5)) for t in [0] + keep},
}
json.dump(out, open(os.path.join(here, "truth_ref.json"), "w"))
