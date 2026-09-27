import sys

import numpy as np

for path in sys.argv[1:]:
    d = np.load(path)
    st = d["stats"]
    print(f"{path}: N={d['pos'].shape[1]} frames={len(st)} T {st[0,1]:.3f}->{st[-1,1]:.3f} "
          f"E/N {st[0,3]:.5f}->{st[-1,3]:.5f} (drift {st[-1,3]-st[0,3]:+.2e}, max dev {np.abs(st[:,3]-st[0,3]).max():.2e}) "
          f"P {st[0,4]:.3f}->{st[-1,4]:.3f}")
