import sys

import numpy as np

d = np.load(sys.argv[1])
for k in d.files:
    a = d[k]
    print(k, a.shape, a.dtype)
