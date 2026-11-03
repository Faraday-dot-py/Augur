import sys
import numpy as np

for f in sys.argv[1:]:
    d = np.load(f, allow_pickle=True)
    print(f, [(k, d[k].shape, str(d[k].dtype)) for k in d.files])
