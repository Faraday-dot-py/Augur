import json
import sys

c = json.load(open(sys.argv[1]))["lam0.5"]["closure"]
print("lam0.5 closure", c)
sys.exit(0 if all(v is not None and v >= 0.30 for v in c.values()) else 1)
