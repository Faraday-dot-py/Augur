import json
import sys

print(json.load(open(sys.argv[1]))["t0"])
