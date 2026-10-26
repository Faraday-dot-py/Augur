"""Idempotent patch for scripts/scatter_field.py of the tier-0 integration code base: adds `vblind` / `norec` switches to ScatterField
(velocity channels of the scatter zeroed / recurrent field channels zeroed at the network input). Usage: python this.py path/to/scatter_field.py"""
import sys

p = sys.argv[1]
s = open(p).read()
if "self.vblind" in s:
    print("already patched")
    sys.exit(0)
a = "        x = self.scatter(spos, vel, smass, mask)\n        if ctx is not None:\n            x = self.t0.after_scatter(ctx, x)\n"
b = "        x = self.scatter(spos, vel * (0.0 if self.vblind else 1.0), smass, mask)\n        if ctx is not None:\n            x = self.t0.after_scatter(ctx, x)\n"
assert s.count(a) == 1
s = s.replace(a, b)
a = "        if self.recurrent:\n            x = torch.cat([x, field], 1)\n        if self.epsin:"
b = "        if self.recurrent:\n            x = torch.cat([x, field * (0.0 if self.norec else 1.0)], 1)\n        if self.epsin:"
assert s.count(a) == 1
s = s.replace(a, b)
a = "    def __init__(self, grid=64, extent=64.0, net=\"local\""
assert s.count(a) == 1
s = s.replace(a, "    vblind = False\n    norec = False\n\n" + a)
open(p, "w").write(s)
print("patched")
