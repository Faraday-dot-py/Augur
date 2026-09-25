p = "scripts/dual_tree.py"
s = open(p).read()
s = s.replace("def dual_accel(tr, fl, theta, cap):", "def dual_accel(tr, fl, theta, cap, accept_fn=None, collect=None):")
s = s.replace("        acc = (torch.maximum(sa, sb) < theta * dist) & (ga != gb)\n",
              "        acc = (torch.maximum(sa, sb) < theta * dist) & (ga != gb)\n        if accept_fn is not None:\n            acc = accept_fn(ga, gb, acc)\n")
s = s.replace("        if acc.any():\n            r_, cb_, ga_ = r[acc], cb[acc].double(), ga[acc]\n",
              "        if acc.any():\n            if collect is not None:\n                collect.append((ga[acc], gb[acc]))\n            r_, cb_, ga_ = r[acc], cb[acc].double(), ga[acc]\n")
s = s.replace("        self.count = torch.cat([x[\"count\"] for x in lv])\n", "        self.count = torch.cat([x[\"count\"] for x in lv])\n        self.q = torch.cat([x[\"q\"] for x in lv])\n")
open(p, "w").write(s)
