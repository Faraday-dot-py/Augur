"""Render docs/class/token-model-proposal.md (paragraphs with **bold** run-in
headings) to a one-page US-Letter PDF: 11 pt Liberation Serif, 1 inch margins.

Usage: python3 docs/class/render_proposal_pdf.py
"""
import re
import sys

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

SRC = "docs/class/token-model-proposal.md"
OUT = "docs/class/token-model-proposal.pdf"
FONT, SIZE, LEADING, GAP = "Liberation Serif", 11.0, 12.6, 5.0
W, H, M = 8.5, 11.0, 1.0
matplotlib.rcParams["pdf.fonttype"] = 42

fig = plt.figure(figsize=(W, H))
renderer = fig.canvas.get_renderer()


def width(word, bold):
    t = fig.text(0, 0, word, family=FONT, size=SIZE, weight="bold" if bold else "normal")
    w = t.get_window_extent(renderer).width / fig.dpi
    t.remove()
    return w


def tokens(par):
    out = []
    for i, part in enumerate(re.split(r"\*\*", par)):
        out += [(w, i % 2 == 1) for w in part.split()]
    return out


space = width("a a", False) - width("aa", False)
x0, x1 = M, W - M
y = H - M
pars = [p.strip() for p in open(SRC).read().split("\n\n") if p.strip()]
for par in pars:
    x, y = x0, y - LEADING / 72
    for word, bold in tokens(par):
        w = width(word, bold)
        if x + w > x1 and x > x0:
            x, y = x0, y - LEADING / 72
        fig.text(x / W, y / H, word, family=FONT, size=SIZE, weight="bold" if bold else "normal", va="baseline")
        x += w + space
    y -= GAP / 72
if y < M:
    sys.exit(f"overflow: text ends at y={y:.2f} in, below the bottom margin")
print(f"ok: text ends at y={y:.2f} in (bottom margin at {M} in)")
fig.savefig(OUT, format="pdf")
