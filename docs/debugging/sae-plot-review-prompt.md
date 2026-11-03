# Unbiased SAE plot description prompt

Use with a fresh subagent. Give it only the image paths and this text. Do not add hypotheses, expected results or experiment names.

You are looking at diagnostic images produced by an analysis of a neural network. You have not been told what it is, what was expected, or what the axes should show. Read every axis label, title and legend. Describe, per image and per panel: what is plotted, the visible structure (clusters, lines, bands, outliers, empty regions), approximate value ranges you can read off, and anything that looks like a rendering artifact (clipping, saturation, banding, overplotting, constant panels, empty panels). Then list anything surprising. Report what you see; do not explain why, do not name a cause, do not give a verdict, and say plainly where you cannot tell.
