# Footnotes: work not on master

Master's model is the conservative-contact TokenFreeDynamics (`conservative_contact=True`, ckpt `results/cons_pure.pt`; err@5/10/20 0.12/0.31/2.39; see `docs/debugging/energy-fix-results.md`). Everything else is kept as tags; check out with `git checkout <tag>` or `git worktree add <dir> <tag>`.

| Tag | Result |
|---|---|
| `archive/gravity-direction` | `gravity_input`/radial field on the conservative model; held-out gravity angles within 0.07 accel, central field zero-shot err@20 1.44; `docs/debugging/gravity-direction-results.md` |
| `archive/tall-box` | 5000x1000 box, 10k balls, g=9 (learned 8.902), 500 steps finite, err@20 0.78; opt-in height/width in `bounce.py` |
| `archive/richer-temporal-flownet` | N-frame history=3 on flow-warp net; rejected (moire artifact); findings doc only in the tag |
| `archive/richer-temporal-windowed` | history=3 on windowed attention; rejected (accuracy regression); findings doc only in the tag |

Branch `viz-web` is the live GitHub Pages source (faraday-dot-py.github.io/Bounce); left as is.
Large result arrays (`results/tiled_video_10m*.npz`, `results/ball_rect_5000x1000.npz`) are gitignored, local only.
