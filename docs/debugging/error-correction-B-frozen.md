# B-frozen: option-B geometry branch on a fully frozen Exp E (2026-10-13)

Branch `worktree-agent-a61e7dc4af33d7032` (on fix-finetune-dE). Remote dir `~/augur-bfrozen`. Seed 4738, protocol of scatter-contact.md (train 800 scenes x 80 ticks, mix box15/box_var/post_circle/post_rect, eval seeds 9100/9200/9300 x 24 scenes, 100 ticks, held-out disc and post_poly), lr 2e-4, loss cap 100, k 4 -> 20, batch 16, 8000 it, init Exp E (E_ms_kp_pot_v_g128).

## What is built (all opt-in)
- `ContactScatterField(obstacle="sdf", frozen_geo=True)`, `cs.build("Bf", ...)`: with no scene geometry set the force is `base * gscale` only (gvec, wall MLP, pair head skipped), so no-geometry behaviour == Exp E bitwise (tests in `scripts/bfrozen_test.py`).
- Training: `--model Bf --train-only gvec,wmlp` (arm 1: SDF branch) or `gvec,wmlp,cmlp` (arm 2: + ball-contact pair head). Everything else (U-Net, kmlp, ppmlp, gscale = 1) is `requires_grad=False`. Geometry never enters the U-Net.
- gvec: Exp E has no uniform-gravity term (the contact world's g = 9 toward +x is the only thing it lacks besides walls), so the 2-parameter learned gravity vector of option B is trained in both arms; it is only active with geometry set. This is a deviation from "only the SDF MLP": without it nothing supplies gravity. Eval-only control `E+g` (gvec = (.9, 0), no walls) added to separate gravity from walls.
- Harness note (anomaly found): `cs.build` makes the contact model with dt = .15 (the contact-world step) and `scatter_regress.py --contact` / `ft_de_probe.py --contact` did not reset it to the harness dt .1, so the earlier free-space harness rows for C / Cfz (ft-dE doc) were integrated at the wrong dt. Fixed here (`model.dt = args.dt`); C / Cfz harness rows are re-run.
- Eval additions (`scatter_contact_scenes.py eval_family`): per-scene median err@5/10/20/100, off-grid fraction @100 (|x| or |y| > 32), runaway rate (any ball |dv|/dt > 200 after tick 20, fraction of scenes), reversal-based restitution (`*_rev`, apex after first reversal; the old probe returned None when the ball never reached the wall surface).

## Predictions (written before any run)
- Unit tests pass (P .9); no-geometry path == E bitwise, harness nr100@1000 dE and bh300@1000 identical to E (P .95).
- Arm 1 (SDF only): tunnel@100 < 3% on all six families incl. held-out (P .6). err@20 on box15 5.5-7.5 (E zero-shot 7.0, B2 5.1; E's mutual-gravity attraction and no pair repulsion remain) (P .55). err@100 median near the floor (~5-6), runaway rate 0 (P .7), off-grid < 2% (P .7).
- Arm 2 (SDF + pair head): err@20 within 1.2x of B2 on in-distribution families (P .5); better than arm 1 (P .7). tunnel < 3% (P .65).
- Over-damping: the cause in B2 is plausibly the multi-step MSE objective (chaos makes the conditional mean damped), not the architecture, so frozen-E branches stay over-damped: wall restitution at v = 10-15 below .8 (P .65), KE ratio @100 < .05 (P .7). Not fixed by freezing.
- dE rows for the no-geometry path: nr100@1000 .017 and bh300@1000 .48 as E (P .95). Contrast: C and Cfz free-space rows (dt fixed) fail nr100 dE (C by orders, Cfz at .8-1 from gscale 1.28) (P .8).
- Frozen-C comparison: frozen branch Bf beats Cfz on tunnel@20 (<10% vs 59%) (P .8), because the SDF gather MLP sees the ball position/velocity directly, not through a frozen first-conv.

(Results below.)
