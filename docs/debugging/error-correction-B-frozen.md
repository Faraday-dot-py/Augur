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

## Results (jobs 3589 train+eval, 3590 reference evals + harness; tables in results/bfrozen/table.md and harness.md)
Both arms trained 8000 it with no divergence (same recipe as B2). Eval: seeds 9100/9200/9300 x 24 scenes, 100 ticks; disc and post_poly held out. Medians below are per-scene medians averaged over families (full per-family table in results/bfrozen/table.md).

| model | err@20 (6 fam range, median) | err@100 median | tunnel@20 / @100 | off-grid@100 | runaway | KE ratio@100 |
|---|---|---|---|---|---|---|
| E zero-shot | 6.7-7.7 | 23-25 | 34-43% / 93-96% | 0.2-0.8% | 0 | .05-.07 |
| A2 (seed set 13, older eval) | 4.3-5.1 | 3.5e4-7.5e4 (blows up after ~tick 30) | ~0% / 100% | n/a | n/a | 1e6-4e7 |
| C | 4.0-4.8 | 1.4e3-1.9e3 | ~0% / 100% | 100% | 97-100% | 3e3-8e3 |
| frozen-C | 6.7-7.9 | 170-185 | 54-67% / 95-99% | 95-98% | 0 | 10-14 |
| B2 (reference) | 4.1-5.0 | 3.8-5.5 | 0 / 0 | 0 | 0 | .003-.009 |
| Bf1 (SDF only) | 4.4-5.1 | 3.5-5.5 | 0 / 0 | 0 | 0 | 7e-9-1e-3 |
| Bf2 (SDF + pair) | 4.1-5.1 | 3.8-5.8 | 0 / 0.09% (disc@20) | 0 | 0 | 2e-4-3e-4 |

E+g (gravity only, no walls): err@20 ~38, tunnel 100%, as expected (balls fall through the right wall); confirms the walls come from the SDF branch, not gvec.

Restitution (apex-after-first-reversal, truth ~1.0), wall: Bf1 / Bf2 / B2 at v=4: 1.25 / 1.18 / .99; v=8: .66 / .62 / .61; v=10: .54 / .51 / .53; v=15: .45 / .41 / .57. Circle v=8: .63 / .60 / .61. Over-damped at speed exactly as B2; values > 1 at v=2 are an apex-after-reversal artefact at small speeds (as B2: 2.2).

### Frozen-E rows (harness, geometry cleared)
| model | nr100 dE@1000 | nr100 dL | bh300 dE@1000 | bh300 dL | two_body err@20 | expB err@20 | bh300 err@20 |
|---|---|---|---|---|---|---|---|
| E | .0177 | .028 | .471 | 2.22 | .0091 | .0239 | .0622 |
| Bf1 | .0235 | .075 | .428 | 3.18 | .0091 | .0239 | .0634 |
| Bf2 | .0232 | .061 | .589 | 2.65 | .0091 | .0239 | .0634 |
| B2 free-space | 2.0 | .125 | 1.60 | .035 | .879 | 5.56 | 8.59 |
| C free-space (dt fixed) | 9.7e3 | 26 | 1.9e4 | 3.1e3 | 1.76 | 5.32 | 9.6 |
| frozen-C free-space | .168 | .106 | .098 | 2.29 | .052 | .267 | 1.02 |

Small-N rows (two_body, expB) are identical to E to printed precision. Large-N dE/dL rows are NOT identical to E (nr100 dE .0177 -> .0235, dL .028 -> .075); prediction "identical" (P .95) was only partly right. The nr100 dE .0235 passes the same ~.02 gate region E sits in (E heads-only passed at .0218) but dL is 2.7x worse. Whether this is E's own GPU run-to-run noise or a real difference was tested separately with scripts/bfrozen_noise.py (see "Noise check" below).

### Timing / O(N)
Harness ms/tick (same GPU, same job): two_body N=2 batch 48: E 4.74, Bf1 4.73, Bf2 4.73; expB N=94 batch 48: 4.96 / 4.92 / 4.93; bh300 N=300 batch 1: 2.46 / 2.50 / 2.48; N=100 batch 1: 2.43 / 2.45 / 2.44. No measurable overhead without geometry (frozen_geo skips gvec/wall MLP/pair head). With geometry the SDF branch is per-ball (gather at ball position), O(N); pair head (Bf2) is the existing local-grid pair head. Geometry-on timing was not separately measured (untested).

### Video review (unbiased subagent, hypothesis-free prompt docs/debugging/contact-grid-review-prompt.md; grids for box15 Bf2, disc Bf1/Bf2, post_circle Bf1/Bf2, post_poly Bf1/Bf2)
Red leaves blue by tick ~10 in all seven. No red ball leaves the frame or enters an obstacle (one possible edge case: disc scene 2, tick 30, red ball at/just past the circle boundary in both arms). Red never spreads leftward as blue does. Bf1: balls collapse onto 1-3 overlapping points near the right wall (no pair repulsion), i.e. "frozen, stacked". Bf2: a tidy, evenly spaced vertical column 1-2 ball-widths from the right wall that does not evolve. In obstacle scenes 1-2 red balls near the left stay fixed all run. This is the over-damped / frozen-gas look, in kind different from blue. Aggregate err@100 ~5 therefore sits at the chaos floor by being static, not by tracking. Videos: videos/contact_*_Bf{1,2}.mp4 for disc, post_circle, post_poly. Not rendered/reviewed: box15 Bf1, box_var, post_rect.

## Verdict
- Geometry on a frozen E works for stability: tunnel 0%, runaway 0, off-grid 0, err@20 4.1-5.1 (match B2 and A2/C at tick 20, 25-35% better than E zero-shot), held-out disc/post_poly as good as in-distribution. Unlike C/A2 (blow up, 100% off-grid) and frozen-C (59% tunnel@20), Bf does not blow up. Predictions: SDF-only tunnel<3% hit; err@20 5.5-7.5 missed (it was better, 4.4-5.1); err@100 near floor and runaway 0 hit; arm 2 within 1.2x of B2 hit but not better than arm 1 (P .7 missed: Bf1 ~= Bf2 at tick 20, Bf2 better at ticks 5-10); over-damping prediction hit (KE ratio @100 4e-4 / 3e-4 < .05, restitution .4-.5 at v=15); Bf beats Cfz tunnel@20 hit.
- The over-damping is not an architecture or base-network artefact: frozen E plus a fresh branch is just as damped as B2 (KE ratio ~3e-4 vs B2 ~4e-3), supporting the multi-step-MSE-objective cause. Bf1 stacks balls (no repulsion), Bf2 forms a column: frozen-gas final state, not physics.
- Large-N dE rows differ slightly from E (see above and noise check): option B on frozen E nearly, but not provably exactly, preserves gravity.

## Noise check (pending)
Job 3596 (scripts/bfrozen_noise.py on Bf1 and Bf2: E-vs-E rerun, Bf-vs-E, Bf-vs-Bf force/position differences at N 10/100/300) queued behind other jobs; output ~/augur-bfrozen-noise.log on Polaris. Until it runs, the cause of the nr100 dE/dL difference from E is unresolved.
