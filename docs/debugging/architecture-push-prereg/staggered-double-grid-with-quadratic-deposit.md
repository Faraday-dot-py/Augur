# staggered-double-grid-with-quadratic-deposit (prereg)

Status: stage 1 ready to run; FULL RUN NOT YET AUTHORISED until stage 1 is read by the supervisor (job script scripts/polaris_staggered-double-grid-with-quadratic-deposit-full.sh refuses to start without the file named in its header).

## Spec lines (copied verbatim)

Change: "replace bilinear deposit and gather with a quadratic (three-cell) shape, and run two grids offset by half a cell through the same U-Net, averaging the gathered force. Trained that way, not only run that way."

Line: "PASS err@20 at most 0.75x control AND shift-probe value at most 0.50x control AND dE@100 at most 1.10x AND cost at most 2.2x. KILL err@20 above 0.95x or dE@100 above 1.5x. Fails pass: err@20 0.85x."

Cost in spec: 0.3 GPU-h stage 1 / 3 GPU-h full. Worker estimate for full: about 5-6 GPU-h (two grids roughly double the 2.5 h the frozen single-law gravity checkpoint run; UNVERIFIED, from the logged the frozen single-law gravity checkpoint 9000 s budget). Flagged needs-user.

Rules from the 2026-11-09 decisions: keep the control's iteration count (60769) with a plateau guard; if the candidate is still falling the result is INCONCLUSIVE-undertrained and one double-length extension is allowed for candidate and control together. Full run starts automatically if stage 1 does not close it.

## Design

Stage 1 (scripts/shift_average_stage1.py): frozen the frozen single-law gravity checkpoint, 1 / 2 / 4 passes (offsets 0; 0 and (0.5,0.5); 0, x, y, diagonal in half cells), each pass own recurrent field; measures err@5/20/100, dE@5/20/100, shift probes, sec per tick. 0.3 GPU-h.

Full: ScatterField(deposit="quad", dual=True) via variant ms_kp_pot_v_g128_quad_dual: three-cell (TSC, triangular-shaped-cloud) weights 0.5(0.5-d)^2, 0.75-d^2, 0.5(0.5+d)^2 in each axis; second grid sees positions shifted by (h/2, h/2), h = cell size; gathered grid forces averaged; recurrent field holds 2*cf channels; pair term and momentum fix applied once on unshifted positions. Same recipe as the frozen single-law gravity checkpoint (init_B warm start, lr 5e-4, batch 16, k 4 to 30, 1500 scenes x 50 steps, seed 4738) but fixed 60769 iterations instead of the 9000 s time budget.
Control: frozen the frozen single-law gravity checkpoint, ~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt (3200305 B). Not retrained (same checkpoint on all seeds). Polaris path of candidate output: ~/augur_staggered_double_grid/ckpt/.

## Seeds and tie-break

Scenes: 9100/9200/9300 first, then fresh 12000/12100/12200 (expB 8 + cluster 4 scenes per seed). Training seed 4738. Ratios are pooled: mean over seeds of candidate divided by mean over seeds of control. Tie-break: a value exactly on a line counts as within the PASS line and as not over the KILL line (at most / above wording). Shift-probe value for the line = probe_half (median relative force change under half-cell x, y and diagonal shifts) (PROPOSED-BY-WORKER interpretation; probe_off is reported, not gated).
Verdict on 9100/9200/9300 decides; 12000/12100/12200 confirms. If they disagree the report says so and the verdict is INCONCLUSIVE, not the better one.

## Plateau guard

INCONCLUSIVE-undertrained if the mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window (scripts/plateau_guard.py; validation loss logged to <ckpt>.val.jsonl every --val-every iterations on scenes seeded seed+1). A guard fires only on non-PASS verdicts.

## Stage-1 closure rule (PROPOSED-BY-WORKER)

Stage 1 closes the experiment (no full run) only if both the 2-pass and the 4-pass err@20 pooled ratios exceed 0.95 (the KILL value). A value that would still not close it: err@20 ratio 0.94 on either. Anything else proceeds to the full run automatically (spec 10a item 9). The spec does not define closure; the supervisor may amend.

## Prediction

Stated: probably INCONCLUSIVE (the shift-probe halving is plausible, but err@20 at most 0.75x is a large ask; the cost 2.2x is tight because two grids double the U-Net work). Rough: PASS 20 / INCONCLUSIVE 50 / KILL 30 (worker guess, not from the spec).

## Constraints check

- pp=2 with knn=16: unchanged (pair term kept, applied once).
- No hard-wired conservation: none added; momentum fix is the existing optional one, off in the variant as in the frozen single-law gravity checkpoint.
- O(N): deposit and gather are 9 cells per body each pass, two passes: constant per body.
- Opt-in: default constructor stays deposit="cic", dual=False; existing checkpoints load unchanged.
- One change per run: quad deposit and dual grid are one spec change; variants _quad and _dual exist for later separation, not run.
- Checkpoint/resume/SIGTERM: train_scatter_field.py saves atomically every --log-every iterations (and a .itN snapshot every 5000), resumes from the checkpoint, and with --save-on-sigterm saves and exits 143.

## Additions not in the spec line (disclosed)

- Any non-finite candidate rollout counts as KILL.
- The plateau guard downgrades any non-PASS verdict (including KILL) to INCONCLUSIVE-undertrained; PROPOSED-BY-WORKER, matches the 10a item 1 wording that an undertrained candidate is not a result.
- A NO-DATA plateau guard (too few validation points) fails the job script loudly; it is never read as plateaued.
