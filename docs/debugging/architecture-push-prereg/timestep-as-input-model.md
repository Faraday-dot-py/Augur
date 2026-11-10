# timestep-as-input-model: pre-registration (stage 1 and full run)

Written before any run. Spec: experiment 16, section 10a overriding. Priority 3.

Definitions. dt: step size of one tick (0.1 is the training step). err@20: mean per-body position error at tick 20. dE@100: |E_model - E_truth| / |E_truth| at tick 100, truth energy at the same tick. single-law-gravity-control: single-law gravity checkpoint, the control.

## Design
Change: step size enters as one extra input (`dtin`: log(dt / substeps / 0.1), added to the kernel and pair features and as one extra U-Net channel); training mixes dt in {0.05, 0.1, 0.2} (uniform per iteration). Variant `ms_kp_pot_v_g128_dtin`; zero-padded columns make the warm start identical to the control at every dt (test_scatter_timestep_input.py). Not trained on: dt 0.075 (evaluation only).
Control: single-law-gravity-control, Polaris `~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt` (verified). Warm start of the candidate: `~/augur/checkpoints/scatter_bh/init_B_ms_kp_pot_v_g128.pt` (verified, 3200305 bytes), the same warm start single-law-gravity-control used. Recipe: single-law-gravity-control (60000 iterations, lr 5e-4, batch 16, k 4 to 30), eager mode.
Stage 1 (about 1 GPU-h): truth at three step sizes plus 0.075 for evaluation on the single-law-gravity-control scene distribution, then the frozen single-law-gravity-control reference at each dt (inference only). Full run (about 5 GPU-h): train and evaluate.

## Lines (verbatim, spec section 7 experiment 16)
- "PASS at dt 0.1, err@20 at most 1.05x control; at dt 0.05, err@20 at most 0.70x and dE@100 at most 0.50x of control at dt 0.1; step sizes not trained on (0.075) interpolate within 15% of the line between neighbours. KILL err@20 at dt 0.1 above 1.15x. Fails pass: 0.80x at dt 0.05."
- Operational definition of the 0.075 clause (builder-fixed): the line is the linear interpolation of the candidate's own err@20 at dt 0.05 and 0.1 evaluated at 0.075; PASS needs the candidate's 0.075 err@20 within 15% of it.
- Between the lines: INCONCLUSIVE. Default scorecard cost clause (section 6): cost at most 1.5x control; the model's cost per tick is unchanged (one extra channel), measured in the full run.
- Stage 1 produces no verdict; it produces the control's numbers at each dt, the denominators of the lines.

## Seeds and noise
Eval seeds 9100/9200/9300 (48 scenes, 100 ticks each); fresh 12000/12100/12200 before the result is called real (new truth sets for those seeds, generated with `--eval-seeds`); training seed 4738, second seed 4739 only after a pass. Validation set for the plateau guard is the seed-9100 eval truth at the three training dts (never used for selection).
Noise floor: seed-to-seed fine-tune variation is about 10-15%, so the 1.05x line at dt 0.1 is inside the floor at one seed; a pass at 4738 is provisional until 4739. The 0.70x and 0.50x lines at dt 0.05 are outside the floor. Control comparisons are against the frozen control run in the same job family (frozen_reference.json), same seeds.

## Tie-break and selection
No selection: the final-iteration checkpoint is scored. Pass lines use `<=`, kill lines `>`. A ratio exactly on a line meets PASS.

## Plateau guard (spec 10a item 1)
Mean validation loss (k=20 rollout loss, three dts) over the last 10% of iterations more than 3% below the mean over the preceding 10% window gives INCONCLUSIVE-undertrained (`plateau_falling`, validation every 250 iterations, total 60000). Iteration count kept at the control's 60000; one extension at double length for candidate and control together if flagged.

## Prediction (my estimate)
At dt 0.1 the ratio lands near 1.0-1.1x. At dt 0.05 err@20 about 0.6-0.8x (retraining at half step cut err@20 about 40%). The dE@100 line at 0.50x is the hardest. Probability of PASS: 35%. The 0.075 interpolation clause probably passes if dt 0.05 is learned.

## Constraints check
- pp=2, knn=16 unchanged. No hard-wired conservation; dE against truth E(t). O(N): one extra channel; cost per unit simulated time falls with larger dt (the stated requirement).
- Matched budget: 60000 iterations and the same recipe as the control.
- Anomaly kept in notes at lower priority: truth at three step sizes is generated once (train scenes 1500 per dt, 50 ticks); training reuses them, so the candidate sees fixed scenes where single-law-gravity-control regenerated them. If the control's data source differs, the comparison is not exactly matched.

## Artifacts
Stage 1: scripts/polaris_timestep-as-input-model-stage-1.sh (needs 15 truth sets: 3 train + 12 eval, checked). Full run skeleton, NOT YET AUTHORISED (exits 64 unless FULL_RUN_AUTHORISED=yes): scripts/polaris_timestep-as-input-model-full-run-not-yet-authorised.sh. Code: scripts/train_scatter_timestep.py. Tarball: `bash scripts/build_architecture_push_stage1_code_tarball.sh`. Estimates: stage 1 about 1.0 GPU-h, full about 5 GPU-h (spec figures, untimed).
