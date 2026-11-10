# law-blind-unet-trunk: stage 1 pre-registration

Spec: docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md (sections 5, 6, 7; 10a overrides). Registered before launch.

## Lines (copied from the spec)
Stage 1 (audit row 16, the cheaper never-run variant: Exp E frozen, small per-law kernel and pair heads):
- PASS: gravity err@20 at most 1.05x Exp E AND the other laws within 2x their single-law comparators.
- KILL: gravity err@20 above 1.15x Exp E.
Full run (not authorised here): PASS gravity err@20 at most 0.0509 on seeds 4738 and 4739 AND no dead decoder level AND every trained law's tick-1 force within 1.25x of main_s4739. KILL gravity above 0.106. Fails the pass line: 0.07.

## Control and checkpoints
- Gravity control: Exp E, `augur_nfr/kern/E_frozen.pt` (verified, 3200305 B).
- Multi-law reference: `augur_nfr/ckpt/main_s4739.pt` (verified, 3659299 B).
- Single-law comparators (verified): `augur_nfr/ckpt/cmp_pow1.5_single`, `cmp_yuk32_single` (held-out laws pow1.5 and yuk32 only).
- Verified files: cmp_pow1.5_single.pt (3661021 B), cmp_yuk32_single.pt (3660775 B), main_s4739.pt (3659299 B), each with a .done marker.

## What is run
- Model: MultiScatterField with `--law-blind` (the law vector theta is zeroed on the U-Net input channel; kernel MLP and pair MLP still receive theta) and `--freeze-trunk` (U-Net weights not trained). Initialised from Exp E with init_from_e. Trained: kernel MLP and pair MLP heads, width 128 (the existing size; fixed, not tuned).
- Budget: 80000 iterations, the control's (main) count; same data and loss recipe as main (train_scatter_multi.py defaults), seed 4738.
- Validation: every 1000 iterations, 24 scenes per shape, k=10 rollout, saved in the checkpoint and .done file.
- Eval: scatter_multi_eval on seeds 9100/9200/9300, 48 scenes per case, cases grav, pow1, pow3, yuk8, yuk128, lj, HELD_pow1.5, HELD_yuk32, for candidate, main_s4739, comparators, and E on grav.

## Seeds and reality rule
Eval seeds 9100/9200/9300; fresh 12000/12100/12200 before a result is called real; second training seed 4739 only after a pass.

## Tie-break rule
No argmin selection. The reported checkpoint is the final one (iteration 80000); no best-of selection. Equal values: the lower seed index is reported first.

## Ratio lines against the reference noise floor
Gravity ratio = mean over seeds of candidate err@20 / E err@20 per seed, printed with E's own spread (max - min) / mean. Held-out ratios use force_rel against the matching comparator, with the comparator's spread. A ratio inside the spread is stated as such.

## Comparator gap (stated, not hidden)
Single-law comparators exist only for held-out pow1.5 and yuk32. For the trained laws pow1, pow3, yuk8, yuk128, lj no single-law comparator exists, so "other laws within 2x their comparators" can only be scored on the two held-out laws. The trained laws are reported against main_s4739 as information, not as the line. Proposed amendment: train single-law comparators for those five laws (needs-user; not done).

## Prediction (stated before launch)
Gravity err@20 ratio vs E about 1.0 to 1.1 (the trunk is E's own, the heads start from E). PASS on the 1.05 line is a coin flip; KILL above 1.15 is unlikely (about 15%). The held-out 2x line is at risk because the heads see theta only through two small MLPs and held-out laws sit between trained ones.

## Plateau guard (section 10a)
INCONCLUSIVE-undertrained if the mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window. The iteration count stays the control's (80000). One extension at double length for candidate and control together is allowed. Caveat: main_s4739 has no validation history (no hook in that run), so the extension would need the control rerun with the new hook; flagged as needs-user.

## Constraints check (spec section 5)
- pp=2, knn=16: unchanged. Satisfied.
- No hard-wired conservation: none. Satisfied.
- One change against the frozen control: theta withheld from the U-Net, trunk frozen. These are two coupled switches of one stage-1 variant as written in the spec (frozen trunk with per-law heads). Flagged, not separated.
- Matched budget: 80000 iterations. Satisfied.
- No sweeps. Satisfied.
- Seeds: 4738 first. Satisfied.
- At most 2 GPUs: one GPU job.
- Over 1 h: full-state checkpoint every 500 iterations, --resume, SIGTERM saves. Satisfied.
- Diagnostic render plus fresh reviewer after the job: the supervisor runs them.
- Requirements Gen (generalises across laws), Int: stated in the spec.
- Names: descriptive kebab-case.

## Cost
Spec 3.5 GPU-h. From the logged rate (job 7296: 21214 s for 80000 iterations, about 0.26 s/it, 5.9 GPU-h) my estimate for a frozen trunk is 4.7 to 5.9 GPU-h. Untested; no timing job was run. Script requests 8 h.

## Job
scripts/polaris_law-blind-unet-trunk.sh. Full run draft: scripts/polaris_law-blind-unet-trunk-full-run-draft.sh (not authorised).

## Caveats (from the fresh code review)
- "Law-blind" means theta is zeroed on the U-Net's own channel only. The U-Net still receives the kernel-field input `ak`, which is computed by the kernel MLP from theta, so law information can still reach the trunk through `ak`. Stage 1 tests the spec's variant as written.
