# Queued: gravity-direction generalization agent

Launch after the energy-fix agent (worktree branch, docs/debugging/energy-fix-results.md) completes; 1 GPU job at a time and the gravity input would confound the energy result. Fill the implementation-agent-prompt.md template with:

TASK: give the token model a gravity input and test generalization to gravity directions/fields not in training. Build on the best energy-fix branch.
INVESTIGATION_DOC: docs/debugging/energy-fix-results.md, docs/debugging/energy-conservation-investigation.md (y-velocity bias, delta_head dvel_y = -0.099; mirror flag from v20).
FIXES, in order:
1. Gravity vector as model input (per-token or global conditioning), gravity=9 magnitude, direction randomized per-trajectory; train on the four cardinal directions (0/90/180/270 deg via box rotation) and hold out arbitrary angles (e.g. 30, 45, 135 deg) for eval. Check the y-bias disappears and energy behavior is unchanged vs the base branch.
2. Central gravity (toward frame center), held out entirely from training; test zero-shot and after light fine-tune, compare with CentralForceDynamics-style extrapolation.
Report err@5/10/20, energy drift vs truth, y/x velocity bias, per held-out direction. Note: the walls/floor stay fixed; only the field direction changes, so wall physics must still hold.
