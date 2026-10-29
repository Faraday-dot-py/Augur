# Neuron-graph loss: theory notes (2026-10-29)

Current loss (scripts/train_neuron_graph_smoke.py:68-82, on branch neuron-graph-smoke-test): per step, MSE on channels 0-2 (voltage, adaptation, synaptic state) + 0.5 x BCE on the spike logit with SPIKE_POS_WEIGHT, averaged over sequence length; teacher-forced stage then short rollout-in-loop stage. Never compared against an alternative loss.

Concerns (untested):
- Spike is a threshold function of voltage in the teacher, so BCE and voltage MSE are partly redundant and can compete near threshold/reset.
- Per-step MSE regresses toward the conditional mean; under chaos that favours smooth, low-rate predictions (consistent with firing-rate error barely beating silence).
- Teacher chaos limit 77-104 ticks vs learned 13-22: trajectory matching is ill-posed past the predictability horizon.
- Evaluation metrics (spike F-score within +-2 ticks, spike-time error) are not what is optimised.

Candidate directions: timing-tolerant spike loss (smoothed / van Rossum-style) matching the F-score; horizon-capped multi-step loss; statistics-matching term (rate, ISI, pairwise correlation) for horizons past chaos, same spirit as energy-term = similarity to truth; probabilistic outputs with NLL. Per-channel breakdown of the current loss is the missing diagnostic.
