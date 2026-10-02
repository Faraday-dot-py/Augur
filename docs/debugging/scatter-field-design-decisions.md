# Scatter-field (particle-in-cell, learned gravity) design decisions

Source: user (awebb) words only. Sessions: A = 4b1db5e9 (design + first experiments, 2026-10-02), B = 91b161f0 (overnight handoff, 2026-10-02). Assistant proposals not confirmed by the user are in section (c).
Other recent transcripts (aaf27cff, ebfd094f, 38e38eed, f177f127 and older) contain no scatter-field design statements by the user.

## (a) Explicit decisions

### Motivation / medium
- Replace CFD/LJ architectures with a kernel/field (medium) method. Argument: information must propagate at finite speed; all-pairs attention violates this. "the planet will move distance/speed of light seconds after the star disappears" (A #8/#39/#69).
- Use that causal argument to avoid O(N^2): "information propigation is spread out over several steps" (A #69).
- First test: 2-body sim ("Have an agent try it with a 2-body sim", A #99). Delete-the-star is the probe.

### State / tokens
- Tokens carry {x, y, dx, dy, m}. Each step tokens are scattered onto the x/y plane of the field and become {dx, dy, m}; x, y are spatial. (A #119/#128/#157/#178)
- Model outputs a new diff field of {ddx, ddy, dm}; deltas are added to the matching terms; repeat. (A #128)
- Energy is not a token trait: "If energy isn't a trait, then we don't store it in the token and we let the emergent dynamics of the sim/model represent it." (A #190)
- Mass-like conserved traits: dm initialised to 0, not random: "dm should be instantiated as 0 instead of random" (A #128). Clarified as zero-init of the head: "Yes, I was talking about a zero-init." (A #190)

### Field net and persistence
- The diff field input is persistent state: "The diff field input is the last one, a sort of persistent state." (A #190)
- Adaptive/hierarchical grid, not a full grid: "don't want to be computing the field for space that doesn't have anything in it." (A #190)

### Sub-cell resolution
- Accept softening at h_leaf (gravity is softened at the finest leaf size). "Accept softening at h_leaf, go with A+B" (A #197).
- A = moments/Taylor gather (scatter m, dipole, quadrupole; field gives value + 2x2 gradient; token acceleration = a0 + J(x - x_c)). B = adaptive refinement until each leaf holds <= k tokens, with a depth cap. (A #197; option definitions came from the assistant, selection is the user's)
- Not chosen: C as stand-alone (only as the stated resolution limit) and D near-field token-token pass (skipped).

### Scope
- Gravity only; "We're not focusing on CFD/LJ at all right now." (A #190)

### Conservation (see also b)
- Conservation encouraged via training data/loss, not hard-baked into the architecture: "we can encourage energy/mass conservation in the training data and architecture of this without hard-baking it into the architecture." (A #128)
- Energy may move between channels (e.g. heat to luminosity) as additional channels. (A #128)

### Training / eval / process
- Run the Polaris tests. "Run the polaris tests" (A #243).
- Lockout override for the Polaris auth failure was explicitly requested once: "Try it again, override the lockout." (A #143; the sub-agent declined it because the user did not make the request itself)
- Overnight (B #64, #76): iteratively refine the architecture; first have a subagent extract these design decisions to a file; keep fresh notes and delete stale ones ("don't hold onto information you don't need").
- Duo: "Send the duo request now, I'll approve it." (B #76)
- Success bar confirmed: err@5/10/20 against the CentralForceDynamics baseline, step-100 stability, curl, self-force, momentum/energy drift, held-out seeds. "That's correct." (B #76)
- Goal: "Try to get it past CFD if you can. Keep it within the 100-step limit." (B #76)
- Design decisions are NOT fixed (B #76, supersedes the earlier implied lock): "you must not make a decision unless the data backs it up." Log each change. Keep following improving trends, do not abandon them.
- "Avoid parameter sweeps when you're trying to solve a problem." Allowed only when right (obvious OOD, robustness). Saved to memory. (B #76)
- Assistant plan items confirmed by "The rest of your assumptions here are correct" (B #76): spawn the subagent first; unbiased subagent review of rollout renders after each change; one Polaris GPU job at a time with periodic full-state checkpoints and seed 4738; one live notes file; commit scripts + results after each experiment with the key metric; do not push; stop only on a real blocker; 200-400 word morning report.

## (b) Preferences / constraints stated

- Drift is acceptable: "I'm okay with drift over long rollouts, as long as it survives for ~100 frames." (A #190)
- Interpretability is a core reason: "You can trace x straight to dx, and see causal ablation's effects per-channel." (A #178)
- No hard-baked conservation; soft/learned (A #128).
- Zero-init of dm: it can be trained out ("We could also always train this out of the model") and the user floated a learned kernel as an alternative ("Would that be better represented with a learned kernel then?"). Assistant answered keep zero-init heads, no special kernel; user did not object. (A #190)
- Hard-zeroing dm: user prefers not to, but allows if needed: "I'd rather you didn't, see if you can find a way around it. If you need to, go for it." (B #76). This is the latest position and stands in place of the earlier assistant default of hard-zero.
- Define acronyms before using them in messages (aaf27cff #12, same day; applies to reports).
- CFD/LJ: parked, not to be worked on now (A #190).

## (c) Left open / assistant proposals NOT confirmed by the user

- Local-only receptive field vs U-Net/FNO (causality): assistant said it would default to local stencils per level with hierarchy for long range; user did not answer. Results later show global-receptive-field nets react in 0 steps (acausal). Causality of the hierarchy (coarse levels updating every 2^l steps) was deferred ("Fine to defer"), user never said whether causality is a goal of this run.
- Per-token gather MLP for contacts, and first benchmark choice (2-body delete-the-star vs bounce-ball contacts): asked, not answered directly (2-body was started earlier by the user at #99).
- Adjoint CIC scatter/gather (same weights both ways), explicit x += dx*dt kinematic position update, semi-implicit gather position, multi-step BPTT (5-20 step) rollout loss, self-force probe, curl probe, separation sweep d/h_leaf 0.25 to 16: assistant proposals; the sweep was relayed and run, the rest unconfirmed by the user.
- Block-sparse multilevel 8x8 tiles, prolongation init of new tiles, depth cap, k_leaf: assistant's design (shape given in A #194, user said "go with A+B" only).
- Explicit energy / luminosity channel `e`: assistant suggestion; user said no energy trait (A #190).
- Momentum conservation losses and the momentum fix: assistant's.
- Post-result next steps (CIC deposit with smooth blend between leaf levels, learned pairwise near-field correction, longer training of multilevel): assistant recommendations; user only asked what the blend means (B #56) and did not approve any.
- Success of the architecture: result so far (A) was baseline CentralForceDynamics beating all scatter variants; the user has not stated a conclusion.

## Reversals / changes (latest kept)

- dm: zero-init (A #128/#190) -> hard-zero proposed by assistant -> user (B #76) prefers not, allowed if needed.
- "Design decisions fixed" implied (A) -> explicitly not fixed, data-gated, trend-aware (B #76).
- Parameter sweeps discouraged for problem solving (B #76).
