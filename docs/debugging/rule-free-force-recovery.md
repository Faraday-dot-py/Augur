# Rule-free force recovery (uncommitted)

Verdict, pre-registered lines applied unchanged: level two (discovered formula vs truth) KILL; level one (measured curve vs truth) INCONCLUSIVE_PARTIAL. Fresh reviewer reproduced every number. Pre-registration: docs/debugging/experiment-log.md, 2026-11-08 rule-free-force-recovery entries.

## What was done
- Job 7190 (one GPU, 37 min, about 0.62 GPU-h): radial force curves from ckpt/main_s4738.pt and ckpt/main_s4739.pt for pow1, pow1.5, pow2, pow3, yuk8, yuk32, yuk128, lj (pow1.5 and yuk32 are held out of training). Probe: two bodies at separation d, warmed field, 1024 placements, median.
- Discovery: 18-candidate library (six bases x three wrappers), differential evolution seed 4738, chosen by BIC with a fewest-parameters tie-break, fitted on 56 distances, scored on 55 held-out distances against exact fp64 force. Truth-blind: nothing from the truth law enters selection.
- Floor: same pipeline on the exact curves (results/rule_free_force_exact.json).

## Floor (pipeline on exact force)
Exact formula recovered for pow1, pow1.5, pow2, pow3, yuk8, yuk32, lj (saturated difference of d^-7 and d^-13). yuk128 comes back as a softened power with exponent 2.04: its range is unidentifiable at 1% resolution (error 0.0064). Denominator for every ratio is 0.01. Validity gate 8 of 8.

## Model results
Level two: 2 PASS (4738 yuk128, 4739 pow2), 6 MARGINAL, 8 KILL of 16 cells, which meets the pre-registered KILL line (8 or more). Level one: 5 PASS, 9 MARGINAL, 2 KILL (4738 pow1.5 ratio 14.4, 4739 pow1 ratio 20.3).

| Law (seed 4738) | Discovered expression, in words | Ratio to floor |
|---|---|---|
| pow2 | softened power, exponent 1.96, softening 0.46 (truth 2, 0.5) | 4.0 marginal |
| pow3 | softened power, exponent 3.09, softening 0.53 | marginal |
| pow1 | saturated flat term plus d^-2.5 (wrong family) | 18.6 kill |
| pow1.5 (held out) | saturated softened power, exponent 1.9 (truth 1.5) | 28.5 kill |
| yuk8 | softened power times exponential, range 10.6, exponent 2.2 (truth 8, 2) | marginal |
| yuk32 (held out) | range 14.6, exponent 1.88 (truth 32); unstable across seeds | 6.4 marginal |
| yuk128 | plain softened power, exponent 2.02 (range unidentifiable) | 1.5 pass |
| lj | saturated two-power, exponents about 14.9 and 5.8, cap 15.7 (truth 13, 7, 20) | 107.7 kill |

Seed 4739 kills: pow1 (29.0, power times exponential, exponent 0.29), pow1.5 (20.3), pow3 (122; blown-up softened power, exponent 14.1), yuk32 (10.6, borderline), lj (125.1).

## Unidentifiable at sampled resolution
yuk128 (range beyond the probed distances; exact-run tied set too), yuk32 range (flagged on 4738), lj (flagged on 4738), 4739 pow2, 4739 yuk8, 4739 yuk128 (replicate or parameter-spread flags). Far-field range of a Yukawa law cannot be told from a pure power law once exp(-r/lambda) is within about 1% over the probed range.

## Why it fails
The reviewer found the fitter is not the cause: refits with a different optimizer seed gave the same expressions and errors. The model curves themselves deviate from truth, mainly for pow3 and lj (e_curve 1.3 for both, against floor 0.27 and 0.17), and for pow1 (0.34 vs 0.017). Where the curve is close (pow2, yuk128) a sensible formula comes out; where it is not, a wrong family is chosen. Held-out laws are not systematically worse than trained ones.

## Prediction check
Predicted INCONCLUSIVE_PARTIAL with 5-7 of 16 level-two PASS. Actual KILL with 2 PASS. Predicted 4738 pow3 PASS, got MARGINAL. Prediction was wrong; lines were not loosened.

## Caveats
- The law descriptor theta is a model input, so the model was told which law it was probing; this is not a blind identification.
- Two checkpoints, one placement-seed family, no second discovery grid.
- Nothing about a menu-free Yukawa lambda is supported: recovered ranges (10.6 for 8, 14.6 for 32) are off.

## Level three (design only, not run)
Not run: level two is KILL, so the result is not clear enough. If revisited after curves improve: train on a law set excluding one family (for example a Gaussian-screened or a log-potential force), probe with theta set to a nearest-neighbor mixture, and require the same pipeline to name a library expression whose held-out ratio is under 3 against the exact floor. The library would need that family held out of it too, otherwise the test is circular.

## Files
scripts/rule_free_force_curves.py, scripts/polaris_rule-free-force-curve-extraction.sh, scripts/rule_free_force_discovery.py, scripts/score_rule_free_force_recovery.py; results/rule_free_force_curves/, rule_free_force_exact.json, rule_free_force_model.json, rule_free_force_recovery_scores.json.
