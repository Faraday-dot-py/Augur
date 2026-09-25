# Core requirements

1. Generalization: the method must work with any particle sim, including ones governed by different rules. No sim-specific traits baked in (audit: sim-specific-assumptions.md).
2. Tileable: a single model must simulate a larger area than it was trained on.
3. O(N) or less time complexity.
4. Interpretability: the underlying rules of the sim must be recoverable from the model. UNTESTED.

Applied to decisions: a LUT force table (1D central-force assumption) is not the main method; see debugging/lut-adaptive-substep-benchmark.md.
