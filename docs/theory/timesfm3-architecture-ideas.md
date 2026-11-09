# TimesFM-3 architecture ideas for Augur (2026-10-29)

Research only. Nothing here was implemented or run. Every TimesFM claim is tagged READ (with URL, content seen via WebFetch of that page) or INFERRED. Anything not found in a source is listed as "not stated", not filled in.

Abbreviations: TimesFM = Time Series Foundation Model (Google Research); HF = Hugging Face; CPM = Contiguous Patch Masking; RevIN = reversible instance normalisation; TiRex = an xLSTM (extended long short-term memory) forecaster, the source of CPM; GRU = gated recurrent unit; BL = tiled baseline; dE@100 = relative energy drift at tick 100; err@k = mean error after k autoregressive ticks; F-score@k = spike F-score (within +-2 ticks) over a k-tick rollout; BCE = binary cross-entropy; NLL = negative log-likelihood; MSE = mean squared error; O(N) = cost linear in body/cell count.

## 1. Verified facts

### Existence
TimesFM-3 exists. Google Research blog (READ): https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/ . HF model card (READ): https://huggingface.co/google/timesfm-3.0-pytorch . GitHub README (READ): https://github.com/google-research/timesfm , which dates 3.0 to "August 2026". No dedicated TimesFM-3 paper was found or linked from the blog, card or README; the card cites only the 2023 paper arXiv:2310.10688 ("A decoder-only foundation model for time-series forecasting", ICML 2024) without saying 3.0 is described there. Search-result snippets only (pages not opened): Google Cloud BigQuery post https://cloud.google.com/blog/products/data-analytics/bigquery-timesfm-3-0-with-multivariate-time-series-forecasting , MarkTechPost https://www.marktechpost.com/2026/08/31/google-ai-releases-timesfm-3-a-330m-parameter-zero-shot-foundation-model-for-multivariate-time-series-forecasting/ ; third-party blogs (mindstudio) not used, they add unverified details.

### TimesFM-3 (blog + card + README)
| Item | Finding | Source |
|---|---|---|
| Backbone | decoder-only transformer, builds on predecessors | blog READ |
| Size | 330M parameters (blog); card: 0.3B, 20 layers, model dim 1280, 16 heads | blog, card READ |
| Card architecture string | "Stacked Mixing Transformer with Variate Attention and CPM Iterative RevIN" | card READ |
| Patching | 32 time steps per patch; target and past-covariate series one patch per token; context patch length 32, forecast horizon patch length 64 | blog, card READ |
| Input stage | an input residual block precedes the transformer stack | blog READ |
| Multivariate | main stack is a grid of alternating layers: causal temporal attention (a token sees only earlier tokens of its own series) and full variate attention (a token sees all series at the same time step). Number of alternating layers: not stated | blog READ |
| Covariates | past covariates (history only) and past-future covariates (future known); the latter use a "lookahead" token concatenating the current patch with future patches | blog READ |
| Decoding | non-autoregressive: whole horizon in a single forward pass via CPM, with masked placeholder tokens; target and past-covariate horizons masked, past-future covariates visible | blog READ |
| Outputs | 9 quantiles (0.1 to 0.9) per target per horizon step; point forecasts supported (how derived: not stated); card: median at quantile index 4 | blog, card, README examples READ |
| Context | README: contexts longer than `global_context` (15,360) are truncated to most recent points; maximum otherwise not stated | README READ |
| Horizon | no maximum stated; README examples use 12-128, and horizons of 128 or more span several output patches | README READ |
| Normalisation | per-series, "similar to that of TimesFM-2.5"; card name says "CPM Iterative RevIN"; mechanics of the iterative part: not read | blog, card READ |
| Pretraining data | blog: more than 1 trillion time points, real-world plus synthetic; card: GiftEvalPretrain (minus fev-bench overlaps), Wikipedia Pageviews (cutoff Nov 2023), Google Trends top queries (cutoff end 2022), synthetic and augmented data. Mix ratios: not stated | blog, card READ |
| Loss | not stated for TimesFM-3 | blog, card READ |
| Positional encoding | not stated | blog READ |
| Scaling behaviour | not stated | blog, card READ |
| Inference cost | blog claims lower latency and no compounding error (no loop); no numbers in text (a chart exists, unreadable) | blog READ |
| Benchmarks | top average rank among pretrained models on GIFT-Eval, fev-bench, TIME, per Google's own reporting; compared with Chronos-2, Toto 2.0 family, TimesFM-2.5 | blog READ |
| License | weights non-commercial (separate from Apache-2.0 code) | card, README READ |

### CPM source (not a TimesFM paper)
The blog credits CPM to arXiv:2505.23719 (READ abstract page https://arxiv.org/abs/2505.23719 and HTML https://arxiv.org/html/2505.23719v2 up to character 100,000 of 148,706), which is the TiRex paper (Auer et al., NeurIPS 2025). From it (READ): CPM "randomly masks full and consecutive patches in pre-training"; masked patches enter as missing values with a binary mask indicator concatenated before the input residual block, so training and multi-patch inference see the same input form; this lets "internal memory" propagate predictive state across patches instead of feeding point estimates back; per-output-token nine-quantile pinball-style loss, "missing values in the output window are ignored for the loss". Max run length and masking probability of the final model: not read. This describes TiRex; whether TimesFM-3 uses the identical recipe is INFERRED from the blog naming CPM.

### Earlier versions (README READ)
- 1.0 and 2.0: code archived in `v1`; the only figures given are 2.0 = 500M parameters, 2048 context. Paper arXiv:2310.10688. No other 1.0/2.0 architecture detail was read.
- 2.5 (released 2025-09-15): 200M parameters, up to 16k context, continuous quantile forecast up to 1k horizon via an optional 30M quantile head, frequency indicator removed; covariate support (XReg) added back 2025-10-29. Multivariate: not stated. All 2.5/3.0 checkpoints were univariate before 3.0 (from search summary of the blog; not re-verified in the page text read).

## 2. What transfers and what does not

Does NOT transfer:
- Univariate/multivariate series forecasting is statistical pattern extrapolation learned from 10^12 points. Our simulators are small models of a known, local, mechanistic update rule; the training signal is exact one-step dynamics, not a corpus.
- Full variate attention is O(V^2) in the number of series. Our requirement is O(N) and tileable, so any cross-entity mixing must stay graph- or halo-local.
- One-shot 64-tick output without feeding back. For particles, Verlet integration with a force evaluated each tick is the physical structure (energy, momentum conservation, halo exchange); emitting several ticks from one state breaks dt-consistency, the potential-gradient force head and the momentum fix. Not recommended for the tiered particle model.
- Pretraining-scale zero-shot generalisation and the data mix: no analogue; our generalization requirement is across force laws and system sizes, not domains.
- Per-series normalisation as done for time series assumes one scale per series; particle systems need physically consistent units (momentum, energy) across tiles.
- The known-future-covariate lookahead token: the neuron teacher's external drive is already given to the model each tick (`model/neuron_graph.py` step takes `ext`; alignment with teacher indices not checked), so nothing to add (INFERRED).
- TimesFM tells us nothing about particle energy drift. The tiered drift problem is a visited-state effect already addressed by visited-state-label-replay-tier-0 (supervisor-log 10-29 16:00: PASS on all pre-registered lines, dE@100 1.78-2.29x BL against a 2.5x line). Honest read: TimesFM contributes ideas to the neuron-graph problem; for the tiered particle model only the two weakest ideas apply.

## 3. Ideas, ranked

Sources of "already tried": `docs/debugging/experiment-log.md` (particle/grid history), `docs/theory/supervisor-log.md` (tiered), the neuron entries on branch `neuron-graph-smoke-test` (its experiment-log, entries dated 2026-10-29), `docs/theory/tiered-research-directions-2026-10-29.md`. Scorecard letters: Gen = generalization, Til = tileable, ON = O(N), Int = interpretability. Baseline numbers (neuron, full-synaptic-messages arm, N=10 / N=100, seed 4738): err@20 9.03 / 10.26 mV, F-score@20 0.81 / 0.79, F-score@100 0.42 / 0.36, divergence horizon 22.6 / 20.7 ticks (teacher predictable 77-104), per-cell rate slope at 300 ticks 0.15 / 0.03, rate correlation 0.24 / 0.04. One seed; all ideas below would need seeds 4738-4740 only if they pass on 4738.

### Rank 1: slow-patch-rate-token-with-patch-count-head (neuron graph)
- TimesFM does: groups 32 ticks into one token and predicts at patch granularity (blog READ); patch-level targets are timing-tolerant by construction.
- Mapping: add a slow per-cell token updated every 16 ticks from a summary of that patch (spike count, mean voltage, mean synaptic state) through a small residual MLP; it feeds the per-tick GRU as an extra input and also drives an auxiliary head predicting the next patch's per-cell spike count (loss: count MSE or Poisson NLL; trained teacher-forced and in the rollout stage). Per-tick spike emission and the BCE with positive weight 5 stay unchanged (baseline's BCE calibrates the 0.5 threshold, per log: it was what the three killed loss variants lacked). This is a temporal tier, matching our spatial-tier idea.
- Requirements: Gen neutral; Til neutral (per cell); ON yes (O(cells + edges), slow token is 1/16 extra compute); Int helps (the slow token is an inspectable per-cell rate state).
- Duplicates? Partly adjacent to statistics-matching-rate-and-correlation (KILL: rate collapse 0.01-0.03, divergence horizon 17-18, because the term acted on probabilities not emitted spikes) and to smoothed-spike-timing-tolerant-van-rossum (KILL, silent). Differences: architecture plus a count target, loss on emitted-spike counts not smoothed probabilities, BCE retained. Not a loss swap, so it respects the prefer-architecture rule. Also adjacent to the 10x-training finding that rate error doubles (rate identity, not step accuracy, is the failing quantity).
- Expected gain: rate slope at 300 ticks from 0.03-0.15 to above 0.3; little change in divergence horizon (a per-step error limit). Probability of passing: about 30%.
- Cost: about 0.3 GPU-h per arm at the existing 1500+1500-iteration budget (3-min arms); one new module, about 60 lines.
- Pass line (fixed now): full arm per-cell rate slope at 300 ticks at least 0.3 and correlation at least 0.3 at both N; predicted mean rate within 0.7-1.3 of true; err@20 no more than 3% worse; F-score@20 no more than 0.03 lower. Kill: slope below 0.2 at either N, or rate ratio below 0.5.
- Sequencing caveat: the pre-registered calibrated-emission follow-up (job from the neuron architecture/emission entry) had no result at the time of reading; read it first, since it decides whether emission calibration alone moves rates.

### Rank 2: masked-block-multi-tick-readout (neuron graph)
- TimesFM does: fills a 64-step output patch in one pass from masked placeholder tokens, no feedback inside the block (blog, card READ); CPM trains with contiguous masked runs plus a mask-flag input so train and inference match (TiRex READ).
- Mapping: head emits the next B ticks (B = 8) of voltage and spike logits from the token state, with the same-tick drive as the only input inside the block; the rollout feeds back every B ticks, so 3 feedback loops reach tick 24 instead of 24. Keep the one-tick head as an auxiliary (B = 1) so the per-tick rule remains inspectable. Mask-flag input marks "no observed state this tick".
- Requirements: Gen neutral; Til neutral; ON yes; Int partial violation (the within-block rule is not a one-step mechanism; the B = 1 head mitigates).
- Duplicates? Closest are rollout-in-loop stage (already in baseline) and horizon-capped multi-step loss (dropped untested as tuning-only). This is an output-structure change, not a horizon hyperparameter. No exact prior.
- Expected gain: fewer compounding steps, but one-step error is about 4x the chaos floor (4.3 vs 1.1 mV at k=1), so independent errors inside a block may stay as large. Probability of passing: about 20%. Neuron data show errors saturate at the memoryless MLP level by k=20, which argues for modest gain.
- Cost: about 0.4 GPU-h; moderate code (new head, block-aware training windows).
- Pass: divergence horizon at least 30 ticks (+30%) at both N with err@20 no worse than +3% and F-score@20 within 0.03. Kill: horizon below 25 at either N.

### Rank 3: windowed-causal-temporal-attention-memory (neuron graph; optional tier-0 variant)
- TimesFM does: alternating causal temporal attention (own series, past only) and variate attention (blog READ).
- Mapping: replace the GRU token with causal attention over the last W = 32 ticks of the cell's own (state, spike, drive) tokens; replace full variate attention with the existing graph-masked message sum (O(edges)). Alternate the two for 2-3 layers.
- Requirements: Gen neutral; Til ok if variate step stays graph/halo-masked; ON yes (O(N x W)); Int worse than a single GRU, the attention maps are inspectable, the recurrent state is not.
- Duplicates? Not for neurons. For the grid/particle side, multi-frame history was tried and regressed (3-frame history: moire, step-1 1.29x; richer-history left onset unchanged, experiment-log 2026-09-22/24) and removing velocity and recurrent inputs hurt tier 0 (1.56-1.85x, supervisor-log 10-26 18:10), so history matters but this exact form is untried.
- Expected gain: modest; spike-tick offset is a message-sum scale effect, not memory. Probability of passing: about 15%.
- Cost: about 1 GPU-h; largest code change of the neuron ideas.
- Pass: err@20 at least 5% below baseline and divergence horizon at least 15% longer at both N. Kill: err@20 within 3% of baseline.

### Rank 4: quantile-pinball-bimodal-voltage-head (neuron graph)
- TimesFM does: 9 quantiles per step, trained with a quantile loss (loss itself READ only for TiRex; TimesFM-3's loss not stated).
- Mapping: voltage head outputs quantiles 0.1-0.9 for the next voltage; point value is the median; the near-threshold regime is bimodal (reset about -61 mV versus continuing), which a single Gaussian cannot represent. Spike logit unchanged.
- Requirements: Gen/Til/ON neutral; Int neutral.
- Duplicates? Close to gaussian-nll-voltage-head (KILL: err@20 +1.7% / -3.3%, F-score@20 -0.09 at N=10, rate error up). Differs only by representing bimodality; the bias source found later (message sum adds +26-31 mV at spike ticks) is architectural, so a better likelihood is unlikely to fix it.
- Expected gain: under 3%. Probability of passing: about 10%.
- Cost: 0.1 GPU-h.
- Pass: err@20 at least 5% lower at both N without F-score@20 dropping more than 0.03. Kill: within 3% of baseline. Run only if rank 1 or 2 shows hedging at threshold is still the dominant error.

### Rank 5: per-scene-input-normalisation-for-dense-generalisation (tiered tier 0)
- TimesFM does: per-series normalisation (blog READ) with an "iterative" RevIN variant (mechanics not read).
- Mapping: normalise tier-0 inputs by a per-tile density/length scale and rescale outputs, aimed at the dense-evaluation gap (eval N 727-1098 versus training quantiles up to 273, tiered-research-directions item 5).
- Requirements: Gen helps (scale-free inputs); Til neutral if the scale is computed per tile plus halo (tile-consistent scales needed, risk at seams); ON yes; Int neutral.
- Duplicates? Substantially: the scene-constant geometric-mean mass unit in `docs/theory/mass-conditioned-scatter-field.md` (`norm="gmean"`, untrained) and the pending dense-core-refinement-residual-tier. Also scatter-review fixes for scaling/out-of-distribution already failed the 15% gate (seven fixes).
- Expected gain: small, unrelated to the drift gap. Probability of passing: about 10%.
- Cost: about 3 GPU-h (tier-0 fine-tune).
- Pass: dense-eval err@20 at most 1.15x BL with sparse-eval err@20 within 1.10x of e44b796. Kill: dense err@20 above 1.25x BL or sparse regression above 1.15x.

### Rank 6: random-length-self-feed-blocks-with-mask-flag (tiered tier 0)
- TimesFM/TiRex do: CPM masks random-length contiguous runs so the model trains on the same input pattern it sees at inference (TiRex READ).
- Mapping: in closed-loop fine-tuning, replace the fixed 28-tick rollout with randomly sampled contiguous self-fed runs (random start, random length up to 60) with a flag channel; without gradient labelling of visited states beyond what the replay run did.
- Requirements: Gen/Til/ON/Int unchanged.
- Duplicates? Largely: closed-loop tiled tier 0 (28-tick rollouts, e44b796), rollout-in-loop (no gain, supervisor-log 10-26 19:10), stepwise curriculum (no gain, experiment-log v19), and visited-state-label-replay-tier-0 (PASS) already reach ticks 28-100. Included for completeness; it is the lowest-value idea.
- Expected gain: under 5% on dE@100. Probability of passing: about 5%.
- Cost: 1.5-3 GPU-h.
- Pass: dE@100 at most 0.8x the replay result (about 1.6-1.8x BL) with err@20 within 1.05x. Kill: otherwise.

## 4. Which first and why
Rank 1, slow-patch-rate-token-with-patch-count-head. Reasons: (a) it targets the one neuron failure that more training made worse (per-cell rate error 9.7 to 24.0 Hz at 10x training) and that loss swaps could not fix; (b) it is an architectural cause, not a hyperparameter or loss swap, per the standing preference; (c) cheapest meaningful test (about 0.3 GPU-h, one changed module); (d) it keeps the baseline BCE, avoiding the threshold-calibration collapse that killed two earlier arms; (e) its pass lines are measurable on the existing 10-circuit evaluation. Read the calibrated-emission result before launching so the two are not confounded.

Ranks 5-6 apply to the tiered particle model and are low priority; the current tiered plan (diagnoses 1, 2, 4 then fix arms) is stronger evidence-based work than anything TimesFM offers there.

Not claimed anywhere above: that TimesFM's single-pass decoding beats autoregression for chaotic dynamics (the blog claims no compounding error for forecasting benchmarks; chaotic particle/neuron systems were not evaluated).
