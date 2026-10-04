# Scatter-field OOD stress (plan agent 7), 2026-10-04

Zero-shot, one axis at a time, Exp E checkpoint (`~/bounce/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt`), inference flags `kcache,fastio,cl,graph` (scatter-opt). Tool: `scripts/scatter_ood.py`, Polaris wrapper `scripts/polaris_scatter_ood.sh`, results `results/scatter_ood.json`. Research only, no model code changed.

## Training distribution (what "in" means), from code
Exp E: unit masses only; N 10-200 uniform box (spread 5 sqrt(N/8), v ~ N(0, .5 (N/8)^.25)) plus 50% cold Gaussian clusters (N 80-300, sigma 5-10, per-component speed vfac sqrt(N/(4 sigma)), vfac .1-.6), Newtonian, eps .5, dt .1, extent 64 (grid 128, h .5), open boundary. Fine-tuned on 50-100-step windows. BH eval (c=10, N=300, vfac .3) is relativistic but never trained on it (model is fed v(p)).
Architecture facts relevant to OOD: mass enters only as the scattered density m and m*v (U-Net input, not normalised), the learned isotropic kernel K(r) (acceleration linear in source mass), and the pair term (linear in m_j); `momfix` removes mass-weighted net force. No eps, dt, c or charge input. Tokens outside +-32 get zero mesh force (pair term still acts). No wall/periodic support.

## Method
Per point 4 scenes (seeds 9100/9200/9300/4738), 10 time units. Truth: float64 KDK, h=.025 (4 substeps at dt .1), relativistic momentum state when c given, masses supported (checked against `gs.rollout_torch`). err@t = mean per-body position error at t = .5/1/2/5 (ticks 5/10/20/50 at dt .1; err@10 saved but unreliable: chaos ceiling). Floor = truth with 1e-5 x rms-radius position perturbation. Ballistic = constant-velocity coasting. dE/E_scale = max_t |E-E0| / (|T0|+|U0|), also truth's own. dP, dL normalised by sum m|p|, sum m|r||p|. "t>1" = first time err exceeds 1.0 (2 eps). Reference in-distribution scenes: Gaussian N=100 sigma 8 vfac .3 (Newtonian) and BH N=300 c=10.

## Predictions (written before any run)
Ranked expected severity, most to least:
1. softening eps in truth (model has eps=.5 baked into K and pair net): err grows with |log(eps/.5)|, but only for scenes with close encounters; cluster collapse breaks at eps .05-.1 (err@2 > .3), eps 1-2 mild. Architectural (no eps input) / data coverage.
2. mass ratio / dominant mass: 1e3:1 breaks. Reason: U-Net input scale (m up to 1e3 per cell vs ~1 in training) and heavy-body near field (pair net saw only m_j=1). Predict equal-mass scaling m=.1-.3 and 3 mildly worse (input scale 10x), m>=30 and heavy >=100 break (err@2 >> floor, ballistic-level by t~2-5). Heavy body error itself will be small (it barely moves); light-body errors dominate. Mostly data coverage (force is linear in mass by construction), fixable by fine-tune with mixed masses; U-Net input normalisation would be the architectural fix.
3. close passes / tight binaries d <= .25 (inside softening core, orbital period ~ dt scale) : orbital phase error grows linearly, err@5 ~ 1 at d .125-.25, fine for d >= 1. Predict tight binary d<=.25 breaks, hierarchical triple with inner a=.2 breaks. Data coverage + integrator (dt .1 vs period).
4. dt: model learned dt .1 Verlet compensation. dt .05 and .2 roughly 2-5x worse err@2 than reference; dt .4 breaks (err@5 ~ ballistic), dt .025 also degrades (recurrent field input out of regime). Architectural-ish (dt baked in via integrator compensation and the recurrent field).
5. boundary/offset: cluster offset by >= 24 puts mass off-grid: mesh force zero for off-grid tokens, err explodes once > ~30% of mass is outside; shift 8-16 fine. Hot clusters (vfac >= 2.5) expand out of the grid by t~3-5 and lose the mesh force: err grows once escapers cross +-32. Architectural (fixed domain, open boundary = zero force, not a monopole tail).
6. velocity scale: zero-velocity cold collapse (vfac 0) is a harder collapse than any training (trained vfac >= .1): err@5 several x reference; hot (vfac 4-8) is easy dynamics (nearly ballistic): small err because ballistic coasting is accurate until escapers leave the grid. Relativistic c: c >= 10 as reference; c=3-5 with N=300 (R_s > cluster) hard because truth is far from Newtonian; c>=30 behaves as Newtonian = fine.
7. density: sigma 1-2 (denser than training min 5, N=100): breaks (collapse timescale ~ 1 time unit, pair near-field and grid h=.5 resolution); sigma 16-32 sparse: easy (nearly ballistic). Plummer a=.25-.5 (core below grid resolution) degraded; a>=2 near equilibrium, long-lived: fine, probably better than cluster collapse.
8. IC shapes: shell/disk/two-cluster moderate: err within 2-3x of reference (they are still gravity with N 100, unit mass); shell collapse at rest (focusing) is the worst (violent crossing), filament next.
9. float precision: fp32 vs fp64 model: identical to within the floor at t<=2 (no sensitivity); tf32/bf16: bf16 worse (~2x err@2), half16 ~ +1.5% as in scatter-optimization.md. Not a failure axis.
Non-finite: predicted none anywhere except possibly mass 1e4:1 and eps .05 (large forces); the model output is a bounded U-Net + MLP so blow-up risk is low.

## Results
(filled below after the run)
