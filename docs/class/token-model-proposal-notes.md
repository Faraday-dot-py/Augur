# Token model: class proposal notes

Purpose: everything needed to write the 1-page proposal for the applied ML project. Written 2026-10-26. Bullets only.

## 0. Read this first (honesty and policy)

- User confirmed (2026-10-26): submitting pre-existing work is allowed. Still disclose it briefly as prior work and frame the proposal around new semester work (done in the draft).
- Model name to use: "conservative-contact token model" (one state token per ball; learned distance-only pair-force MLP, wall-force MLP, learned uniform gravity, 8 velocity-Verlet substeps; NO attention, no recurrent state). The attention + GRU path (`model/token_free.py::_core`, older "soup B") is a separate, superseded variant used only as a baseline. Do not call the deployed model a transformer.
- Accuracy is real only at small N and short horizon (about 20 ticks). Past about 30-50 ticks positions decorrelate from truth in chaotic contact scenes (expected; the truth sim is chaotic). Large-N results are stability/scale checks. The 1 billion ball run was speed and memory only, on the older soup B model.

## Chosen checkpoint: `results/cons_pure_100.pt` (the checkpoint named by the live site's `weights.json`)

- Family: conservative_contact. Candidates: `results/cons_pure.pt` (trained with scenes up to about 120 balls, n<=200) and `results/cons_pure_100.pt` (retrained with at most 100 balls, 50 epochs, seed 4738, per memory notes; commit fbbe239 on 2026-09-28 is the last commit touching it). Soup B and all attention variants are worse on every energy row (see below), so they are excluded.
- Why `cons_pure_100.pt`: it is the checkpoint the live demo serves (`weights.json` on the site names `cons_pure_100.pt`, checked), and it is the one tested to 50,000 balls. Its err@1/5/10 at 50k balls (0.019/0.107/0.280) are similar to `cons_pure.pt` at 1k balls (0.018/0.106/0.281), but those are different scenes, so the choice is by deployment and test coverage, not by a measured accuracy win. Be honest that the two were not compared head to head on the same small-N seeds.
- Re-read from results JSON this session (no torch locally, nothing re-run):
  - `results/ball_50k_rollout.json` (cons_pure_100, 50,000 balls, box n=2237, 100 steps, from true start): err@1/5/10/20/50/100 = 0.0188/0.1065/0.2799/0.9374/9.32/71.53; kinetic energy per ball at tick 100: 6323 model vs 6433 truth (1.7 percent low); finite for all steps; about 1.2-1.3 s total for model and truth.
  - `results/ball_1k_fix.json` (`cons_pure.pt`, 1,000 balls, 500 steps, soup B for comparison): cons_pure err@1/5/10/20/50 = 0.0178/0.1056/0.2807/1.3828/32.9, never non-finite in 500 steps; soup B 0.0381/0.1904/0.3884/2.1408/56.4, first non-finite at step 75.
  - `results/eval_energy_fix.json` (`cons_pure.pt` vs soup B, 4 balls and 100 balls; see section 5 facts): conservative wins on err@5/10/20, energy at tick 100, restitution and pair rows.
- Not available: a small-N held-out eval of `cons_pure_100.pt` itself in results/. Week 2 of the plan reruns `scripts/eval_energy_fix.py` on it (needs the GPU environment; torch is not installed on this machine).

## 1. Rubric checklist (score 1 needs all four; any missing = 0 and 0 on final project)

| Rubric criterion | Where satisfied in proposal | Check |
|---|---|---|
| Clear ML project idea | Project idea paragraph: learn a particle-contact simulator from simulated trajectories, evaluate vs baselines | one-sentence pitch present |
| Feasible method/pipeline | Methods: GPU-generated truth data, small model (about 9k params deployed), 50 epochs in about 40 min on one GPU, named baselines, metrics, held-out seeds | numbers are from repo (section 5) |
| Clear implementation/code plan | Implementation: built vs new split, 5-week plan with weekly deliverables | new items are labeled "to build" |
| Clear demo plan | Demo: live browser page (exists) with truth-vs-model side by side plus rollout videos plus results table | URL and files in section 7 |
| Also required | Project type line; related work with references; max 1 page, PDF, 11 pt, 1 inch margins | section 9 |
| Meaningful ML | Training a learned force model with multi-step unrolled loss, baseline comparison, ablations | do not present it as only "an engineering port" |

## 2. Project type

- Applied project (build and explain an ML application/system, working prototype, demo).

## 3. Project idea

- Topic: a learned simulator for 2D bouncing balls (walls, ball-ball contact, gravity). Each ball is one token (position, velocity); a small network predicts the next state; rollouts are compared against an exact physics simulator.
- Why interesting (3-5 bullets):
  - Learned simulators can run fast and at large N; this one is O(N) per step by construction (cell-list neighbor graph) and was run to 50,000 balls in the tests listed in section 5.
  - It is trained on at most 100 balls in small boxes and tested on far more balls and larger boxes (generalization in N).
  - Energy is a known failure mode of learned simulators; here the previous attention-based model drifted in energy, and a physics-structured fix (learned potential-derived forces, exact pair momentum) removed most of it. The proposal can test that claim properly.
  - Weights are tiny (8.7k floats), so the whole model runs in plain JavaScript in the browser, with editable weights and per-ball internals visible: a good demo for peers.
  - Honest negative results are documented (many failed variants, chaos floor), which suits a "build and explain" presentation.
- One-sentence pitch: "We build and evaluate a one-token-per-ball learned simulator for bouncing balls that runs in linear time and in the browser, and test it against a physics integrator and a graph-network baseline on held-out scenes."

## 4. Related work (verified by fetching arXiv abstract pages on 2026-10-26)

- Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec, Battaglia. "Learning to Simulate Complex Physics with Graph Networks." ICML 2020. https://arxiv.org/abs/2002.09405. Relevance: the standard particle-graph learned simulator; our GNN baseline follows its encode-process-decode idea.
- Battaglia, Pascanu, Lai, Rezende, Kavukcuoglu. "Interaction Networks for Learning about Objects, Relations and Physics." NIPS 2016. https://arxiv.org/abs/1612.00222. Relevance: pairwise-interaction networks on n-body and collision systems; closest ancestor of the pair-force design.
- Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia. "Learning Mesh-Based Simulation with Graph Networks." ICLR 2021. https://arxiv.org/abs/2010.03409. Relevance: graph-network simulators on meshes; cited as the mesh counterpart (secondary).
- Kipf, Fetaya, Wang, Welling, Zemel. "Neural Relational Inference for Interacting Systems." ICML 2018. https://arxiv.org/abs/1802.04687. Relevance: infers interactions in particle systems; related to the interpretability goal (secondary).
- Greydanus, Dzamba, Yosinski. "Hamiltonian Neural Networks." NeurIPS 2019. https://arxiv.org/abs/1906.01563. Relevance: energy-conserving learned dynamics; motivates our potential-derived forces and energy-drift metric.
- Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho. "Lagrangian Neural Networks." arXiv 2020 (ICLR 2020 workshop on Deep Differential Equations). https://arxiv.org/abs/2003.04630. Relevance: same motivation without canonical coordinates (secondary).
- Vaswani et al. "Attention Is All You Need." arXiv:1706.03762 (NeurIPS 2017). https://arxiv.org/abs/1706.03762. Relevance: attention used in the radius-graph variant. Venue year 2017 from my knowledge; the fetched page showed only the arXiv record.
- Code reference: google-deepmind/deepmind-research, `learning_to_simulate` folder (GNS reference code). https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate. Verified it exists and is the GNS code; the fetched page did not state the framework or license (appears TensorFlow-based; unverified). Do not run it; implement the baseline ourselves in PyTorch.
- Dropped (not verified, do not cite without checking): MeshGraphNets code repo, PDE-Refiner, any "particle transformer simulator" paper, Fourier/neural-PDE surrogates, track-query/MOTR (used in the repo history, not needed).

## 5. Proposed methods / pipeline

Read from code: `model/token_free.py`, `model/token_model.py`, `scripts/train_conservative.py`, `scripts/energy_probe2.py` (truth), `scripts/ball_1k_rollout.py` (truth forces), `scripts/eval_energy_fix.py`, `scripts/ball_50k_rollout.py`, `web/token/README.md`, `docs/debugging/energy-fix-results.md`.

**Data**
- Ground truth: explicit force-based simulator (soft penalty contact, spring-like wall and ball-ball forces with stiffness 400, gravity 9 along +x, ball radius 0.75, dt 0.15, 8 substeps), in 64-bit on GPU (`scripts/energy_probe2.py::truth_states`; same forces as `augur.py`).
- Training scenes are generated on the GPU at train time from a seeded sampler: single ball vs walls (25%), head-on/offset pairs (25%), small 20x20 boxes with 2-6 balls (15%), wider boxes with up to 100 balls and speeds up to 40 (35%). 1,200 training scenes and 100 validation scenes (seed 9000), 24 steps each, seed 4738 (`scripts/train_conservative.py`).

**Model (deployed variant: conservative contact)**
- Input per ball: position, velocity (hidden state unused). Output: next position, velocity.
- Pair force: distance-only MLP (1-64-64-1, tanh) times penetration, applied along the pair line with opposite signs (exact pair momentum, potential-derived). Wall force: same style MLP on penetration into each of 4 walls. Learned uniform acceleration (initialized at 0, learned about 8.9-9.1 for true 9). Neighbor graph by cell lists, radius 2r (O(N)). 8 velocity-Verlet substeps per tick.
- Prior built in (disclose): Verlet integrator, 8 substeps, a force scale of 100 that mirrors the truth stiffness, and a distance-only force form. The repo notes flag this as "maybe too much prior".
- Attention variant (older, soup B): radius-graph single-head attention (key/query/value of node velocity, wall features, relative position) plus a GRU cell plus a delta head, with extra wall and pair-impulse heads. Not conservative.

**Training**
- Loss: per-step state loss on position and velocity (`token_state_loss`, velocity weight 0.1, speed weight 0.3), unrolled K=1 (epochs 0-9), 4 (10-29), 8 (30-49) steps, 50 epochs, Adam lr 1e-3 (3e-2 for the gravity scalar), grad-clip 1.0, lr x0.1 in last 20 percent, batch 8 scenes. From scratch with zero-initialized forces. Reported at about 39 min on one GPU for the first training run (`docs/debugging/energy-fix-results.md`).
- Deployed checkpoint `results/cons_pure_100.pt` retrained with training ball count capped at 100 (per memory notes; training log not re-read).

**Evaluation plan**
- Metric err@k = mean position error per ball at tick k, in simulation length units ("cells"), starting both models from the same true state. Also: energy per ball vs truth, wall restitution vs incoming speed, pair head-on kinetic-energy ratio, momentum error, first non-finite step.
- Baselines: (1) constant velocity (exists in repo for some tests, `scripts/eval_trivial_baselines.py` and the 1k test); (2) exact simulator with perturbed parameters (to build, gives the chaos floor); (3) attention variant soup B (exists, `checkpoints/token_model_soup_b.pt`); (4) GNS-style message-passing GNN baseline, same data and K-step loss (to build; no GNN baseline exists in the repo today).
- Held-out seeds: sets 4738, 9000, 12000 (selection on a different seed set from the reported one).
- Ablations (to run): substeps {1, 2, 4, 8}; conservative-only vs with attention residual (residual never trained); fixed vs fine-tuned gravity; training unroll length.
- Proposed pass/kill criterion (record before running): the conservative model must beat the GNN baseline on err@10 and on energy drift at 100 balls on all 3 seed sets; otherwise report it as a negative result.
- Caveat: chaos limits comparability. Per the repo log, err past about tick 20-30 is dominated by trajectory divergence, so judge ticks 1-20 and use energy/physical-law metrics beyond.

**Compute**
- University GPU cluster, a single GPU for training and evaluation (each run well under 1 hour at this model size). Video rendering and the web demo run on a laptop.

**Facts and sources (honest numbers; each read from the file named)**
- Eval of `cons_pure` (the version before the 100-ball cap) vs soup B, from truth start, held-out seeds 9000 and 12000, same seeds both models, `results/eval_energy_fix.json` (read directly) and `docs/debugging/energy-fix-results.md`:
  - 4 balls, 20x20 box (24 scenes): err@1/5/10/20 = 0.026/0.123/0.315/2.39 (conservative) vs 0.065/0.281/0.667/3.19 (soup B); err@50/100 about 7.3/7.9 vs 7.2/8.1 (no gain at long horizon, chaos).
  - 100 balls, n=100 box (6 scenes): err@1/5/10/20 = 0.021/0.113/0.298/1.90 vs 0.047/0.236/0.494/3.27; err@50 about 44 for both.
  - Energy per ball at tick 100: 4 balls -81.8 (conservative) vs -113.5 (soup B) vs truth -77.7; 100 balls -387.7 vs -778.1 vs truth -363.7.
  - Small sample (24 and 6 scenes), single training seed: say so.
- Wall restitution speeds 3-80: conservative 1.00-1.04, soup B 0.41-1.65 (truth 1.0), from `docs/debugging/energy-fix-results.md`. Speeds above 40 were not in training.
- Scaling tests, from project memory notes (source files: `scripts/ball_1k_rollout.py`, `scripts/ball_50k_rollout.py`, `results/ball_50k_rollout.npz`; I did not recompute these numbers in this session, mark as unverified until rerun): 1,000 balls, 500 steps, conservative model finite throughout, err@5/10/20 = 0.11/0.28/1.38; 50,000 balls x 100 steps (trained on at most 100 balls), err@1/5/10/20/50/100 = 0.02/0.11/0.28/0.94/9.3/71.5, KE per ball within about 2 percent of truth, about 1.2-1.3 s total for truth and model on one GPU. Late-time floor pile looks mottled vs truth.
- Linear scaling and 1B run (older soup B model, ball sim, speed only, no accuracy): 1M balls 4.39 s, 10M 40 s, 100M 384 s for 300 ticks, 1B 3,825 s (12.75 s/tick, 121 GB peak, fp16 hidden state), fitted exponent 0.96-0.99 for 1M-100M, from `results/bench_scaling_*.json` and README.md. Not the deployed model; do not feature in a 1-page proposal beyond "scales linearly".
- Browser parity: JS port vs PyTorch one-step max abs diff about 2e-6 (below the 1e-4 gate); in dense contact the float32 rollout diff reaches about 4e-3; free-run diffs at step 20 are chaotic. Source: `web/token/README.md` (not re-run here).
- Known failures (state them in the report/demo): attention model failed gravity at 100-1000 bodies (a different model, `CentralForceDynamics`, was needed; not part of this proposal); 1k-ball rollouts with the older model collapsed onto the floor; the open gravity offset of about 0.88 in pair-momentum error comes from learned g_x 8.90 vs true 9.


## 6. Planned implementation

Term: a full semester, about 15 weeks (user confirmed).

**Already built (disclose as prior work)**
- Truth simulator (`augur.py`, GPU version in `scripts/energy_probe2.py`), token model code (`model/token_free.py`, `model/token_model.py`), conservative-contact training and eval scripts, 1k and 50k rollout scripts, JS port and 3D inspector page (`web/token/`), verification tests (`tests/test_token_free.py`, `web/token/tests/`), rollout videos.

**To build or adapt this term (the planned work)**
1. One-command reproducible pipeline (data generation, train, evaluate, export weights, verify web parity) with a README, tested on a clean clone.
2. GNS-style GNN baseline in PyTorch (encoder, 3-5 message-passing layers, decoder to acceleration, same data and unrolled loss).
3. Single evaluation harness: err@k, energy drift, momentum, restitution, non-finite step, 3 seed sets, bootstrap intervals over scenes; results table plus plots.
4. Ablations: substeps, residual attention path (never trained), gravity fine-tune (open repo item), unroll length.
5. Scaling study: ball count 100 to 50k at constant density, box size, speeds beyond training (>40), timing vs N.
6. Demo polish: results panel in the browser page, new side-by-side video, final slides.

**Week plan (15 weeks, milestones in bold)**
- **Weeks 1-2: proposal approved**; freeze `cons_pure_100.pt` and scripts; write pass/kill criteria; one-command pipeline; rerun small-N eval on `cons_pure_100.pt`; clean-clone test.
- Weeks 3-5: GNN baseline implemented, unit-tested (shapes, permutation invariance, pair momentum), trained on the same scenes.
- **Weeks 6-7: baselines milestone**: evaluation harness, all models (constant velocity, soup B, GNN, conservative) on 3 seed sets, N in {4, 100, 1000}; results table v1.
- **Weeks 8-10: ablations milestone**: substeps, residual path, gravity fine-tune, unroll length; ablation table; log every run, failures included.
- **Weeks 11-12: scaling study milestone**: 100 to 50k balls, box size, speed beyond training; timing; floor-pile failure analysis.
- Weeks 13-14: **demo polish**: browser results panel, side-by-side truth/model/GNN video (new), unbiased frame review of the video, report draft.
- Week 15: **final presentation** and report; buffer for slips.

**Risks and fallbacks**
- GNN baseline beats the structured model: report it; compare trade-offs (energy, size, speed, interpretability).
- GPU queue delays: each training is under 1 hour; schedule in batches.
- Chaos makes long-horizon comparison noisy: restrict accuracy to ticks <=20 and use physical metrics beyond.
- Time: drop the scaling study (prior 1k/50k numbers stay labeled as prior), then the unroll-length ablation.
- Web demo breaks: videos and results table are the fallback demo.

## 7. Planned demo

- Live browser demo (already deployed, public): https://faraday-dot-py.github.io/Augur/web/token/ (HTTP 200 checked 2026-10-26; loads `weights.json` from `cons_pure_100.pt`). Spawn balls with the mouse, ground-truth ghost overlay beside the model, 3D per-ball token cubes, edit weights live, force-curve and energy plots. Source: `web/token/`. Note the landing page and URL contain the author's GitHub handle (public repo).
- Rollout videos to show (check they exist and are the conservative model before use; many videos in `videos/` are older models): `videos/ball_1k_500.mp4` (1,000 balls, older soup B), `videos/ball_50k.mp4` and `videos/ball_50k_grid.png` (50,000 balls, conservative), `videos/ball_1k_energy_fix_grid.png` (soup B vs conservative vs truth grid; its mp4 `ball_1k_energy_fix.mp4` is gitignored and not in `videos/` now), `videos/token_model_soup_b_100x100_100balls_300steps.mp4` (soup B). Mark each with its model.
- New for the term: side-by-side truth vs model vs GNN video from the final checkpoint; results table slide; ablation plot.
- Structure of the talk (about 10 min): problem, token design (diagram), data and training, results table and failure cases, live demo, limits.


## 8. One-page proposal (source and PDF)

- Source text: `docs/class/token-model-proposal.md` (about 550 words incl. headings). Edit it, then rerun `python3 docs/class/render_proposal_pdf.py`.
- PDF: `docs/class/token-model-proposal.pdf`, 1 page, US Letter, 1-inch margins, Liberation Serif 11 pt (Times-metric). Last render: text ends 2.5 in above the page bottom (1.5 in of slack above the 1-inch margin).
- No pandoc, LaTeX or LibreOffice on this machine; the PDF is produced by matplotlib (embedded Liberation Serif, TrueType). Layout is plain (no hyphenation, ragged-right words with fixed spacing). For a more polished final, paste the text into Word/Google Docs (11 pt, 1-inch margins) and export to PDF.
- Verified with pypdf: 1 page, 612x792 pt, fonts LiberationSerif and LiberationSerif-Bold embedded. Font size 11 is set in the script (`SIZE = 11.0`), not read back from the PDF.

## 9. Format checklist

- Required: PDF, minimum 11 pt font, 1-inch margins, maximum 1 page, with: project type, project idea, related work/GitHub references, methods/pipeline, planned implementation, planned demo (all in the source text).
- Alternative production: LaTeX `\documentclass[11pt]{article}\usepackage[margin=1in]{geometry}` with `pdflatex`; or pandoc `pandoc proposal.md -o proposal.pdf -V fontsize=11pt -V geometry:margin=1in`.
- Verify: page count 1 (`pdfinfo file.pdf | grep Pages`, or `pypdf`), fonts at 11 pt, headings not smaller than 11 pt, no small-font tables or URLs.

**Open questions for you (remaining)**
1. GNN baseline vs a cheaper "MLP per ball without neighbors" baseline if time runs short: keep the GNN as the plan?
2. The small-N table in section 5 is for `cons_pure.pt`; OK to rerun the eval on `cons_pure_100.pt` in week 2 and cite those numbers in the final report?
