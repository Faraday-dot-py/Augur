# Token model: class proposal notes

Purpose: everything needed to write the 1-page proposal for the applied ML project. Written 2026-10-26. Bullets only.

## 0. Read this first (honesty and policy)

- This is existing work (about 5 weeks old, 500+ commits). The proposal says "plan", so it must (a) state what already exists and (b) propose new, concrete work for the course term. Do not hide prior work.
- Check the course/instructor policy on submitting pre-existing work before submitting. Ask in writing; disclose the prior work in the proposal either way (the draft in section 8 does).
- Naming trap: the project calls the model a "token-per-ball" model, but the version that is deployed in the browser (conservative-contact, `results/cons_pure_100.pt`) has NO attention. It is one state token per ball, with a learned distance-only pair-force MLP, a wall-force MLP, a learned uniform gravity, and 8 velocity-Verlet substeps (8,708 exported floats). The attention + GRU path (radius-graph single-head attention, `model/token_free.py::_core`) exists in the code and in the older "soup B" model, and is kept as an optional residual (`contact_residual=True`, never trained). Do not call the deployed model "a transformer" in the proposal. Say "token-per-ball graph model (attention variant plus physics-structured variant)".
- Accuracy is real only at small N and short horizon (about 20 ticks). Past about 30-50 ticks, positions decorrelate from truth in chaotic contact scenes (expected; the truth sim itself is chaotic). Large-N results (1k, 50k) are stability/scale checks, not accuracy claims. The 1 billion ball run was speed and memory only, on an older model (soup B), not the deployed one.

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

Assumption: 4-6 weeks available (5-week plan below; shrink by dropping week-4 ablations and the second seed).

**Already built (disclose as prior work)**
- Truth simulator (`augur.py`, GPU version in `scripts/energy_probe2.py`), token model code (`model/token_free.py`, `model/token_model.py`), conservative-contact training and eval scripts, 1k and 50k rollout scripts, JS port and 3D inspector page (`web/token/`), verification tests (`tests/test_token_free.py`, `web/token/tests/`), several rollout videos.

**To build or adapt in this term (the planned work)**
1. One-command reproducible pipeline (data generation, train, evaluate, export weights, verify web parity) with a README, tested on a clean clone. Today the steps are spread across scripts.
2. GNS-style GNN baseline in PyTorch (encoder, 3-5 message-passing layers, decoder to acceleration, same Verlet-free Euler update, same data and loss).
3. Single evaluation harness: err@k, energy drift, momentum, restitution, non-finite step, across 3 seed sets, mean and bootstrap interval over scenes; one results table plus plots.
4. Ablations: substeps, residual path, gravity fine-tune (open item in the repo), unroll length.
5. Demo polish: results panel in the browser page, side-by-side truth vs model video (new, from the final checkpoint), slides.

**Week plan**
- Week 1: freeze baseline checkpoint and scripts; write the pass/kill criterion; build the single pipeline script; clean-clone run. Deliverable: pipeline README.
- Week 2: GNN baseline implemented and trained; unit tests (shape, permutation invariance, momentum check). Deliverable: baseline checkpoint and curve.
- Week 3: evaluation harness; run all models on 3 seed sets, N in {4, 100, 1000}; add 10k-50k stability check. Deliverable: results table v1.
- Week 4: ablations and gravity fine-tune; rerun table; honest write-up of failures. Deliverable: ablation table.
- Week 5: demo: rendered videos, web results panel, rehearsal, final report/slides.

**Risks and fallbacks**
- GNN baseline beats the structured model: report it honestly; the project then compares trade-offs (energy, size, speed, interpretability).
- GPU queue delays: all trainings are under 1 hour; the small-N tests also run on CPU (slowly).
- Chaos makes long-horizon comparison noisy: restrict to ticks <=20 and physical metrics.
- Time: drop the ablations and the 50k rerun (the prior numbers stay labeled as prior).
- Web demo breaks: videos and table are the fallback demo.

## 7. Planned demo

- Live browser demo (already deployed, public): https://faraday-dot-py.github.io/Augur/web/token/ (HTTP 200 checked 2026-10-26; loads `weights.json` from `cons_pure_100.pt`). Spawn balls with the mouse, ground-truth ghost overlay beside the model, 3D per-ball token cubes, edit weights live, force-curve and energy plots. Source: `web/token/`. Note the landing page and URL contain the author's GitHub handle (public repo).
- Rollout videos to show (check they exist and are the conservative model before use; many videos in `videos/` are older models): `videos/ball_1k_500.mp4` (1,000 balls, older soup B), `videos/ball_50k.mp4` and `videos/ball_50k_grid.png` (50,000 balls, conservative), `videos/ball_1k_energy_fix_grid.png` (soup B vs conservative vs truth grid; its mp4 `ball_1k_energy_fix.mp4` is gitignored and not in `videos/` now), `videos/token_model_soup_b_100x100_100balls_300steps.mp4` (soup B). Mark each with its model.
- New for the term: side-by-side truth vs model vs GNN video from the final checkpoint; results table slide; ablation plot.
- Structure of the talk (about 10 min): problem, token design (diagram), data and training, results table and failure cases, live demo, limits.

## 8. Draft of the one-page proposal (about 480 words; verify it fits one page at 11 pt, trim related work first if not)

**Project type:** Applied project.

**Project idea.** We will build and evaluate a learned simulator for 2D bouncing balls in which each ball is a single token (position, velocity) and a small neural network predicts the next state. Such learned simulators matter because physics solvers become costly at scale, and learned models can be faster but often drift in energy and fail on long rollouts. We ask how a physics-structured token model compares with a generic graph-network baseline in accuracy, energy behavior and generalization to many more balls than it was trained on. The result is a working prototype with a live browser demo.

**Prior work disclosure.** I started this system before the course (about 5 weeks of work). It already contains the exact simulator, a token-per-ball model with a learned pair-force and wall-force network and Verlet integration, a first training/evaluation script set, and a browser port. The course work below is the new part: a reproducible pipeline, a graph-network baseline, a systematic evaluation, ablations and a finished demo.

**Related work.** Graph Network-based Simulators (Sanchez-Gonzalez et al., ICML 2020; DeepMind reference code), Interaction Networks (Battaglia et al., NIPS 2016), MeshGraphNets (Pfaff et al., ICLR 2021), Neural Relational Inference (Kipf et al., ICML 2018), Hamiltonian Neural Networks (Greydanus et al., NeurIPS 2019), Lagrangian Neural Networks (Cranmer et al., 2020), and Attention Is All You Need (Vaswani et al., 2017).

**Methods and pipeline.** Data: a seeded, GPU-based force simulator (walls, ball contact, gravity) generates trajectories of 1 to 100 balls in boxes of several sizes; about 1,200 training scenes of 24 steps, 100 validation scenes. Model: per-ball tokens; a distance-only pair-force network and a wall-force network (about 9k parameters) with eight velocity-Verlet substeps and a learned gravity, built on a linear-time neighbor graph. Training: multi-step unrolled loss on position and velocity (1, 4, then 8 steps), 50 epochs, one GPU, under 1 hour. Baselines: constant velocity, a previous attention-plus-GRU token model, and a new message-passing GNN trained on the same data. Evaluation: mean position error per ball at ticks 1 to 20 (err@k), energy drift, momentum, wall restitution versus speed, and stability at 100, 1,000 and 50,000 balls, on three held-out seed sets with confidence intervals. We use ticks up to 20 for accuracy because rollouts are chaotic beyond that. Ablations: number of substeps, residual attention path, learned versus fixed gravity, unroll length.

**Planned implementation.** Week 1: one-command pipeline (generate, train, evaluate, export) and clean-clone test. Week 2: PyTorch GNN baseline with tests. Week 3: evaluation harness and results table. Week 4: ablations and a gravity fine-tune. Week 5: demo and report. All training uses a university GPU cluster (single GPU). Fallback: if the structured model does not beat the baseline, we report that and compare the trade-offs.

**Planned demo.** A live browser page (already built) that runs the model next to the exact simulator with an energy plot and editable weights, a side-by-side truth/model/GNN rollout video, and a results table with ablations.

(Count the words after editing; trim the related-work line first if over a page.)

## 9. Format checklist and open questions

- Required: PDF, minimum 11 pt font, 1-inch margins, maximum 1 page.
- Produce: Google Docs or Word (set margins 1 inch under Page setup, font 11 pt, e.g. Times New Roman or Arial) and export to PDF. Or LaTeX: `\documentclass[11pt]{article}\usepackage[margin=1in]{geometry}` and `pdflatex`. Pandoc: `pandoc proposal.md -o proposal.pdf -V fontsize=11pt -V geometry:margin=1in`.
- Verify: open the PDF, check page count is 1 (`pdfinfo proposal.pdf | grep Pages`), check fonts at 11 pt (`pdffonts`, or zoom to 100 percent), and that bold headings are also at least 11 pt. Tables and URLs in small fonts can violate the 11 pt rule; avoid them.
- Needed: Project type; Project idea; Related work / GitHub references; Proposed methods / pipeline; Planned implementation; Planned demo (all in the draft).
- Open questions for you:
  1. Is submitting pre-existing work allowed? If not, can the scope be limited to the new baseline/evaluation work as the project?
  2. Which model is "the token model" for the proposal: the deployed physics-structured one (no attention) or the attention variant? The draft uses the structured one and keeps the attention one as a baseline.
  3. OK to put the public demo URL (contains the GitHub handle) and the repo in the proposal?
  4. Is the term 5 weeks? Adjust the weekly plan.
  5. Do you want the GNN baseline, or a cheaper "MLP per ball without neighbors" baseline if time is short?
  6. Whether to reproduce the small-N numbers for `cons_pure_100.pt` (the deployed checkpoint): the table in section 5 is for `cons_pure.pt`; the 100-ball-capped checkpoint was verified at 1k/50k per notes only.
