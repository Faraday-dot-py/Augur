import re, sys

SRC = "/home/awebb/.claude/projects/-home-awebb-Research-Augur/49e2af9c-5146-46c5-9cde-06c4d9bb6fc9/tool-results/artifact-8e6d0dc8-1793006807-d0eb.html"
OUT = "/tmp/claude-1000/-home-awebb-Research-Augur/49e2af9c-5146-46c5-9cde-06c4d9bb6fc9/scratchpad/supervisor-log-refresh.html"

h = open(SRC, encoding="utf-8").read()

def sub1(old, new):
    global h
    assert h.count(old) == 1, (h.count(old), old[:70])
    h = h.replace(old, new)

K = '<span class="v k">KILL</span>'
P = '<span class="v p">PASS</span>'
M = '<span class="v m">MIXED</span>'

sub1("log current through</span><span class=\"v2\">2026-10-26 01:55 PDT",
     "log current through</span><span class=\"v2\">2026-10-26 14:40 PDT")
sub1("(branch descriptive-experiment-names, not yet merged)", "(merged to master at ed036be)")

# Agents table -> live status table
a0 = h.index("<table>\n<thead>\n<tr>\n<th>Agent</th>\n<th>Job</th>")
a1 = h.index("</table>", a0) + len("</table>")
rows = [
 ("tier-k lever continuation", "running", "no-velocity retrain: job 4050 (seed 4738) at iteration 41,300 of 60,000, job 4051 (second seed) running; then four-arm hierarchy rerun"),
 ("planetesimal-disk solar-system test", "finished", "out of range for our models (unit-mass only); proxy KILL; reference numbers logged (0f19cf5)"),
 ("loss analysis and error-distribution diagnostic", "finished", "tail terms dropped for tiered arms; local work-bias loss term recommended (75d8622); not yet run"),
 ("local work-bias loss term test", "running", "code and 7 CPU tests done; job 4052 (seed 4738, three concurrent arms) submitted, pending behind the retrain"),
 ("planetesimal-disk video and per-body-mass support", "finished", "video rendered and reviewed; mass-conditioning design, curriculum trainer and 10 CPU tests committed (0f1c7d8, 79a6c70); not trained, queued for a free GPU"),
 ("comparison videos", "finished", "six videos in videos/, reviewed by fresh reviewers, logged (0220a18)"),
 ("all earlier agents (hole-1, tier-0 extensions, upper-tier module, naming, page restyle, sandbox wrapper)", "finished or stopped", "results are in the Log"),
]
t = "<table>\n<thead>\n<tr>\n<th>Agent</th>\n<th>State</th>\n<th>Status at 14:40 10-26</th>\n</tr>\n</thead>\n<tbody>\n"
for a, s, d in rows:
    t += "<tr>\n<td><strong>%s</strong></td>\n<td>%s</td>\n<td>%s</td>\n</tr>\n" % (a, s, d)
t += "</tbody>\n</table>\n<p>Pending on retrain result: frozen-base residual corrector test (user approved); cold-collapse scoping.</p>"
h = h[:a0] + t + h[a1:]

# verdict panels
sub1("<li>Tier 0 from scratch is the first to tile better than it runs untiled (0.54x).</li>",
     "<li>Tier 0 from scratch is the first to tile better than it runs untiled (0.54x). Matched-budget retrain (60,000 iterations) passes every pre-registered line: err@20 1.13x / 1.12x baseline, mid beta -0.118 / -0.052, tiled over untiled 0.95. The earlier mid-density bias was mostly undertraining.</li>")
sub1("<li>Density-stratified harvest at 18.75k iterations: err@20 1.34x baseline, mid beta worse.</li>",
     "<li>Density-stratified harvest at 18.75k iterations: err@20 1.34x baseline, mid beta worse.</li>\n"
     "<li>End-to-end hierarchy (tier 0 with external field from upper tiers), both with the 10k-iteration and the matched-budget tier 0: joint fine-tune 1.8-2.4x the tiled baseline at tick 20, drift 3.5-13x. Gap is the field-aware tier 0 in 64-unit windows, not the upper-tier supply.</li>\n"
     "<li>Planetesimal-disk proxy for our unit-mass models: err@100 1.73, energy drift 1.23 vs 0.003 reference, disk survivors 0.00 vs 0.39.</li>")
sub1("<li>Mid-density bias in every window-trained tier 0. It is real, not a test artefact (imposed-field re-score).</li>\n<li>Cause may be undertraining: from-scratch runs had 18,750 iterations against about 60k for the baseline. Untested.</li>",
     "<li>Mid-density bias in the earlier window-trained tier 0 checkpoints was real, not a test artefact (imposed-field re-score). Undertraining is now the supported cause (matched-budget run, one seed).</li>\n"
     "<li>Matched-budget tier 0 caveats: sparse beta sits on its 0.15 limit, dense 16-32 beta +0.48 near the 0.5 kill line, plain dE@100 0.03 vs 0.014.</li>\n"
     "<li>Retrain without velocity and recurrent inputs (jobs 4050, 4051) running; pass if exact-field err@20 at most 1.2x baseline and drift at most 2x on every seed.</li>\n"
     "<li>Local work-bias loss term (job 4052) and per-body-mass training: queued behind the retrain.</li>")
sub1("<li>Tier-0 co-adaptation drives joint-fine-tune drift. A frozen tier 0 loses the far-field gain.</li>\n<li>Long-horizon energy penalty is the only lever that cut drift and kept gains (0.0212 against a 0.0125 line). Checkpointed k up to 40 run pending.</li>",
     "<li>Joint fine-tune drift: the co-adaptation explanation is unconfirmed (one seed per arm). Frozen sequential training stays the default; test a regularised joint variant if the no-velocity tier 0 still drifts.</li>")

# headline rows
extra = [
 ("tier0-from-scratch-density-stratified-harvest-matched-iterations", "Same recipe trained 60,000 iterations (5.06 GPU-hours, seed 4738)", "err@20 1.13x / 1.12x baseline; mid beta -0.118 / -0.052; tiled/untiled 0.95; tiled dE@100 0.019", P),
 ("hierarchy-external-field-tier0-matched-budget", "Tiered hierarchy with matched-budget tier 0, four arms, seeds 9100/9200/9300", "joint fine-tune 2.0 / 2.1 / 1.8x tiled baseline at tick 20; drift 3.5-5x", K),
 ("gravity-tier0-no-velocity-no-recurrent-inputs-in-hierarchy", "Velocity and recurrent inputs removed from gravity tier 0, standard and position-only upper tiers", "force error -35 to -38%; position error at tick 50 -32 to -41%; dE@100 0.0145 / 0.0165 vs 0.035", P),
 ("planetesimal-disk-central-force-proxy", "Star plus disk proxy with unit-mass central-force model", "err@100 1.73; dE 1.23 vs 0.003; survivors 0.00 vs 0.39", K),
]
r = ""
for n, w, k, v in extra:
    r += "<tr>\n<td><strong>%s</strong></td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n</tr>\n" % (n, w, k, v)
sub1("<tr>\n<td><strong>contact-head-arms</strong>", r + "<tr>\n<td><strong>contact-head-arms</strong>")
h = h.replace('<td>Tier 1 trained over frozen baseline tier 0</td>', '<td>Tier 1 trained over frozen baseline tier 0</td>')

# log entries
E = [
 ("tier0", "02:05", "tier0-extensions", "Density-coverage diagnosis: windows-trained tier 0 mid-density bias is not edge, taper or radial; larger for denser surroundings. Density-stratified harvest failed (plain err@20 1.34x, mid beta -1.46 / -1.78) but gives the best Plummer gravity model. Suspected undertraining (18,750 vs about 60k iterations); matched-budget run sent."),
 ("sup", "02:05", "supervisor", "User requests: (1) summary page redesign through the graphic-design skill, static HTML from a declarative source, no JS or Python; (2) verify the naming rule as agents report; (3) commit-time filter hook that spawns an advisory reviewer (security, quality, verifiability, traceability, trust) that cannot override supervisor purview. Dispatched a summary-redesign agent and a commit-review-hook agent. Concern (non-blocking): reviewer cost against repeated usage-limit hits; hook builder told to throttle, use a cheap model and add a one-switch disable. Note: 'website' read as the published summary artifact, not the viz-web site."),
 ("sup", "02:05", "supervisor", "Resumed naming-audit, summary-redesign (asked to show converter vetting before publish) and commit-review-hook agents from saved work. The three experiment agents still running."),
 ("sup", "02:30", "supervisor", "Naming rule landed on master (ed036be): convention, glossary and a name-check tool in docs/theory/experiment-naming.md; master fast-forwarded from descriptive-experiment-names. To confirm with owning agents: the unmapped tier-energy experiment, the joint-training ablation names, the tier-0 fine-tune names."),
 ("sup", "02:30", "supervisor", "Summary page rebuilt with the signed Ubuntu cmark-gfm converter after the npm marked package failed the no-error-margin vetting. cmark-gfm C source not audited; trust rests on the signed archive."),
 ("sup", "02:30", "supervisor", "Commit-review hook installed (advisory only, always exits 0): stage one deterministic, stage two a haiku reviewer capped at 10 per day, about $0.01-0.02 each. Disable with a single marker file. First real commit under the hook is this one."),
 ("sup", "02:30", "supervisor", "Dispatched the prompting-architecture case report agent (persuasive artifact titled Prompting Architecture Case; may spawn read-only research subagents and ask running agents for facts). Evidence-only rule; project lead referred to by role."),
 ("sup", "02:40", "supervisor", "Summary page restyle requested: dark CRT and analog color theme, blueprint kept, decorative-only garnishes, color-research note before palette, contrast computed. Sent to the summary page agent."),
 ("sup", "02:55", "supervisor", "User requested every log line link to a redacted Claude Code conversation file in a new private GitHub repo, links obfuscated per a security research subagent. Dispatched the conversation-linked-log agent (security research, mapping design, redaction and scanner pipeline, private repo, page republish). Concern (non-blocking): transcripts may hold secrets; pipeline requires zero scanner findings and excludes any file it cannot redact with certainty. Authorization: the user's explicit request; no other pushes."),
 ("lj", "02:30", "lj-potential-gradient", "Hole-1 round 5 closed (branch hole1-potential-field, unpushed): gravity trained at half the timestep cut position error at 20 ticks about 40% and relative energy drift about 3.5x (0.0149 to 0.0043) at 1.9x training cost, " + "PASS" + " on the pre-registered bar; quarter timestep drifts less but costs 3.8x; substepping only the pair term fails for gravity. LJ retrain with cached pair substeps passes (solid error 0.0035 at 1.24x cost) but loses the coexist scene to whole-step halving (0.0617 vs 0.0498). Contact recurrent-field ablation shows no effect and the original runaway did not reproduce on a same-recipe rerun, so it needs a second seed; no usable contact arm. Naming slip: the report still used legacy labels."),
 ("tiers", "04:20", "upper-tier", "Tier-k round: removing velocity and recurrent inputs from the gravity tier 0 (job 4024) " + "PASS" + " on its pre-registered rule with standard and with position-only upper tiers: force error down 35-38%, clump-mean error down 57-83%, position error at tick 50 down 32-41% versus the same tier 0 alone. Energy drift at tick 100 is 0.0145 and 0.0165 against a same-tier baseline near 0.035, but this tier 0 drifts 3.3x more than the earlier density-stratified one alone (0.0145 is about 1.4x its 0.0106 baseline). One training seed per arm; dropping velocity from upper tiers gives no drift benefit. Upper-tier external-field hierarchy implemented and smoke-passed (job 4044); full run 4045 submitted, unscored because Polaris was locked out after three failed reconnects. Naming slip: the report still cited legacy labels."),
 ("tiers", "10:25", "upper-tier", "End-to-end hierarchy (scratch-trained tier 0 given an external field from upper tiers, 4x4 windows of 64 units, seeds 9100/9200/9300) " + "KILL" + " on its pre-registered rule: joint fine-tune is about 2.4x the density-stratified tiled baseline at tick 20 (0.159 / 0.170 / 0.180 vs 0.066 / 0.067 / 0.076), 2x at tick 50, 7-13x energy drift at tick 100, clump-mean error 0.96-1.14 vs 0.23-0.42. Upper tiers add only 4-10% error over an exact external field, so the bottleneck is the external-field-aware tier 0 itself. Next lever is that tier 0, not upper-tier supply."),
 ("tier0", "10:27", "tier0-extensions", "Matched-budget tier 0 (job 4041, density-stratified recipe, 60,000 iterations, 5.06 GPU-hours, seed 4738 only) " + "PASS" + " on every pre-registered criterion: plain err@20 1.13x / 1.12x the 60k-iteration baseline (limit 1.25x; the 10k-iteration run was 1.34x), mid-density beta -0.118 / -0.052 (limit 0.15; was -1.46 / -1.78), tiled over untiled err@20 0.95 (limit 1.5x), tiled energy drift at tick 100 0.019. The earlier mid-density bias was mostly undertraining, not coverage or architecture, so the larger-grid and edge-distance variants were not run. Caveats: sparse beta on the 0.15 limit, dense 16-32 beta +0.48 near the 0.5 kill line (not a pass criterion), plain err@100 2.92 vs 2.82, plain dE@100 0.03 vs 0.014, one seed. Supersedes both earlier from-scratch tier-0 checkpoints."),
 ("tiers", "11:30", "upper-tier", "End-to-end hierarchy rerun with the matched-budget tier 0 (job 4046, same seeds, arms and rule) " + "KILL" + " again but closer: with the exact external field the tier 0 is 1.5-1.7x the tiled baseline error at tick 20 (10k-iteration tier 0: 2.4x) and 4-5x its energy drift (was 11x); sequential-fit upper tiers 1.2-1.3x the exact-field arm; joint fine-tune 2.0 / 2.1 / 1.8x baseline at tick 20, 1.5-1.7x at tick 50, drift 3.5-5x, clump-mean error 0.75 / 0.86 / 0.88 vs 0.23 / 0.29 / 0.42; joint fine-tune gives nothing over sequential. The gap is the external-field-aware tier 0 in 64-unit tiled windows. No frame review (no renderer for the tiled hierarchy)."),
 ("tiers", "11:40", "upper-tier", "Diagnosis of the tiled hierarchy gap (teacher-forced, 1.66M particle-ticks): dense-region energy-work error is 2.1x baseline (3.3x in ticks 10-30; dense bins hold 76% of it, force error only +9%) and is the drift; recurrent state adds 13-16% force error by ticks 30-100 (windows keep independent recurrent state, no halo exchange); low-density particles have 3x force error but only 3.6% of squared error. Ruled out: seam distance, owned-core size, external field and upper tiers, the plus/minus 18 sampling grid, the dropped 4-unit ring. Fix approved: retrain the matched-budget tier 0 without velocity and recurrent inputs; pass if exact-field error at tick 20 is at most 1.2x baseline and energy drift at most 2x baseline on every seed."),
 ("tiers", "12:30", "upper-tier", "Retrain without velocity and recurrent inputs launched on the matched-budget tier 0 (job 4050, seed 4738, running about 1.3 h at this entry; second seed on the other GPU), then the four-arm hierarchy rerun on seeds 9100/9200/9300. User asked whether frozen sequential training is right: current answer is keep it as the default (joint fine-tuning drifted about 6x on the earlier tier 0 and gave nothing over sequential on both hierarchy runs), but the co-adaptation explanation is unconfirmed and evidence is one seed per arm; if the no-velocity tier 0 still drifts, test a regularised joint variant (energy-similarity term or alternating freeze). User approved a frozen-base residual corrector test (is the tiered drift correctable by a small head on a frozen base), held until the retrain reports."),
 ("harness", "13:40", "eval-harness", "Comparison videos rendered and reviewed (six, in videos/, experiment-log entry 0220a18). Fresh reviewers: gravity clustered scene, models collapse at ticks 20-40, tiled worst (err@100 about 14 vs about 7); multiclump 128, untiled fails first then tiled overtakes around tick 50, no seam pattern; gravity generations near-identical (err@100 7.3-7.8); wall contact, the existing set-up and the kernel-convolution variant blow up at ticks 30-35, the sdf-gather variant stays bounded but freezes into a clump; liquid-argon solid and coexist, no artifacts, smooth error growth. Planetesimal-disk solar-system test started on the sandbox GPU."),
 ("sup", "13:52", "supervisor", "Planetesimal-disk solar-system test (star plus disk around a central mass vs direct N-body) finished, " + "KILL" + " on the in-range proxy and out of range for the real design: our models take no per-body mass (unit-mass only, gravitational constant 1, softening 0.5, timestep 0.1, 100-1000 bodies), so the 1000:1 star-to-particle ratio and the 100:1 radius range are untestable and no model claim is made. Direct N-body reference for the real design (star, 1000 disk particles, one Jupiter-mass body at 5 AU, 100 years) is valid: relative energy drift 2.4e-5 (limit 1e-3), median semi-major axis change 6.4%, median eccentricity 0.0096 to 0.059, Jupiter 4.995 to 4.924 AU, 54% of bodies near 5 AU remain, sticky merging leaves 618 of 1000. Proxy (300 unit-mass core plus 700 disk bodies, central-force model): err@20 0.016 passes (limit 0.1), err@100 1.73 (kill above 1.0), relative energy drift 1.23 vs 0.003 reference (kill above 0.3), disk survivors 0.00 vs 0.39 (kill if difference above 0.2); a heavier-core proxy also KILL (err@100 4.68). Caveats: proxy reference disks are violent not cold, no scatter-field or tier-0 checkpoint run, no frame review. Needs per-body-mass support in the model before any real test."),
 ("sup", "14:10", "supervisor", "Loss analysis finished (75d8622, docs/debugging/loss-analysis-2026-10-26.md, CPU rollouts only), started 13:47 at user request after the question whether the loss hides how the model is wrong. Worst 1% of particles hold under 20% of squared position error at tick 50 in all six cells and at tick 20 for the tiered arms (0.11-0.21), so the tail-term idea is dropped for the tiered arms; it only fails for the old baseline at tick 20 (0.25 and 0.39, a few close-encounter particles). Tiered error is mostly coherent (whole neighbourhoods displaced together, share 0.36-0.54); the baseline's is incoherent scatter. The densest 10% of particles hold only 12-26% of squared position error but carry the largest negative kinetic-energy error in every arm (N=1000 tick 50: matched tier 0 -46.9, tiled -37.5, baseline -8.9). Recommendation, not run: ranked 1, a local signed work-bias term (squared per-density-bin mean of kinetic-energy error; differs from the failed scene-total energy-similarity term), pass if dense-bin energy-work error at most 1.5x baseline (now 2.1x), tiled drift at tick 100 at most 0.012 (now 0.019), position error at tick 20 at most 1.10x control; ranked 2, a clump-centre (16-nearest-neighbour mean) error term; dropped: tail or top-k, heteroscedastic weighting, more noise injection, Huber. Cost about 0.3 GPU-hours per short fine-tune arm, 5.1 for a full retrain. Literature review (25 arXiv abstracts read, not full text) has no verified sources for several loss families; treat as pointers only."),
 ("sup", "14:30", "supervisor", "User asked to pursue the local work-bias loss direction, add per-body-mass support, and produce a video of the planetesimal-disk run. Dispatched: local work-bias loss term test (separate arms for the work-bias and clump-centre terms, queued behind the retrain for a Polaris GPU), and an agent for the disk video plus per-body-mass design, mass-ratio curriculum dataset and training code (no training until a GPU frees). Retrain job 4050 at about 2 h 40 min, still running."),
 ("sup", "14:40", "supervisor", "Planetesimal-disk video rendered (videos/protodisk_real_reference_faceon_edgeon_plus_central_force_proxy.mp4, 500 frames, sandbox GPU, server stopped) and per-body-mass support designed (docs/theory/mass-conditioned-scatter-field.md, commits 0f1c7d8 and 79a6c70). Fresh reviewer: real-design panels change slowly (disk eccentricity 10th / 50th / 90th percentile 0.004 / 0.010 / 0.018 to 0.020 / 0.059 / 0.146); proxy panels start identical and diverge by about tick 1800, reference thins out while the model stays a tight central clump; no stripes or black frames; overlay text small and overlapping in corners, not fixed. Design: mass-weighted deposit and pair term already linear in source mass; earlier mass fine-tunes failed the in-distribution gate; new are two mass-range input channels, soft-clipped inputs and a per-scene geometric-mean mass unit (keeps tiling seamless); new input columns start at zero so unit-mass behaviour is reproduced exactly (unit-tested). Pass and kill lines pre-registered (planet semi-major-axis error at most 0.02, kill above 0.15; disk at most 0.05, kill above 0.25; in-distribution gate 1.15x, kill above 1.5x). Curriculum reaches 1000:1 star-to-planet and 1e4:1 star-to-disk-particle; the real design's about 6.7e4:1 needs a later stage. Training code ready (3 arms, 7 stages x 600 iterations, about 1 GPU-hour estimated), needs a tier-0 checkpoint as the initial checkpoint; GPU path untested; not submitted, waiting for a free GPU behind jobs 4050, 4051 and 4052."),
]

def fix(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\bKILL\b", K, s)
    s = re.sub(r"\bPASS(es)?\b", lambda m: P + (m.group(1) or ""), s)
    return s

out = ""
for g, tm, who, txt in E:
    out += '<p class="e g-%s"><time>%s</time><span class="t"><b class="who">%s</b>%s</span></p>\n' % (g, tm, who, fix(txt))

marker = 'Running agents told. Legacy labels remain in older log text until the glossary lands.</span></p>\n'
sub1(marker, marker + out)

sub1("results from jobs 4021 and 4024 (unread).",
     "results from jobs 4050, 4051 and 4052 (running or pending). Transcript links are not regenerated for entries after 01:55 on 2026-10-26.")
h = h.replace("Dispatched 01:55 10-26; glossary written on its branch", "Dispatched 01:55 10-26; glossary written on its branch")
open(OUT, "w", encoding="utf-8").write(h)
print(len(h))
