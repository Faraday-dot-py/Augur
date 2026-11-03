# Re-merging hetero-tiered branches into hetero-tiered-integration

Worktree: .claude/worktrees/hetero-tiered-integration (branch hetero-tiered-integration, off master 59c271c). Nothing pushed. Never delete worktrees/branches; no reset --hard on others' worktrees.

## Source branches (last integrated head)
| Part | Branch | Integrated head |
|---|---|---|
| A LJ energy-gradient mesh, poly-cell potential | hole1-potential-field | 56f5111 |
| B baseline checks | worktree-agent-ac8d0c1d33e4ac2c7 | b1898a5 |
| C model/tiered, train_tiered | worktree-agent-a4059904ca1ecb03c | 61247b3 |
| D tiered_* harness | worktree-agent-a8b09cd8f67d9fd44 | 46480ac |
| E tier-0 extensions | worktree-agent-a96914a7f2c210071 | 160c60a |

## Steps
1. `git log --oneline HEAD..<branch>` per branch (from the integration worktree); skip if empty.
2. `git merge --no-ff <branch>` in order A, B, E, C, D (order used). Commit your own work only in the integration worktree.
3. Known conflict points:
   - docs/theory/heterogeneous-tiered.md: untracked in main checkout; branches carry copies. Keep the union of all status/result sections, de-duplicate, one section per branch (tier-0 ext, upper tiers, harness, hole-1 outcomes, measured baselines).
   - docs/debugging/experiment-log.md: append-only; keep every branch's entries, drop duplicate headings.
   - tests/test_tiered.py (C) vs D's harness tests: D's is tests/test_tiered_harness.py. Keep the rename.
   - scripts/scatter_field.py: A (lj branch, apply_opts/pp_mode from master) and E (8 inert t0 hooks) both edit it; keep both.
4. Copy the main checkout's docs/theory/supervisor-log.md over the worktree copy (main is the live one); never write to the main checkout.
5. Tests on Polaris (CPU, debug partition, no GPU): tar `git ls-files scripts model tests` to ~/polaris-mcp-files, upload with polaris_upload, submit a job that
   - extracts master's scripts/scatter_field.py (`git show master:scripts/scatter_field.py`) and runs `python scripts/scatter_tier0_tests.py --device cpu --make-ref --orig <that> --ckpt E_ms_kp_pot_v_g128.pt`, then `--ckpt ... --skip-train` (flags-off bitwise regression must pass);
   - runs pytest on tests/test_tiered.py, test_tiered_harness.py, test_poly_cell_potential.py, test_scatter_lj.py, test_scatter_lj_energy.py.
6. Commit the merge with the test job id in the message.
