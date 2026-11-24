# Polaris home inventory and deletion table (2026-11-23, INCOMPLETE)

Read-only. Nothing deleted, moved, written, submitted or run on a GPU on Polaris. Job 13236 was RUNNING (gpu01, elapsed 1:45 at check) and nothing it uses was touched.

## What the available tools could and could not do

The only Polaris tools in this session were polaris_ls, polaris_download, polaris_job_status and polaris_status. There is no shell, so there is no du, md5sum, mtime, or quota query. Consequences:

- Home listing is capped at 1000 entries and came back truncated. Home holds more than 1000 top-level entries; the rest are unseen. Counts therefore reconcile only for the 1000 seen.
- Directory sizes are not bytes (the tool returns entry counts for directories). Directory byte sizes, newest-file dates, total home usage and quota are NOT available.
- Polaris-side md5 cannot be computed, so "local copy" for files is name plus byte size match only, against ~/polaris-mcp-files and ~/Research/archive/Augur (110412 local files). Directories were matched by name only; contents were not enumerated (191 project directories would need 191+ narrow listings and still give no md5).
- Because directory contents are unverified, no directory is marked DELETE-CANDIDATE. A real deletion table needs a shell on Polaris (du -sb, find -printf, md5sum) or an approved job-free session; see Next step.

## Reconciliation (1000 entries seen = 214 directories + 786 files)

| Class | Dirs | Files | Notes |
|---|---|---|---|
| NOT-TOUCHED | 4 | 9 | stress-test material, names only below |
| KEEP-IN-USE | 2 | 2 | job 13236 |
| EXCLUDED-NON-PROJECT | 17 | 6 | dotdirs, bin, node, conda, shell files |
| DOWNLOADED (Polaris-only, copied) | 0 | 33 | 81.2 MB, md5 in polaris-only-downloads-md5.txt |
| DELETE-CANDIDATE (file name+size match local) | 0 | 230 | 107.6 MB total; md5 unverified on Polaris side |
| DOWNLOAD-FIRST | 191 | 506 | 191 dirs not enumerated; 506 files listed below by group |
| Total | 214 | 786 | 1000 |

Per-entry classification for all 1000: docs/archive/polaris-inventory-rows.tsv.

## Table by group

| Path group | Size | Producing job/experiment | Local copy | Polaris-only items | Action | Reason |
|---|---|---|---|---|---|---|
| augur_architecture_push_stage1_mass_double_iterations_2026_11_23 | 6 entries (ckpt dir, scripts, results, model, tests) | job 13236 | n/a | live run state | KEEP-IN-USE | running job work dir |
| augur/ (98 entries) | not measured | holds checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt, the warm-start init used by the stage-1 mass launcher (scripts/polaris_learned-per-body-property-channels-mass-conditioned-stage-1.sh line 25) | E_ms_kp_pot_v_g128.pt: yes, ~/polaris-mcp-files/E_ms_kp_pot_v_g128.pt, 3200305 bytes, same as Polaris size (size only) | unknown | KEEP-IN-USE | contains the init checkpoint 13236 likely needs |
| augur-architecture-push-stage-one-code.tgz | 798412 B | code tarball most plausibly used by 13236 (unconfirmed; exact tarball name not in repo) | not checked | unknown | KEEP-IN-USE | do not remove until 13236 ends |
| augur-...-stage-1-double-iterations-13236.log | 0 B | job 13236 | n/a | live log | KEEP-IN-USE | running |
| augur_architecture_push_stage1_{mass,distribution}_2026_11_23_fresh, *_mass, *_distribution, *_lawzoo | entries 8-10 each | jobs 13166/13167, stale 13154/13155 runs | not found by name | probe results (scored in commit 888b89f) | DOWNLOAD-FIRST | verify results json/ckpt copied locally before any deletion |
| 191 other project dirs (augur-*, augur_*, bounce*, t0*, edge_ck, truthval, etc.) | not measured | experiment-log.md jobs, not mapped per directory in this pass | 11 have a same-name local dir (__pycache__, augur-contact, augur-fix, augur-smoke, augur-two-body-upload, gpuaudit, recabl_stage, scatterpc, scripts, tln, truthval), contents unverified; 180 have no same-name local dir | unknown | DOWNLOAD-FIRST | cannot show local copy without enumeration and md5 |
| 230 files with local name+size match (mostly logs, small scripts, tarballs, jsons) | 107.6 MB | various | yes, name+size only | none | DELETE-CANDIDATE (conditional) | md5 not confirmable; confirm with a Polaris-side md5sum before deleting |
| 33 Polaris-only files | 81.2 MB | tiered c2 diagnosis jsons, lt/lf/lw/le/sx/fb results and code tars, gns_style.pt, softcap_E_old.pt, scaling_submitter logs, scripts | now yes, ~/polaris-mcp-files/archive-polaris-only-2026-11-23/ (33 files, MD5SUMS) | none after download | DELETE-CANDIDATE after review of the download | size verified by the tool on transfer |
| p1e8_main_results.tgz | 6,779,612,978 B | 1e8 main results (job not mapped) | no | yes, only on Polaris | DOWNLOAD-FIRST | 6.78 GB, over 200 MB cap, not downloaded |
| 5 files same name, size differs (polaris_scatter_lj_h1.sh, polaris_sae_interp.sh, polaris_slj_hv.sh, slj_hv_code.tgz, ftde_code.tgz) | small | | partial, versions differ | possibly | DOWNLOAD-FIRST | compare versions |
| 495 Polaris-only job .log files, bounce, bounce-opt, bounce-ood, two zero-byte files | 65.0 MB for the 495 logs | slurm logs | no copy by name | logs only | DOWNLOAD-FIRST | small; Pilot to say whether logs matter |
| 17 dirs + 6 files (.ssh, .config, .cache, .triton*, .inductor-cache-d3, .apptainer, .jupyter, .npm, .vnc, .esmtp_queue, Desktop, bin, ondemand, py312hdr_conda, node-v24.14.1-linux-x64, node.tar.xz, dotfiles) | not measured | not project | n/a | n/a | EXCLUDED-NON-PROJECT | not Augur material |

## Not touched (stress tests, by name only)

Directories: augur-nvlink-peer-bandwidth-and-allreduce-stress-on-granted-gpus, augur-pcie-bus-host-device-and-cross-group-peer-stress-on-granted-gpus, augur_cross_group_pairwise_nccl, augur_exact_all_pairs_1e7. Plus 9 files whose names match the stress patterns (names in polaris-inventory-rows.tsv). augur-bh-1e10-stress was not in the 1000 entries seen. None were listed in detail.

## Anomalies

- Home listing exceeds 1000 entries; the unseen remainder may include more project material (for example the 1e10 stress dir and the stage-one tarball variants).
- Total home usage and quota: unavailable.
- Name and size matches are weak evidence; extracted tarballs and renamed local copies would show as "none".

## Next step needed for a real deletion table

A job-free shell on Polaris (or the Pilot running one command block) producing: du -sb per top-level entry, newest mtime per entry, md5sum of result jsons and checkpoints, and quota. Then compare md5 against the local archive and promote DOWNLOAD-FIRST rows.
