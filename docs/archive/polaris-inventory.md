# Polaris home inventory and deletion table (2026-11-23)

Read-only. Nothing deleted, moved or overwritten on Polaris. One CPU-only measurement job (13237, debug partition, no GPU, 6 min, exit 0) wrote one new report file; job 13236 was running throughout and nothing it uses was measured or touched. Raw report: docs/archive/polaris-measurement-report-job-13237.txt. Per-entry table (1263 rows, all columns): docs/archive/polaris-inventory-rows.tsv.

## Method and limits

- Measured per project entry: du -sb, newest file date, md5 of every *.pt, *.json, *.npz under 2 GB (6590 files), and size, date and md5 of top-level files (md5 skipped for 14 files over 2 GB inside directories, listed below).
- "Verified local" means the Polaris md5 equals the md5 of a file with the same basename found under ~/polaris-mcp-files or ~/Research/archive/Augur (28186 local candidates hashed). Basename-restricted, so a local copy stored under a different name would show as unverified (conservative).
- Directory DELETE-CANDIDATE needs every pt/json/npz key output verified. Logs, scripts and tarballs inside directories were not md5ed. 57 directories qualify; 41 of them hold no pt/json/npz at all (scripts and logs only), so "key outputs verified" is vacuous there and they need a review of the code before deletion.
- Total home usage: sum of du over the 1263 measured entries is 327.7 GB (excludes stress, job 13236, dotdirs and tool dirs, which were not measured). No per-user quota exists: quota and lfs are unavailable; the filesystem is shared NFS (/home, 931T size, 18T used, 2% use, all users).
- Producing job is filled for only 58 of 259 directories (first job id in experiment-log.md or a scripts/polaris_*.sh reference); the rest are not mapped.

## Reconciliation

| Class | Dirs | Files | Bytes |
|---|---|---|---|
| Measured (1263 entries) | 259 | 1004 | 327.7 GB |
| DELETE-CANDIDATE | 57 | 345 | 0.61 GB dirs, 0.26 GB files |
| DOWNLOAD-FIRST | 200 | 659 | 319.9 GB dirs, 6.88 GB files |
| EXCLUDED-NON-PROJECT (MCP infrastructure: polaris-mcp-files, polaris_mcp_jobs) | 2 | 0 | 8.6 MB |
| Sum | 259 | 1004 | |
| Not measured by design | Stress tests (names only, below), job 13236 files (KEEP-IN-USE), dotdirs, bin, Desktop, ondemand, py312hdr_conda, node. Seen in the earlier 1000-entry listing: 13 stress, 4 keep-in-use, 23 system. The job did not count its exclusions, so the full excluded total is not known. | | |

## Findings

1. Only 57 of 259 directories can be deleted on current evidence, and they are small (0.61 GB). The 200 DOWNLOAD-FIRST directories hold 319.9 GB, dominated by a few huge ones (below).
2. 4455 key outputs (2040 json, 1366 npz, 1049 pt) in the 200 directories have no md5 match locally. Whether they are absent or merely stored under another name/version is not resolved. Downloading thousands of files one by one through the MCP is impractical; proposal: one more read-only job that writes a single tar of the unverified small key files (new file under ~), then one download. Not done, needs approval.
3. Files over 2 GB with no md5 (not downloaded, over the 200 MB cap): showcase 1e9 and kpadext passes (8.0 GB each, 8 files), augur-showcase-kpadext/ckpt/showcase_bh_1e10_kpadext (80.6 GB, this is a showcase checkpoint, not the stress test), augur-solar-generalized scenes_256_0.pt and scenes_256_64.pt (2.5 and 3.1 GB), augur-solar-more-moons, augur-solar-translation and augur-solar-translation-arms scenes_256_0 (2.5 GB each). Plus top-level p1e8_main_results.tgz (6.78 GB, md5 b88d75a7b4b4923821bd1f485bd298bb): not found locally.
4. Top-level files: 345 DELETE-CANDIDATE (md5 match or name and size match local; 167 of these are name and size only), 659 DOWNLOAD-FIRST: 658 logs and scripts with no local name+size match (102.8 MB total, not downloaded, not key outputs) and p1e8_main_results.tgz.
5. Downloaded now (read-only copies, 41 files, 117 MB incl. report, md5 in docs/archive/polaris-only-downloads-md5.txt): the 33 files from the first pass plus 8 more small Polaris-only top-level result and code files. 30 of the downloads with a Polaris-side md5 in the report all matched.

## Largest DOWNLOAD-FIRST directories (key outputs unverified/total)

| Dir | Size | Newest | Producing job | Unverified/total key outputs | Note |
|---|---|---|---|---|---|
| augur-showcase-kpadext | 118.14 GB | 2026-10-28 |  | 2/10 | 5 file(s) over 2 GB, no md5 |
| augur-showcase | 36.70 GB | 2026-10-26 | polaris_showcase_solar_system_orbit_ft2.sh | 5/6 | 4 file(s) over 2 GB, no md5 |
| augur-galaxy-3861 | 33.27 GB | 2026-10-21 |  | 2/6 |  |
| augur-p1e8 | 25.23 GB | 2026-10-22 |  | 27/87 |  |
| augur-galaxy-tile-3873 | 16.67 GB | 2026-10-22 |  | 7/12 |  |
| augur-solar10m | 13.15 GB | 2026-10-19 |  | 215/223 |  |
| augur-galaxy-3855 | 8.64 GB | 2026-10-21 |  | 2/2 |  |
| augur-bhtime | 6.08 GB | 2026-10-12 |  | 9/36 |  |
| augur-solar-generalized | 5.74 GB | 2026-11-02 |  | 3/7 | 2 file(s) over 2 GB, no md5 |
| augur-galaxy-3862 | 5.13 GB | 2026-10-21 |  | 16/24 |  |
| augur_escape | 3.83 GB | 2026-10-21 |  | 254/264 |  |
| augur_slj_hv | 3.37 GB | 2026-10-26 |  | 215/257 |  |
| augur-galaxy-3857 | 3.21 GB | 2026-10-21 |  | 3/3 |  |
| augur-solar-translation-arms | 2.69 GB | 2026-11-02 |  | 8/8 | 1 file(s) over 2 GB, no md5 |
| augur-solar-more-moons | 2.66 GB | 2026-11-02 |  | 3/8 | 1 file(s) over 2 GB, no md5 |
| augur-solar-translation | 2.66 GB | 2026-11-02 |  | 3/7 | 1 file(s) over 2 GB, no md5 |
| augur-galaxy-3856 | 2.56 GB | 2026-10-21 |  | 12/12 |  |
| augur_tiered_c2 | 2.24 GB | 2026-10-26 |  | 60/60 |  |
| a3 | 1.67 GB | 2026-10-10 |  | 18/19 |  |
| augur_h1 | 1.52 GB | 2026-10-24 |  | 62/79 |  |

All other directories, with every column, are in polaris-inventory-rows.tsv.

## KEEP-IN-USE (job 13236, not measured)

augur_architecture_push_stage1_mass_double_iterations_2026_11_23, augur/ (holds the init checkpoint checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt, local copy 3200305 bytes matching size), augur-architecture-push-stage-one-code.tgz (probable tarball, unconfirmed), the 13236 log.

## Not touched (stress tests, by name only)

augur-nvlink-peer-bandwidth-and-allreduce-stress-on-granted-gpus, augur-pcie-bus-host-device-and-cross-group-peer-stress-on-granted-gpus, augur_cross_group_pairwise_nccl, augur_exact_all_pairs_1e7, the job 11245/11246/11247/11252 outputs and any augur-bh-1e10-stress directory, excluded from measurement by name pattern.
