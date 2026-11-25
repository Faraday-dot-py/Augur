# Secrets scan report

Scan run 2026-11-23. Read-only: nothing was scrubbed, rewritten or removed, and no matched value is printed here (only file paths, counts and pattern names). gitleaks, trufflehog and detect-secrets are not installed, so this is a regex scan written for the task (Python, `re`), not a certified scanner.

## Scope

- Working tree: the main checkout /home/awebb/Research/Augur, excluding `.claude/` and the git directory; 2,204 text files of 5 MB or less (tracked and untracked).
- History: every blob reachable from any ref (`git rev-list --objects --all`): 6,116 blobs, of which 6,095 are 5 MB or smaller and 5,624 of those are text; all 5,624 were scanned. 21 blobs over 5 MB and all binary blobs (npz, npy, pt, images, pyc) were not scanned.
- Not scanned: /home/awebb/polaris-mcp-files, /home/awebb/Research/Augur-backups, sibling worktrees, git commit messages and tag annotations (commit author and committer addresses were checked separately), Polaris itself.
- Patterns: private key blocks, AWS access key ids, GitHub tokens, `sk-` style keys, Slack tokens, Hugging Face tokens, quoted key/secret/token/password/passcode assignments, ssh public key lines, ssh identity flags and key file names, email addresses, ALCF/Polaris host names, Polaris cluster paths (/lus, /eagle, /grand), the username path (/home/awebb, /mnt/c/Users/awebb), Duo/MFA references. File names also checked for `.env*`, `.mcp.json`, `*.pem`, `*.key`, `id_rsa*`, `*credentials*`, `*.token`, `.remember/`, `.claude/`, `.superpowers/` in any commit.

## Result in one line

No credentials found: zero hits for private keys, cloud or GitHub or Slack or Hugging Face or `sk-` tokens, quoted secret assignments, ssh public keys, Polaris host names and Polaris cluster paths, in both the working tree and history. No sensitive-named file (`.env*`, `.mcp.json`, key or credential files) was ever committed on any branch. Findings are personal-information and local-path exposure only.

## Findings (none is a credential)

| id | finding | where | count | severity | note |
| --- | --- | --- | --- | --- | --- |
| 1 | Author and committer email adamwebb@cpp.edu on every commit | all refs | 1,552 commits (one distinct address) | personal data in a public repo | cannot be removed without rewriting history; the README already carries the same address as its contact line (README.md, 1 occurrence on this branch) |
| 2 | Second personal address (a gmail address) in older README text and in docs/web/scatter-page-update-2026-10-03.md; also in the untracked `.remember/today-2026-09-24.done.md` | history: 2 blobs; main checkout README.md on its branch: 1; untracked: 1 | 4 | personal data | not present in the README on this branch (HEAD); present on the main checkout's branch |
| 3 | noreply@anthropic.com | docs/superpowers/plans/2026-09-28-cfd-contact-generalization.md and commit trailers | 8 in file, 65 across history blobs | none | co-author trailer text, public address |
| 4 | Local username in absolute paths (/home/awebb, /mnt/c/Users/awebb) | working tree: 66 files, 766 hits (docs/debugging 18 files, docs/theory 6, scripts 6, `.remember/` 10, `scripts/.remember/` 5, others); history: 24 files, 26 hits | 766 | low | reveals the local account name `awebb` and the WSL layout; no secret |
| 5 | Duo/MFA mentions | working tree 10 files (8 in untracked `.remember/`, docs/theory/supervisor-log.md and docs/theory/supervisor-log-page/supervisor-log-page-reference.html); history 4 blobs | 16 | low | prose about needing a Duo push or avoiding the 3-failure lockout; no passcode digits seen in the tracked lines (checked with digit masking) |
| 6 | Link to a second GitHub repository, Faraday-dot-py/agent-transcript-archive, from docs/theory/supervisor-log-page/supervisor-log-page-reference.html | tracked docs | 36 links in 1 file | check | that repository holds agent conversation transcripts; its visibility and content were not checked here; needs-user |
| 7 | `.mcp.json` in the main checkout root | untracked, not in `.gitignore` | 1 file, 220 bytes | low | contains one MCP server command with an absolute local path; no key or token (checked with value masking) |
| 8 | `.remember/` (596 files) and `scripts/.remember/` | untracked, not in `.gitignore` | 600 files, about 1 MiB | medium if ever added | session handoff notes and hook logs with local paths, Duo mentions and one personal email; not tracked on any branch |
| 9 | Prior audit exists | docs/debugging/branch-push-audit-2026-10-29.md (history only, commit 117398f) | 1 | none | describes an earlier push audit and its secrets/credential regex; its text matched the key-file-name pattern but it is prose, not a key |

Polaris: 160 tracked files mention Polaris by name and 30 scripts do; none contains a host name, user@host string, cluster path or allocation account string that the patterns could find. Job ids, queue names and module names appear, which are not secrets. This does not prove absence: a project or allocation name written in another form would not match.

## Recommendations (not applied)

1. Decide whether the cpp.edu address on all commits is acceptable in the public archive (finding 1). If not, only a history rewrite changes it; none was done.
2. Add `.remember/`, `scripts/.remember/` and `.mcp.json` to `.gitignore` before anyone runs `git add -A` in the main checkout (findings 7, 8). `.gitignore` was not edited here (out of scope).
3. Confirm the visibility of agent-transcript-archive before the main repo is public (finding 6).
4. Optionally run gitleaks over the private full copy and polaris-mcp-files before archiving; they were outside this scan.
