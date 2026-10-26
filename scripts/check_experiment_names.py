"""Fail on single-letter / letter-number experiment, arm and agent labels. See docs/theory/experiment-naming.md.

Default: added lines in `git diff BASE` (docs, scripts, tests) must be clean, and added argparse
choices must not be labels. --all scans the whole tree and prints counts instead (audit mode).
--root scans another checkout read-only. A label inside "(legacy: ...)" is allowed once per entry.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

KEYWORD = r"(?i:arms?|agents?|experiments?|exp|runs?|variants?|levers?|branch|tier-?arm)"
LABEL = r"(?:[A-Z][0-9][a-zA-Z]?|[A-Z](?:\+[A-Z])+|[A-Z][a-z]?[0-9]*(?![a-z]))"
CONTEXT = re.compile(rf"\b{KEYWORD}\s+({LABEL})\b")
BARE = re.compile(
    r"\b(?:[A-F][0-9]|F[ab]|Fb[LH]\w*|S\+[RJ]|J[LETMA]|ENMT?|Bh[1-6]|P2C?|B1g|H[1-3]|S[12]|gv_(?:R|NG|EN)|Exp ?[0-9]+)\b"
)
LEGACY = re.compile(r"\((?:legacy|formerly)[^)]*\)|`[^`]*`")
CHOICES = re.compile(r"choices=\[([^\]]*)\]")
SHORT = re.compile(r"^[A-Za-z]\d*$|^[A-Za-z]\d+[A-Za-z]$|^[A-Za-z][+\-][A-Za-z]$")
SUFFIXES = {".md", ".sh", ".py", ".txt"}
DIRS = ("docs", "scripts", "tests", "model")
SELF = {"docs/theory/experiment-naming.md", "scripts/check_experiment_names.py", "scripts/experiment_label_allowlist.txt"}


def load_allowlist(root):
    path = root / "scripts" / "experiment_label_allowlist.txt"
    if not path.exists():
        return set()
    return {l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")}


def line_hits(line, allow):
    line = LEGACY.sub("", line)
    hits = [m.group(1) for m in CONTEXT.finditer(line)]
    hits += [m.group(0) for m in BARE.finditer(line)]
    return [h for h in hits if h not in allow]


def choice_hits(line):
    hits = []
    for m in CHOICES.finditer(line):
        for tok in re.findall(r"""["']([^"']*)["']""", m.group(1)):
            if SHORT.match(tok):
                hits.append(tok)
    return hits


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def added_lines(root, base):
    cur, out = None, []
    for ln in git(root, "diff", "-U0", base, "--", *DIRS).splitlines():
        if ln.startswith("+++ b/"):
            cur = ln[6:]
        elif ln.startswith("+") and not ln.startswith("+++") and cur:
            out.append((cur, ln[1:]))
    return out


def all_lines(root):
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and rel.startswith(DIRS) and p.suffix in SUFFIXES and ".claude" not in rel:
            for ln in p.read_text(errors="ignore").splitlines():
                yield rel, ln


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="master")
    ap.add_argument("--root", default=".")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--commits", action="store_true", help="also check commit subjects since BASE")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    allow = load_allowlist(root)
    lines = all_lines(root) if args.all else added_lines(root, args.base)
    bad, counts = [], {}
    for rel, ln in lines:
        if rel in SELF:
            continue
        hits = line_hits(ln, allow) + choice_hits(ln)
        for h in hits:
            counts[h] = counts.get(h, 0) + 1
        if hits:
            bad.append((rel, ln.strip()[:110], hits))
    if args.commits:
        for subj in git(root, "log", "--format=%h %s", f"{args.base}..HEAD").splitlines():
            h = line_hits(subj, allow)
            if h:
                bad.append(("commit", subj[:110], h))
    if args.all:
        print(f"{len(bad)} lines, {sum(counts.values())} labels, {len(counts)} distinct")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"{v:5d} {k}")
        return 0
    for rel, ln, hits in bad:
        print(f"{rel}: {hits}: {ln}")
    if bad:
        print(f"{len(bad)} line(s) use single-letter or letter-number experiment labels; see docs/theory/experiment-naming.md")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
