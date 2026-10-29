import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

COLUMNS = ["id", "kind", "name", "owner", "rating", "rationale", "gpu_slot",
           "state", "queued_since", "updated", "log"]
KINDS = {"job", "agent"}
STATES = {"running", "pending", "idle", "released", "done"}
SLOTS = {"1", "2", "none", "unknown"}
ACTIVE = {"running", "pending", "idle"}
WAITING = {"pending", "idle"}
TIME_FORMAT = "%Y-%m-%d %H:%M"
DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "docs/debugging/priority-registry.md"


def parse_rows(text):
    rows = []
    header_seen = False
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and set(cells[0]) <= set("- :"):
            continue
        if not header_seen:
            header_seen = True
            if cells != COLUMNS:
                return rows, [f"line {number}: header must be {' | '.join(COLUMNS)}"]
            continue
        rows.append((number, cells))
    if not header_seen:
        return rows, ["no table header found"]
    return rows, []


def parse_time(value):
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError:
        return None


def check(text, now, starve_hours=12.0, stale_hours=12.0):
    errors, warnings = [], []
    rows, header_errors = parse_rows(text)
    errors.extend(header_errors)
    if header_errors:
        return errors, warnings
    seen = set()
    records = []
    for number, cells in rows:
        where = f"line {number}"
        if len(cells) != len(COLUMNS):
            errors.append(f"{where}: {len(cells)} cells, expected {len(COLUMNS)}")
            continue
        row = dict(zip(COLUMNS, cells))
        where = f"{where} ({row['id']})"
        if not row["id"]:
            errors.append(f"{where}: empty id")
        elif row["id"] in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(row["id"])
        if row["kind"] not in KINDS:
            errors.append(f"{where}: kind must be one of {sorted(KINDS)}")
        if row["state"] not in STATES:
            errors.append(f"{where}: state must be one of {sorted(STATES)}")
        if row["gpu_slot"] not in SLOTS:
            errors.append(f"{where}: gpu_slot must be one of {sorted(SLOTS)}")
        if not row["owner"]:
            errors.append(f"{where}: empty owner")
        if not row["rationale"]:
            errors.append(f"{where}: empty rationale")
        rating = None
        if row["rating"].isdigit() and 1 <= int(row["rating"]) <= 5:
            rating = int(row["rating"])
        elif row["state"] in ACTIVE:
            warnings.append(f"UNRATED {where}: {row['state']} with rating '{row['rating']}'")
        else:
            errors.append(f"{where}: rating must be 1-5")
        queued = parse_time(row["queued_since"])
        updated = parse_time(row["updated"])
        if queued is None:
            errors.append(f"{where}: queued_since not {TIME_FORMAT}")
        if updated is None:
            errors.append(f"{where}: updated not {TIME_FORMAT}")
        if queued and updated and updated < queued:
            errors.append(f"{where}: updated before queued_since")
        if updated and updated > now + timedelta(minutes=5):
            errors.append(f"{where}: updated is in the future")
        if row["state"] in ACTIVE:
            if updated and now - updated > timedelta(hours=stale_hours):
                warnings.append(f"STALE {where}: not updated for {(now - updated).total_seconds() / 3600:.1f} h")
            if row["state"] in WAITING and queued and now - queued > timedelta(hours=starve_hours):
                warnings.append(f"STARVING {where}: waiting {(now - queued).total_seconds() / 3600:.1f} h, rating {rating}")
            if row["state"] == "running" and row["kind"] == "job" and row["owner"] == "unknown":
                warnings.append(f"UNOWNED {where}: running job with unknown owner")
        row["rating_value"] = rating
        row["queued_time"] = queued
        records.append(row)
    holders = {}
    for row in records:
        if row["state"] == "running" and row["gpu_slot"] in ("1", "2") and row["owner"] != "unknown":
            holders.setdefault(row["owner"], set()).add(row["gpu_slot"])
    for owner, slots in holders.items():
        if slots == {"1", "2"}:
            warnings.append(f"BOTH_GPUS: owner '{owner}' holds GPU 1 and GPU 2 at once")
    running_slots = {r["gpu_slot"] for r in records if r["state"] == "running"}
    for slot in ("1", "2"):
        if slot not in running_slots and "unknown" not in running_slots:
            waiting = [r["id"] for r in records if r["state"] in WAITING and r["kind"] == "job"]
            warnings.append(f"IDLE_SLOT: no running row on GPU {slot}" + (f"; waiting jobs {waiting}" if waiting else ""))
    return errors, warnings


def ranking(text, now):
    rows, _ = parse_rows(text)
    ranked = []
    for _, cells in rows:
        if len(cells) != len(COLUMNS):
            continue
        row = dict(zip(COLUMNS, cells))
        if row["state"] not in ACTIVE or not row["rating"].isdigit():
            continue
        queued = parse_time(row["queued_since"]) or now
        waited = (now - queued).total_seconds() / 3600
        bonus = 2 if waited >= 24 else 1 if waited >= 12 else 0
        base = int(row["rating"])
        effective = base if base >= 5 else min(4, base + bonus) if row["state"] in WAITING else base
        ranked.append((-effective, queued, row["id"], effective))
    ranked.sort()
    return [(rid, eff) for _, _, rid, eff in ranked]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate docs/debugging/priority-registry.md")
    parser.add_argument("registry", nargs="?", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--now", help=f"override current time, {TIME_FORMAT} (testing)")
    parser.add_argument("--starve-hours", type=float, default=12.0)
    parser.add_argument("--stale-hours", type=float, default=12.0)
    parser.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    args = parser.parse_args(argv)
    now = parse_time(args.now) if args.now else datetime.now()
    if now is None:
        parser.error(f"--now must be {TIME_FORMAT}")
    text = Path(args.registry).read_text()
    errors, warnings = check(text, now, args.starve_hours, args.stale_hours)
    for line in errors:
        print("ERROR", line)
    for line in warnings:
        print("WARN", line)
    print("ranking (effective rating, ties by queued_since):", ", ".join(f"{i}={e}" for i, e in ranking(text, now)))
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
