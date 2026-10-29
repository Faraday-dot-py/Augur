import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_priority_registry as cpr

HEADER = "| " + " | ".join(cpr.COLUMNS) + " |\n| " + " | ".join("---" for _ in cpr.COLUMNS) + " |\n"
NOW = datetime(2026, 10, 27, 12, 0)


def row(id="a", kind="agent", owner="x", rating="3", slot="1", state="running",
        queued="2026-10-27 10:00", updated="2026-10-27 11:00"):
    cells = [id, kind, "name", owner, rating, "why", slot, state, queued, updated, "-"]
    return "| " + " | ".join(cells) + " |\n"


def run(*rows):
    return cpr.check(HEADER + "".join(rows), NOW)


def test_clean_registry():
    errors, warnings = run(row("a", slot="1"), row("b", owner="y", slot="2"))
    assert errors == [] and warnings == []


def test_bad_header():
    errors, _ = cpr.check("| id | kind |\n| --- | --- |\n", NOW)
    assert errors


def test_format_errors():
    errors, _ = run(row(kind="thing"), row("b", state="sleeping"), row("c", slot="3"),
                    row("d", updated="yesterday"), row("a"))
    assert len(errors) == 5


def test_unrated_running_is_flagged():
    _, warnings = run(row(rating="-"))
    assert any(w.startswith("UNRATED") for w in warnings)


def test_rating_out_of_range_on_done_is_error():
    errors, _ = run(row(rating="9", state="done"))
    assert errors


def test_starving_and_stale():
    _, warnings = run(row(state="pending", slot="none", queued="2026-10-26 20:00", updated="2026-10-26 20:00"))
    assert any(w.startswith("STARVING") for w in warnings)
    assert any(w.startswith("STALE") for w in warnings)


def test_both_gpus_one_owner():
    _, warnings = run(row("a", owner="x", slot="1"), row("b", owner="x", slot="2"))
    assert any(w.startswith("BOTH_GPUS") for w in warnings)


def test_idle_slot_hint():
    _, warnings = run(row("a", slot="1"), row("b", kind="job", state="pending", slot="none"))
    assert any(w.startswith("IDLE_SLOT") and "b" in w for w in warnings)


def test_ranking_orders_by_rating_then_age():
    text = HEADER + row("low", rating="2", state="pending", slot="none") + row("high", rating="4") + \
        row("older", rating="4", queued="2026-10-27 09:00")
    assert [i for i, _ in cpr.ranking(text, NOW)] == ["older", "high", "low"]


def test_aging_bonus_capped_at_four():
    text = HEADER + row("w", rating="3", state="pending", slot="none", queued="2026-10-26 10:00", updated="2026-10-27 11:00")
    assert cpr.ranking(text, NOW) == [("w", 4)]
