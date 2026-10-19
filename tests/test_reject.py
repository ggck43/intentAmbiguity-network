"""Rejection scoring (src/reject.py)."""
from src.reject import outcome, rejection_rates


def test_outcome_classes():
    assert outcome({"parse_ok": True, "parsed": {"slos": []}}) == "rejected"
    assert outcome({"parse_ok": True, "parsed": {"slos": [{"metric": "x"}]}}) == "accepted"
    assert outcome({"parse_ok": False, "parsed": None}) == "unparseable"
    assert outcome({"parse_ok": True, "parsed": {"service": "a"}}) == "unparseable"


def test_rates():
    rows = [{"category": "ambiguous", "outcome": o} for o in ("rejected", "accepted", "unparseable", "rejected")]
    r = rejection_rates(rows)
    assert r["ambiguous"]["n"] == 4 and r["ambiguous"]["rate"] == 0.75 and r["ambiguous"]["strict_rate"] == 0.5
    assert r["overall"]["accepted"] == 1
