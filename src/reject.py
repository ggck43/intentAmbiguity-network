"""Rejection scoring for the 195 invalid intents (Intent Engine paper, Fig 6 / Eq. 8).

A response is one of:
  rejected      parsed JSON with an empty `slos` list
  unparseable   no JSON object with a `slos` list could be extracted (counted as "no SLO returned")
  accepted      at least one SLO returned (a false accept: the intent is invalid)

Rejection rate (paper-style) = (rejected + unparseable) / N_category.
The strict variant counts only `rejected`.
"""
import json
from pathlib import Path

from src.data_io import load_records

CATEGORIES = ["ambiguous", "conflicting", "malformed", "unsupported"]


def outcome(pred):
    """pred is one predictions.jsonl row."""
    parsed = pred.get("parsed")
    if not pred.get("parse_ok") or not isinstance(parsed, dict) or not isinstance(parsed.get("slos"), list):
        return "unparseable"
    return "rejected" if len(parsed["slos"]) == 0 else "accepted"


def rejection_rows(run_dir):
    """One dict per invalid record: index, category, level, source, outcome, n_slos."""
    run_dir = Path(run_dir)
    recs = {r["index"]: r for r in load_records() if r["meta"]["status"] != "valid"}
    rows = []
    with open(run_dir / "predictions.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            r = recs.get(p["index"])
            if r is None:
                continue
            n = len((p.get("parsed") or {}).get("slos") or []) if isinstance(p.get("parsed"), dict) else 0
            rows.append({"index": p["index"], "category": r["meta"]["status"], "level": r["meta"]["level"],
                         "source": r["meta"]["source"], "outcome": outcome(p), "n_slos": n,
                         "truncated": bool(p.get("truncated")), "completion_tokens": p.get("completion_tokens")})
    return rows


def rejection_rates(rows):
    """{category: {n, rejected, unparseable, accepted, rate, strict_rate}} plus an 'overall' entry."""
    out = {}
    for cat in CATEGORIES + ["overall"]:
        sub = [r for r in rows if cat == "overall" or r["category"] == cat]
        n = len(sub)
        c = {k: sum(r["outcome"] == k for r in sub) for k in ("rejected", "unparseable", "accepted")}
        out[cat] = {"n": n, **c,
                    "rate": (c["rejected"] + c["unparseable"]) / n if n else float("nan"),
                    "strict_rate": c["rejected"] / n if n else float("nan")}
    return out
