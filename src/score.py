"""S0 raw-strict scoring, micro-F1 slices and template-cluster bootstrap.

Usage: python -m src.score outputs/runs/<run_dir>
Pure function of predictions.jsonl + dataset.json: re-running gives identical scores.json.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data_io import DATASET_PATH, load_records, template_id

BOOT_B = 10_000
BOOT_SEED = 20260101
PAPER_TARGET = None  # no paper number is recorded for this model; do not invent one


# ---------- canonicalisation ----------
def _to_value(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v if v is None else str(v)  # unparseable value stays a string and just won't match


def canon_slo(slo):
    if not isinstance(slo, dict):
        return ("invalid", None, None, None, repr(slo), None)
    t = slo.get("target")
    t = t if isinstance(t, dict) else {}
    typ, metric, unit = slo.get("type"), slo.get("metric"), slo.get("unit")
    cmp_, op, val = t.get("cmp"), t.get("op"), t.get("value")
    if typ == "placement_location":
        return (typ, None, "set", None, str(val).lower(), None)
    ext = op if op in ("highest", "lowest") else None
    if cmp_ is not None:
        rel = cmp_
    elif op in ("<=", ">=", "="):
        rel = op
    elif op == "highest":
        rel = ">="
    elif op == "lowest":
        rel = "<="
    else:
        rel = None
    return (typ, metric, rel, ext, _to_value(val), unit)


def canon_list(slos):
    return [canon_slo(s) for s in (slos or [])]


# ---------- per-record counts ----------
def record_counts(pred, gold):
    """pred/gold: lists of canonical tuples. Multiset comparison."""
    p, g = Counter(pred), Counter(gold)
    inter = sum((p & g).values())
    union = sum((p | g).values())
    tp = inter
    return {
        "tp": tp, "fp": len(pred) - tp, "fn": len(gold) - tp,
        "em": 1.0 if p == g else 0.0,
        "jaccard": 1.0 if union == 0 else inter / union,
    }


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}


# ---------- IO ----------
def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[r["index"]] = r
    return preds


# ---------- scoring ----------
def per_record(records, preds, only_predicted=False):
    """One row per valid record (missing prediction counts as unparseable unless only_predicted)."""
    rows = []
    for rec in records:
        if rec["meta"]["status"] != "valid":
            continue
        if only_predicted and rec["index"] not in preds:
            continue
        pr = preds.get(rec["index"])
        ok = bool(pr and pr.get("parse_ok") and pr.get("parsed"))
        pred = canon_list(pr["parsed"]["slos"]) if ok else []
        c = record_counts(pred, canon_list(rec["label"]["slos"]))
        c.update(index=rec["index"], source=rec["meta"]["source"], template=template_id(rec),
                 parse_ok=ok, truncated=bool(pr and pr.get("truncated")))
        rows.append(c)
    return rows


def bootstrap_ci(rows, B=BOOT_B, seed=BOOT_SEED):
    templates = sorted({r["template"] for r in rows})
    t_idx = {t: i for i, t in enumerate(templates)}
    nT = len(templates)
    stats = {}
    for name, keep in (("total", lambda r: True), ("real", lambda r: r["source"] == "real"),
                       ("synthetic", lambda r: r["source"] == "synthetic")):
        a = np.zeros((nT, 3))
        for r in rows:
            if keep(r):
                a[t_idx[r["template"]]] += (r["tp"], r["fp"], r["fn"])
        stats[name] = a
    idx = np.random.RandomState(seed).randint(0, nT, size=(B, nT))
    out = {}
    for name, a in stats.items():
        s = a[idx].sum(axis=1)  # (B, 3)
        denom = 2 * s[:, 0] + s[:, 1] + s[:, 2]
        f1 = np.divide(2 * s[:, 0], denom, out=np.zeros(B), where=denom > 0)
        lo, hi = np.percentile(f1, [2.5, 97.5])
        out[name] = {"f1_lo": float(lo), "f1_hi": float(hi)}
    out.update(B=B, seed=seed, n_templates=nT)
    return out


def score_run(run_dir, dataset_path=DATASET_PATH, B=BOOT_B, seed=BOOT_SEED, only_predicted=False):
    run_dir = Path(run_dir)
    records = load_records(dataset_path)
    preds = load_predictions(run_dir / "predictions.jsonl")
    rows = per_record(records, preds, only_predicted)
    n = len(rows)

    def pooled(keep):
        sel = [r for r in rows if keep(r)]
        return prf(sum(r["tp"] for r in sel), sum(r["fp"] for r in sel), sum(r["fn"] for r in sel))

    scores = {
        "total": pooled(lambda r: True),
        "real": pooled(lambda r: r["source"] == "real"),
        "synthetic": pooled(lambda r: r["source"] == "synthetic"),
        "em": sum(r["em"] for r in rows) / n,
        "jaccard": sum(r["jaccard"] for r in rows) / n,
        "parse_rate": sum(r["parse_ok"] for r in rows) / n,
        "trunc_rate": sum(r["truncated"] for r in rows) / n,
        "n_valid_scored": n,
        "n_valid_with_prediction": sum(1 for r in rows if r["index"] in preds),
        "bootstrap_ci": bootstrap_ci(rows, B, seed) if B else None,
        "paper_target": PAPER_TARGET,
        "regime": _first(preds, "regime"),
        "model": _first(preds, "model"),
    }
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    (run_dir / "table7_row.md").write_text(table_row(scores))
    return scores


def _first(preds, key):
    vals = {p.get(key) for p in preds.values()}
    return vals.pop() if len(vals) == 1 else sorted(map(str, vals))


REGIME_LABEL = {"zs_cot": "zero-shot CoT", "zs": "zero-shot", "zs_think": "zero-shot + thinking"}


def table_row(s, name=None):
    model = s.get("model") if isinstance(s.get("model"), str) else "model"
    name = name or f"{model.split('/')[-1]} (re-run)"
    regime = REGIME_LABEL.get(s.get("regime"), s.get("regime"))
    ci = s["bootstrap_ci"]
    cell = ((lambda k: f"{s[k]['f1']:.3f}") if ci is None else
            (lambda k: f"{s[k]['f1']:.3f} [{ci[k]['f1_lo']:.3f}, {ci[k]['f1_hi']:.3f}]"))
    return (f"| Model ({regime}, micro-F1, S0 strict) | Real | Synthetic | Total |\n"
            "|---|---|---|---|\n"
            f"| {name} | {cell('real')} | {cell('synthetic')} | {cell('total')} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--no-ci", action="store_true", help="skip the template-cluster bootstrap (point estimates only)")
    ap.add_argument("--only-predicted", action="store_true",
                    help="score only valid records that have a prediction (for smoke runs)")
    a = ap.parse_args()
    s = score_run(a.run_dir, B=0 if a.no_ci else BOOT_B, only_predicted=a.only_predicted)
    ci = s["bootstrap_ci"]
    print(f"valid scored: {s['n_valid_scored']} | parse {s['parse_rate']:.3f} | trunc {s['trunc_rate']:.3f}")
    for k in ("total", "real", "synthetic"):
        rng = "" if ci is None else f" [{ci[k]['f1_lo']:.3f}, {ci[k]['f1_hi']:.3f}]"
        print(f"{k:>9}: F1 {s[k]['f1']:.3f}{rng}  "
              f"P {s[k]['precision']:.3f} R {s[k]['recall']:.3f}  TP/FP/FN {s[k]['tp']}/{s[k]['fp']}/{s[k]['fn']}")
    print(f"EM {s['em']:.3f} | Jaccard {s['jaccard']:.3f}")


if __name__ == "__main__":
    main()
