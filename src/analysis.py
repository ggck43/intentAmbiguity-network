"""Error analysis helpers for comparing runs (all S0 strict, valid records only)."""
from collections import Counter

import numpy as np
import pandas as pd

from src.data_io import metric_table
from src.score import BOOT_B, BOOT_SEED, canon_list, per_record

FIELDS = ("rel", "ext", "value", "unit")  # tuple slots 2..5 of a canonical SLO


def record_frame(records, preds, regime):
    """per_record rows + level + regime."""
    df = pd.DataFrame(per_record(records, preds))
    level = {r["index"]: r["meta"]["level"] for r in records}
    return df.assign(level=df["index"].map(level), regime=regime)


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def grouped_f1(df, by):
    g = df.groupby(by)[["tp", "fp", "fn"]].sum()
    g["F1"] = [f1(*x) for x in g[["tp", "fp", "fn"]].to_numpy()]
    g["n"] = df.groupby(by).size()
    g["EM"] = df.groupby(by)["em"].mean()
    return g


def paired_diff_ci(df_a, df_b, B=BOOT_B, seed=BOOT_SEED):
    """F1(b) - F1(a) per slice, with a paired template-cluster bootstrap (same resample for both runs)."""
    a = df_a.set_index("index")
    b = df_b.set_index("index").loc[a.index]
    templates = sorted(a["template"].unique())
    t_idx = a["template"].map({t: i for i, t in enumerate(templates)}).to_numpy()
    idx = np.random.RandomState(seed).randint(0, len(templates), size=(B, len(templates)))
    out = {}
    for name, mask in (("real", a["source"] == "real"), ("synthetic", a["source"] == "synthetic"),
                       ("total", a["source"].notna())):
        m = mask.to_numpy()
        res = []
        for d in (a, b):
            arr = np.zeros((len(templates), 3))
            np.add.at(arr, t_idx[m], d.loc[m, ["tp", "fp", "fn"]].to_numpy())
            s = arr[idx].sum(axis=1)
            den = 2 * s[:, 0] + s[:, 1] + s[:, 2]
            res.append((np.divide(2 * s[:, 0], den, out=np.zeros(B), where=den > 0),
                        f1(*arr.sum(axis=0))))
        diff = res[1][0] - res[0][0]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        out[name] = {"F1 (a)": res[0][1], "F1 (b)": res[1][1], "diff (b - a)": res[1][1] - res[0][1],
                     "CI low": lo, "CI high": hi, "P(diff <= 0)": float((diff <= 0).mean())}
    return pd.DataFrame(out).T


def _key(t):
    return (t[0], t[1])  # (type, metric)


def per_metric(records, preds):
    """TP/FP/FN per (type, metric) bucket; buckets partition the SLOs, so sums equal the totals."""
    acc = Counter()
    for rec in records:
        if rec["meta"]["status"] != "valid":
            continue
        pr = preds.get(rec["index"])
        pred = canon_list(pr["parsed"]["slos"]) if pr and pr.get("parse_ok") else []
        gold = canon_list(rec["label"]["slos"])
        for k in {_key(t) for t in pred + gold}:
            p = Counter(t for t in pred if _key(t) == k)
            g = Counter(t for t in gold if _key(t) == k)
            tp = sum((p & g).values())
            acc[(k, "tp")] += tp
            acc[(k, "fp")] += sum(p.values()) - tp
            acc[(k, "fn")] += sum(g.values()) - tp
    keys = sorted({k for k, _ in acc}, key=lambda k: (str(k[0]), str(k[1])))
    df = pd.DataFrame([{"type": k[0], "metric": k[1] or "-", "tp": acc[(k, "tp")], "fp": acc[(k, "fp")],
                        "fn": acc[(k, "fn")]} for k in keys])
    df["gold"] = df.tp + df.fn
    df["recall"] = df.tp / df.gold.where(df.gold > 0)
    df["precision"] = df.tp / (df.tp + df.fp).where(df.tp + df.fp > 0)
    return df.set_index(["type", "metric"])


def error_taxonomy(records, preds):
    """Classify every unmatched SLO.

    An unmatched prediction and an unmatched gold SLO with the same (type, metric) are paired;
    the pair is labelled by which fields differ (e.g. 'value', 'rel+ext'). Leftovers are
    'missed' (gold only) or 'spurious' (prediction only). Unparseable records count as 'unparseable'.
    """
    cats = Counter()
    for rec in records:
        if rec["meta"]["status"] != "valid":
            continue
        pr = preds.get(rec["index"])
        gold = canon_list(rec["label"]["slos"])
        if not (pr and pr.get("parse_ok")):
            cats[("unparseable", "-")] += len(gold)
            continue
        pred = canon_list(pr["parsed"]["slos"])
        p, g = Counter(pred), Counter(gold)
        common = p & g
        up = list((p - common).elements())
        ug = list((g - common).elements())
        for gt in ug:
            match = next((pt for pt in up if _key(pt) == _key(gt)), None)
            if match is None:
                cats[("missed", gt[1] or gt[0])] += 1
                continue
            up.remove(match)
            diff = [f for f, a, b in zip(FIELDS, match[2:], gt[2:]) if a != b]
            cats[("wrong " + "+".join(diff), gt[1] or gt[0])] += 1
        for pt in up:
            cats[("spurious", pt[1] or pt[0])] += 1
    return pd.DataFrame([{"error": e, "metric": m, "count": c} for (e, m), c in cats.items()],
                        columns=["error", "metric", "count"])


def snapshot_extrema(snapshot):
    rows = metric_table(snapshot)
    out = {}
    for m in rows[0][2]:
        for scope in ("all", "edge", "cloud"):
            vals = [v[m] for _, layer, v in rows if v[m] is not None and scope in ("all", layer)]
            if vals:
                out[(m, scope, "highest")] = max(vals)
                out[(m, scope, "lowest")] = min(vals)
    return out


def extremum_choices(records, preds, snapshot):
    """For each gold highest/lowest SLO: which snapshot value did gold and the model use?

    Values are compared after converting the model's value to the gold unit where the unit
    differs only by cores vs millicores, so this shows the grounding choice, not the scoring.
    """
    ext = snapshot_extrema(snapshot)
    rows = []
    for rec in records:
        if rec["meta"]["status"] != "valid":
            continue
        gold = canon_list(rec["label"]["slos"])
        layer = next((t[4] for t in gold if t[0] == "placement_location"), None)
        pr = preds.get(rec["index"])
        pred = canon_list(pr["parsed"]["slos"]) if pr and pr.get("parse_ok") else []
        for gt in gold:
            if gt[3] not in ("highest", "lowest"):
                continue
            metric, op = gt[1], gt[3]
            g_all, g_layer = ext.get((metric, "all", op)), ext.get((metric, layer, op)) if layer else None

            def which(v, present=True):
                if not present:
                    return "metric not predicted"
                if v is None:
                    return "null value"
                if not isinstance(v, float):
                    return "non-numeric"
                hits = [n for n, ref in (("global", g_all), ("layer", g_layer)) if ref is not None
                        and abs(v - ref) <= 1e-6 * max(1.0, abs(ref))]
                if len(hits) == 2:
                    return "global = layer"
                return hits[0] if hits else "other value"

            cand = [t for t in pred if t[1] == metric and t[0] == gt[0]]
            pv = None
            if cand:
                t = next((c for c in cand if c[3] == op), cand[0])
                pv = t[4]
                if metric == "cpu_capacity" and t[5] == "cores" and isinstance(pv, float):
                    pv *= 1000.0
            rows.append({"index": rec["index"], "source": rec["meta"]["source"], "metric": metric, "op": op,
                         "layer": layer, "gold uses": which(gt[4]), "model uses": which(pv, bool(cand))})
    return pd.DataFrame(rows)


def cpu_capacity_units(records, preds):
    """Units the model used for cpu_capacity, and how many would match gold after cores->millicores."""
    out = Counter()
    for rec in records:
        if rec["meta"]["status"] != "valid":
            continue
        pr = preds.get(rec["index"])
        if not (pr and pr.get("parse_ok")):
            continue
        gold = canon_list(rec["label"]["slos"])
        for t in canon_list(pr["parsed"]["slos"]):
            if t[1] != "cpu_capacity":
                continue
            fixed = t
            if t[5] == "cores" and isinstance(t[4], float):
                fixed = t[:4] + (t[4] * 1000.0, "millicores")
            out[(t[5], "matches gold as emitted" if t in gold else
                 "matches gold after x1000" if fixed in gold else "no match")] += 1
    return pd.DataFrame([{"unit emitted": u, "outcome": o, "count": n} for (u, o), n in out.items()],
                        columns=["unit emitted", "outcome", "count"])


def diagnostic_f1(records, preds, transform, name):
    """Micro-F1 per slice after applying `transform` to every canonical tuple (gold and pred).

    DIAGNOSTIC ONLY: it answers "how much of the gap is this one field?", it is not a score.
    """
    acc = {k: np.zeros(3) for k in ("real", "synthetic", "total")}
    for rec in records:
        if rec["meta"]["status"] != "valid":
            continue
        pr = preds.get(rec["index"])
        pred = [transform(t) for t in canon_list(pr["parsed"]["slos"])] if pr and pr.get("parse_ok") else []
        gold = [transform(t) for t in canon_list(rec["label"]["slos"])]
        p, g = Counter(pred), Counter(gold)
        tp = sum((p & g).values())
        row = np.array([tp, len(pred) - tp, len(gold) - tp])
        acc[rec["meta"]["source"]] += row
        acc["total"] += row
    return pd.Series({k: f1(*v) for k, v in acc.items()}, name=name)


def drop_ext(t):
    return t[:3] + (None,) + t[4:]


def drop_ext_and_unit(t):
    t = drop_ext(t)
    if t[1] == "cpu_capacity" and t[5] == "cores" and isinstance(t[4], float):
        t = t[:4] + (t[4] * 1000.0, "millicores")
    return t


def round_sig(t, n=9):
    """DIAGNOSTIC transform: round a float value to n significant digits (absorbs 1-ulp float noise)."""
    if isinstance(t[4], float):
        t = t[:4] + (float(f"{t[4]:.{n}g}"),) + t[5:]
    return t


def extremum_outcomes(records, preds):
    """Outcome of each gold highest/lowest SLO, matched to the model's SLO with the same (type, metric)
    (preferring one with the same op). cpu_capacity in cores is credited as correct after x1000 here,
    so this isolates the grounding behaviour from the unit issue."""
    out = Counter()
    for rec in records:
        if rec["meta"]["status"] != "valid":
            continue
        pr = preds.get(rec["index"])
        parsed = bool(pr and pr.get("parse_ok"))
        pred = canon_list(pr["parsed"]["slos"]) if parsed else []
        for g in canon_list(rec["label"]["slos"]):
            if g[3] not in ("highest", "lowest"):
                continue
            cand = [t for t in pred if t[:2] == g[:2]]
            if not parsed:
                out["unparseable record"] += 1
            elif not cand:
                out["metric not predicted"] += 1
            else:
                t = next((c for c in cand if c[3] == g[3]), cand[0])
                v = t[4] * 1000.0 if t[1] == "cpu_capacity" and t[5] == "cores" and isinstance(t[4], float) else t[4]
                if t[3] != g[3]:
                    out["wrong op (e.g. threshold / opposite extreme)"] += 1
                elif v is None:
                    out["right op, value null (snapshot not read)"] += 1
                elif not isinstance(v, float):
                    out["right op, non-numeric value"] += 1
                elif abs(v - g[4]) <= 1e-9 * max(1.0, abs(g[4])):
                    out["right op, correct number"] += 1
                else:
                    out["right op, wrong number"] += 1
    return pd.Series(out, name="count", dtype=int)
