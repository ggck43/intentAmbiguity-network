"""Error counts per run for the midterm error analysis -> results/error_analysis.json"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_midterm_results import MODELS, CONDS, find
from src.data_io import load_records
from src.score import load_predictions
from src.analysis import error_taxonomy, extremum_outcomes


def group(e):
    if e in ("missed", "spurious", "unparseable"):
        return e
    f = e.replace("wrong ", "").split("+")
    return "wrong op/cmp" if ("rel" in f or "ext" in f) else "wrong value/unit"


def main():
    records = load_records()
    out = {}
    for m, tag in MODELS.items():
        for c, pat in CONDS.items():
            preds = load_predictions(Path(find(pat.format(m=tag))) / "predictions.jsonl")
            t = error_taxonomy(records, preds)
            t["g"] = t["error"].map(group)
            ext = extremum_outcomes(records, preds)
            out[f"{m}|{c}"] = {"errors": t.groupby("g")["count"].sum().to_dict(),
                               "by_metric": t.groupby("metric")["count"].sum().sort_values(ascending=False).head(4).to_dict(),
                               "extremum": ext.to_dict()}
    (ROOT / "results/error_analysis.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v["errors"], v["extremum"])


if __name__ == "__main__":
    main()
