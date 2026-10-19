"""Prompt-variant ablation for Qwen3.5-4B/9B from existing runs -> results/prompt_ablation.json, tables/tab_ablation.tex"""
import glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_io import load_records
from src.score import load_predictions, per_record, prf
from src.analysis import per_metric, error_taxonomy, extremum_outcomes

MODELS = {"4B": "qwen35-4b", "9B": "qwen35-9b"}
# (label, run prefix without thinking, run prefix with thinking, shift prefix)
VARIANTS = [
    ("P0 paper zero-shot", "zs_", "zs_think_", None),
    ("P0 + CoT phrase", "zs_cot_", None, None),
    ("P1 millicores", "zs_p_mc_", "zs_think_p_mc_", "zs_p_mc_shift{}_"),
    ("P2 few-shot fs1", "fs-f1_", "fs-f1_think_", "fs-f1_shift{}_"),
    ("P3 few-shot fs2", "fs-f2_", "fs-f2_think_", "fs-f2_shift{}_"),
]
records = load_records()


def run(prefix, tag):
    d = glob.glob(str(ROOT / "outputs/runs" / f"{prefix}{tag}_2*"))
    assert len(d) == 1, (prefix, tag, d)
    return load_predictions(Path(d[0]) / "predictions.jsonl")


def stats(preds):
    rows = per_record(records, preds)
    f1 = prf(sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows))["f1"]
    pm = per_metric(records, preds)
    def bucket(metrics):
        s = pm[pm.index.get_level_values("metric").isin(metrics)][["tp", "fp", "fn"]].sum()
        return prf(s.tp, s.fp, s.fn)["f1"]
    et = error_taxonomy(records, preds)
    ext_err = int(et[et.error.str.contains("ext")]["count"].sum())
    ex = extremum_outcomes(records, preds)
    return {"f1": f1, "em": sum(r["em"] for r in rows) / len(rows), "cpu_cap_f1": bucket(["cpu_capacity"]),
            "memsto_cap_f1": bucket(["memory_capacity", "storage_capacity"]), "wrong_tag": ext_err,
            "null_lookups": int(ex.get("right op, value null (snapshot not read)", 0)),
            "right_lookups": int(ex.get("right op, correct number", 0))}


out = {}
for m, tag in MODELS.items():
    for lab, p, pt, ps in VARIANTS:
        r = {"off": stats(run(p, tag))}
        if pt:
            r["on"] = stats(run(pt, tag))
        if ps:
            r["shift_f1"] = [stats(run(ps.format(k), tag))["f1"] for k in (1, 2)]
        out[f"{m}|{lab}"] = r
        print(m, lab, {k: (round(v["f1"], 3) if isinstance(v, dict) else [round(x, 3) for x in v]) for k, v in r.items()},
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["off"].items()})
(ROOT / "results/prompt_ablation.json").write_text(json.dumps(out, indent=1))

L = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
     "Prompt variant & F1 & F1 & CPU-cap & Mem/sto & Wrong & Null & Shift \\\\",
     " & off & on (+T) & F1 & cap F1 & tag & lookups & F1 \\\\", "\\midrule"]
for m in MODELS:
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{Qwen3.5-{m}}}}} \\\\")
    for lab, *_ in VARIANTS:
        r = out[f"{m}|{lab}"]; o = r["off"]
        on = f"{r['on']['f1']:.3f}" if "on" in r else "--"
        sh = f"{sum(r['shift_f1'])/2:.3f}" if "shift_f1" in r else "--"
        L.append(f"{lab} & {o['f1']:.3f} & {on} & {o['cpu_cap_f1']:.2f} & {o['memsto_cap_f1']:.2f} & {o['wrong_tag']} & {o['null_lookups']} & {sh} \\\\")
    if m == "4B":
        L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "tables/tab_ablation.tex").write_text("\n".join(L) + "\n")
