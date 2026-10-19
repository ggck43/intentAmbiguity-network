"""Collect the midterm results (Table 7 style, level-wise F1, EM/Jaccard) from the run directories.
Writes results/midterm_results.json. Does not touch scores.json in the run dirs."""
import glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_io import load_records
from src.score import load_predictions, per_record, prf

MODELS = {"Qwen3.5-4B": "qwen35-4b", "Qwen3.5-9B": "qwen35-9b", "Gemma 4 E2B": "gemma-4-e2b-it", "Gemma 4 E4B": "gemma-4-e4b-it"}
CONDS = {"zero-shot": "zs_p_mc_{m}_2*", "zero-shot + thinking": "zs_think_p_mc_{m}_2*",
         "few-shot": "fs-f2_{m}_2*", "few-shot + thinking": "fs-f2_think_{m}_2*"}


def find(pattern):
    d = [x for x in glob.glob(str(ROOT / "outputs/runs" / pattern)) if not any(t in x for t in ("shift", "energy", "deploy", "invalid"))]
    assert len(d) == 1, (pattern, d)
    return d[0]


def pooled(rows):
    return prf(sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows))["f1"]


def main():
    records = load_records()
    level = {r["index"]: r["meta"]["level"] for r in records}
    out = {}
    for m, tag in MODELS.items():
        for c, pat in CONDS.items():
            d = find(pat.format(m=tag))
            rows = per_record(records, load_predictions(Path(d) / "predictions.jsonl"))
            for r in rows:
                r["level"] = level[r["index"]]
            n = len(rows)
            res = {"run": Path(d).name,
                   "f1": {s: pooled([r for r in rows if s == "total" or r["source"] == s]) for s in ("real", "synthetic", "total")},
                   "em": sum(r["em"] for r in rows) / n, "jaccard": sum(r["jaccard"] for r in rows) / n,
                   "parse_rate": sum(r["parse_ok"] for r in rows) / n, "trunc_rate": sum(r["truncated"] for r in rows) / n,
                   "level_f1": {s: {L: pooled([r for r in rows if r["level"] == L and (s == "total" or r["source"] == s)]) for L in range(1, 6)}
                                for s in ("real", "synthetic", "total")},
                   "level_n": {L: sum(r["level"] == L for r in rows) for L in range(1, 6)},
                   "level_em": {L: sum(r["em"] for r in rows if r["level"] == L) / max(1, sum(r["level"] == L for r in rows)) for L in range(1, 6)}}
            out[f"{m}|{c}"] = res
    # paper-faithful prompts (cores in the system prompt; fs1 = paper's three examples + three of ours) for the baseline check
    for m in ("Qwen3.5-4B", "Qwen3.5-9B"):
        for c, pat in {"paper zero-shot": "zs_{m}_2*", "paper few-shot (fs1)": "fs-f1_{m}_2*"}.items():
            d = find(pat.format(m=MODELS[m]))
            rows = per_record(records, load_predictions(Path(d) / "predictions.jsonl"))
            out[f"{m}|{c}"] = {"run": Path(d).name, "em": sum(r["em"] for r in rows) / len(rows),
                               "f1": {s: pooled([r for r in rows if s == "total" or r["source"] == s]) for s in ("real", "synthetic", "total")}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/midterm_results.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if "level_f1" not in v:
            print(f"{k:42s} R {v['f1']['real']:.3f} S {v['f1']['synthetic']:.3f} T {v['f1']['total']:.3f}")
            continue
        print(f"{k:42s} R {v['f1']['real']:.3f} S {v['f1']['synthetic']:.3f} T {v['f1']['total']:.3f} EM {v['em']:.3f} J {v['jaccard']:.3f}")


if __name__ == "__main__":
    main()
