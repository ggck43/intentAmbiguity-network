"""Recompute the report numbers that do not come from the make_midterm_* scripts -> results/report_claims.json"""
import glob, json, re, statistics, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as A
from src import fewshot as F
from src import grammar as G
from src.data_io import load_records, load_snapshot
from src.parse import parse_response
from src.score import canon_list, load_predictions, per_record, prf

records = load_records(); snap = load_snapshot()
valid = [r for r in records if r["meta"]["status"] == "valid"]
expr = lambda r: r["input"]["intent"]["expression"]
svc = lambda r: r["input"]["service"] if isinstance(r["input"].get("service"), str) else r["input"]["intent"].get("service", "")
out = {}


def P(pattern):
    d = [x for x in glob.glob(str(ROOT / "outputs/runs" / pattern)) if "invalid" not in x]
    assert len(d) == 1, (pattern, d)
    return load_predictions(Path(d[0]) / "predictions.jsonl")


def pooled(preds):
    rows = per_record(records, preds)
    return prf(sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows))


# 1 habit tag (paper prompt, 4B zero-shot): predicted highest/lowest tag on an SLO whose gold has none
p = P("zs_qwen35-4b_2*")
tag = 0
for r in valid:
    pr = p.get(r["index"])
    if not (pr and pr.get("parse_ok")): continue
    gold_ext = {(t[0], t[1]): t[3] for t in canon_list(r["label"]["slos"])}
    tag += sum(1 for t in canon_list(pr["parsed"]["slos"]) if t[3] in ("highest", "lowest") and (t[0], t[1]) in gold_ext and gold_ext[(t[0], t[1])] is None)
out["habit_tag_on_thresholds_4B_paper_zs"] = tag
out["f1_4B_paper_zs"] = pooled(p)["f1"]
out["f1_4B_paper_zs_ignoring_tag"] = float(A.diagnostic_f1(records, p, A.drop_ext, "x")["total"])

# 2 units
gold = [canon_list(r["label"]["slos"]) for r in valid]
out["gold_cpu_capacity_slos"] = sum(1 for g in gold for t in g if t[1] == "cpu_capacity")
out["gold_cpu_capacity_units"] = dict(Counter(t[5] for g in gold for t in g if t[1] == "cpu_capacity"))
conv = Counter(); gb_req = 0
for r in valid:
    m = re.search(r"(\d+(?:\.\d+)?)\s*GB", expr(r))
    for t in canon_list(r["label"]["slos"]):
        if t[1] in ("memory_capacity", "storage_capacity") and t[3] is None and m and isinstance(t[4], float):
            gb_req += 1
            conv["binary" if abs(t[4] - float(m.group(1)) * 2**30) < 1 else "other"] += 1
out["gold_GB_requests"] = gb_req; out["gold_GB_binary"] = conv["binary"]


def decimal_gb(preds):  # every GB number in the intent is a candidate
    n = 0
    for r in valid:
        pr = preds.get(r["index"]); nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*GB", expr(r))]
        if not (nums and pr and pr.get("parse_ok")): continue
        for t in canon_list(pr["parsed"]["slos"]):
            if t[1] in ("memory_capacity", "storage_capacity") and t[3] is None and isinstance(t[4], float) and any(abs(t[4] - x * 1e9) < 1 for x in nums):
                n += 1
    return n
out["decimal_GB_9B_fs1_think"] = decimal_gb(P("fs-f1_think_qwen35-9b_2*"))


def mib1000(preds):  # "1 GB = 1000 MiB" slips: N x 1000 x 2^20
    n = 0
    for r in valid:
        pr = preds.get(r["index"]); m = re.search(r"(\d+(?:\.\d+)?)\s*GB", expr(r))
        if not (m and pr and pr.get("parse_ok")): continue
        for t in canon_list(pr["parsed"]["slos"]):
            if t[1] in ("memory_capacity", "storage_capacity") and t[3] is None and isinstance(t[4], float) and abs(t[4] - float(m.group(1)) * 1000 * 2**20) < 1:
                n += 1
    return n
out["gb_as_1000MiB_E4B_fs2"] = mib1000(P("fs-f2_gemma-4-e4b-it_2*"))

# 3 null lookups (zero-shot millicores)
for k, pat in (("9B", "zs_p_mc_qwen35-9b_2*"), ("E4B", "zs_p_mc_gemma-4-e4b-it_2*")):
    out[f"null_lookups_{k}_zs"] = int(A.extremum_outcomes(records, P(pat)).get("right op, value null (snapshot not read)", 0))

# 4 scope conflict
sc = 0; recs_sc = set()
ext = A.snapshot_extrema(snap)
ex = A.extremum_choices(records, P("fs-f2_think_qwen35-9b_2*"), snap)
d = ex[(ex["gold uses"] == "global") & ex["layer"].isin(["edge", "cloud"])]  # a layer is requested but gold uses the global extreme
out["scope_conflict_slos"] = int(len(d)); out["scope_conflict_records"] = int(d["index"].nunique())

# ceiling diagnostics (as in notebooks 4 and 6)
for k, pat in (("Qwen4B", "fs-f2_think_qwen35-4b_2*"), ("Qwen9B", "fs-f2_think_qwen35-9b_2*"), ("E2B", "fs-f2_think_gemma-4-e2b-it_2*"), ("E4B", "fs-f2_think_gemma-4-e4b-it_2*")):
    preds = P(pat); t = pooled(preds); ex = A.extremum_choices(records, preds, snap)
    dd = ex[ex["gold uses"].isin(["global", "layer"])]; n = int(((dd["gold uses"] == "global") & (dd["model uses"] == "layer")).sum())
    out[f"ceiling_{k}"] = {"FN": t["fn"], "scope_conflict_unmatched": n,
                           "diag_f1": 2 * (t["tp"] + n) / (2 * (t["tp"] + n) + (t["fp"] - n) + (t["fn"] - n))}

# 5 CoT: 9B answers with bare JSON
p = P("zs_cot_qwen35-9b_2*")
out["9B_cot_bare_json"] = sum(1 for v in p.values() if v["raw_text"].strip().startswith("{") and v["raw_text"].strip().endswith("}"))

# 6 example coverage (notebook 3 definition: gold SLOs whose (type, metric, cmp, ext) is shown)
key = lambda t: (t[0], t[1], t[2], t[3])
paper = [parse_response(v.split("Output:\n", 1)[1])["parsed"]["slos"] for v in F.PAPER_EXAMPLES.values()]
mine = [e["slos"] for e in F.EXAMPLES.values()]
tot = sum(len(g) for g in gold)
cov = lambda sets_: sum(1 for g in gold for t in g if key(t) in {key(x) for s in sets_ for x in canon_list(s)}) / tot
out["coverage_paper3"] = cov(paper); out["coverage_fs2"] = cov(mine)

# 7 prompt tokens (median over records)
for k, pat in (("zs", "zs_p_mc_qwen35-4b_2*"), ("fs2", "fs-f2_qwen35-4b_2*")):
    out[f"prompt_tokens_median_{k}"] = statistics.median(v["prompt_tokens"] for v in P(pat).values())

# 8 thinking cap (4B zero-shot thinking, millicores)
p = P("zs_think_p_mc_qwen35-4b_2*")
out["4B_zs_think_forced_close"] = sum(1 for v in p.values() if v.get("forced_close")); out["n"] = len(p)

# 9 vocabulary and re-wording loss
out["word_types_valid"] = len({t for r in valid for t in G.tokens(expr(r), svc(r))})
loss = {}
for m in ("4b", "9b"):
    for e in ("fs-f2", "fs-f2_think"):
        base = pooled(P(f"{e}_qwen35-{m}_2*"))["f1"]
        sh = [pooled(P(f"{e}_shift{k}_qwen35-{m}_2*"))["f1"] for k in (1, 2)]
        loss[f"{e} {m}"] = {"base": base, "shift": sh, "max_loss": max(base - s for s in sh)}
out["reword"] = loss

(ROOT / "results/report_claims.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
