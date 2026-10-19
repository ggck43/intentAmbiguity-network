"""LaTeX table snippets for the midterm report -> tables/*.tex"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results/midterm_results.json").read_text())
E = json.loads((ROOT / "results/error_analysis.json").read_text())
MODELS = ["Qwen3.5-4B", "Qwen3.5-9B", "Gemma 4 E2B", "Gemma 4 E4B"]
f3 = lambda x: f"{x:.3f}"
(ROOT / "tables").mkdir(exist_ok=True)

# baseline check against the paper (GPT-4.1 mini, Table 7)
paper = {"zs": (.296, .345, .331), "fs": (.711, .768, .751)}
L = []
def row(name, v, ref):
    d = (v[2] - ref[2]) / ref[2] * 100
    L.append(f"{name} & {f3(v[0])} & {f3(v[1])} & {f3(v[2])} & {d:+.1f}\\% \\\\")
L.append("\\textit{Zero-shot} & & & & \\\\")
L.append(f"GPT-4.1 mini (paper) & {f3(paper['zs'][0])} & {f3(paper['zs'][1])} & {f3(paper['zs'][2])} & -- \\\\")
for m in MODELS[:2]:
    f = R[f"{m}|paper zero-shot"]["f1"]; row(f"{m} (ours)", (f["real"], f["synthetic"], f["total"]), paper["zs"])
L.append("\\textit{Few-shot} & & & & \\\\")
L.append(f"GPT-4.1 mini (paper) & {f3(paper['fs'][0])} & {f3(paper['fs'][1])} & {f3(paper['fs'][2])} & -- \\\\")
for m in MODELS[:2]:
    f = R[f"{m}|paper few-shot (fs1)"]["f1"]; row(f"{m} (ours)", (f["real"], f["synthetic"], f["total"]), paper["fs"])
(ROOT / "tables/tab_baseline_body.tex").write_text("\n".join(L) + "\n")

# main table: model x thinking, zero-shot and few-shot, Real/Synthetic/Total F1
L = []
best = {}
for s in ("real", "synthetic", "total"):
    for p in ("zero-shot", "few-shot"):
        best[(p, s)] = max(R[f"{m}|{p}{t}"]["f1"][s] for m in MODELS for t in ("", " + thinking"))
for m in MODELS:
    for t, tl in (("", "off"), (" + thinking", "on")):
        cells = []
        for p in ("zero-shot", "few-shot"):
            for s in ("real", "synthetic", "total"):
                v = R[f"{m}|{p}{t}"]["f1"][s]
                cells.append(f"\\textbf{{{f3(v)}}}" if abs(v - best[(p, s)]) < 1e-12 else f3(v))
        L.append(f"{m if tl == 'off' else ''} & {tl} & " + " & ".join(cells) + " \\\\")
    if m != MODELS[-1]:
        L.append("\\midrule")
(ROOT / "tables/tab_main_body.tex").write_text("\n".join(L) + "\n")

# error table
L = []
SET = [("", "ZS"), (" + thinking", "ZS+T")]
for m in MODELS:
    for p, pl in (("zero-shot", "ZS"), ("few-shot", "FS")):
        for t, tl in (("", ""), (" + thinking", "+T")):
            e = E[f"{m}|{p}{t}"]; g = e["errors"]; x = e["extremum"]
            L.append(f"{m if (p == 'zero-shot' and not t) else ''} & {pl}{tl} & {g.get('missed',0)+g.get('unparseable',0)} & {g.get('spurious',0)} & {g.get('wrong op/cmp',0)} & {g.get('wrong value/unit',0)} & {x.get('right op, value null (snapshot not read)',0)} & {x.get('right op, correct number',0)} \\\\")
    if m != MODELS[-1]:
        L.append("\\midrule")
(ROOT / "tables/tab_errors_body.tex").write_text("\n".join(L) + "\n")
print("ok")


# wrap each body into a complete tabular so the report can \\input it outside a tabular
WRAP = {
    "tab_baseline": ("lcccr", " & Real & Synth. & Total & vs. paper \\\\"),
    "tab_main": ("llcccccc", None),
    "tab_errors": ("llrrrrrr", None),
}
HEAD_MAIN = ("\\toprule\n & & \\multicolumn{3}{c}{Zero-shot (millicores prompt)} & \\multicolumn{3}{c}{Few-shot (fs-f2)} \\\\\n"
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nModel & Thinking & Real & Synthetic & Total & Real & Synthetic & Total \\\\\n\\midrule")
HEAD_ERR = "\\toprule\nModel & Run & Missed & Spur. & Op/cmp & Val./unit & Null & Right \\\\\n\\midrule"
for name, (cols, head) in WRAP.items():
    body = (ROOT / f"tables/{name}_body.tex").read_text()
    if name == "tab_baseline":
        h = "\\toprule\n" + head + "\n\\midrule"
    else:
        h = HEAD_MAIN if name == "tab_main" else HEAD_ERR
    (ROOT / f"tables/{name}.tex").write_text(f"\\begin{{tabular}}{{{cols}}}\n{h}\n{body}\\bottomrule\n\\end{{tabular}}\n")
