"""Rejection results for the report -> results/rejection_results.json, figures/fig_rejection.pdf, tables/tab_reject.tex"""
import glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.reject import CATEGORIES, rejection_rates, rejection_rows

MODELS = {"Qwen3.5-4B": "qwen35-4b", "Qwen3.5-9B": "qwen35-9b"}
CONDS = {"zero-shot": "zs_p_mc_invalid_", "zero-shot + thinking": "zs_think_p_mc_invalid_",
         "few-shot": "fs-f2_invalid_", "few-shot + thinking": "fs-f2_think_invalid_"}
out = {}
for m, tag in MODELS.items():
    for c, pre in CONDS.items():
        d = sorted(glob.glob(str(ROOT / "outputs/runs" / f"{pre}{tag}_2*")))
        assert len(d) == 1, (m, c, d)
        rows = rejection_rows(d[0])
        out[f"{m}|{c}"] = {"run": Path(d[0]).name, "rates": rejection_rates(rows),
                           "mean_slos_when_accepted": float(np.mean([r["n_slos"] for r in rows if r["outcome"] == "accepted"] or [0]))}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/rejection_results.json").write_text(json.dumps(out, indent=1))

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
cols = ["#b5bdc4", "#7f8c8d", "#7fb0e6", "#2c6fbb"]
x = np.arange(len(CATEGORIES) + 1)
for ax, m in zip(axes, MODELS):
    for i, c in enumerate(CONDS):
        ax.bar(x + (i - 1.5) * 0.2, [out[f"{m}|{c}"]["rates"][k]["rate"] for k in CATEGORIES + ["overall"]], 0.2, color=cols[i], label=c)
    ax.set_xticks(x); ax.set_xticklabels(CATEGORIES + ["overall"], fontsize=7)
    ax.set_title(m, fontsize=8); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("rejection rate")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); (ROOT / "figures").mkdir(exist_ok=True); fig.savefig(ROOT / "figures/fig_rejection.pdf"); plt.close(fig)

L = ["\\begin{tabular}{llrrrrr}", "\\toprule", "Model & Run & Ambig. & Confl. & Malf. & Unsupp. & All \\\\", "\\midrule"]
ab = {"zero-shot": "ZS", "zero-shot + thinking": "ZS+T", "few-shot": "FS", "few-shot + thinking": "FS+T"}
for m in MODELS:
    for c in CONDS:
        r = out[f"{m}|{c}"]["rates"]
        L.append(f"{m if c == 'zero-shot' else ''} & {ab[c]} & " + " & ".join(f"{r[k]['rate']:.2f}" for k in CATEGORIES + ["overall"]) + " \\\\")
    if m != list(MODELS)[-1]:
        L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "tables/tab_reject.tex").write_text("\n".join(L) + "\n")
for k, v in out.items():
    print(k, {c: round(x["rate"], 2) for c, x in v["rates"].items()}, "unparse", v["rates"]["overall"]["unparseable"], "slos", round(v["mean_slos_when_accepted"], 2))
