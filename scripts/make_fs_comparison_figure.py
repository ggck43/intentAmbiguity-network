"""Single-column bar chart: our few-shot + thinking (fs-f2) vs the paper's few-shot CoT (Table 7) -> figures/fig_fs_vs_paper.pdf"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results/midterm_results.json").read_text())
ours = [(m, R[f"{m}|few-shot + thinking"]["f1"]["total"]) for m in ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen3.5-4B", "Qwen3.5-9B"]]
paper = [("DeepSeek V4-Flash", .756), ("GPT-4.1 mini", .758), ("Claude Sonnet 4.5", .778)]  # paper Table 7, few-shot CoT, total
rows = [(n, v, "paper") for n, v in paper] + [(n, v, "ours") for n, v in ours]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.4, 3.0))
y = range(len(rows))
ax.barh(list(y), [v for _, v, _ in rows], color=["#c9a227" if s == "paper" else "#2c6fbb" for *_, s in rows], height=0.65)
for i, (n, v, s) in enumerate(rows):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
ax.set_yticks(list(y)); ax.set_yticklabels([n for n, *_ in rows], fontsize=7)
ax.axvline(0.941, ls="--", color="gray", lw=0.8)
ax.set_xlim(0, 1.08); ax.set_xlabel("total micro-F1 (521 valid intents)")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color="#c9a227", label="paper: few-shot CoT"), Patch(color="#2c6fbb", label="ours: few-shot + thinking"),
                   Line2D([], [], ls="--", color="gray", lw=0.8, label="paper: Intent Engine, GPT-4.1 mini (0.941)")],
          fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=1)
fig.tight_layout(); fig.savefig(ROOT / "figures/fig_fs_vs_paper.pdf")
print("ok")
