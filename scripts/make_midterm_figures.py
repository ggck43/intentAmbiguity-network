"""Figures 4 and 5 equivalents from results/midterm_results.json -> figures/*.pdf"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results/midterm_results.json").read_text())
MODELS = ["Qwen3.5-4B", "Qwen3.5-9B", "Gemma 4 E2B", "Gemma 4 E4B"]
CONDS = ["zero-shot", "zero-shot + thinking", "few-shot", "few-shot + thinking"]
STY = {"zero-shot": ("#7f8c8d", "o", "-"), "zero-shot + thinking": ("#7f8c8d", "s", "--"),
       "few-shot": ("#2c6fbb", "o", "-"), "few-shot + thinking": ("#2c6fbb", "s", "--")}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
(ROOT / "figures").mkdir(exist_ok=True)

# Figure 4: level-wise F1, rows = Real / Synthetic
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True)
for i, src in enumerate(["real", "synthetic"]):
    for j, m in enumerate(MODELS):
        ax = axes[i, j]
        for c in CONDS:
            col, mk, ls = STY[c]
            ax.plot(range(1, 6), [R[f"{m}|{c}"]["level_f1"][src][str(L)] for L in range(1, 6)], color=col, marker=mk, ls=ls, ms=3, lw=1.1, label=c)
        ax.set_ylim(0, 1.05); ax.grid(alpha=.3)
        if i == 0: ax.set_title(m, fontsize=8)
        if j == 0: ax.set_ylabel(f"{src.capitalize()}  F1")
        if i == 1: ax.set_xlabel("level"); ax.set_xticks(range(1, 6))
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(ROOT / "figures/fig_levelwise_f1.pdf"); plt.close(fig)

# Figure 5: EM and Jaccard
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
x = np.arange(len(MODELS)); w = 0.2
cols = ["#b5bdc4", "#7f8c8d", "#7fb0e6", "#2c6fbb"]
for ax, key, name in zip(axes, ["em", "jaccard"], ["Exact match", "Jaccard"]):
    for k, c in enumerate(CONDS):
        ax.bar(x + (k - 1.5) * w, [R[f"{m}|{c}"][key] for m in MODELS], w, color=cols[k], label=c)
    ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=7); ax.set_title(name, fontsize=8)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("score (521 valid intents)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(ROOT / "figures/fig_em_jaccard.pdf"); plt.close(fig)
print("ok")
