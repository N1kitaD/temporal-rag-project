import json
import numpy as np
import matplotlib.pyplot as plt

with open("results/main_table.json", "r", encoding="utf-8") as f:
    table = json.load(f)

by_type = table["by_type"]
categories = list(by_type.keys())
configs = ["Plain", "Naive Penalty", "Two-Stage", "Oracle"]
colors = ["#95A5A6", "#E67E22", "#27AE60", "#8E44AD"]

data = {cfg: [float(by_type[cat][cfg].strip("%")) for cat in categories] for cfg in configs}

x = np.arange(len(categories))
width = 0.2

fig, ax = plt.subplots(figsize=(13, 7))
for i, cfg in enumerate(configs):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, data[cfg], width, label=cfg, color=colors[i])
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5)

ax.set_xlabel("Query Type", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy by Query Type Across Four Configurations", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(fontsize=11, loc="upper left")
ax.set_ylim(0, 90)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("diagrams/fig_6_1_main_results_bar.png", dpi=200)
print("Saved fig_6_1_main_results_bar.png")