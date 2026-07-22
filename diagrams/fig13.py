import json
import matplotlib.pyplot as plt

with open("results/main_table.json", "r", encoding="utf-8") as f:
    table = json.load(f)

by_type = table["by_type"]
configs = ["Plain", "Naive Penalty", "Two-Stage", "Oracle"]

static_key = [k for k in by_type if "Static" in k][0]
hist_key = [k for k in by_type if "Historical" in k][0]

static_vals = [float(by_type[static_key][c].strip("%")) for c in configs]
hist_vals = [float(by_type[hist_key][c].strip("%")) for c in configs]

fig, ax = plt.subplots(figsize=(11, 7))
ax.plot(configs, static_vals, marker="o", markersize=10, linewidth=2.5,
        color="#4C72B0", label="Static Questions")
ax.plot(configs, hist_vals, marker="s", markersize=10, linewidth=2.5,
        color="#C44E52", label="Dynamic-Historical Questions")

for i, (s, h) in enumerate(zip(static_vals, hist_vals)):
    ax.annotate(f"{s:.1f}%", xy=(i, s), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold", color="#4C72B0")
    ax.annotate(f"{h:.1f}%", xy=(i, h), xytext=(0, -18), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold", color="#C44E52")

# highlight the collapse point
naive_idx = configs.index("Naive Penalty")
ax.annotate("Sharp collapse under\nunconditional penalty",
            xy=(naive_idx, hist_vals[naive_idx]),
            xytext=(naive_idx + 0.3, hist_vals[naive_idx] - 15),
            fontsize=10, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred"))

ts_idx = configs.index("Two-Stage")
ax.annotate("Recovers with\nclassifier gate",
            xy=(ts_idx, hist_vals[ts_idx]),
            xytext=(ts_idx - 0.9, hist_vals[ts_idx] + 12),
            fontsize=10, color="darkgreen",
            arrowprops=dict(arrowstyle="->", color="darkgreen"))

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("The Drop-and-Recovery Pattern: Static vs Dynamic-Historical Questions",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 90)
plt.tight_layout()
plt.savefig("diagrams/fig_6_2_drop_and_recovery.png", dpi=200)
print("Saved fig_6_2_drop_and_recovery.png")