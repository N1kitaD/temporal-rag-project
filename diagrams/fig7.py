import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")

# Root box
root = mpatches.FancyBboxPatch((4.2, 8), 3.6, 1, boxstyle="round,pad=0.1",
                                 edgecolor="black", facecolor="#2C3E50", linewidth=2)
ax.add_patch(root)
ax.text(6, 8.5, "Benchmark: 105 Questions", ha="center", va="center",
        fontsize=13, fontweight="bold", color="white")

groups = [
    ("Group 1\nStatic\n(30 Q)", "#4C72B0", 0.5),
    ("Group 2\nDynamic-Current\n(25 Q)", "#DD8452", 2.7),
    ("Group 3\nDynamic-Historical\n(25 Q)", "#55A868", 4.9),
    ("Group 4\nUndated-Dynamic\n(15 Q)", "#C44E52", 7.1),
    ("Group 5\nNo-date Control\n(10 Q)", "#8172B2", 9.3),
]

for label, color, x in groups:
    box = mpatches.FancyBboxPatch((x, 4.5), 2.0, 1.8, boxstyle="round,pad=0.08",
                                    edgecolor="black", facecolor=color, linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + 1.0, 5.4, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white")
    # arrow from root to group
    ax.annotate("", xy=(x + 1.0, 6.3), xytext=(6, 8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5,
                                connectionstyle="arc3,rad=0.0"))

# Purpose labels below each group
purposes = [
    "Tests penalty\nstays OFF",
    "Tests penalty\ncorrectly ON",
    "Tests binary vs\n3-class classifier",
    "Tests semantic vs\nsyntactic gate",
    "Tests no-date\nedge case",
]
for (label, color, x), purpose in zip(groups, purposes):
    ax.text(x + 1.0, 3.8, purpose, ha="center", va="top", fontsize=8.5,
            style="italic", color="#333333")

ax.set_xlim(0, 12)
ax.set_ylim(2.5, 9.5)
plt.title("Benchmark Structure: Five Question Groups", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("diagrams/fig_4_1_benchmark_structure.png", dpi=200, bbox_inches="tight")
print("Saved fig_4_1_benchmark_structure.png")