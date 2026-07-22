import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(11, 7))
ax.axis("off")

steps = [
    ("105 Benchmark\nQuestions", "#2C3E50", "white"),
    ("Run through\n4 Configurations", "#3498DB", "white"),
    ("Check Answer\nCorrectness", "#F39C12", "white"),
    ("Aggregate by\nQuery Type & Group", "#27AE60", "white"),
    ("Build Results\nTables (6.1, 6.2, 6.3)", "#8E44AD", "white"),
]

y_positions = [6.5, 5.2, 3.9, 2.6, 1.3]

for (label, color, text_color), y in zip(steps, y_positions):
    box = mpatches.FancyBboxPatch((3.5, y - 0.4), 5, 0.8, boxstyle="round,pad=0.1",
                                    edgecolor="black", facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(6, y, label, ha="center", va="center", fontsize=11,
            color=text_color, fontweight="bold")

for i in range(len(y_positions) - 1):
    ax.annotate("", xy=(6, y_positions[i+1] + 0.45), xytext=(6, y_positions[i] - 0.4),
                arrowprops=dict(arrowstyle="->", lw=1.8))

# side annotations
side_notes = [
    "Written blind,\nbefore any testing",
    "Plain, Naive, Two-Stage,\nOracle — same corpus",
    "Substring match against\nrecorded correct answer",
    "By Static / Dynamic-Current /\nDynamic-Historical",
    "Section 6 of the report",
]
for note, y in zip(side_notes, y_positions):
    ax.text(9.2, y, note, ha="left", va="center", fontsize=9, style="italic", color="#444444")

ax.set_xlim(2, 13)
ax.set_ylim(0.5, 7.3)
plt.title("Evaluation Pipeline: From Questions to Results Tables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("diagrams/fig_5_2_evaluation_flow.png", dpi=200, bbox_inches="tight")
print("Saved fig_5_2_evaluation_flow.png")