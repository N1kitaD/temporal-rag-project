import json
import matplotlib.pyplot as plt

with open("results/main_table.json", "r", encoding="utf-8") as f:
    table = json.load(f)

per_type = table["classifier_by_type"]
overall = table["classifier_overall"]

labels = list(per_type.keys()) + ["Overall"]
values = [float(v.strip("%")) for v in per_type.values()] + [float(overall.strip("%"))]
colors = ["#4C72B0", "#DD8452", "#55A868", "#2C3E50"]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(labels, values, color=colors)
for bar, val in zip(bars, values):
    ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 5), textcoords="offset points", ha="center",
                fontsize=11, fontweight="bold")

ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
ax.set_title("Classifier Accuracy by True Query Type", fontsize=14, fontweight="bold")
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("diagrams/fig_6_3_classifier_accuracy.png", dpi=200)
print("Saved fig_6_3_classifier_accuracy.png")