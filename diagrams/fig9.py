import json
import matplotlib.pyplot as plt

with open("data/benchmark/benchmark.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

group_names = {
    1: "Static",
    2: "Dynamic-Current",
    3: "Dynamic-Historical",
    4: "Undated-Dynamic",
    5: "No-date Control",
}

rows = []
for g in range(1, 6):
    questions = [q["question"] for q in benchmark if q["group"] == g][:2]  # take 2 examples
    for q_text in questions:
        rows.append([group_names[g], q_text[:55]])

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=["Group", "Example Question"],
                  cellLoc="left", loc="center", colWidths=[0.25, 0.75])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for i in range(2):
    table[0, i].set_facecolor("#4C72B0")
    table[0, i].set_text_props(color="white", fontweight="bold")

# color-code rows by group
group_colors = {"Static": "#EAF0F8", "Dynamic-Current": "#FBEEE6",
                "Dynamic-Historical": "#EAF5EC", "Undated-Dynamic": "#FBEAEB",
                "No-date Control": "#F0EDF5"}
for i, row in enumerate(rows, start=1):
    color = group_colors.get(row[0], "#FFFFFF")
    table[i, 0].set_facecolor(color)
    table[i, 1].set_facecolor(color)

plt.title("Sample Questions From Each Benchmark Group", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("diagrams/table_4_2_example_questions.png", dpi=200, bbox_inches="tight")
print("Saved table_4_2_example_questions.png")