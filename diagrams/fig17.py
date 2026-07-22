import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/evaluation_results.csv")

# Pick one clean example per type where naive fails but two-stage succeeds (or similar clear pattern)
examples = []

static_row = df[(df["true_type"] == "STATIC") & (df["ts_correct"] == True)].iloc[0]
examples.append(static_row)

dynamic_row = df[(df["true_type"] == "DYNAMIC_CURRENT") & (df["ts_correct"] == True) & (df["plain_correct"] == False)]
if len(dynamic_row) > 0:
    examples.append(dynamic_row.iloc[0])
else:
    examples.append(df[df["true_type"] == "DYNAMIC_CURRENT"].iloc[0])

hist_candidates = df[(df["true_type"] == "DYNAMIC_HISTORICAL") &
                     (df["naive_correct"] == False) & (df["ts_correct"] == True)]
if len(hist_candidates) > 0:
    examples.append(hist_candidates.iloc[0])
else:
    examples.append(df[df["true_type"] == "DYNAMIC_HISTORICAL"].iloc[0])

rows = []
for row in examples:
    rows.append([
        row["question"][:40],
        row["true_type"].replace("_", "-").title(),
        "✓" if row["plain_correct"] else "✗",
        "✓" if row["naive_correct"] else "✗",
        "✓" if row["ts_correct"] else "✗",
        "✓" if row["oracle_correct"] else "✗",
    ])

columns = ["Question", "Type", "Plain", "Naive", "Two-Stage", "Oracle"]

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center",
                  colWidths=[0.34, 0.18, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 2.3)

for i in range(len(columns)):
    table[0, i].set_facecolor("#2C3E50")
    table[0, i].set_text_props(color="white", fontweight="bold")

# color check/cross cells
for r in range(1, len(rows) + 1):
    for c in range(2, 6):
        val = rows[r-1][c]
        color = "#C8E6C9" if val == "✓" else "#FFCDD2"
        table[r, c].set_facecolor(color)

plt.title("Sample Question Walkthroughs Across Configurations", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("diagrams/table_6_4_sample_walkthroughs.png", dpi=200, bbox_inches="tight")
print("Saved table_6_4_sample_walkthroughs.png")
for row in rows:
    print(row)