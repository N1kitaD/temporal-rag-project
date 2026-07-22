import matplotlib.pyplot as plt

data = [
    ["Wikipedia 2019", "0.85", "0.02", "0.621", "3rd"],
    ["Wikipedia 2024", "0.80", "0.55", "0.731", "1st"],
    ["India Governance Article", "0.71", "0.40", "0.625", "2nd"],
    ["Indian Constitution", "0.60", "0.10", "0.462", "4th"],
]
columns = ["Document", "Cosine", "Freshness", "Final Score", "Rank"]

fig, ax = plt.subplots(figsize=(9, 3))
ax.axis("off")
table = ax.table(cellText=data, colLabels=columns, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

for i in range(len(columns)):
    table[0, i].set_facecolor("#4C72B0")
    table[0, i].set_text_props(color="white", fontweight="bold")

# highlight the winning row
for i in range(len(columns)):
    table[2, i].set_facecolor("#D6E8F5")   # row 2 = Wikipedia 2024, the winner

plt.title("Worked Example: P = 0.92, Query about Current Prime Minister",
          fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("diagrams/table_3_2_worked_example.png", dpi=200, bbox_inches="tight")
print("Saved table_3_2_worked_example.png")