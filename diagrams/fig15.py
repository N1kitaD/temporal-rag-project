import json
import matplotlib.pyplot as plt

with open("results/failure_taxonomy.json", "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

counts = taxonomy["counts"]
type_labels = {
    "A": "Classifier Error",
    "B": "Retriever/Penalty Miss",
    "C": "Reranker Regression",
    "D": "Generator Failure",
    "E": "Corpus Gap",
}

labels = [type_labels.get(k, k) for k in counts.keys()]
sizes = list(counts.values())
colors = ["#E74C3C", "#F39C12", "#9B59B6", "#3498DB", "#95A5A6"][:len(labels)]

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct=lambda p: f"{p:.0f}%\n({int(round(p*sum(sizes)/100))})",
    colors=colors, startangle=90, textprops={"fontsize": 12}
)
for autotext in autotexts:
    autotext.set_fontweight("bold")

ax.set_title(f"Failure Taxonomy: Two-Stage System\n(Total failures: {sum(sizes)})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("diagrams/fig_6_4_failure_taxonomy.png", dpi=200)
print("Saved fig_6_4_failure_taxonomy.png")