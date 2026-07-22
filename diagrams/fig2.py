import json
import matplotlib.pyplot as plt

with open("data/corpus/articles_full.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

static_count = sum(1 for a in articles if a["topic_type"] == "static")
dynamic_count = sum(1 for a in articles if a["topic_type"] == "dynamic")

labels = ["Static Topics", "Dynamic Topics"]
sizes = [static_count, dynamic_count]
colors = ["#4C72B0", "#DD8452"]

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct=lambda p: f"{p:.1f}%\n({int(p*sum(sizes)/100)})",
    colors=colors, startangle=90, textprops={"fontsize": 12}
)
ax.set_title(f"Corpus Composition (Total: {sum(sizes)} Articles)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("diagrams/fig_3_2_corpus_composition.png", dpi=200)
print("Saved fig_3_2_corpus_composition.png")
print(f"Static: {static_count}, Dynamic: {dynamic_count}")