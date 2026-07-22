import matplotlib.pyplot as plt

stages = ["Initial Batch", "First Top-up\n(gap-filling)", "Second Top-up\n(400 articles)", "Final Total"]
article_counts = [562, 573, 833, 1131]
chunk_counts = [6424, 6794, 15423, 23594]

fig, ax1 = plt.subplots(figsize=(10, 6))

x = range(len(stages))
ax1.bar([i - 0.2 for i in x], article_counts, width=0.4, label="Articles", color="#4C72B0")
ax1.set_ylabel("Number of Articles", fontsize=12, color="#4C72B0")
ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=10)

ax2 = ax1.twinx()
ax2.bar([i + 0.2 for i in x], chunk_counts, width=0.4, label="Chunks", color="#DD8452")
ax2.set_ylabel("Number of Chunks", fontsize=12, color="#DD8452")

for i, v in enumerate(article_counts):
    ax1.text(i - 0.2, v + 15, str(v), ha="center", fontsize=9)
for i, v in enumerate(chunk_counts):
    ax2.text(i + 0.2, v + 200, str(v), ha="center", fontsize=9)

plt.title("Corpus Growth Across Build Stages", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig("diagrams/table_3_1_corpus_growth.png", dpi=200)
print("Saved table_3_1_corpus_growth.png")