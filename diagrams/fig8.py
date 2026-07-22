import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("data/benchmark/benchmark.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Pick the Assam CM question as the example, or fall back to first Group 2 question with a revision pair
example = None
for q in benchmark:
    if q.get("group") == 2 and q.get("has_revision_pair"):
        if "Assam" in q["question"]:
            example = q
            break
if example is None:
    for q in benchmark:
        if q.get("group") == 2 and q.get("has_revision_pair"):
            example = q
            break

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

# Question box at top
ax.text(6, 5.6, f'Question: "{example["question"]}"', ha="center", fontsize=13,
        fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0EDE6", edgecolor="black"))

# Old revision box (left)
old_box = mpatches.FancyBboxPatch((0.5, 2.2), 5, 2.4, boxstyle="round,pad=0.15",
                                    edgecolor="black", facecolor="#F5D0C5", linewidth=2)
ax.add_patch(old_box)
ax.text(3, 4.2, "OLD REVISION", ha="center", fontsize=11, fontweight="bold")
ax.text(3, 3.7, f'Date: {example.get("old_doc_date", "N/A")}', ha="center", fontsize=10)
ax.text(3, 3.2, f'Says: "{example.get("old_wrong_answer", "N/A")}"', ha="center", fontsize=10, style="italic")
ax.text(3, 2.6, "(WRONG for today)", ha="center", fontsize=10, color="darkred", fontweight="bold")

# New revision box (right)
new_box = mpatches.FancyBboxPatch((6.5, 2.2), 5, 2.4, boxstyle="round,pad=0.15",
                                    edgecolor="black", facecolor="#C5E8D0", linewidth=2)
ax.add_patch(new_box)
ax.text(9, 4.2, "CURRENT REVISION", ha="center", fontsize=11, fontweight="bold")
ax.text(9, 3.7, f'Date: {example.get("new_doc_date", "N/A")}', ha="center", fontsize=10)
ax.text(9, 3.2, f'Says: "{example["correct_answer"]}"', ha="center", fontsize=10, style="italic")
ax.text(9, 2.6, "(CORRECT today)", ha="center", fontsize=10, color="darkgreen", fontweight="bold")

# arrows from question down to both boxes
ax.annotate("", xy=(3, 4.6), xytext=(5.5, 5.2),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(9, 4.6), xytext=(6.5, 5.2),
            arrowprops=dict(arrowstyle="->", lw=1.5))

# bottom note
ax.text(6, 1.3, "Both documents are semantically similar to the question.\n"
                "Only the reranker's freshness signal distinguishes which is correct.",
        ha="center", fontsize=10, style="italic", color="#444444")

ax.set_xlim(0, 12)
ax.set_ylim(0.5, 6.2)
plt.title("Example: Evolving Semantic Conflict (Group 2)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("diagrams/fig_4_2_revision_pair_example.png", dpi=200, bbox_inches="tight")
print("Saved fig_4_2_revision_pair_example.png")
print(f"\nUsed question: {example['question']}")