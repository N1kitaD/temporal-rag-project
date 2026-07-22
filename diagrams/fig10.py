import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(13, 8))
ax.axis("off")

configs = [
    ("Plain", False, False, "#95A5A6"),
    ("Naive Penalty", False, True, "#E67E22"),
    ("Two-Stage\n(Our System)", True, True, "#27AE60"),
    ("Oracle", "GT", True, "#8E44AD"),
]

col_width = 2.8
start_x = 0.3

for i, (name, clf_active, rerank_active, color) in enumerate(configs):
    x = start_x + i * (col_width + 0.3)

    # Title
    ax.text(x + col_width/2, 7.3, name, ha="center", fontsize=12, fontweight="bold")

    # Question box
    q_box = mpatches.FancyBboxPatch((x + 0.6, 6.3), 1.6, 0.6, boxstyle="round,pad=0.05",
                                      edgecolor="black", facecolor="white", linewidth=1.2)
    ax.add_patch(q_box)
    ax.text(x + col_width/2, 6.6, "Question", ha="center", va="center", fontsize=8.5)

    # Retriever box (always active)
    r_box = mpatches.FancyBboxPatch((x, 5.0), col_width, 0.7, boxstyle="round,pad=0.05",
                                      edgecolor="black", facecolor="#3498DB", linewidth=1.2, alpha=0.85)
    ax.add_patch(r_box)
    ax.text(x + col_width/2, 5.35, "Retriever", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")

    # Classifier box (active or greyed)
    clf_color = "#F39C12" if clf_active is True else ("#B39DDB" if clf_active == "GT" else "#DADADA")
    clf_text = "Classifier" if clf_active is True else ("Ground Truth\nLabel" if clf_active == "GT" else "Classifier\n(SKIPPED)")
    clf_alpha = 0.9 if clf_active else 0.4
    c_box = mpatches.FancyBboxPatch((x, 3.7), col_width, 0.9, boxstyle="round,pad=0.05",
                                      edgecolor="black", facecolor=clf_color, linewidth=1.2, alpha=clf_alpha)
    ax.add_patch(c_box)
    ax.text(x + col_width/2, 4.15, clf_text, ha="center", va="center",
            fontsize=8.5, color="black" if not clf_active else "white", fontweight="bold")

    # Reranker box
    rerank_color = color if rerank_active else "#DADADA"
    rerank_text = "Reranker\n(Active)" if rerank_active else "Reranker\n(SKIPPED)"
    r2_box = mpatches.FancyBboxPatch((x, 2.4), col_width, 0.9, boxstyle="round,pad=0.05",
                                       edgecolor="black", facecolor=rerank_color, linewidth=1.2,
                                       alpha=0.9 if rerank_active else 0.4)
    ax.add_patch(r2_box)
    ax.text(x + col_width/2, 2.85, rerank_text, ha="center", va="center",
            fontsize=8.5, color="white" if rerank_active else "black", fontweight="bold")

    # Generator box
    g_box = mpatches.FancyBboxPatch((x, 1.1), col_width, 0.7, boxstyle="round,pad=0.05",
                                      edgecolor="black", facecolor="#16A085", linewidth=1.2, alpha=0.85)
    ax.add_patch(g_box)
    ax.text(x + col_width/2, 1.45, "Generator", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")

    # Arrows connecting boxes vertically
    for y1, y2 in [(6.3, 5.7), (5.0, 4.6), (3.7, 3.3), (2.4, 1.8)]:
        ax.annotate("", xy=(x + col_width/2, y2), xytext=(x + col_width/2, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2))

ax.set_xlim(0, 12.7)
ax.set_ylim(0.5, 7.8)
plt.title("Four Pipeline Configurations Compared", fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("diagrams/fig_5_1_four_configurations.png", dpi=200, bbox_inches="tight")
print("Saved fig_5_1_four_configurations.png")