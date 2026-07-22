import numpy as np
import matplotlib.pyplot as plt

# Features from Table 2.1, converted to 1 (yes) / 0 (no)
features = [
    "Distinguishes\ncurrent vs historical",
    "Continuous\nconfidence scaling",
    "Semantic (not\nkeyword) detection",
    "No retriever\ntraining required",
    "Works on any\npre-trained retriever",
    "Explicit 'no\nadjustment needed' state",
]

systems = {
    "Wu et al. (2024)":     [0, 0, 1, 0, 0, 0],
    "Grofsky":              [0, 0, 0, 1, 1, 0],
    "An et al. (FRESCO)":   [0, 0, 0, 1, 1, 0],
    "TempRetriever":        [0, 0, 1, 0, 0, 0],
    "T-GRAG":               [0, 0, 0, 1, 0, 1],
    "Our System":           [1, 1, 1, 1, 1, 1],
}

num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

colors = ["#95A5A6", "#E67E22", "#F1C40F", "#3498DB", "#9B59B6", "#27AE60"]
linewidths = [1.2, 1.2, 1.2, 1.2, 1.2, 3.0]
alphas_fill = [0.05, 0.05, 0.05, 0.05, 0.05, 0.2]

for (name, values), color, lw, fill_alpha in zip(systems.items(), colors, linewidths, alphas_fill):
    vals = values + values[:1]
    ax.plot(angles, vals, color=color, linewidth=lw, label=name)
    ax.fill(angles, vals, color=color, alpha=fill_alpha)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=10)
ax.set_yticks([0, 1])
ax.set_yticklabels(["No", "Yes"], fontsize=9)
ax.set_ylim(0, 1.1)

plt.title("Feature Comparison: Our System vs Prior Work", fontsize=14, fontweight="bold", pad=30)
plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig("diagrams/fig_7_1_radar_comparison.png", dpi=200, bbox_inches="tight")
print("Saved fig_7_1_radar_comparison.png")