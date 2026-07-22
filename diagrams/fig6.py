import numpy as np
import matplotlib.pyplot as plt

alpha = 0.7
P = np.linspace(0, 1, 100)
semantic_weight = (1 - P) + P * alpha
freshness_weight = P * (1 - alpha)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(P, semantic_weight, label="Effective Semantic Weight", color="#4C72B0", linewidth=2.5)
ax.plot(P, freshness_weight, label="Effective Freshness Weight", color="#DD8452", linewidth=2.5)

for p_val in [0.0, 0.5, 1.0]:
    sw = (1 - p_val) + p_val * alpha
    fw = p_val * (1 - alpha)
    ax.plot(p_val, sw, "o", color="#4C72B0", markersize=8)
    ax.plot(p_val, fw, "o", color="#DD8452", markersize=8)
    ax.annotate(f"{sw:.2f}", xy=(p_val, sw), xytext=(p_val, sw + 0.04), fontsize=9, ha="center")
    ax.annotate(f"{fw:.2f}", xy=(p_val, fw), xytext=(p_val, fw + 0.04), fontsize=9, ha="center")

ax.set_xlabel("Classifier Confidence (P)", fontsize=12)
ax.set_ylabel("Effective Weight in Final Score", fontsize=12)
ax.set_title("How Classifier Confidence Controls the Penalty Strength", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("diagrams/fig_3_4_formula_behavior.png", dpi=200)
print("Saved fig_3_4_formula_behavior.png")