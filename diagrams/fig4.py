import numpy as np
import matplotlib.pyplot as plt

h = 180  # half-life used in the project
age_days = np.linspace(0, 1500, 500)
freshness = 0.5 ** (age_days / h)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(age_days, freshness, color="#4C72B0", linewidth=2.5)

# mark the half-life point
ax.axvline(x=h, color="gray", linestyle="--", linewidth=1)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
ax.plot(h, 0.5, "o", color="#DD8452", markersize=10)
ax.annotate(f"Half-life point\n({h} days, freshness = 0.5)",
            xy=(h, 0.5), xytext=(h + 150, 0.65),
            fontsize=11, arrowprops=dict(arrowstyle="->", color="black"))

# mark a few more reference points
for days in [365, 730, 1460]:
    f_val = 0.5 ** (days / h)
    ax.plot(days, f_val, "o", color="#55A868", markersize=6)
    ax.annotate(f"{days}d\n{f_val:.3f}", xy=(days, f_val),
                xytext=(days, f_val + 0.05), fontsize=9, ha="center")

ax.set_xlabel("Document Age (days)", fontsize=12)
ax.set_ylabel("Freshness Score", fontsize=12)
ax.set_title("Exponential Freshness Decay (h = 180 days)", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("diagrams/fig_3_3_freshness_decay.png", dpi=200)
print("Saved fig_3_3_freshness_decay.png")