"""
Plot RMSE and skill score vs. lead time, comparing a single sample against
the 5-sample ensemble mean (UNet_FM_OceanDDPM, 2023 NRT altimetry).

Values below are the ones already computed via the analysis notebook's
compute_stat_scores pipeline -- this script only plots them, it doesn't
recompute anything from the .nc files.

Usage:
    python plot_ensemble_vs_single_sample.py
"""

import matplotlib.pyplot as plt

leadtimes = [0, 3, 5]

rmse_1sample = [0.06715344660804204, 0.08625269854998535, 0.08925251251868802]
rmse_5sample = [0.06512738766854802, 0.08197889947634215, 0.08552454167038318]

score_1sample = [0.5319418948653352, 0.3988205120476589, 0.3779118720169725]
score_5sample = [0.546063482900061, 0.42860880136379, 0.40389575012585677]

COLOR_1SAMPLE = "#2a78d6"
COLOR_5SAMPLE = "#eb6834"

fig, (ax_rmse, ax_score) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

for ax, y1, y5, ylabel, title in [
    (ax_rmse, rmse_1sample, rmse_5sample, "RMSE (m)", "RMSE vs. lead time (lower is better)"),
    (ax_score, score_1sample, score_5sample, "Skill score", "Skill score vs. lead time (higher is better)"),
]:
    ax.plot(leadtimes, y1, marker="o", markersize=7, linewidth=2, color=COLOR_1SAMPLE, label="1 sample")
    ax.plot(leadtimes, y5, marker="o", markersize=7, linewidth=2, color=COLOR_5SAMPLE, label="5-sample mean")

    for x, v in zip(leadtimes, y1):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(-16, 9),
                    ha="right", fontsize=9, color=COLOR_1SAMPLE)
    for x, v in zip(leadtimes, y5):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(16, -13),
                    ha="left", fontsize=9, color=COLOR_5SAMPLE)

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

ax_score.set_xlabel("Lead time (days)")
ax_score.set_xticks(leadtimes)

fig.suptitle("UNet_FM_OceanDDPM: 1 sample vs. 5-sample ensemble mean (2023 NRT altimetry)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = "ensemble_vs_single_sample.png"
fig.savefig(out_path, dpi=150)
print("Saved:", out_path)
