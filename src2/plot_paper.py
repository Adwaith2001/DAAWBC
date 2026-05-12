"""
plot_paper.py
4 best plots to represent the DAAWBC project in the IEEE paper.

Plot 1: Training curves — AC Shared vs MAPPO Shared (all 5 agents, mean±std)
Plot 2: Budget pacing over time from bidding log (key contribution)
Plot 3: Evaluation comparison — pacing, utilization, stability
Plot 4: Version progression + per-advertiser final performance

Run from anywhere. Paths point to outputs/ and uploads/ directories.
Saves to: outputs/paper_plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ======================================================
# PATHS
# ======================================================
DATA_DIR  = Path("D:/Research Methodology/DAAWBC/dynamic_ad_allocation")
OUT_DIR   = DATA_DIR / "outputs" / "paper_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AC_DIR    = DATA_DIR / "outputs" / "final_experiments_shared"
MAPPO_DIR = DATA_DIR / "outputs" / "final_experiments_shared_mappo"
BIDLOG    = DATA_DIR / "outputs" / "bidding_log_shared.csv"

# ======================================================
# CONFIG
# ======================================================
ADV_IDS  = ["1458", "2259", "3386", "2997", "3476"]
BUDGETS  = {"1458":18000,"2259":14000,"3386":2000,"2997":20000,"3476":10000}
SEEDS    = [0, 1, 2, 3, 4]

COLORS = {
    "1458": "#2196F3", "2259": "#F44336",
    "3386": "#4CAF50", "2997": "#FF9800", "3476": "#9C27B0"
}
M_COLORS = {"AC Shared": "#2196F3", "MAPPO Shared": "#F44336"}

plt.rcParams.update({
    "font.family":     "Times New Roman",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":      200,
})

# ======================================================
# LOAD DATA
# ======================================================
def load_dfs(folder, prefix):
    dfs = []
    for s in SEEDS:
        p = Path(folder) / f"{prefix}_seed_{s}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    return dfs

dfs_ac   = load_dfs(AC_DIR,    "actor_critic_shared")
dfs_mp   = load_dfs(MAPPO_DIR, "mappo_shared")
episodes = dfs_ac[0]["episode"].values if dfs_ac else np.arange(200)

def get_clicks(dfs, adv):
    return np.stack([df[f"clicks_{adv}"].values for df in dfs])

def get_utils(dfs, adv):
    return np.stack([df[f"utilization_{adv}"].values for df in dfs])

# ======================================================
# PLOT 1: Training Curves (AC vs MAPPO, total clicks)
# ======================================================
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

for idx, adv in enumerate(ADV_IDS):
    ax = axes[idx]

    for dfs, label, color in [
        (dfs_ac, "AC Shared",    "#2196F3"),
        (dfs_mp, "MAPPO Shared", "#F44336"),
    ]:
        if not dfs:
            continue
        clicks = get_clicks(dfs, adv)
        m = clicks.mean(axis=0)
        s = clicks.std(axis=0)
        ax.plot(episodes, m, color=color, linewidth=1.5, label=label)
        ax.fill_between(episodes, m-s, m+s, alpha=0.15, color=color)

    ax.set_title(
        f"Advertiser {adv}\n"
        f"(Budget {BUDGETS[adv]:,})",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel("Clicks" if idx == 0 else "", fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")

    # Final mean annotation
    if dfs_ac and dfs_mp:
        ac_final = get_clicks(dfs_ac, adv).mean(axis=0)[-20:].mean()
        mp_final = get_clicks(dfs_mp, adv).mean(axis=0)[-20:].mean()
        ax.axhline(ac_final, color="#2196F3", linestyle=":", alpha=0.5, linewidth=1)
        ax.axhline(mp_final, color="#F44336", linestyle=":", alpha=0.5, linewidth=1)

    if idx == 0:
        ax.legend(loc="lower right", fontsize=8)

plt.suptitle(
    "Fig. 1 — Training Curves: AC Shared vs MAPPO Shared\n"
    "(Mean ± Std, 5 Seeds, 200 Episodes | TRUE Shared Auction Stream)",
    fontsize=11, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_training_curves.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("Fig 1 saved: fig1_training_curves.png")

# ======================================================
# PLOT 2: Budget Pacing Over Time (bidding log)
# ======================================================
if BIDLOG.exists():
    bidlog = pd.read_csv(BIDLOG)
    MAX_STEPS = 2000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Budget depletion vs ideal linear
    ax = axes[0]
    steps = bidlog["step"].values

    for adv in ADV_IDS:
        budget     = BUDGETS[adv]
        cum_cost   = bidlog[f"cum_cost_{adv}"].values
        budget_pct = cum_cost / budget * 100
        ax.plot(steps, budget_pct, color=COLORS[adv],
                linewidth=1.5, label=adv)

    # Ideal linear pacing line
    ax.plot(steps, (steps / MAX_STEPS) * 100, "k--",
            linewidth=2, label="Ideal linear pace", alpha=0.7)

    ax.set_xlabel("Auction Step")
    ax.set_ylabel("Budget Spent (%)")
    ax.set_title("Budget Depletion vs Ideal Linear Pacing\n"
                 "(MAPPO Shared — Agents learn to follow ideal curve)",
                 fontweight="bold")
    ax.legend(title="Advertiser", fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(0, MAX_STEPS)
    ax.set_ylim(0, 110)

    # Right: Pacing error over time (rolling average)
    ax = axes[1]
    window = 100
    for adv in ADV_IDS:
        pe = bidlog[f"pacing_err_{adv}"].abs()
        rolling = pe.rolling(window=window, min_periods=1).mean()
        ax.plot(steps, rolling, color=COLORS[adv],
                linewidth=1.5, label=adv)

    ax.axhline(y=0.1, color="orange", linestyle="--",
               linewidth=1.5, label="10% tolerance", alpha=0.8)
    ax.set_xlabel("Auction Step")
    ax.set_ylabel(f"|Pacing Error| (Rolling {window}-step avg)")
    ax.set_title("Pacing Error Over Time\n"
                 "(Lower = better budget discipline)",
                 fontweight="bold")
    ax.legend(title="Advertiser", fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(0, MAX_STEPS)
    ax.set_ylim(0, 0.8)

    plt.suptitle(
        "Fig. 2 — Budget Pacing Analysis: Key RL Contribution\n"
        "(Episode Demo — MAPPO Shared Agents, 2000 Auction Steps)",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_budget_pacing.png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print("Fig 2 saved: fig2_budget_pacing.png")
else:
    print("Fig 2 skipped: bidding_log_shared.csv not found")

# ======================================================
# PLOT 3: Evaluation Comparison (3-panel)
# ======================================================
methods  = ["Fixed\nBid", "Linear\npCTR", "AC\nShared", "MAPPO\nShared"]
clicks   = [181.5,  1055.0, 699.0,  688.8]
stds     = [10.7,   11.2,   14.9,   10.1]
utils    = [21.8,   86.2,   93.8,   92.4]
pacing   = [63.5,   17.7,   8.0,    9.8]
colors   = ["#9E9E9E", "#FF9800", "#2196F3", "#F44336"]
rl_mask  = [False, False, True, True]

fig = plt.figure(figsize=(16, 5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

# Panel A: Budget Pacing Error
ax = fig.add_subplot(gs[0])
bars = ax.bar(methods, pacing, color=colors, alpha=0.85,
              width=0.55, edgecolor="white", linewidth=1.2)
for bar, v, rl in zip(bars, pacing, rl_mask):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.8,
            f"{v:.1f}%",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold",
            color="#1B5E20" if rl else "#424242")
    if rl:
        ax.text(bar.get_x() + bar.get_width()/2,
                1.5, "LEARNED",
                ha="center", fontsize=7,
                color="white", fontweight="bold")

# Improvement arrow
ax.annotate("",
    xy=(2.1, pacing[2]+1), xytext=(1.1, pacing[1]+1),
    arrowprops=dict(arrowstyle="->",
                    color="green", lw=2))
ax.text(1.6, 19, "54.8%\nbetter",
        ha="center", fontsize=9,
        color="green", fontweight="bold")
ax.set_ylabel("Pacing Error (%)")
ax.set_title("(a) Budget Pacing\nDiscipline ↓",
             fontweight="bold")
ax.set_ylim(0, 82)
ax.grid(axis="y", alpha=0.25, linestyle="--")

# Panel B: Budget Utilization
ax = fig.add_subplot(gs[1])
bars = ax.bar(methods, utils, color=colors, alpha=0.85,
              width=0.55, edgecolor="white", linewidth=1.2)
ax.axhline(y=80, color="orange", linestyle="--",
           linewidth=2, label="80% target", alpha=0.9)
ax.axhline(y=100, color="gray", linestyle=":",
           linewidth=1, alpha=0.5)
for bar, v in zip(bars, utils):
    ax.text(bar.get_x() + bar.get_width()/2,
            v + 1, f"{v:.1f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_ylabel("Budget Utilization (%)")
ax.set_title("(b) Budget Utilization ↑\n(AC Shared = best)",
             fontweight="bold")
ax.set_ylim(0, 115)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.25, linestyle="--")

# Panel C: Policy Stability
ax = fig.add_subplot(gs[2])
bars = ax.bar(methods, stds, color=colors, alpha=0.85,
              width=0.55, edgecolor="white", linewidth=1.2)
for bar, v, rl in zip(bars, stds, rl_mask):
    ax.text(bar.get_x() + bar.get_width()/2,
            v + 0.15, f"{v:.1f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="#1B5E20" if rl else "#424242")

# MAPPO stability arrow
ax.annotate("",
    xy=(3.1, stds[3]+0.1), xytext=(1.1, stds[1]+0.1),
    arrowprops=dict(arrowstyle="->", color="green",
                    lw=2, connectionstyle="arc3,rad=-0.3"))
ax.text(3.15, 11.6,
        "9.8%\nmore\nstable",
        ha="center", fontsize=8,
        color="green", fontweight="bold")
ax.set_ylabel("Std Dev of Total Clicks")
ax.set_title("(c) Policy Stability ↓\n(MAPPO = most consistent)",
             fontweight="bold")
ax.set_ylim(0, max(stds) * 1.7)
ax.grid(axis="y", alpha=0.25, linestyle="--")

# Shared legend
patches = [
    mpatches.Patch(color="#9E9E9E", label="Fixed Bid"),
    mpatches.Patch(color="#FF9800", label="Linear pCTR"),
    mpatches.Patch(color="#2196F3", label="AC Shared (RL)"),
    mpatches.Patch(color="#F44336", label="MAPPO Shared (RL)"),
]
fig.legend(handles=patches, loc="lower center",
           ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.06),
           frameon=True)

plt.suptitle(
    "Fig. 3 — Evaluation Results: Key RL Contributions\n"
    "Pacing Discipline, Budget Utilization & Policy Stability "
    "(10 episodes, 5 methods)",
    fontsize=11, fontweight="bold"
)
plt.savefig(OUT_DIR / "fig3_evaluation_comparison.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("Fig 3 saved: fig3_evaluation_comparison.png")

# ======================================================
# PLOT 4: Version Progression + Per-Advertiser Heatmap
# ======================================================
fig = plt.figure(figsize=(16, 6))
gs  = GridSpec(1, 2, figure=fig, wspace=0.4, width_ratios=[1, 1.3])

# Left: Version progression
ax = fig.add_subplot(gs[0])
versions = [
    ("v3\n(Base AC)", 757.0, "#BDBDBD"),
    ("v4\n(Enhanced)", 695.0, "#90A4AE"),
    ("v6\n(Hetero\nBudgets)", 983.6, "#2196F3"),
    ("MAPPO\n(Centralized\nCritic)", 999.8, "#F44336"),
    ("AC\nShared\n(True Comp.)", 694.6, "#4CAF50"),
    ("MAPPO\nShared\n(True Comp.)", 863.4, "#FF9800"),
]
names, vals, cols = zip(*versions)
bars = ax.bar(names, vals, color=cols, alpha=0.85,
              width=0.6, edgecolor="white", linewidth=1.2)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 8,
            f"{v:.0f}",
            ha="center", fontsize=9, fontweight="bold")

# Vertical separator: separate vs shared
ax.axvline(x=3.5, color="gray", linestyle="--",
           linewidth=1.5, alpha=0.6)
ax.text(1.5, 1060, "Separate Logs",
        ha="center", fontsize=8, color="gray", style="italic")
ax.text(4.5, 1060, "Shared Auction",
        ha="center", fontsize=8, color="gray", style="italic")
ax.set_ylabel("Total Clicks (All Agents)")
ax.set_title("(a) Performance Across Framework Versions",
             fontweight="bold")
ax.set_ylim(0, 1150)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.tick_params(axis="x", labelsize=8)

# Right: Per-advertiser heatmap (AC Shared final performance)
ax = fig.add_subplot(gs[1])

if dfs_ac and dfs_mp:
    ac_clicks  = [get_clicks(dfs_ac, adv).mean() for adv in ADV_IDS]
    mp_clicks  = [get_clicks(dfs_mp, adv).mean() for adv in ADV_IDS]
    ac_utils   = [get_utils(dfs_ac, adv).mean() for adv in ADV_IDS]
    mp_utils   = [get_utils(dfs_mp, adv).mean() for adv in ADV_IDS]

    x     = np.arange(len(ADV_IDS))
    width = 0.35

    b1 = ax.bar(x - width/2, ac_clicks, width,
                label="AC Shared",
                color="#2196F3", alpha=0.85,
                edgecolor="white")
    b2 = ax.bar(x + width/2, mp_clicks, width,
                label="MAPPO Shared",
                color="#F44336", alpha=0.85,
                edgecolor="white")

    for bar, v in zip(list(b1)+list(b2), ac_clicks+mp_clicks):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 2,
                f"{v:.0f}",
                ha="center", fontsize=8, fontweight="bold")

    # Add utilization labels below x-axis
    util_labels = [
        f"{adv}\nAC:{ac_utils[i]:.0f}%  MP:{mp_utils[i]:.0f}%"
        for i, adv in enumerate(ADV_IDS)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(util_labels, fontsize=8)
    ax.set_ylabel("Mean Clicks (200 Episodes, 5 Seeds)")
    ax.set_title("(b) Per-Advertiser Performance\n"
                 "AC Shared vs MAPPO Shared (Util% shown)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(0, max(ac_clicks + mp_clicks) * 1.25)

plt.suptitle(
    "Fig. 4 — Framework Evolution and Per-Advertiser Analysis\n"
    "Shared auction enables true competitive evaluation; "
    "4/5 agents achieve >90% budget utilization",
    fontsize=11, fontweight="bold"
)
plt.savefig(OUT_DIR / "fig4_version_and_peragent.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("Fig 4 saved: fig4_version_and_peragent.png")

print(f"\nAll 4 plots saved to: {OUT_DIR}")
print("Include in paper as:")
print("  Fig 1 — Training curves (convergence evidence)")
print("  Fig 2 — Budget pacing (KEY CONTRIBUTION)")
print("  Fig 3 — Evaluation 3-panel (pacing + util + stability)")
print("  Fig 4 — Version progression + per-advertiser breakdown")
