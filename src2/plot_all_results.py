"""
plot_all_results.py
Generates ALL plots for the DAAWBC project paper.
Run from: dynamic_ad_allocation/src2/
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# DATA PATHS
# ======================================================
V4_DIR     = ROOT / "outputs" / "final_experiments_5agents_v4"
V6_DIR     = ROOT / "outputs" / "final_experiments_5agents_v6"
MAPPO_DIR  = ROOT / "outputs" / "final_experiments_5agents_mappo"
SHARED_AC  = ROOT / "outputs" / "final_experiments_shared"
SHARED_MP  = ROOT / "outputs" / "final_experiments_shared_mappo"
BIDLOG     = ROOT / "outputs" / "bidding_log_shared.csv"
EVAL_CSV   = ROOT / "outputs" / "evaluation_results_shared.csv"

# ======================================================
# CONFIG
# ======================================================
SEEDS       = [0, 1, 2, 3, 4]
ADV_V6      = ["1458", "2259", "3386", "2997", "3358"]
ADV_SHARED  = ["1458", "2259", "3386", "2997", "3476"]
BUDGETS_V6  = [20000, 12000, 20000, 25000, 18000]
BUDGETS_SH  = [18000, 14000, 10000, 28000, 23000]

COLORS = {
    "1458": "#2196F3", "2259": "#F44336",
    "3386": "#4CAF50", "2997": "#FF9800",
    "3358": "#9C27B0", "3476": "#00BCD4"
}

METHOD_COLORS = {
    "Fixed Bid":             "#9E9E9E",
    "Uniform Thresh (0.15)": "#2196F3",
    "Random Threshold":      "#FF9800",
    "AC Shared":             "#4CAF50",
    "MAPPO Shared":          "#F44336",
    "AC v6":                 "#673AB7",
    "MAPPO":                 "#E91E63",
}

plt.rcParams["figure.dpi"]       = 150
plt.rcParams["font.size"]        = 10
plt.rcParams["axes.titlesize"]   = 11
plt.rcParams["axes.labelsize"]   = 10
plt.rcParams["legend.fontsize"]  = 9


def load_dfs(folder, prefix, seeds=SEEDS):
    dfs = []
    for s in seeds:
        p = Path(folder) / f"{prefix}_seed_{s}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    return dfs


def get_clicks(dfs, adv):
    return np.stack([df[f"clicks_{adv}"].values for df in dfs])


def get_costs(dfs, adv):
    return np.stack([df[f"cost_{adv}"].values for df in dfs])


# ======================================================
# LOAD ALL DATA
# ======================================================
dfs_v6    = load_dfs(V6_DIR,    "actor_critic_5adv_v6")
dfs_mappo = load_dfs(MAPPO_DIR, "actor_critic_5adv_mappo")
dfs_ac_sh = load_dfs(SHARED_AC, "actor_critic_shared")
dfs_mp_sh = load_dfs(SHARED_MP, "mappo_shared")

episodes  = dfs_v6[0]["episode"].values if dfs_v6 else np.arange(200)
bidlog    = pd.read_csv(BIDLOG) if BIDLOG.exists() else None
eval_df   = pd.read_csv(EVAL_CSV) if EVAL_CSV.exists() else None

print("Data loaded successfully!")

# ======================================================
# PLOT 1: Training curves v6 (line + shaded)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, adv in enumerate(ADV_V6):
    ax = axes[idx]
    if dfs_v6:
        clicks = get_clicks(dfs_v6, adv)
        m, s   = clicks.mean(axis=0), clicks.std(axis=0)
        ax.plot(episodes, m, color=COLORS[adv], linewidth=1.5,
                label=f"Adv {adv}")
        ax.fill_between(episodes, m-s, m+s,
                        alpha=0.2, color=COLORS[adv])
    ax.set_title(f"Advertiser {adv}", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Clicks")
    ax.grid(True, alpha=0.3)
    ax.legend()

ax = axes[5]
for adv in ADV_V6:
    if dfs_v6:
        clicks = get_clicks(dfs_v6, adv)
        ax.plot(episodes, clicks.mean(axis=0),
                color=COLORS[adv], linewidth=1.5, label=adv)
ax.set_title("All Advertisers", fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Clicks")
ax.grid(True, alpha=0.3)
ax.legend(title="Advertiser")

plt.suptitle("Training Curves — Actor-Critic v6 (Mean ± Std, 5 Seeds)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_training_curves_v6.png",
            bbox_inches="tight")
plt.close()
print("Plot 01: Training curves v6")

# ======================================================
# PLOT 2: Training curves MAPPO Shared
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, adv in enumerate(ADV_SHARED):
    ax = axes[idx]
    if dfs_mp_sh:
        clicks = get_clicks(dfs_mp_sh, adv)
        m, s   = clicks.mean(axis=0), clicks.std(axis=0)
        ax.plot(episodes, m, color=COLORS[adv], linewidth=1.5)
        ax.fill_between(episodes, m-s, m+s,
                        alpha=0.2, color=COLORS[adv])
    ax.set_title(f"Advertiser {adv} "
                 f"(Budget {BUDGETS_SH[idx]:,})",
                 fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Clicks")
    ax.grid(True, alpha=0.3)

ax = axes[5]
for adv in ADV_SHARED:
    if dfs_mp_sh:
        clicks = get_clicks(dfs_mp_sh, adv)
        ax.plot(episodes, clicks.mean(axis=0),
                color=COLORS[adv], linewidth=1.5, label=adv)
ax.set_title("All Advertisers", fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Clicks")
ax.grid(True, alpha=0.3)
ax.legend(title="Advertiser")

plt.suptitle(
    "Training Curves — MAPPO Shared (Mean ± Std, 5 Seeds)\n"
    "TRUE Shared Auction | Neutral Stream (3427)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_training_curves_mappo_shared.png",
            bbox_inches="tight")
plt.close()
print("Plot 02: Training curves MAPPO Shared")

# ======================================================
# PLOT 3: Grouped bar — methods vs clicks
# ======================================================
if eval_df is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods = eval_df["method"].tolist()
    clicks  = eval_df["clicks"].tolist()
    stds    = eval_df["clicks_std"].tolist()
    cpcs    = eval_df["cpc"].tolist()
    utils   = eval_df["budget_util"].tolist()
    colors  = [METHOD_COLORS.get(m, "#607D8B") for m in methods]

    # Clicks
    ax = axes[0]
    bars = ax.barh(methods, clicks, xerr=stds,
                   color=colors, capsize=5, alpha=0.85)
    for bar, v in zip(bars, clicks):
        ax.text(v + 2, bar.get_y() + bar.get_height()/2,
                f"{v:.0f}", va="center", fontsize=9,
                fontweight="bold")
    ax.set_xlabel("Total Clicks")
    ax.set_title("Total Clicks per Method", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # CPC
    ax = axes[1]
    bars = ax.barh(methods, cpcs,
                   color=colors, alpha=0.85)
    for bar, v in zip(bars, cpcs):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=9,
                fontweight="bold")
    ax.set_xlabel("Cost Per Click (CPC)")
    ax.set_title("CPC per Method (Lower = Better)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Evaluation Results — Shared Auction Environment",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_evaluation_comparison.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 03: Evaluation comparison")

# ======================================================
# PLOT 4: Radar chart
# ======================================================
if eval_df is not None:
    metrics_list = ["clicks", "budget_util"]
    labels       = ["Clicks\n(normalized)", "Budget\nUtil%"]
    n_metrics    = len(labels)
    angles       = np.linspace(0, 2*np.pi, n_metrics,
                               endpoint=False).tolist()
    angles      += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))

    max_clicks = eval_df["clicks"].max()

    for _, row in eval_df.iterrows():
        vals = [
            row["clicks"] / max_clicks * 100,
            row["budget_util"],
        ]
        vals += vals[:1]
        color = METHOD_COLORS.get(row["method"], "#607D8B")
        ax.plot(angles, vals, "o-", linewidth=2,
                color=color, label=row["method"])
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Method Comparison — Radar Chart",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right",
              bbox_to_anchor=(1.35, 1.15))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_radar_chart.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 04: Radar chart")

# ======================================================
# PLOT 5: Budget depletion timeline
# ======================================================
if bidlog is not None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for adv in ADV_SHARED:
        col = f"budget_{adv}"
        if col in bidlog.columns:
            ax.plot(bidlog["step"], bidlog[col],
                    color=COLORS[adv], linewidth=1.5,
                    label=f"{adv} (Budget {BUDGETS_SH[ADV_SHARED.index(adv)]:,})")

    ax.set_xlabel("Auction Step")
    ax.set_ylabel("Remaining Budget")
    ax.set_title("Budget Depletion Over Time — MAPPO Shared",
                 fontweight="bold", fontsize=13)
    ax.legend(title="Advertiser")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x:,.0f}")
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_budget_depletion.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 05: Budget depletion")

# ======================================================
# PLOT 6: pCTR distribution histogram per agent
# ======================================================
if bidlog is not None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, adv in enumerate(ADV_SHARED):
        ax  = axes[idx]
        col = f"pctr_{adv}"
        if col in bidlog.columns:
            data = bidlog[col]
            ax.hist(data[data > 0], bins=30,
                    color=COLORS[adv], alpha=0.8,
                    edgecolor="white")
            ax.set_title(f"Adv {adv}", fontweight="bold")
            ax.set_xlabel("pCTR Value")
            ax.set_ylabel("Count" if idx == 0 else "")
            ax.grid(True, alpha=0.3)
            ax.text(0.95, 0.95,
                    f"Mean: {data.mean():.3f}\n"
                    f"Zeros: {(data==0).sum()/len(data)*100:.0f}%",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round",
                              facecolor="wheat", alpha=0.5))

    plt.suptitle("pCTR Distribution per Advertiser "
                 "(non-zero values)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_pctr_distribution.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 06: pCTR distribution")

# ======================================================
# PLOT 7: Threshold violin plot
# ======================================================
if bidlog is not None:
    fig, ax = plt.subplots(figsize=(10, 6))

    data   = []
    labels = []
    cols   = []

    for adv in ADV_SHARED:
        col = f"threshold_{adv}"
        if col in bidlog.columns:
            data.append(bidlog[col].values)
            labels.append(f"{adv}\n(Budget {BUDGETS_SH[ADV_SHARED.index(adv)]:,})")
            cols.append(COLORS[adv])

    parts = ax.violinplot(data, showmeans=True,
                          showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(cols[i])
        pc.set_alpha(0.7)

    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("red")

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Threshold Value")
    ax.set_title("Learned Threshold Distribution per Advertiser\n"
                 "(Black=Mean, Red=Median)",
                 fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_threshold_violin.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 07: Threshold violin")

# ======================================================
# PLOT 8: Decision rate (BID vs SKIP) pie charts
# ======================================================
if bidlog is not None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for idx, adv in enumerate(ADV_SHARED):
        ax  = axes[idx]
        col = f"decision_{adv}"
        if col in bidlog.columns:
            vc    = bidlog[col].value_counts()
            bids  = vc.get("BID", 0)
            skips = vc.get("SKIP", 0)
            ax.pie([bids, skips],
                   labels=["BID", "SKIP"],
                   colors=[COLORS[adv], "#E0E0E0"],
                   autopct="%1.1f%%",
                   startangle=90)
            ax.set_title(f"Adv {adv}\n"
                         f"pCTR={bidlog[f'pctr_{adv}'].mean():.3f}",
                         fontweight="bold")

    plt.suptitle("BID vs SKIP Decision Rate per Advertiser",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_bid_skip_pie.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 08: BID vs SKIP pie charts")

# ======================================================
# PLOT 9: Scatter — pCTR vs threshold (bid/skip boundary)
# ======================================================
if bidlog is not None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, adv in enumerate(ADV_SHARED):
        ax       = axes[idx]
        pctr_col = f"pctr_{adv}"
        thr_col  = f"threshold_{adv}"
        dec_col  = f"decision_{adv}"

        if all(c in bidlog.columns
               for c in [pctr_col, thr_col, dec_col]):
            sample = bidlog.sample(min(1000, len(bidlog)),
                                   random_state=42)
            bids  = sample[sample[dec_col] == "BID"]
            skips = sample[sample[dec_col] == "SKIP"]

            ax.scatter(bids[pctr_col], bids[thr_col],
                       alpha=0.3, s=5,
                       color=COLORS[adv], label="BID")
            ax.scatter(skips[pctr_col], skips[thr_col],
                       alpha=0.1, s=5,
                       color="gray", label="SKIP")
            ax.plot([0, 1], [0, 1], "r--",
                    linewidth=1, label="pCTR=threshold")
            ax.set_xlabel("pCTR")
            ax.set_ylabel("Threshold" if idx == 0 else "")
            ax.set_title(f"Adv {adv}", fontweight="bold")
            ax.legend(markerscale=3, fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.suptitle("pCTR vs Threshold — Bid/Skip Decision Boundary",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_pctr_vs_threshold_scatter.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 09: pCTR vs threshold scatter")

# ======================================================
# PLOT 10: Winner distribution bar chart
# ======================================================
if bidlog is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    won     = bidlog[bidlog["winner"] != "None"]
    wc      = won["winner"].value_counts()
    cc      = won.groupby("winner")["click"].sum()
    costc   = won.groupby("winner")["cost_paid"].sum()

    ax = axes[0]
    bars = ax.bar(wc.index, wc.values,
                  color=[COLORS.get(str(a), "#607D8B")
                         for a in wc.index],
                  alpha=0.85)
    for bar, v in zip(bars, wc.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{v:,}", ha="center", fontsize=9)
    ax.set_xlabel("Advertiser")
    ax.set_ylabel("Number of Wins")
    ax.set_title("Auction Wins per Advertiser",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(costc.index, costc.values,
                  color=[COLORS.get(str(a), "#607D8B")
                         for a in costc.index],
                  alpha=0.85)
    for bar, v in zip(bars, costc.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{v:,.0f}", ha="center", fontsize=9)
    ax.set_xlabel("Advertiser")
    ax.set_ylabel("Total Cost Paid")
    ax.set_title("Total Cost Paid per Advertiser",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Auction Outcome Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_winner_distribution.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 10: Winner distribution")

# ======================================================
# PLOT 11: Market price distribution
# ======================================================
if bidlog is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(bidlog["market_price"], bins=50,
            color="#2196F3", alpha=0.8, edgecolor="white")
    ax.axvline(bidlog["market_price"].mean(),
               color="red", linestyle="--",
               linewidth=2, label=f"Mean={bidlog['market_price'].mean():.1f}")
    ax.axvline(bidlog["market_price"].median(),
               color="green", linestyle="--",
               linewidth=2, label=f"Median={bidlog['market_price'].median():.1f}")
    ax.set_xlabel("Market Price")
    ax.set_ylabel("Count")
    ax.set_title("Market Price Distribution",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    won  = bidlog[bidlog["cost_paid"] > 0]["cost_paid"]
    ax.hist(won, bins=40,
            color="#4CAF50", alpha=0.8, edgecolor="white")
    ax.axvline(won.mean(), color="red", linestyle="--",
               linewidth=2, label=f"Mean={won.mean():.1f}")
    ax.set_xlabel("Cost Paid per Win")
    ax.set_ylabel("Count")
    ax.set_title("Cost Paid Distribution (Winning Auctions)",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Market Price & Cost Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_market_price_distribution.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 11: Market price distribution")

# ======================================================
# PLOT 12: Scatter — clicks vs CPC (efficiency frontier)
# ======================================================
if eval_df is not None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in eval_df.iterrows():
        color = METHOD_COLORS.get(row["method"], "#607D8B")
        ax.scatter(row["cpc"], row["clicks"],
                   s=200, color=color,
                   zorder=5, alpha=0.9)
        ax.annotate(row["method"],
                    (row["cpc"], row["clicks"]),
                    textcoords="offset points",
                    xytext=(8, 5),
                    fontsize=9,
                    fontweight="bold",
                    color=color)

    ax.set_xlabel("Cost Per Click (CPC) — Lower is Better",
                  fontsize=11)
    ax.set_ylabel("Total Clicks — Higher is Better",
                  fontsize=11)
    ax.set_title("Efficiency Frontier: Clicks vs CPC\n"
                 "Best = Top-Left Corner",
                 fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Annotate best region
    ax.annotate("Best Region",
                xy=(eval_df["cpc"].min(),
                    eval_df["clicks"].max()),
                xytext=(eval_df["cpc"].min() + 5,
                        eval_df["clicks"].max() - 20),
                fontsize=10, color="green",
                arrowprops=dict(arrowstyle="->",
                                color="green"))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_clicks_vs_cpc_scatter.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 12: Clicks vs CPC scatter")

# ======================================================
# PLOT 13: Rolling win rate over time
# ======================================================
if bidlog is not None:
    fig, ax = plt.subplots(figsize=(12, 6))

    window = 200
    for adv in ADV_SHARED:
        col = f"decision_{adv}"
        if col in bidlog.columns:
            wins = (bidlog["winner"] == adv).astype(int)
            rolling_wr = wins.rolling(window=window).mean() * 100
            ax.plot(bidlog["step"], rolling_wr,
                    color=COLORS[adv], linewidth=1.5,
                    label=f"Adv {adv}")

    ax.set_xlabel("Auction Step")
    ax.set_ylabel(f"Win Rate % (Rolling {window} steps)")
    ax.set_title("Rolling Win Rate Over Time per Advertiser",
                 fontweight="bold", fontsize=13)
    ax.legend(title="Advertiser")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "13_rolling_win_rate.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 13: Rolling win rate")

# ======================================================
# PLOT 14: Version progression (v3→v4→v6→MAPPO)
# ======================================================
methods_prog = {
    "v3\n(Base AC)":          757.0,
    "v4\n(Enhanced)":         695.0,
    "v6\n(3386+CTR Budget)":  983.6,
    "MAPPO\n(Centralized)":   999.8,
    "AC Shared\n(True Comp)": 831.4,
    "MAPPO Shared\n(Best)":   831.8,
}

fig, ax = plt.subplots(figsize=(12, 6))
x      = np.arange(len(methods_prog))
bars   = ax.bar(x,
                list(methods_prog.values()),
                color=["#9E9E9E", "#2196F3", "#4CAF50",
                       "#F44336", "#FF9800", "#9C27B0"],
                alpha=0.85, width=0.6)

for bar, v in zip(bars, methods_prog.values()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f"{v:.0f}", ha="center",
            fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(list(methods_prog.keys()), fontsize=10)
ax.set_ylabel("Total Clicks (All Agents Combined)")
ax.set_title("Performance Progression Across All Versions",
             fontweight="bold", fontsize=13)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(methods_prog.values()) * 1.2)

# Add annotation for shared env note
ax.axvline(x=3.5, color="gray", linestyle="--",
           linewidth=1.5, alpha=0.7)
ax.text(3.6, max(methods_prog.values()) * 1.1,
        "Shared\nEnvironment",
        fontsize=9, color="gray")
ax.text(-0.4, max(methods_prog.values()) * 1.1,
        "Separate\nLogs",
        fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(OUT_DIR / "14_version_progression.png",
            bbox_inches="tight")
plt.close()
print("Plot 14: Version progression")

# ======================================================
# PLOT 15: Budget utilization comparison
# ======================================================
if eval_df is not None:
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = eval_df["method"].tolist()
    utils   = eval_df["budget_util"].tolist()
    colors  = [METHOD_COLORS.get(m, "#607D8B") for m in methods]

    bars = ax.bar(methods, utils, color=colors, alpha=0.85)
    ax.axhline(y=80, color="orange", linestyle="--",
               linewidth=2, label="80% threshold")
    ax.axhline(y=100, color="red", linestyle="--",
               linewidth=1.5, label="100% budget")

    for bar, v in zip(bars, utils):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Budget Utilization (%)")
    ax.set_title("Budget Utilization per Method",
                 fontweight="bold", fontsize=13)
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "15_budget_utilization.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 15: Budget utilization")

# ======================================================
# PLOT 16: AC vs MAPPO variance comparison
# ======================================================
if dfs_v6 and dfs_mappo:
    fig, ax = plt.subplots(figsize=(10, 5))

    x     = np.arange(len(ADV_SHARED))
    width = 0.35

    v6_stds    = [get_clicks(dfs_v6,    adv).std()
                  for adv in ADV_V6]
    mappo_stds = [get_clicks(dfs_mappo, adv).std()
                  for adv in ADV_V6]

    ax.bar(x - width/2, v6_stds,    width,
           label="AC v6",  color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, mappo_stds, width,
           label="MAPPO",  color="#F44336",  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ADV_V6)
    ax.set_ylabel("Standard Deviation of Clicks")
    ax.set_title("Policy Variance: AC v6 vs MAPPO\n"
                 "Lower = More Stable Policy",
                 fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "16_variance_comparison.png",
                bbox_inches="tight")
    plt.close()
    print("Plot 16: Variance comparison")

# ======================================================
# SUMMARY
# ======================================================
plots = list(OUT_DIR.glob("*.png"))
print(f"\n{'='*55}")
print(f" ALL PLOTS GENERATED!")
print(f"{'='*55}")
print(f"Total plots : {len(plots)}")
print(f"Output dir  : {OUT_DIR}")
print(f"\nPlots saved:")
for p in sorted(plots):
    size = p.stat().st_size / 1024
    print(f"  {p.name:<45} {size:>6.1f} KB")
