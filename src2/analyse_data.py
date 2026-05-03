"""
analyse_data.py
Phase 1 - Data Analysis for all 5 advertisers

Run from src2/:
    python analyse_data.py

Outputs:
    - Console summary per advertiser
    - outputs/plots_5agents_v3/data_analysis/ folder with plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# PATHS
# ======================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ipinyou"
OUT_DIR  = ROOT / "outputs" / "plots_5agents_v3" / "data_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADV_IDS = ["1458", "2259", "2821", "2997", "3358"]
COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# ======================================================
# LOAD DATA
# ======================================================
dfs = {}
for adv in ADV_IDS:
    path = DATA_DIR / adv / "final_sample_log_with_pctr.txt"
    dfs[adv] = pd.read_csv(path, sep="\t")
    print(f"✅ Loaded {adv}: {len(dfs[adv]):,} rows")

# ======================================================
# 1. BASIC STATS PER ADVERTISER
# ======================================================
print("\n" + "="*70)
print(" BASIC STATS PER ADVERTISER")
print("="*70)

summary = []
for adv in ADV_IDS:
    df = dfs[adv]
    stats = {
        "advertiser"   : adv,
        "total_rows"   : len(df),
        "total_clicks" : df["click"].sum(),
        "ctr"          : df["click"].mean() * 100,
        "market_min"   : df["market_price"].min(),
        "market_max"   : df["market_price"].max(),
        "market_mean"  : df["market_price"].mean(),
        "market_median": df["market_price"].median(),
        "market_p75"   : df["market_price"].quantile(0.75),
        "market_p90"   : df["market_price"].quantile(0.90),
        "market_p95"   : df["market_price"].quantile(0.95),
        "pctr_mean"    : df["pctr"].mean(),
        "pctr_max"     : df["pctr"].max(),
    }
    summary.append(stats)

summary_df = pd.DataFrame(summary).set_index("advertiser")

print("\n--- Impression Volume & CTR ---")
print(summary_df[["total_rows", "total_clicks", "ctr"]].to_string())

print("\n--- Market Price Distribution ---")
print(summary_df[["market_min", "market_mean", "market_median",
                   "market_p75", "market_p90", "market_p95", "market_max"]].to_string())

print("\n--- pCTR Stats ---")
print(summary_df[["pctr_mean", "pctr_max"]].to_string())

# ======================================================
# 2. BID GRID RECOMMENDATION
# ======================================================
print("\n" + "="*70)
print(" BID GRID RECOMMENDATION")
print("="*70)

for adv in ADV_IDS:
    df = dfs[adv]
    p95 = df["market_price"].quantile(0.95)
    p75 = df["market_price"].quantile(0.75)
    print(f"\nAdvertiser {adv}:")
    print(f"  75th percentile market price : {p75:.2f}")
    print(f"  95th percentile market price : {p95:.2f}")
    print(f"  Current bid grid max         : 5.0  ← {'✅ OK' if p75 <= 5.0 else '❌ TOO LOW'}")
    print(f"  Recommended bid grid max     : {p95:.1f}")

# ======================================================
# 3. CLICK VALUE RECOMMENDATION
# ======================================================
print("\n" + "="*70)
print(" CLICK VALUE RECOMMENDATION")
print("="*70)

for adv in ADV_IDS:
    df = dfs[adv]
    avg_cost_per_click = df["market_price"].mean() / (df["click"].mean() + 1e-9)
    print(f"\nAdvertiser {adv}:")
    print(f"  Avg market price             : {df['market_price'].mean():.4f}")
    print(f"  CTR                          : {df['click'].mean()*100:.4f}%")
    print(f"  Avg cost per click           : {avg_cost_per_click:.2f}")
    print(f"  Recommended CLICK_VALUE      : {avg_cost_per_click * 1.5:.2f}  (1.5x cost per click for profit)")

# ======================================================
# PLOT 1: MARKET PRICE DISTRIBUTION PER ADVERTISER
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    df = dfs[adv]
    p95 = df["market_price"].quantile(0.95)

    ax = axes[idx]
    ax.hist(
        df["market_price"].clip(upper=p95 * 1.5),
        bins=50, color=COLORS[idx], alpha=0.8, edgecolor="white"
    )
    ax.axvline(df["market_price"].mean(),   color="red",    linestyle="--", label=f"Mean: {df['market_price'].mean():.2f}")
    ax.axvline(df["market_price"].median(), color="orange", linestyle="--", label=f"Median: {df['market_price'].median():.2f}")
    ax.axvline(5.0, color="black", linestyle="-", linewidth=2, label="Current bid max: 5.0")
    ax.set_title(f"Advertiser {adv}", fontsize=12)
    ax.set_xlabel("Market Price")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.4)

axes[-1].set_visible(False)
fig.suptitle("Market Price Distribution per Advertiser", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "market_price_distribution.png", dpi=300)
plt.close()
print("\n✅ Saved: market_price_distribution.png")

# ======================================================
# PLOT 2: pCTR DISTRIBUTION PER ADVERTISER
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    df = dfs[adv]
    ax = axes[idx]
    ax.hist(df["pctr"], bins=50, color=COLORS[idx], alpha=0.8, edgecolor="white")
    ax.axvline(df["pctr"].mean(), color="red", linestyle="--",
               label=f"Mean pCTR: {df['pctr'].mean():.4f}")
    ax.set_title(f"Advertiser {adv}", fontsize=12)
    ax.set_xlabel("pCTR")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)

axes[-1].set_visible(False)
fig.suptitle("pCTR Distribution per Advertiser", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "pctr_distribution.png", dpi=300)
plt.close()
print("✅ Saved: pctr_distribution.png")

# ======================================================
# PLOT 3: CTR PER ADVERTISER BAR CHART
# ======================================================
ctrs  = [dfs[adv]["click"].mean() * 100 for adv in ADV_IDS]
vols  = [len(dfs[adv]) for adv in ADV_IDS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(ADV_IDS, ctrs, color=COLORS)
ax1.set_xlabel("Advertiser ID")
ax1.set_ylabel("CTR (%)")
ax1.set_title("Click-Through Rate per Advertiser")
ax1.grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(ctrs):
    ax1.text(i, v + 0.001, f"{v:.3f}%", ha="center", fontsize=9)

ax2.bar(ADV_IDS, vols, color=COLORS)
ax2.set_xlabel("Advertiser ID")
ax2.set_ylabel("Impressions")
ax2.set_title("Impression Volume per Advertiser")
ax2.grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(vols):
    ax2.text(i, v + 1000, f"{v:,}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "ctr_and_volume.png", dpi=300)
plt.close()
print("✅ Saved: ctr_and_volume.png")

# ======================================================
# PLOT 4: MARKET PRICE vs pCTR SCATTER (sample)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    df = dfs[adv].sample(min(5000, len(dfs[adv])), random_state=42)
    ax = axes[idx]
    ax.scatter(df["market_price"], df["pctr"],
               alpha=0.3, s=5, color=COLORS[idx])
    ax.set_title(f"Advertiser {adv}", fontsize=12)
    ax.set_xlabel("Market Price")
    ax.set_ylabel("pCTR")
    ax.set_xlim(0, df["market_price"].quantile(0.95))
    ax.grid(linestyle="--", alpha=0.4)

axes[-1].set_visible(False)
fig.suptitle("Market Price vs pCTR (5k sample)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "market_price_vs_pctr.png", dpi=300)
plt.close()
print("✅ Saved: market_price_vs_pctr.png")

print(f"\n✅ All analysis plots saved to: {OUT_DIR}")
print("\n🎯 Use the BID GRID and CLICK VALUE recommendations above to configure train_v3.py")