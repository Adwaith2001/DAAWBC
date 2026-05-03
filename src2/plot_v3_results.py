"""
plot_v3_results.py
Plots v3 training results for all 5 advertisers across 5 seeds
Run from: src2/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# PATHS
# ======================================================
ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "final_experiments_5agents_v3"
OUT_DIR  = ROOT / "outputs" / "plots_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADV_IDS = ["1458", "2259", "2821", "2997", "3358"]
SEEDS   = [0, 1, 2, 3, 4]

COLORS = {
    "1458": "#2196F3",
    "2259": "#F44336",
    "2821": "#4CAF50",
    "2997": "#FF9800",
    "3358": "#9C27B0",
}

# ======================================================
# LOAD DATA
# ======================================================
dfs = []
for seed in SEEDS:
    df = pd.read_csv(DATA_DIR / f"actor_critic_5adv_v3_seed_{seed}.csv")
    dfs.append(df)

episodes = dfs[0]["episode"].values

def get_metric(metric):
    return np.stack([df[metric].values for df in dfs])  # shape (5, 200)

clicks  = {adv: get_metric(f"clicks_{adv}")  for adv in ADV_IDS}
rewards = {adv: get_metric(f"reward_{adv}")  for adv in ADV_IDS}
costs   = {adv: get_metric(f"cost_{adv}")    for adv in ADV_IDS}


# ======================================================
# PLOT 1: Clicks per advertiser (mean ± std)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    ax = axes[idx]
    m  = clicks[adv].mean(axis=0)
    s  = clicks[adv].std(axis=0)
    ax.plot(episodes, m, color=COLORS[adv], linewidth=1.5, label=f"Advertiser {adv}")
    ax.fill_between(episodes, m - s, m + s, alpha=0.2, color=COLORS[adv])
    ax.set_title(f"Advertiser {adv}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Clicks")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

ax = axes[5]
for adv in ADV_IDS:
    m = clicks[adv].mean(axis=0)
    ax.plot(episodes, m, color=COLORS[adv], linewidth=1.5, label=adv)
ax.set_title("All Advertisers", fontsize=12, fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Clicks")
ax.grid(True, alpha=0.3)
ax.legend(title="Advertiser", fontsize=9)

plt.suptitle("v3 Training — Clicks per Advertiser (Mean ± Std, 5 Seeds)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "v3_clicks_per_advertiser.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Plot 1: clicks_per_advertiser")


# ======================================================
# PLOT 2: Rewards per advertiser (mean ± std)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    ax = axes[idx]
    m  = rewards[adv].mean(axis=0)
    s  = rewards[adv].std(axis=0)
    ax.plot(episodes, m, color=COLORS[adv], linewidth=1.5)
    ax.fill_between(episodes, m - s, m + s, alpha=0.2, color=COLORS[adv])
    ax.set_title(f"Advertiser {adv}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

ax = axes[5]
for adv in ADV_IDS:
    m = rewards[adv].mean(axis=0)
    ax.plot(episodes, m, color=COLORS[adv], linewidth=1.5, label=adv)
ax.set_title("All Advertisers", fontsize=12, fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.grid(True, alpha=0.3)
ax.legend(title="Advertiser", fontsize=9)

plt.suptitle("v3 Training — Reward per Advertiser (Mean ± Std, 5 Seeds)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "v3_rewards_per_advertiser.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Plot 2: rewards_per_advertiser")


# ======================================================
# PLOT 3: Total clicks across all advertisers
# ======================================================
fig, ax = plt.subplots(figsize=(10, 5))
total = sum(clicks[adv] for adv in ADV_IDS)  # (5, 200)
m = total.mean(axis=0)
s = total.std(axis=0)

ax.plot(episodes, m, color="#2196F3", linewidth=2, label="Total clicks (all 5 agents)")
ax.fill_between(episodes, m - s, m + s, alpha=0.2, color="#2196F3", label="±1 Std")
ax.set_title("v3 — Total Clicks Across All 5 Advertisers", fontsize=13, fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Clicks")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "v3_total_clicks.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Plot 3: total_clicks")


# ======================================================
# PLOT 4: Final episode bar chart summary
# ======================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

final_clicks_mean  = [clicks[adv][:, -1].mean()  for adv in ADV_IDS]
final_clicks_std   = [clicks[adv][:, -1].std()   for adv in ADV_IDS]
final_rewards_mean = [rewards[adv][:, -1].mean() for adv in ADV_IDS]
final_rewards_std  = [rewards[adv][:, -1].std()  for adv in ADV_IDS]
final_costs_mean   = [costs[adv][:, -1].mean()   for adv in ADV_IDS]

colors_list = [COLORS[a] for a in ADV_IDS]

ax = axes[0]
ax.bar(ADV_IDS, final_clicks_mean, yerr=final_clicks_std,
       color=colors_list, capsize=5, alpha=0.85)
ax.set_title("Final Episode — Clicks", fontweight="bold")
ax.set_ylabel("Clicks")
ax.set_xlabel("Advertiser")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.bar(ADV_IDS, final_rewards_mean, yerr=final_rewards_std,
       color=colors_list, capsize=5, alpha=0.85)
ax.set_title("Final Episode — Reward", fontweight="bold")
ax.set_ylabel("Reward")
ax.set_xlabel("Advertiser")
ax.grid(axis="y", alpha=0.3)

ax = axes[2]
ax.bar(ADV_IDS, final_costs_mean, color=colors_list, alpha=0.85)
ax.axhline(y=20000, color="red", linestyle="--", linewidth=1.5, label="Budget")
ax.set_title("Final Episode — Cost", fontweight="bold")
ax.set_ylabel("Cost")
ax.set_xlabel("Advertiser")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.suptitle("v3 Final Episode Performance Summary (Mean ± Std, 5 Seeds)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "v3_final_performance.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Plot 4: final_performance")


# ======================================================
# PLOT 5: Per-seed training curves (clicks)
# ======================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for seed_idx, seed in enumerate(SEEDS):
    total_seed = sum(clicks[adv][seed_idx] for adv in ADV_IDS)
    ax.plot(episodes, total_seed, linewidth=1.2, alpha=0.8, label=f"Seed {seed}")
ax.set_title("Total Clicks — Per Seed", fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Clicks")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for seed_idx, seed in enumerate(SEEDS):
    total_seed = sum(rewards[adv][seed_idx] for adv in ADV_IDS)
    ax.plot(episodes, total_seed, linewidth=1.2, alpha=0.8, label=f"Seed {seed}")
ax.set_title("Total Reward — Per Seed", fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("v3 Training — Per-Seed Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "v3_per_seed_curves.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Plot 5: per_seed_curves")


# ======================================================
# SUMMARY TABLE
# ======================================================
print("\n" + "="*65)
print(" v3 TRAINING SUMMARY (all episodes, all seeds)")
print("="*65)
print(f"{'Advertiser':<12} {'Clicks':>18} {'Reward':>18} {'Cost':>10}")
print("-"*65)
for adv in ADV_IDS:
    m_c  = clicks[adv].mean()
    s_c  = clicks[adv].std()
    m_r  = rewards[adv].mean()
    s_r  = rewards[adv].std()
    m_co = costs[adv].mean()
    print(f"{adv:<12} {m_c:>8.1f} ± {s_c:<6.1f}   {m_r:>8.1f} ± {s_r:<6.1f}   {m_co:>8.0f}")
print("="*65)
print(f"\n✅ All plots saved to: outputs/plots_v3/")