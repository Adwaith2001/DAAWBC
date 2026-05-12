import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# PATHS
# ======================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "final_experiments_5agents_v2"
OUT_DIR = ROOT / "outputs" / "plots_5agents_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
ADV_IDS = ["1458", "2259", "2821", "2997", "3358"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
SMOOTH_WINDOW = 10  # rolling average window

# ======================================================
# LOAD ALL SEED LOGS
# ======================================================
dfs = []
for seed in SEEDS:
    f = DATA_DIR / f"actor_critic_5adv_v2_seed_{seed}.csv"
    df = pd.read_csv(f)
    dfs.append(df)

episodes = dfs[0]["episode"].values

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def stack_metric(metric, adv_id):
    col = f"{metric}_{adv_id}"
    return np.stack([df[col].values for df in dfs])

def smooth(x, window=SMOOTH_WINDOW):
    return pd.Series(x).rolling(window, min_periods=1).mean().values


# ======================================================
# PLOT 1: CLICKS PER ADVERTISER (mean ± std, smoothed)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    clicks = stack_metric("clicks", adv)
    mean = smooth(clicks.mean(axis=0))
    std  = smooth(clicks.std(axis=0))

    ax = axes[idx]
    ax.plot(episodes, mean, color=COLORS[idx], label="Mean Clicks")
    ax.fill_between(
        episodes,
        np.clip(mean - std, 0, None),
        mean + std,
        alpha=0.3, color=COLORS[idx], label="±1 Std"
    )
    ax.set_title(f"Advertiser {adv}", fontsize=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Clicks")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.5)

axes[-1].set_visible(False)
fig.suptitle("5-Agent Actor–Critic v2: Clicks per Advertiser (Mean ± Std)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "v2_clicks_per_advertiser.png", dpi=300)
plt.close()
print("✅ Saved: v2_clicks_per_advertiser.png")


# ======================================================
# PLOT 2: COST PER ADVERTISER (mean ± std, smoothed)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    costs = stack_metric("cost", adv)
    mean = smooth(costs.mean(axis=0))
    std  = smooth(costs.std(axis=0))

    ax = axes[idx]
    ax.plot(episodes, mean, color="orange", label="Mean Cost")
    ax.fill_between(
        episodes,
        np.clip(mean - std, 0, None),
        mean + std,
        alpha=0.3, color="orange", label="±1 Std"
    )
    ax.axhline(y=1000, color="red", linestyle="--", alpha=0.5, label="Budget")
    ax.set_title(f"Advertiser {adv}", fontsize=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cost")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.5)

axes[-1].set_visible(False)
fig.suptitle("5-Agent Actor–Critic v2: Cost per Advertiser (Mean ± Std)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "v2_cost_per_advertiser.png", dpi=300)
plt.close()
print("✅ Saved: v2_cost_per_advertiser.png")


# ======================================================
# PLOT 3: REWARD PER ADVERTISER (mean ± std, smoothed)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    rewards = stack_metric("reward", adv)
    mean = smooth(rewards.mean(axis=0))
    std  = smooth(rewards.std(axis=0))

    ax = axes[idx]
    ax.plot(episodes, mean, color="green", label="Mean Reward")
    ax.fill_between(
        episodes,
        mean - std,
        mean + std,
        alpha=0.3, color="green", label="±1 Std"
    )
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.4)
    ax.set_title(f"Advertiser {adv}", fontsize=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.5)

axes[-1].set_visible(False)
fig.suptitle("5-Agent Actor–Critic v2: Reward per Advertiser (Mean ± Std)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "v2_reward_per_advertiser.png", dpi=300)
plt.close()
print("✅ Saved: v2_reward_per_advertiser.png")


# ======================================================
# PLOT 4: ALL ADVERTISERS CLICKS ON ONE PLOT (smoothed)
# ======================================================
plt.figure(figsize=(12, 6))

for idx, adv in enumerate(ADV_IDS):
    clicks = stack_metric("clicks", adv)
    mean = smooth(clicks.mean(axis=0))
    std  = smooth(clicks.std(axis=0))

    plt.plot(episodes, mean, label=f"Adv {adv}", color=COLORS[idx])
    plt.fill_between(
        episodes,
        np.clip(mean - std, 0, None),
        mean + std,
        alpha=0.15, color=COLORS[idx]
    )

plt.xlabel("Episode")
plt.ylabel("Clicks")
plt.title("5-Agent Actor–Critic v2: All Advertisers Clicks Comparison")
plt.legend()
plt.ylim(bottom=0)
plt.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_DIR / "v2_all_advertisers_clicks.png", dpi=300)
plt.close()
print("✅ Saved: v2_all_advertisers_clicks.png")


# ======================================================
# PLOT 5: FINAL EPISODE CLICKS BAR CHART
# ======================================================
final_clicks_mean = []
final_clicks_std  = []

for adv in ADV_IDS:
    clicks = stack_metric("clicks", adv)[:, -1]
    final_clicks_mean.append(clicks.mean())
    final_clicks_std.append(clicks.std())

plt.figure(figsize=(8, 5))
plt.bar(ADV_IDS, final_clicks_mean, yerr=final_clicks_std,
        capsize=6, color=COLORS)
plt.xlabel("Advertiser ID")
plt.ylabel("Clicks (mean ± std)")
plt.title("5-Agent Actor–Critic v2: Final Episode Clicks per Advertiser")
plt.ylim(bottom=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_DIR / "v2_final_clicks_bar.png", dpi=300)
plt.close()
print("✅ Saved: v2_final_clicks_bar.png")


# ======================================================
# PLOT 6: V1 vs V2 CLICKS COMPARISON (last 10 ep avg)
# ======================================================
V1_DIR = ROOT / "outputs" / "final_experiments_5agents"

v1_dfs = []
for seed in SEEDS:
    f = V1_DIR / f"actor_critic_5adv_seed_{seed}.csv"
    if f.exists():
        v1_dfs.append(pd.read_csv(f))

if v1_dfs:
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(ADV_IDS))
    width = 0.35

    v1_means, v1_stds = [], []
    v2_means, v2_stds = [], []

    for adv in ADV_IDS:
        col = f"clicks_{adv}"

        # last 10 episodes average per seed
        v1 = np.array([df[col].values[-10:].mean() for df in v1_dfs])
        v2 = np.array([df[col].values[-10:].mean() for df in dfs])

        v1_means.append(v1.mean())
        v1_stds.append(v1.std())
        v2_means.append(v2.mean())
        v2_stds.append(v2.std())

    bars1 = ax.bar(x - width/2, v1_means, width, yerr=v1_stds,
                   capsize=5, label="v1 (100 ep)", color="#4C72B0", alpha=0.8)
    bars2 = ax.bar(x + width/2, v2_means, width, yerr=v2_stds,
                   capsize=5, label="v2 (200 ep)", color="#DD8452", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ADV_IDS)
    ax.set_xlabel("Advertiser ID")
    ax.set_ylabel("Clicks (last 10 ep avg)")
    ax.set_title("v1 vs v2: Clicks Comparison per Advertiser")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v1_vs_v2_clicks.png", dpi=300)
    plt.close()
    print("✅ Saved: v1_vs_v2_clicks.png")
else:
    print("⚠️  v1 logs not found, skipping v1 vs v2 comparison plot")


# ======================================================
# SUMMARY TABLE
# ======================================================
print("\n=== v2 Final Episode Summary (Mean ± Std across Seeds) ===\n")
print(f"{'Advertiser':<12} {'Clicks':>18} {'Cost':>18} {'Reward':>18}")
print("-" * 70)

for adv in ADV_IDS:
    clicks  = stack_metric("clicks", adv)[:, -1]
    costs   = stack_metric("cost",   adv)[:, -1]
    rewards = stack_metric("reward", adv)[:, -1]

    print(
        f"{adv:<12} "
        f"{clicks.mean():>8.2f} ± {clicks.std():>6.2f}  "
        f"{costs.mean():>8.2f} ± {costs.std():>6.2f}  "
        f"{rewards.mean():>8.2f} ± {rewards.std():>6.2f}"
    )

print(f"\n✅ All plots saved to: outputs/plots_5agents_v2/")