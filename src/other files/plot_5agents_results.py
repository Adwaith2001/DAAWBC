import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# PATHS
# ======================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "final_experiments_5agents"
OUT_DIR = ROOT / "outputs" / "plots_5agents"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
ADV_IDS = ["1458", "2259", "2821", "2997", "3358"]

# ======================================================
# LOAD ALL SEED LOGS
# ======================================================
dfs = []
for seed in SEEDS:
    f = DATA_DIR / f"actor_critic_5adv_seed_{seed}.csv"
    df = pd.read_csv(f)
    dfs.append(df)

episodes = dfs[0]["episode"].values

# ======================================================
# HELPER: stack metric across seeds for one advertiser
# ======================================================
def stack_metric(metric, adv_id):
    col = f"{metric}_{adv_id}"
    return np.stack([df[col].values for df in dfs])


# ======================================================
# PLOT 1: CLICKS PER ADVERTISER (mean ± std)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    clicks = stack_metric("clicks", adv)
    mean = clicks.mean(axis=0)
    std = clicks.std(axis=0)

    ax = axes[idx]
    ax.plot(episodes, mean, label="Mean Clicks")
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.3, label="±1 Std")
    ax.set_title(f"Advertiser {adv}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Clicks")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.5)

# hide the 6th unused subplot
axes[-1].set_visible(False)

fig.suptitle("5-Agent Actor–Critic: Clicks per Advertiser (Mean ± Std)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "clicks_per_advertiser.png", dpi=300)
plt.close()
print("✅ Saved: clicks_per_advertiser.png")


# ======================================================
# PLOT 2: COST PER ADVERTISER (mean ± std)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    costs = stack_metric("cost", adv)
    mean = costs.mean(axis=0)
    std = costs.std(axis=0)

    ax = axes[idx]
    ax.plot(episodes, mean, color="orange", label="Mean Cost")
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.3, color="orange", label="±1 Std")
    ax.set_title(f"Advertiser {adv}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cost")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.5)

axes[-1].set_visible(False)

fig.suptitle("5-Agent Actor–Critic: Cost per Advertiser (Mean ± Std)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "cost_per_advertiser.png", dpi=300)
plt.close()
print("✅ Saved: cost_per_advertiser.png")


# ======================================================
# PLOT 3: REWARD PER ADVERTISER (mean ± std)
# ======================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, adv in enumerate(ADV_IDS):
    rewards = stack_metric("reward", adv)
    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)

    ax = axes[idx]
    ax.plot(episodes, mean, color="green", label="Mean Reward")
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.3, color="green", label="±1 Std")
    ax.set_title(f"Advertiser {adv}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.5)

axes[-1].set_visible(False)

fig.suptitle("5-Agent Actor–Critic: Reward per Advertiser (Mean ± Std)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "reward_per_advertiser.png", dpi=300)
plt.close()
print("✅ Saved: reward_per_advertiser.png")


# ======================================================
# PLOT 4: FINAL EPISODE CLICKS - ALL ADVERTISERS (bar)
# ======================================================
final_clicks_mean = []
final_clicks_std = []

for adv in ADV_IDS:
    clicks = stack_metric("clicks", adv)
    final = clicks[:, -1]  # last episode for each seed
    final_clicks_mean.append(final.mean())
    final_clicks_std.append(final.std())

plt.figure(figsize=(8, 5))
bars = plt.bar(ADV_IDS, final_clicks_mean, yerr=final_clicks_std, capsize=6,
               color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
plt.xlabel("Advertiser ID")
plt.ylabel("Clicks (mean ± std)")
plt.title("5-Agent Actor–Critic: Final Episode Clicks per Advertiser")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_DIR / "final_clicks_bar.png", dpi=300)
plt.close()
print("✅ Saved: final_clicks_bar.png")


# ======================================================
# PLOT 5: ALL ADVERTISERS CLICKS ON ONE PLOT
# ======================================================
plt.figure(figsize=(10, 5))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

for idx, adv in enumerate(ADV_IDS):
    clicks = stack_metric("clicks", adv)
    mean = clicks.mean(axis=0)
    std = clicks.std(axis=0)
    plt.plot(episodes, mean, label=f"Adv {adv}", color=colors[idx])
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=colors[idx])

plt.xlabel("Episode")
plt.ylabel("Clicks")
plt.title("5-Agent Actor–Critic: All Advertisers Clicks Comparison")
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_DIR / "all_advertisers_clicks.png", dpi=300)
plt.close()
print("✅ Saved: all_advertisers_clicks.png")


# ======================================================
# SUMMARY TABLE
# ======================================================
print("\n=== Final Episode Summary (Mean ± Std across Seeds) ===\n")
print(f"{'Advertiser':<12} {'Clicks':>15} {'Cost':>15} {'Reward':>15}")
print("-" * 60)

for adv in ADV_IDS:
    clicks = stack_metric("clicks", adv)[:, -1]
    costs  = stack_metric("cost",   adv)[:, -1]
    rewards = stack_metric("reward", adv)[:, -1]

    print(
        f"{adv:<12} "
        f"{clicks.mean():>7.2f} ± {clicks.std():>4.2f}  "
        f"{costs.mean():>7.2f} ± {costs.std():>4.2f}  "
        f"{rewards.mean():>7.2f} ± {rewards.std():>4.2f}"
    )

print(f"\n✅ All plots saved to: outputs/plots_5agents/")