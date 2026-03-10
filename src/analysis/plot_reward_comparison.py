import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ===== CONFIG =====
ACTOR_CRITIC_FILES = [
    ROOT / "outputs/final_experiments/actor_critic_3adv_seed_0.csv",
    ROOT / "outputs/final_experiments/actor_critic_3adv_seed_1.csv",
    ROOT / "outputs/final_experiments/actor_critic_3adv_seed_2.csv",
    ROOT / "outputs/final_experiments/actor_critic_3adv_seed_3.csv",
    ROOT / "outputs/final_experiments/actor_critic_3adv_seed_4.csv",
]

REINFORCE_FILE = ROOT / "outputs/reinforce/reinforce_multiseed.csv"

LAMBDA = 0.001  # use the same lambda you reported

# =========================
# Load Actor–Critic rewards
# =========================
ac_rewards = []

for f in ACTOR_CRITIC_FILES:
    df = pd.read_csv(f)
    reward = (
        df["clicks_0"] + df["clicks_1"] + df["clicks_2"]
        - LAMBDA * (df["cost_0"] + df["cost_1"] + df["cost_2"])
    )
    ac_rewards.append(reward.values)

ac_rewards = np.array(ac_rewards)
ac_mean = ac_rewards.mean(axis=0)
ac_std = ac_rewards.std(axis=0)

# =========================
# Load REINFORCE rewards
# =========================
df_r = pd.read_csv(REINFORCE_FILE)
rf_rewards = df_r.filter(like="return").values.T
rf_mean = rf_rewards.mean(axis=0)
rf_std = rf_rewards.std(axis=0)

episodes = np.arange(1, len(ac_mean) + 1)

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 6))

plt.plot(episodes, rf_mean, label="REINFORCE", color="tab:orange")
plt.fill_between(
    episodes, rf_mean - rf_std, rf_mean + rf_std,
    alpha=0.25, color="tab:orange"
)

plt.plot(episodes, ac_mean, label="Actor–Critic (3 Adv)", color="tab:blue")
plt.fill_between(
    episodes, ac_mean - ac_std, ac_mean + ac_std,
    alpha=0.25, color="tab:blue"
)

plt.xlabel("Episode")
plt.ylabel("Return (Reward)")
plt.title("Reward Comparison: Actor–Critic vs REINFORCE")
plt.legend()
plt.grid(True)

OUT = ROOT / "outputs/analysis/reward_comparison.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200)
plt.show()

print(f"✅ Saved plot to {OUT}")
