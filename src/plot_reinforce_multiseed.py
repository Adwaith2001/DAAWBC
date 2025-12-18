import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# PATHS
# =========================
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# =========================
# LOAD MULTI-SEED LOGS
# =========================
csv_files = sorted(LOG_DIR.glob("reinforce_training_seed_*.csv"))
assert len(csv_files) > 0, "No multi-seed logs found!"

dfs = [pd.read_csv(f) for f in csv_files]

episodes = dfs[0]["episode"].values

# =========================
# STACK METRICS
# =========================
clicks = np.stack([df["clicks"].values for df in dfs])
returns = np.stack([df["return"].values for df in dfs])

clicks_mean = clicks.mean(axis=0)
clicks_std = clicks.std(axis=0)

returns_mean = returns.mean(axis=0)
returns_std = returns.std(axis=0)

# =========================
# PLOT: CLICKS
# =========================
plt.figure(figsize=(7, 4))
plt.plot(episodes, clicks_mean, label="Mean Clicks")
plt.fill_between(
    episodes,
    clicks_mean - clicks_std,
    clicks_mean + clicks_std,
    alpha=0.3,
    label="±1 Std",
)
plt.xlabel("Episode")
plt.ylabel("Clicks")
plt.title("REINFORCE Multi-Seed: Mean ± Std Clicks")
plt.legend()
plt.tight_layout()

clicks_path = OUT_DIR / "reinforce_multiseed_clicks.png"
plt.savefig(clicks_path, dpi=300)
plt.close()

# =========================
# PLOT: RETURNS
# =========================
plt.figure(figsize=(7, 4))
plt.plot(episodes, returns_mean, label="Mean Return")
plt.fill_between(
    episodes,
    returns_mean - returns_std,
    returns_mean + returns_std,
    alpha=0.3,
    label="±1 Std",
)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("REINFORCE Multi-Seed: Mean ± Std Return")
plt.legend()
plt.tight_layout()

returns_path = OUT_DIR / "reinforce_multiseed_return.png"
plt.savefig(returns_path, dpi=300)
plt.close()

print("✅ Multi-seed plots saved to outputs/")
print(f" - {clicks_path.name}")
print(f" - {returns_path.name}")
