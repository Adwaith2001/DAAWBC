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

SEEDS = [0, 1, 2, 3, 4]

dfs = []
for seed in SEEDS:
    df = pd.read_csv(LOG_DIR / f"actor_critic_budget10k_seed_{seed}.csv")
    dfs.append(df)

episodes = dfs[0]["episode"].values

def stack_metric(metric):
    return np.stack([df[metric].values for df in dfs])

# =========================
# STACK METRICS
# =========================
clicks = stack_metric("clicks")
returns = stack_metric("return")
lambdas = stack_metric("lambda")

clicks_mean, clicks_std = clicks.mean(axis=0), clicks.std(axis=0)
returns_mean, returns_std = returns.mean(axis=0), returns.std(axis=0)
lambda_mean, lambda_std = lambdas.mean(axis=0), lambdas.std(axis=0)

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
)
plt.xlabel("Episode")
plt.ylabel("Clicks")
plt.title("Actor–Critic (Budget 10k): Clicks")
plt.tight_layout()
plt.savefig(OUT_DIR / "ac_budget10k_clicks.png", dpi=300)
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
)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Actor–Critic (Budget 10k): Return")
plt.tight_layout()
plt.savefig(OUT_DIR / "ac_budget10k_return.png", dpi=300)
plt.close()

# =========================
# PLOT: LAMBDA
# =========================
plt.figure(figsize=(7, 4))
plt.plot(episodes, lambda_mean, label="Mean Lambda")
plt.fill_between(
    episodes,
    lambda_mean - lambda_std,
    lambda_mean + lambda_std,
    alpha=0.3,
)
plt.xlabel("Episode")
plt.ylabel("Lambda")
plt.title("Actor–Critic (Budget 10k): Lambda Convergence")
plt.tight_layout()
plt.savefig(OUT_DIR / "ac_budget10k_lambda.png", dpi=300)
plt.close()

print("✅ Plots saved to outputs/")
