import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "outputs" / "analysis" / "actor_critic_3adv_aggregated.csv"
PLOT_DIR = ROOT / "outputs" / "analysis" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)
episodes = df["episode"]

# Detect number of agents
agent_ids = sorted(set(c.split("_")[1][1] for c in df.columns if c.startswith("clicks_a")))

# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_metric(prefix, ylabel, filename):
    plt.figure(figsize=(10, 6))

    for a in agent_ids:
        mean = df[f"{prefix}_a{a}_mean"]
        std  = df[f"{prefix}_a{a}_std"]

        plt.plot(episodes, mean, label=f"Advertiser {a}")
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"Actor–Critic (3 Advertisers): {ylabel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out = PLOT_DIR / filename
    plt.savefig(out)
    plt.close()

    print(f"✅ Saved plot: {out}")

# ============================================================
# GENERATE PLOTS
# ============================================================

plot_metric("clicks", "Clicks", "clicks_vs_episode.png")
plot_metric("cost", "Cost", "cost_vs_episode.png")

print("🎉 All plots generated successfully")
