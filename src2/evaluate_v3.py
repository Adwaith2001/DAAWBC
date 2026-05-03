"""
evaluate_v3.py
Evaluation: RL agents vs Fixed Bid vs Linear pCTR
All strategies compete in the SAME 5-agent environment

Run from: src2/
"""

import sys
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))

from simulator.multi_environment_v2 import MultiRTBEnvironmentV2
from policy_network_v2 import ActorCriticNetworkV2

# ======================================================
# CONFIG
# ======================================================
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
ADV_IDS  = ["1458", "2259", "2821", "2997", "3358"]
SEEDS    = [0, 1, 2, 3, 4]
EPISODES = 50   # evaluation episodes per seed

BUDGETS   = [20000.0] * 5
MAX_STEPS = 10000

THRESHOLD_VALUES = list(np.linspace(0.0, 0.9, 51))
NUM_ACTIONS      = len(THRESHOLD_VALUES)

DATA_DIR = ROOT / "data" / "ipinyou"
MDL_DIR  = ROOT / "models" / "5adv_v3"
OUT_DIR  = ROOT / "outputs" / "evaluation_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATHS = {
    0: DATA_DIR / "1458" / "final_sample_log_with_pctr.txt",
    1: DATA_DIR / "2259" / "final_sample_log_with_pctr.txt",
    2: DATA_DIR / "2821" / "final_sample_log_with_pctr.txt",
    3: DATA_DIR / "2997" / "final_sample_log_with_pctr.txt",
    4: DATA_DIR / "3358" / "final_sample_log_with_pctr.txt",
}

# ======================================================
# BASELINE STRATEGIES
# ======================================================
FIXED_BID_THRESHOLD  = 0.0   # bid on everything (no filter)
LINEAR_PCTR_ALPHA    = 0.3   # threshold = alpha × mean_pctr

def fixed_bid_threshold():
    """Always bid on everything — threshold = 0"""
    return 0.0

def linear_pctr_threshold(pctr):
    """Threshold proportional to impression pCTR"""
    return float(np.clip(LINEAR_PCTR_ALPHA * pctr, 0.0, 0.9))


# ======================================================
# LOAD RL MODELS
# ======================================================
def load_rl_agents(seed):
    agents = []
    for adv in ADV_IDS:
        model = ActorCriticNetworkV2(input_dim=10, num_actions=NUM_ACTIONS).to(DEVICE)
        path  = MDL_DIR / f"policy_v3_{adv}_seed_{seed}.pt"
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        agents.append(model)
    return agents


def get_rl_thresholds(agents, states):
    thresholds = []
    for i, agent in enumerate(agents):
        s = torch.tensor(states[i], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits, _ = agent(s)
            action    = Categorical(logits=logits.squeeze(0)).sample()
        thresholds.append(THRESHOLD_VALUES[action.item()])
    return thresholds


# ======================================================
# RUN ONE EVALUATION EPISODE
# ======================================================
def run_episode(env, mode, rl_agents=None):
    """
    mode: 'rl', 'fixed_bid', 'linear_pctr'
    Returns clicks, costs per agent
    """
    states = env.reset()
    done   = False

    while not done:
        if mode == "rl":
            thresholds = get_rl_thresholds(rl_agents, states)

        elif mode == "fixed_bid":
            thresholds = [fixed_bid_threshold()] * 5

        elif mode == "linear_pctr":
            # Use pCTR from current state (index 2)
            thresholds = [linear_pctr_threshold(states[i][2]) for i in range(5)]

        next_states, rewards, done = env.step(thresholds)
        if not done:
            states = next_states

    return env.clicks.copy(), env.costs.copy()


# ======================================================
# MAIN EVALUATION
# ======================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

results = {
    "rl":          {adv: {"clicks": [], "costs": []} for adv in ADV_IDS},
    "fixed_bid":   {adv: {"clicks": [], "costs": []} for adv in ADV_IDS},
    "linear_pctr": {adv: {"clicks": [], "costs": []} for adv in ADV_IDS},
}

for seed in SEEDS:
    print(f"\n=== Evaluating Seed {seed} ===")
    set_seed(seed)

    rl_agents = load_rl_agents(seed)

    env = MultiRTBEnvironmentV2(
        data_paths    = DATA_PATHS,
        budgets       = BUDGETS,
        max_steps     = MAX_STEPS,
        reserve_price = 1.0,
    )

    for ep in range(1, EPISODES + 1):

        # RL evaluation
        clicks_rl, costs_rl = run_episode(env, "rl", rl_agents)

        # Fixed bid evaluation
        clicks_fb, costs_fb = run_episode(env, "fixed_bid")

        # Linear pCTR evaluation
        clicks_lp, costs_lp = run_episode(env, "linear_pctr")

        for i, adv in enumerate(ADV_IDS):
            results["rl"][adv]["clicks"].append(int(clicks_rl[i]))
            results["rl"][adv]["costs"].append(float(costs_rl[i]))

            results["fixed_bid"][adv]["clicks"].append(int(clicks_fb[i]))
            results["fixed_bid"][adv]["costs"].append(float(costs_fb[i]))

            results["linear_pctr"][adv]["clicks"].append(int(clicks_lp[i]))
            results["linear_pctr"][adv]["costs"].append(float(costs_lp[i]))

        if ep % 10 == 0:
            rl_total  = sum(int(clicks_rl[i]) for i in range(5))
            fb_total  = sum(int(clicks_fb[i]) for i in range(5))
            lp_total  = sum(int(clicks_lp[i]) for i in range(5))
            print(
                f"Seed {seed} | Ep {ep:03d} | "
                f"RL={rl_total} | FixedBid={fb_total} | LinearPCTR={lp_total}"
            )


# ======================================================
# AGGREGATE RESULTS
# ======================================================
def mean_std(method, adv, metric):
    vals = results[method][adv][metric]
    return np.mean(vals), np.std(vals)

METHODS       = ["rl", "fixed_bid", "linear_pctr"]
METHOD_LABELS = {"rl": "RL (v3)", "fixed_bid": "Fixed Bid", "linear_pctr": "Linear pCTR"}
METHOD_COLORS = {"rl": "#2196F3", "fixed_bid": "#F44336", "linear_pctr": "#4CAF50"}

# ======================================================
# PLOT 1: Clicks comparison per advertiser
# ======================================================
fig, axes = plt.subplots(1, 5, figsize=(18, 5))

for idx, adv in enumerate(ADV_IDS):
    ax = axes[idx]
    x  = np.arange(len(METHODS))

    means = [mean_std(m, adv, "clicks")[0] for m in METHODS]
    stds  = [mean_std(m, adv, "clicks")[1] for m in METHODS]
    cols  = [METHOD_COLORS[m] for m in METHODS]

    bars = ax.bar(x, means, yerr=stds, color=cols, capsize=5, alpha=0.85)
    ax.set_title(f"Advertiser {adv}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=15, fontsize=8)
    ax.set_ylabel("Clicks" if idx == 0 else "")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Evaluation — Clicks per Advertiser: RL vs Fixed Bid vs Linear pCTR",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "eval_clicks_per_advertiser.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Evaluation Plot 1: clicks_per_advertiser")


# ======================================================
# PLOT 2: Total clicks comparison
# ======================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
total_means = []
total_stds  = []
for method in METHODS:
    total_per_ep = np.array([
        sum(results[method][adv]["clicks"][ep] for adv in ADV_IDS)
        for ep in range(EPISODES * len(SEEDS))
    ])
    total_means.append(total_per_ep.mean())
    total_stds.append(total_per_ep.std())

x    = np.arange(len(METHODS))
cols = [METHOD_COLORS[m] for m in METHODS]
ax.bar(x, total_means, yerr=total_stds,
       color=cols, capsize=5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=10)
ax.set_title("Total Clicks (All 5 Advertisers)", fontweight="bold")
ax.set_ylabel("Total Clicks")
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for i, (m, s) in enumerate(zip(total_means, total_stds)):
    ax.text(i, m + s + 2, f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")


# PLOT 2b: Cost per click
ax = axes[1]
cpc_means = []
cpc_stds  = []
for method in METHODS:
    total_clicks_all = sum(
        np.array(results[method][adv]["clicks"])
        for adv in ADV_IDS
    )
    total_costs_all = sum(
        np.array(results[method][adv]["costs"])
        for adv in ADV_IDS
    )
    cpc = total_costs_all / np.maximum(total_clicks_all, 1)
    cpc_means.append(cpc.mean())
    cpc_stds.append(cpc.std())

ax.bar(x, cpc_means, yerr=cpc_stds,
       color=cols, capsize=5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=10)
ax.set_title("Cost Per Click (lower = better)", fontweight="bold")
ax.set_ylabel("Cost Per Click")
ax.grid(axis="y", alpha=0.3)

for i, (m, s) in enumerate(zip(cpc_means, cpc_stds)):
    ax.text(i, m + s + 1, f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")

plt.suptitle("Evaluation Summary — RL vs Fixed Bid vs Linear pCTR",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "eval_summary.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Evaluation Plot 2: summary")


# ======================================================
# SAVE RESULTS CSV
# ======================================================
rows = []
for method in METHODS:
    for adv in ADV_IDS:
        m_c, s_c = mean_std(method, adv, "clicks")
        m_co, s_co = mean_std(method, adv, "costs")
        cpc = m_co / max(m_c, 1)
        rows.append({
            "method":        METHOD_LABELS[method],
            "advertiser":    adv,
            "mean_clicks":   round(m_c, 2),
            "std_clicks":    round(s_c, 2),
            "mean_cost":     round(m_co, 2),
            "cost_per_click": round(cpc, 2),
        })

df_results = pd.DataFrame(rows)
df_results.to_csv(OUT_DIR / "evaluation_results.csv", index=False)
print("✅ Evaluation results saved to evaluation_v3/evaluation_results.csv")


# ======================================================
# PRINT SUMMARY TABLE
# ======================================================
print("\n" + "="*75)
print(" EVALUATION SUMMARY")
print("="*75)
print(f"{'Method':<14} {'Advertiser':<12} {'Clicks':>12} {'Cost':>12} {'CPC':>10}")
print("-"*75)
for method in METHODS:
    for adv in ADV_IDS:
        m_c, s_c   = mean_std(method, adv, "clicks")
        m_co, s_co = mean_std(method, adv, "costs")
        cpc        = m_co / max(m_c, 1)
        print(
            f"{METHOD_LABELS[method]:<14} {adv:<12} "
            f"{m_c:>8.1f}±{s_c:<4.1f} "
            f"{m_co:>10.0f} "
            f"{cpc:>10.1f}"
        )
    print("-"*75)
print("="*75)