"""
evaluate_v6.py
Compares 4 strategies in the 5-agent RTB environment:
1. Fixed Bid      — threshold=0, always bids
2. Uniform pCTR   — fixed threshold=0.15 for all agents (no dataset knowledge)
3. Random         — random threshold each step
4. Actor-Critic v6 — trained RL models

Run from: dynamic_ad_allocation/src2/
"""

import sys
import random
import numpy as np
import torch
from torch.distributions import Categorical
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))

from simulator.multi_environment_v5 import MultiRTBEnvironmentV5
from policy_network_v2 import ActorCriticNetworkV2

# ======================================================
# CONFIG
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENHANCED_DIR = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output"
)
MDL_DIR = ROOT / "models" / "5adv_v6"

DATA_PATHS = {
    0: ENHANCED_DIR / "1458" / "enhanced" / "final_sample_log_with_pctr.txt",
    1: ENHANCED_DIR / "2259" / "enhanced" / "final_sample_log_with_pctr.txt",
    2: ENHANCED_DIR / "3386" / "enhanced" / "final_sample_log_with_pctr.txt",
    3: ENHANCED_DIR / "2997" / "enhanced" / "final_sample_log_with_pctr.txt",
    4: ENHANCED_DIR / "3358" / "enhanced" / "final_sample_log_with_pctr.txt",
}

NUM_AGENTS       = 5
ADV_IDS          = ["1458", "2259", "3386", "2997", "3358"]
BUDGETS          = [20000.0, 12000.0, 20000.0, 25000.0, 18000.0]
MAX_STEPS        = 10000
STATE_DIM        = 14
THRESHOLD_VALUES = list(np.linspace(0.0, 0.3, 51))
NUM_ACTIONS      = len(THRESHOLD_VALUES)
EVAL_EPISODES    = 10
SEEDS            = [0, 1, 2, 3, 4]

TOTAL_BUDGET     = sum(BUDGETS)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env():
    return MultiRTBEnvironmentV5(
        data_paths    = DATA_PATHS,
        budgets       = BUDGETS,
        max_steps     = MAX_STEPS,
        reserve_price = 1.0,
        state_dim     = STATE_DIM,
    )


# ======================================================
# BASELINE 1: FIXED BID (threshold=0, always bids)
# ======================================================
def run_fixed_bid(episodes=EVAL_EPISODES):
    all_clicks = np.zeros((episodes, NUM_AGENTS))
    all_costs  = np.zeros((episodes, NUM_AGENTS))

    for ep in range(episodes):
        set_seed(ep)
        env    = make_env()
        states = env.reset()
        done   = False
        while not done:
            states, _, done = env.step([0.0] * NUM_AGENTS)
            if states is None:
                break
        all_clicks[ep] = env.clicks
        all_costs[ep]  = env.costs

    return all_clicks, all_costs


# ======================================================
# BASELINE 2: UNIFORM FIXED THRESHOLD (0.15 for all)
# No dataset knowledge — same threshold for everyone
# ======================================================
def run_uniform_threshold(threshold=0.15, episodes=EVAL_EPISODES):
    all_clicks = np.zeros((episodes, NUM_AGENTS))
    all_costs  = np.zeros((episodes, NUM_AGENTS))

    for ep in range(episodes):
        set_seed(ep)
        env    = make_env()
        states = env.reset()
        done   = False
        while not done:
            states, _, done = env.step([threshold] * NUM_AGENTS)
            if states is None:
                break
        all_clicks[ep] = env.clicks
        all_costs[ep]  = env.costs

    return all_clicks, all_costs


# ======================================================
# BASELINE 3: RANDOM THRESHOLD
# Truly naive — random threshold each episode
# ======================================================
def run_random_threshold(episodes=EVAL_EPISODES):
    all_clicks = np.zeros((episodes, NUM_AGENTS))
    all_costs  = np.zeros((episodes, NUM_AGENTS))

    for ep in range(episodes):
        set_seed(ep)
        env    = make_env()
        states = env.reset()
        done   = False
        while not done:
            thresholds = [
                random.uniform(0.0, 0.3)
                for _ in range(NUM_AGENTS)
            ]
            states, _, done = env.step(thresholds)
            if states is None:
                break
        all_clicks[ep] = env.clicks
        all_costs[ep]  = env.costs

    return all_clicks, all_costs


# ======================================================
# RL: ACTOR-CRITIC v6
# ======================================================
def run_actor_critic_v6(episodes=EVAL_EPISODES):
    all_clicks = np.zeros((episodes, NUM_AGENTS))
    all_costs  = np.zeros((episodes, NUM_AGENTS))

    for ep in range(episodes):
        seed = SEEDS[ep % len(SEEDS)]
        set_seed(seed)

        agents = []
        for i, adv in enumerate(ADV_IDS):
            model = ActorCriticNetworkV2(
                input_dim   = STATE_DIM,
                num_actions = NUM_ACTIONS,
            ).to(DEVICE)
            model.load_state_dict(
                torch.load(
                    MDL_DIR / f"policy_v6_{adv}_seed_{seed}.pt",
                    map_location=DEVICE, weights_only=True
                )
            )
            model.eval()
            agents.append(model)

        env    = make_env()
        states = env.reset()
        states_t = [
            torch.tensor(s, dtype=torch.float32, device=DEVICE)
            for s in states
        ]
        done = False

        while not done:
            thresholds = []
            for i in range(NUM_AGENTS):
                with torch.no_grad():
                    logits, _ = agents[i](states_t[i])
                    action    = Categorical(
                        logits=logits.squeeze(0)
                    ).sample()
                thresholds.append(THRESHOLD_VALUES[action.item()])

            next_states, _, done = env.step(thresholds)
            if next_states is not None:
                states_t = [
                    torch.tensor(s, dtype=torch.float32, device=DEVICE)
                    for s in next_states
                ]

        all_clicks[ep] = env.clicks
        all_costs[ep]  = env.costs

    return all_clicks, all_costs


# ======================================================
# COMPUTE METRICS
# ======================================================
def metrics(clicks, costs):
    total_clicks = clicks.sum(axis=1).mean()
    total_costs  = costs.sum(axis=1).mean()
    cpc          = total_costs / total_clicks if total_clicks > 0 else 0
    util         = (total_costs / TOTAL_BUDGET) * 100
    clicks_std   = clicks.sum(axis=1).std()
    return total_clicks, total_costs, cpc, util, clicks_std


# ======================================================
# MAIN
# ======================================================
def main():
    print("=" * 65)
    print(" DAAWBC — Multi-Agent RTB Evaluation")
    print(f" {NUM_AGENTS} Advertisers | {EVAL_EPISODES} Episodes")
    print(" Fair Baseline Comparison (no dataset knowledge)")
    print("=" * 65)

    print("\n[1/4] Running Fixed Bid baseline...")
    fb_clicks, fb_costs = run_fixed_bid()

    print("[2/4] Running Uniform Threshold (0.15) baseline...")
    ut_clicks, ut_costs = run_uniform_threshold(threshold=0.15)

    print("[3/4] Running Random Threshold baseline...")
    rd_clicks, rd_costs = run_random_threshold()

    print("[4/4] Running Actor-Critic v6...")
    rl_clicks, rl_costs = run_actor_critic_v6()

    # ======================================================
    # RESULTS TABLE
    # ======================================================
    print("\n" + "=" * 75)
    print(" EVALUATION RESULTS")
    print("=" * 75)
    print(f"{'Method':<22} | {'Clicks':>8} | {'Std':>6} | "
          f"{'Cost':>10} | {'CPC':>8} | {'Util%':>7}")
    print("-" * 75)

    methods = [
        ("Fixed Bid",           fb_clicks, fb_costs),
        ("Uniform Thresh(0.15)", ut_clicks, ut_costs),
        ("Random Threshold",    rd_clicks, rd_costs),
        ("Actor-Critic v6",     rl_clicks, rl_costs),
    ]

    results = []
    for name, clicks, costs in methods:
        c, cost, cpc, util, std = metrics(clicks, costs)
        print(f"{name:<22} | {c:>8.1f} | {std:>6.1f} | "
              f"{cost:>10.1f} | {cpc:>8.2f} | {util:>6.1f}%")
        results.append({
            "method":       name,
            "clicks":       round(c, 1),
            "clicks_std":   round(std, 1),
            "total_cost":   round(cost, 1),
            "cpc":          round(cpc, 2),
            "budget_util":  round(util, 1),
        })

    print("=" * 75)

    # ======================================================
    # RL ADVANTAGE ANALYSIS
    # ======================================================
    fb_c,  _,  fb_cpc,  _, _  = metrics(fb_clicks,  fb_costs)
    ut_c,  _,  ut_cpc,  _, _  = metrics(ut_clicks,  ut_costs)
    rd_c,  _,  rd_cpc,  _, _  = metrics(rd_clicks,  rd_costs)
    rl_c,  _,  rl_cpc,  _, rl_std = metrics(rl_clicks, rl_costs)

    best_baseline_clicks = max(fb_c, ut_c, rd_c)
    best_baseline_cpc    = min(fb_cpc, ut_cpc, rd_cpc)

    print(f"\n{'='*65}")
    print(f" RL ADVANTAGE SUMMARY")
    print(f"{'='*65}")
    print(f"Clicks vs best baseline : "
          f"{rl_c - best_baseline_clicks:+.1f} "
          f"({(rl_c - best_baseline_clicks)/best_baseline_clicks*100:+.1f}%)")
    print(f"CPC vs Fixed Bid        : "
          f"{rl_cpc - fb_cpc:+.2f} "
          f"({(rl_cpc - fb_cpc)/fb_cpc*100:+.1f}%)")
    print(f"CPC vs Random           : "
          f"{rl_cpc - rd_cpc:+.2f} "
          f"({(rl_cpc - rd_cpc)/rd_cpc*100:+.1f}%)")
    print(f"Policy Std (stability)  : {rl_std:.1f} "
          f"← lower = more consistent")
    print(f"Manual tuning required  : None ← RL learns automatically")
    print(f"{'='*65}")

    # ======================================================
    # PER ADVERTISER BREAKDOWN
    # ======================================================
    print(f"\n{'='*65}")
    print(f" PER ADVERTISER — Actor-Critic v6")
    print(f"{'='*65}")
    print(f"{'Adv':<6} {'Budget':>8} {'Clicks':>8} "
          f"{'Cost':>10} {'CPC':>8} {'Util%':>8}")
    print("-" * 65)

    for i, adv in enumerate(ADV_IDS):
        clicks = rl_clicks[:, i].mean()
        costs  = rl_costs[:, i].mean()
        cpc    = costs / clicks if clicks > 0 else 0
        util   = (costs / BUDGETS[i]) * 100
        flag   = "✅" if util >= 80 else "⚠️"
        print(f"{adv:<6} {int(BUDGETS[i]):>8} {clicks:>8.1f} "
              f"{costs:>10.1f} {cpc:>8.2f} {util:>7.1f}% {flag}")

    print("=" * 65)

    # Save
    out_file = ROOT / "outputs" / "evaluation_results_v6.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n✅ Saved: {out_file}")


if __name__ == "__main__":
    main()