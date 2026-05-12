"""
demo_bidding.py
Step-by-step bidding demonstration using trained v6 models.
Shows exactly how each agent decides to bid or skip each impression.

Run from: dynamic_ad_allocation/src2/
"""

import sys
import random
import numpy as np
import torch
from torch.distributions import Categorical
from pathlib import Path

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

DEMO_STEPS  = 20   # show first 20 impressions
DEMO_SEED   = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(DEMO_SEED)

    print("=" * 70)
    print(" DAAWBC — Live Bidding Demonstration")
    print(f" 5 Advertisers | Trained Actor-Critic v6 | Seed {DEMO_SEED}")
    print("=" * 70)
    print(f"\nAdvertisers: {ADV_IDS}")
    print(f"Budgets    : {[int(b) for b in BUDGETS]}")
    print(f"\nConcept: Each agent learns a pCTR THRESHOLD")
    print(f"         If impression pCTR >= threshold → BID")
    print(f"         If impression pCTR <  threshold → SKIP")
    print(f"         (saves budget for higher quality impressions)")

    # ======================================================
    # LOAD TRAINED MODELS
    # ======================================================
    print(f"\nLoading trained models (seed {DEMO_SEED})...")
    agents = []
    for i, adv in enumerate(ADV_IDS):
        model = ActorCriticNetworkV2(
            input_dim   = STATE_DIM,
            num_actions = NUM_ACTIONS,
        ).to(DEVICE)
        model_path = MDL_DIR / f"policy_v6_{adv}_seed_{DEMO_SEED}.pt"
        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE,
                       weights_only=True)
        )
        model.eval()
        agents.append(model)
        print(f"  ✅ Loaded: policy_v6_{adv}_seed_{DEMO_SEED}.pt")

    # ======================================================
    # INITIALIZE ENVIRONMENT
    # ======================================================
    env = MultiRTBEnvironmentV5(
        data_paths    = DATA_PATHS,
        budgets       = BUDGETS,
        max_steps     = MAX_STEPS,
        reserve_price = 1.0,
        state_dim     = STATE_DIM,
    )

    states = env.reset()
    states_t = [
        torch.tensor(s, dtype=torch.float32, device=DEVICE)
        for s in states
    ]

    total_clicks = np.zeros(NUM_AGENTS, dtype=int)
    total_costs  = np.zeros(NUM_AGENTS)

    print(f"\n{'='*70}")
    print(f" STEP-BY-STEP BIDDING (First {DEMO_STEPS} Impressions)")
    print(f"{'='*70}")

    done = False
    step = 0

    while not done and step < DEMO_STEPS:

        step += 1

        # Get current impression info
        pctr_values   = []
        market_prices = []
        thresholds    = []
        decisions     = []

        for i in range(NUM_AGENTS):
            df  = env.dfs[i]
            idx = env.indices[i]

            if idx < len(df):
                row   = df.iloc[idx]
                pctr  = float(row["pctr"])
                price = float(row["market_price"])
            else:
                pctr  = 0.0
                price = 0.0

            pctr_values.append(pctr)
            market_prices.append(price)

            # Agent decides threshold
            with torch.no_grad():
                logits, _ = agents[i](states_t[i])
                action    = Categorical(
                    logits=logits.squeeze(0)
                ).sample()

            threshold = THRESHOLD_VALUES[action.item()]
            thresholds.append(threshold)
            decisions.append("BID ✅" if pctr >= threshold else "SKIP ❌")

        # Print impression header
        print(f"\n{'─'*70}")
        print(f" Impression #{step:03d}")
        print(f"{'─'*70}")

        # Find winner (highest pCTR among bidders)
        bidders = [
            i for i in range(NUM_AGENTS)
            if pctr_values[i] >= thresholds[i]
            and env.remaining_budget[i] >= market_prices[i]
        ]
        winner = None
        if bidders:
            winner = max(bidders, key=lambda i: pctr_values[i])

        # Print each agent's decision
        for i, adv in enumerate(ADV_IDS):
            budget_left = env.remaining_budget[i]
            budget_pct  = (budget_left / BUDGETS[i]) * 100

            winner_str = " ← WINNER 🏆" if i == winner else ""

            print(
                f"  {adv:>4} | "
                f"pCTR={pctr_values[i]:.4f} | "
                f"Threshold={thresholds[i]:.4f} | "
                f"{decisions[i]:<8} | "
                f"Budget={budget_left:>8.0f} ({budget_pct:.0f}%)"
                f"{winner_str}"
            )

        # Execute step
        next_states, rewards, done = env.step(thresholds)

        # Show outcome
        if winner is not None:
            click = env.clicks[winner] - total_clicks[winner]
            cost  = env.costs[winner]  - total_costs[winner]
            print(f"\n  → {ADV_IDS[winner]} won | "
                  f"Cost={cost:.2f} | "
                  f"Click={'YES 🎯' if click > 0 else 'NO'}")
        else:
            print(f"\n  → No winner (all agents skipped)")

        total_clicks = env.clicks.copy()
        total_costs  = env.costs.copy()

        if next_states is not None:
            states_t = [
                torch.tensor(s, dtype=torch.float32, device=DEVICE)
                for s in next_states
            ]

    # ======================================================
    # CONTINUE TO FULL EPISODE
    # ======================================================
    print(f"\n{'='*70}")
    print(f" Running remaining episode silently...")
    print(f"{'='*70}")

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

    # ======================================================
    # FINAL SUMMARY
    # ======================================================
    print(f"\n{'='*70}")
    print(f" EPISODE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Adv':<6} {'Budget':>8} {'Clicks':>8} "
          f"{'Cost':>10} {'CPC':>8} {'Util%':>8}")
    print(f"{'─'*70}")

    total_clicks_all = 0
    total_cost_all   = 0

    for i, adv in enumerate(ADV_IDS):
        clicks = int(env.clicks[i])
        cost   = float(env.costs[i])
        cpc    = cost / clicks if clicks > 0 else 0
        util   = (cost / BUDGETS[i]) * 100
        flag   = "✅" if util >= 80 else "⚠️"

        print(f"{adv:<6} {int(BUDGETS[i]):>8} {clicks:>8} "
              f"{cost:>10.1f} {cpc:>8.2f} {util:>7.1f}% {flag}")

        total_clicks_all += clicks
        total_cost_all   += cost

    print(f"{'─'*70}")
    total_budget = sum(BUDGETS)
    total_util   = (total_cost_all / total_budget) * 100
    total_cpc    = total_cost_all / total_clicks_all \
                   if total_clicks_all > 0 else 0

    print(f"{'TOTAL':<6} {int(total_budget):>8} "
          f"{total_clicks_all:>8} "
          f"{total_cost_all:>10.1f} "
          f"{total_cpc:>8.2f} "
          f"{total_util:>7.1f}%")
    print(f"{'='*70}")
    print(f"\n🎉 Demo complete!")


if __name__ == "__main__":
    main()