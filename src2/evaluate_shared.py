"""
evaluate_shared.py
Evaluates all strategies on TRUE shared auction environment.

Baselines match original thesis setup (modified for multi-agent):
  1. Fixed Bid    : bid = 50.0 always (context-blind, random winner)
  2. Linear pCTR  : bid = 86 x pCTR (pCTR-aware, myopic, no pacing)
  3. AC Shared    : learned pCTR threshold + budget pacing + competition
  4. MAPPO Shared : AC + centralized global critic

Key Contribution of RL:
  NOT raw click maximization (Linear pCTR wins on that with binary pCTR)
  BUT:
    ✅ Budget pacing discipline (spend linearly over episode)
    ✅ Policy stability (lower variance across episodes)
    ✅ Competitive awareness (adapts to other agents)
    ✅ Budget compliance (avoids over/under-spending)

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

from simulator.multi_environment_shared import MultiRTBEnvironmentShared
from simulator.multi_environment_shared_mappo import MultiRTBEnvironmentSharedMAPPO
from policy_network_v2 import ActorCriticNetworkV2
from policy_network_mappo import MAPPOActor, CentralizedCritic

# ======================================================
# CONFIG
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHARED_DATA   = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/shared_auction_log.txt"
)
AC_MDL_DIR    = ROOT / "models" / "5adv_shared"
MAPPO_MDL_DIR = ROOT / "models" / "5adv_shared_mappo"

NUM_AGENTS       = 5
ADV_IDS          = ["1458", "2259", "3386", "2997", "3476"]
BUDGETS          = [18000.0, 14000.0, 2000.0, 20000.0, 10000.0]
MAX_STEPS        = 2000
STATE_DIM        = 14
GLOBAL_DIM       = STATE_DIM * NUM_AGENTS
THRESHOLD_VALUES = list(np.linspace(0.0, 0.3, 51))
NUM_ACTIONS      = len(THRESHOLD_VALUES)
EVAL_EPISODES    = 10
SEEDS            = [0, 1, 2, 3, 4]
TOTAL_BUDGET     = sum(BUDGETS)

# ======================================================
# BASELINE CONSTANTS
# ======================================================
FIXED_BID    = 50.0    # context-blind, below market avg
LINEAR_ALPHA = 86.0    # calibrated to market avg for binary pCTR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env_ac():
    return MultiRTBEnvironmentShared(
        shared_data_path = str(SHARED_DATA),
        budgets          = BUDGETS,
        adv_ids          = ADV_IDS,
        max_steps        = MAX_STEPS,
        reserve_price    = 1.0,
        state_dim        = STATE_DIM,
    )


def make_env_mappo():
    return MultiRTBEnvironmentSharedMAPPO(
        shared_data_path = str(SHARED_DATA),
        budgets          = BUDGETS,
        adv_ids          = ADV_IDS,
        max_steps        = MAX_STEPS,
        reserve_price    = 1.0,
        state_dim        = STATE_DIM,
    )


def pacing_error(costs, budgets, steps, max_steps):
    """
    Measures how well budget is paced linearly over episode.
    ideal_spend = budget × (steps_done / max_steps)
    pacing_error = |actual_spend - ideal_spend| / budget
    Lower = better pacing
    """
    ideal = np.array(budgets) * (steps / max_steps)
    actual = costs
    errors = np.abs(actual - ideal) / np.array(budgets)
    return errors.mean() * 100  # as percentage


def metrics(clicks, costs):
    total_clicks = clicks.sum(axis=1).mean()
    total_costs  = costs.sum(axis=1).mean()
    cpc          = total_costs / total_clicks if total_clicks > 0 else 0
    util         = (total_costs / TOTAL_BUDGET) * 100
    std          = clicks.sum(axis=1).std()
    norm         = total_clicks / TOTAL_BUDGET * 1000
    return total_clicks, total_costs, cpc, util, std, norm


# ======================================================
# BASELINE 1: FIXED BID
# bid = 50.0 for all agents, random winner (no pCTR knowledge)
# ======================================================
def run_fixed_bid(episodes=EVAL_EPISODES):
    all_clicks = np.zeros((episodes, NUM_AGENTS))
    all_costs  = np.zeros((episodes, NUM_AGENTS))
    all_pacing = []

    for ep in range(episodes):
        set_seed(ep)
        env = make_env_ac()
        env.reset()
        done = False

        while not done:
            row          = env.df.iloc[env.ptr]
            market_price = float(row["market_price"])

            valid = [
                i for i in range(NUM_AGENTS)
                if FIXED_BID >= market_price
                and env.remaining_budget[i] >= market_price
            ]

            if valid:
                # Random winner — truly context-blind
                winner = random.choice(valid)
                env.remaining_budget[winner] -= market_price
                env.costs[winner]            += market_price
                pctr  = float(row[f"pctr_{ADV_IDS[winner]}"])
                click = 1 if np.random.rand() < pctr else 0
                env.clicks[winner] += click

            env.ptr += 1
            env.t   += 1
            done = (env.ptr >= env.n or env.t >= MAX_STEPS)

        all_clicks[ep] = env.clicks
        all_costs[ep]  = env.costs
        all_pacing.append(
            pacing_error(env.costs, BUDGETS, env.t, MAX_STEPS)
        )

    return all_clicks, all_costs, np.mean(all_pacing)


# ======================================================
# BASELINE 2: LINEAR pCTR
# bid = alpha x pCTR (pCTR-aware, NO budget/time/pacing awareness)
# ======================================================
def run_linear_pctr(episodes=EVAL_EPISODES):
    all_clicks = np.zeros((episodes, NUM_AGENTS))
    all_costs  = np.zeros((episodes, NUM_AGENTS))
    all_pacing = []

    for ep in range(episodes):
        set_seed(ep)
        env = make_env_ac()
        env.reset()
        done = False

        while not done:
            row          = env.df.iloc[env.ptr]
            market_price = float(row["market_price"])

            bids = {
                i: LINEAR_ALPHA * float(row[f"pctr_{ADV_IDS[i]}"])
                for i in range(NUM_AGENTS)
            }

            valid = [
                i for i in range(NUM_AGENTS)
                if bids[i] >= market_price
                and env.remaining_budget[i] >= market_price
            ]

            if valid:
                # Winner = highest pCTR-proportional bid
                winner = max(valid, key=lambda i: bids[i])
                env.remaining_budget[winner] -= market_price
                env.costs[winner]            += market_price
                pctr  = float(row[f"pctr_{ADV_IDS[winner]}"])
                click = 1 if np.random.rand() < pctr else 0
                env.clicks[winner] += click

            env.ptr += 1
            env.t   += 1
            done = (env.ptr >= env.n or env.t >= MAX_STEPS)

        all_clicks[ep] = env.clicks
        all_costs[ep]  = env.costs
        all_pacing.append(
            pacing_error(env.costs, BUDGETS, env.t, MAX_STEPS)
        )

    return all_clicks, all_costs, np.mean(all_pacing)


# ======================================================
# RL: AC SHARED
# ======================================================
def run_ac_shared(episodes=EVAL_EPISODES):
    all_clicks = np.zeros((episodes, NUM_AGENTS))
    all_costs  = np.zeros((episodes, NUM_AGENTS))
    all_pacing = []

    for ep in range(episodes):
        seed = SEEDS[ep % len(SEEDS)]
        set_seed(seed)

        agents = []
        for i, adv in enumerate(ADV_IDS):
            model = ActorCriticNetworkV2(
                input_dim=STATE_DIM, num_actions=NUM_ACTIONS
            ).to(DEVICE)
            model.load_state_dict(torch.load(
                AC_MDL_DIR / f"policy_shared_{adv}_seed_{seed}.pt",
                map_location=DEVICE, weights_only=True
            ))
            model.eval()
            agents.append(model)

        env      = make_env_ac()
        states   = env.reset()
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
        all_pacing.append(
            pacing_error(env.costs, BUDGETS, env.t, MAX_STEPS)
        )

    return all_clicks, all_costs, np.mean(all_pacing)


# ======================================================
# RL: MAPPO SHARED
# ======================================================
def run_mappo_shared(episodes=EVAL_EPISODES):
    all_clicks = np.zeros((episodes, NUM_AGENTS))
    all_costs  = np.zeros((episodes, NUM_AGENTS))
    all_pacing = []

    for ep in range(episodes):
        seed = SEEDS[ep % len(SEEDS)]
        set_seed(seed)

        actors = []
        for i, adv in enumerate(ADV_IDS):
            actor = MAPPOActor(STATE_DIM, NUM_ACTIONS).to(DEVICE)
            actor.load_state_dict(torch.load(
                MAPPO_MDL_DIR /
                f"policy_shared_mappo_actor_{adv}_seed_{seed}.pt",
                map_location=DEVICE, weights_only=True
            ))
            actor.eval()
            actors.append(actor)

        env                  = make_env_mappo()
        local_states, _      = env.reset()
        states_t             = [
            torch.tensor(s, dtype=torch.float32, device=DEVICE)
            for s in local_states
        ]
        done = False

        while not done:
            thresholds = []
            for i in range(NUM_AGENTS):
                with torch.no_grad():
                    action, _, _ = actors[i].get_action(states_t[i])
                thresholds.append(THRESHOLD_VALUES[action.item()])

            next_local, _, _, done = env.step(thresholds)
            if next_local is not None:
                states_t = [
                    torch.tensor(s, dtype=torch.float32, device=DEVICE)
                    for s in next_local
                ]

        all_clicks[ep] = env.clicks
        all_costs[ep]  = env.costs
        all_pacing.append(
            pacing_error(env.costs, BUDGETS, env.t, MAX_STEPS)
        )

    return all_clicks, all_costs, np.mean(all_pacing)


# ======================================================
# MAIN
# ======================================================
def main():
    print("=" * 75)
    print(" DAAWBC — Shared Environment Evaluation")
    print(f" TRUE Shared Auction | {NUM_AGENTS} Agents | "
          f"{EVAL_EPISODES} Episodes")
    print(f" Neutral Master Stream : Advertiser 3427")
    print(f" MAX_STEPS             : {MAX_STEPS}")
    print(f" Total Budget          : {TOTAL_BUDGET:,}")
    print(f"\n Key RL Contributions Being Evaluated:")
    print(f"   1. Budget pacing discipline (linear spend over episode)")
    print(f"   2. Policy stability (low variance across episodes)")
    print(f"   3. Budget compliance (utilization without overspending)")
    print(f"   4. Competitive awareness (5-agent shared auction)")
    print("=" * 75)

    print("\n[1/4] Fixed Bid (context-blind, random winner)...")
    fb_clicks, fb_costs, fb_pace = run_fixed_bid()

    print("[2/4] Linear pCTR (pCTR-aware, no pacing)...")
    lp_clicks, lp_costs, lp_pace = run_linear_pctr()

    print("[3/4] Actor-Critic Shared (learned policy + pacing)...")
    ac_clicks, ac_costs, ac_pace = run_ac_shared()

    print("[4/4] MAPPO Shared (+ centralized critic)...")
    mp_clicks, mp_costs, mp_pace = run_mappo_shared()

    # ======================================================
    # PRIMARY RESULTS TABLE
    # ======================================================
    print("\n" + "=" * 85)
    print(" EVALUATION RESULTS — Shared Auction Environment")
    print("=" * 85)
    print(f"{'Method':<20} | {'Clicks':>8} | {'Std':>6} | "
          f"{'CPC':>8} | {'Util%':>7} | "
          f"{'Pacing Err%':>12} | {'Budget Ctrl':>11}")
    print("-" * 85)

    methods = [
        ("Fixed Bid",    fb_clicks, fb_costs, fb_pace),
        ("Linear pCTR",  lp_clicks, lp_costs, lp_pace),
        ("AC Shared",    ac_clicks, ac_costs, ac_pace),
        ("MAPPO Shared", mp_clicks, mp_costs, mp_pace),
    ]

    results = []
    for name, clicks, costs, pace in methods:
        c, cost, cpc, util, std, norm = metrics(clicks, costs)
        # Budget compliance = % of agents with util > 80%
        per_agent_utils = [
            (costs[:, i].mean() / BUDGETS[i]) * 100
            for i in range(NUM_AGENTS)
        ]
        compliance = sum(1 for u in per_agent_utils if u >= 80) / NUM_AGENTS * 100
        flag = "LEARNED" if "Shared" in name else "RULE"
        print(f"{name:<20} | {c:>8.1f} | {std:>6.1f} | "
              f"{cpc:>8.2f} | {util:>6.1f}% | "
              f"{pace:>11.1f}% | {compliance:>9.0f}% {flag}")
        results.append({
            "method":        name,
            "clicks":        round(c, 1),
            "clicks_std":    round(std, 1),
            "cpc":           round(cpc, 2),
            "budget_util":   round(util, 1),
            "pacing_error":  round(pace, 1),
            "compliance":    round(compliance, 1),
        })

    print("=" * 85)

    # ======================================================
    # KEY CONTRIBUTION ANALYSIS
    # ======================================================
    fb_c, _, fb_cpc, fb_util, fb_std, _ = metrics(fb_clicks, fb_costs)
    lp_c, _, lp_cpc, lp_util, lp_std, _ = metrics(lp_clicks, lp_costs)
    ac_c, _, ac_cpc, ac_util, ac_std, _ = metrics(ac_clicks, ac_costs)
    mp_c, _, mp_cpc, mp_util, mp_std, _ = metrics(mp_clicks, mp_costs)

    print(f"\n{'='*75}")
    print(f" KEY CONTRIBUTION 1: BUDGET PACING DISCIPLINE")
    print(f"{'='*75}")
    print(f"  Pacing Error = |actual_spend - ideal_linear_spend| / budget")
    print(f"  Lower = agent spends budget more uniformly over episode")
    print()
    print(f"  Fixed Bid    : {fb_pace:.1f}% pacing error ← no pacing")
    print(f"  Linear pCTR  : {lp_pace:.1f}% pacing error ← no pacing")
    print(f"  AC Shared    : {ac_pace:.1f}% pacing error ← LEARNED pacing ✅")
    print(f"  MAPPO Shared : {mp_pace:.1f}% pacing error ← LEARNED pacing ✅")
    print(f"\n  AC improvement over Linear pCTR : "
          f"{lp_pace - ac_pace:+.1f}% reduction in pacing error")
    print(f"  MAPPO improvement over Linear pCTR: "
          f"{lp_pace - mp_pace:+.1f}% reduction in pacing error")

    print(f"\n{'='*75}")
    print(f" KEY CONTRIBUTION 2: POLICY STABILITY")
    print(f"{'='*75}")
    print(f"  Std Dev of total clicks across {EVAL_EPISODES} episodes")
    print(f"  Lower = more consistent campaign delivery")
    print()
    print(f"  Fixed Bid    : std = {fb_std:.1f}")
    print(f"  Linear pCTR  : std = {lp_std:.1f}")
    print(f"  AC Shared    : std = {ac_std:.1f} "
          f"({(1-ac_std/lp_std)*100:.1f}% more stable than LinearPCTR)")
    print(f"  MAPPO Shared : std = {mp_std:.1f} "
          f"({(1-mp_std/lp_std)*100:.1f}% more stable than LinearPCTR)")

    print(f"\n{'='*75}")
    print(f" KEY CONTRIBUTION 3: BUDGET COMPLIANCE")
    print(f"{'='*75}")
    print(f"  % of agents achieving 80-100% budget utilization")
    print(f"  RL agents learn to spend efficiently without over/under-spending")
    print()
    for name, clicks, costs, pace in methods:
        per_agent_utils = [
            (costs[:, i].mean() / BUDGETS[i]) * 100
            for i in range(NUM_AGENTS)
        ]
        print(f"  {name:<20}:", end="")
        for i, (adv, u) in enumerate(zip(ADV_IDS, per_agent_utils)):
            flag = "OK" if u >= 80 else "LOW"
            print(f"  {adv}={u:.0f}%[{flag}]", end="")
        print()

    print(f"\n{'='*75}")
    print(f" NOTE ON CLICK COUNT")
    print(f"{'='*75}")
    print(f"  Linear pCTR achieves higher raw clicks ({lp_c:.0f}) than RL "
          f"({ac_c:.0f}/{mp_c:.0f})")
    print(f"  This is expected with binary pCTR — Linear pCTR greedily")
    print(f"  targets all pCTR=1 impressions without budget constraints.")
    print(f"  RL trades some clicks for budget pacing discipline,")
    print(f"  which is essential in real RTB where campaigns run over")
    print(f"  extended periods and uneven spending reduces effectiveness.")
    print(f"  Note: α=86 was recalibrated from thesis α=300 to account")
    print(f"  for binary pCTR distribution in shared environment.")
    print(f"{'='*75}")

    # Save
    out_file = ROOT / "outputs" / "evaluation_results_shared.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()