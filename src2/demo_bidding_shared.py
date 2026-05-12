"""
demo_bidding_shared.py
Step-by-step shared auction demo + complete bidding log CSV.
Logs EVERY step correctly including winner, cost, click, pacing.
Run from: dynamic_ad_allocation/src2/
"""

import sys
import random
import numpy as np
import torch
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))

from simulator.multi_environment_shared_mappo import MultiRTBEnvironmentSharedMAPPO
from policy_network_mappo import MAPPOActor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHARED_DATA   = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/shared_auction_log.txt"
)
MAPPO_MDL_DIR = ROOT / "models" / "5adv_shared_mappo"
LOG_DIR       = ROOT / "outputs"
LOG_DIR.mkdir(exist_ok=True)

NUM_AGENTS       = 5
ADV_IDS          = ["1458", "2259", "3386", "2997", "3476"]
BUDGETS          = [18000.0, 14000.0, 2000.0, 20000.0, 10000.0]
MAX_STEPS        = 2000
STATE_DIM        = 14
THRESHOLD_VALUES = list(np.linspace(0.0, 0.3, 51))
NUM_ACTIONS      = len(THRESHOLD_VALUES)
DEMO_STEPS       = 20
DEMO_SEED        = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(DEMO_SEED)

    print("=" * 70)
    print(" DAAWBC — Shared Auction Bidding Demonstration")
    print(f" TRUE Shared Auction | MAPPO Shared | Seed {DEMO_SEED}")
    print("=" * 70)
    print(f"\nAdvertisers  : {ADV_IDS}")
    print(f"Budgets      : {[int(b) for b in BUDGETS]}")
    print(f"Total Budget : {sum(BUDGETS):,}")
    print(f"Max Steps    : {MAX_STEPS}")
    print(f"Master stream: Advertiser 3427 (neutral)")
    print(f"\nConcept:")
    print(f"  ALL agents see THE SAME impression")
    print(f"  Each decides threshold based on own pCTR + budget state")
    print(f"  Winner = highest pCTR above threshold")

    # ======================================================
    # LOAD MODELS
    # ======================================================
    print(f"\nLoading MAPPO Shared models (seed {DEMO_SEED})...")
    actors = []
    for i, adv in enumerate(ADV_IDS):
        actor = MAPPOActor(STATE_DIM, NUM_ACTIONS).to(DEVICE)
        actor.load_state_dict(torch.load(
            MAPPO_MDL_DIR /
            f"policy_shared_mappo_actor_{adv}_seed_{DEMO_SEED}.pt",
            map_location=DEVICE, weights_only=True
        ))
        actor.eval()
        actors.append(actor)
        print(f"  Loaded: policy_shared_mappo_actor_{adv}_seed_0.pt")

    # ======================================================
    # INIT ENVIRONMENT
    # ======================================================
    env = MultiRTBEnvironmentSharedMAPPO(
        shared_data_path = str(SHARED_DATA),
        budgets          = BUDGETS,
        adv_ids          = ADV_IDS,
        max_steps        = MAX_STEPS,
        reserve_price    = 1.0,
        state_dim        = STATE_DIM,
    )

    local_states, global_state = env.reset()
    states_t = [
        torch.tensor(s, dtype=torch.float32, device=DEVICE)
        for s in local_states
    ]

    log_rows    = []
    done        = False
    step        = 0
    prev_clicks = env.clicks.copy()
    prev_costs  = env.costs.copy()

    print(f"\n{'='*70}")
    print(f" STEP-BY-STEP BIDDING (First {DEMO_STEPS} Impressions)")
    print(f"{'='*70}")

    while not done:
        step += 1

        # Current impression
        row          = env.df.iloc[env.ptr]
        market_price = float(row["market_price"])
        hour         = int(row.get("hour", 0))
        weekday      = int(row.get("weekday", 0))
        slot_w       = int(row.get("slot_w", 0))
        slot_h       = int(row.get("slot_h", 0))

        pctr_values  = {
            adv: float(row[f"pctr_{adv}"])
            for adv in ADV_IDS
        }

        # Each agent decides threshold
        thresholds  = []
        decisions   = []
        budget_pcts = []

        for i in range(NUM_AGENTS):
            with torch.no_grad():
                action, _, _ = actors[i].get_action(states_t[i])
            threshold = THRESHOLD_VALUES[action.item()]
            thresholds.append(threshold)
            pctr = pctr_values[ADV_IDS[i]]
            decisions.append("BID" if pctr >= threshold else "SKIP")
            budget_pcts.append(
                env.remaining_budget[i] / BUDGETS[i] * 100
            )

        # Find winner
        valid      = [
            i for i in range(NUM_AGENTS)
            if pctr_values[ADV_IDS[i]] >= thresholds[i]
            and env.remaining_budget[i] >= env.reserve_price
        ]
        winner     = max(
            valid, key=lambda i: pctr_values[ADV_IDS[i]]
        ) if valid else None
        winner_adv = ADV_IDS[winner] if winner is not None else "None"

        # Pacing signal per agent
        pacing_errors = []
        for i in range(NUM_AGENTS):
            br = env.remaining_budget[i] / BUDGETS[i]
            tr = 1.0 - (env.t / MAX_STEPS)
            pe = br - tr
            pacing_errors.append(round(pe, 4))

        # Verbose print for first DEMO_STEPS
        if step <= DEMO_STEPS:
            print(f"\n{'─'*70}")
            print(
                f" Impression #{step:03d} | "
                f"Market Price: {market_price:.2f} | "
                f"Hour: {hour} | Weekday: {weekday} | "
                f"Slot: {slot_w}×{slot_h}"
            )
            print(f"{'─'*70}")
            for i, adv in enumerate(ADV_IDS):
                win_str = " ← WINNER" if i == winner else ""
                bid_str = "BID  " if decisions[i] == "BID" else "SKIP "
                pe_str  = (f"PACE:{pacing_errors[i]:+.2f}"
                           if abs(pacing_errors[i]) > 0.1 else "PACED")
                print(
                    f"  {adv:>4} | "
                    f"pCTR={pctr_values[adv]:.4f} | "
                    f"Thr={thresholds[i]:.4f} | "
                    f"{bid_str} | "
                    f"Budget={env.remaining_budget[i]:>8.0f} "
                    f"({budget_pcts[i]:.0f}%) | "
                    f"{pe_str}"
                    f"{win_str}"
                )

        # Execute step
        next_local, _, rewards, done = env.step(thresholds)

        # Compute cost and click for this step correctly
        cost_paid = 0.0
        click     = 0
        if winner is not None:
            cost_paid = float(env.costs[winner] - prev_costs[winner])
            click     = int(env.clicks[winner] - prev_clicks[winner])

        if step <= DEMO_STEPS:
            print(
                f"\n  Result  : Winner={winner_adv} | "
                f"Cost={cost_paid:.2f} | "
                f"Click={'YES ✅' if click else 'NO ❌'} | "
                f"Reward={rewards[winner] if winner else 0:.3f}"
            )

        # ======================================================
        # LOG ROW — complete, every step
        # ======================================================
        log_row = {
            "step":         step,
            "market_price": round(market_price, 2),
            "hour":         hour,
            "weekday":      weekday,
            "slot_w":       slot_w,
            "slot_h":       slot_h,
            "winner":       winner_adv,
            "cost_paid":    round(cost_paid, 2),
            "click":        click,
        }

        # Per agent columns
        for i, adv in enumerate(ADV_IDS):
            log_row[f"pctr_{adv}"]         = round(pctr_values[adv], 6)
            log_row[f"threshold_{adv}"]    = round(thresholds[i], 4)
            log_row[f"decision_{adv}"]     = decisions[i]
            log_row[f"budget_{adv}"]       = round(
                float(env.remaining_budget[i]), 2)
            log_row[f"budget_pct_{adv}"]   = round(budget_pcts[i], 1)
            log_row[f"pacing_err_{adv}"]   = pacing_errors[i]
            log_row[f"reward_{adv}"]       = round(float(rewards[i]), 4)
            log_row[f"cum_clicks_{adv}"]   = int(env.clicks[i])
            log_row[f"cum_cost_{adv}"]     = round(float(env.costs[i]), 2)

        log_rows.append(log_row)

        # Update tracking
        prev_clicks = env.clicks.copy()
        prev_costs  = env.costs.copy()

        if not done and next_local is not None:
            states_t = [
                torch.tensor(s, dtype=torch.float32, device=DEVICE)
                for s in next_local
            ]

        if step == DEMO_STEPS:
            print(f"\n{'='*70}")
            print(" Running remaining episode silently...")
            print(f"{'='*70}")

    # ======================================================
    # SAVE BIDDING LOG CSV
    # ======================================================
    log_df   = pd.DataFrame(log_rows)
    log_file = LOG_DIR / "bidding_log_shared.csv"
    log_df.to_csv(log_file, index=False)

    # ======================================================
    # DETAILED INTROSPECTION
    # ======================================================
    print(f"\n{'='*70}")
    print(f" DETAILED BIDDING LOG ANALYSIS")
    print(f"{'='*70}")

    won  = log_df[log_df["cost_paid"] > 0]
    skip = log_df[log_df["winner"] == "None"]

    print(f"\n--- BASIC STATS ---")
    print(f"Total steps     : {len(log_df):,}")
    print(f"Total cost paid : {log_df['cost_paid'].sum():,.2f}")
    print(f"Total clicks    : {log_df['click'].sum():,}")
    tc = log_df['click'].sum()
    print(f"Overall CPC     : "
          f"{log_df['cost_paid'].sum()/tc:.2f}" if tc > 0
          else "Overall CPC: N/A")
    print(f"Win rate        : {len(won)/len(log_df)*100:.1f}%")
    print(f"Skip rate       : {len(skip)/len(log_df)*100:.1f}%")

    print(f"\n--- MARKET PRICE DISTRIBUTION ---")
    mp = log_df["market_price"]
    print(f"Mean={mp.mean():.2f} | Std={mp.std():.2f} | "
          f"Min={mp.min():.2f} | Median={mp.median():.2f} | "
          f"Max={mp.max():.2f}")

    print(f"\n--- PER AGENT ANALYSIS ---")
    print(f"{'Agent':<6} {'BIDs':>7} {'SKIPs':>7} {'Wins':>6} "
          f"{'WinRate':>8} {'Clicks':>7} {'Cost':>10} "
          f"{'CPC':>8} {'Util%':>7} {'AvgPacing':>10}")
    print("-" * 85)
    for adv in ADV_IDS:
        bids       = (log_df[f"decision_{adv}"] == "BID").sum()
        skips      = (log_df[f"decision_{adv}"] == "SKIP").sum()
        agent_rows = log_df[log_df["winner"] == adv]
        wins       = len(agent_rows)
        clicks     = agent_rows["click"].sum()
        cost       = agent_rows["cost_paid"].sum()
        cpc        = cost / clicks if clicks > 0 else 0
        wr         = wins / bids * 100 if bids > 0 else 0
        bleft      = log_df[f"budget_{adv}"].iloc[-1]
        util       = (BUDGETS[ADV_IDS.index(adv)] - bleft) / \
                     BUDGETS[ADV_IDS.index(adv)] * 100
        avg_pace   = log_df[f"pacing_err_{adv}"].abs().mean()
        print(f"{adv:<6} {bids:>7,} {skips:>7,} {wins:>6} "
              f"{wr:>7.1f}% {clicks:>7} {cost:>10.2f} "
              f"{cpc:>8.2f} {util:>6.1f}% {avg_pace:>10.4f}")

    print(f"\n--- THRESHOLD ANALYSIS ---")
    print(f"{'Agent':<6} {'Mean':>10} {'Std':>8} "
          f"{'Min':>8} {'Max':>8}  Interpretation")
    print("-" * 65)
    for adv in ADV_IDS:
        t = log_df[f"threshold_{adv}"]
        if t.mean() < 0.05:
            interp = "Very aggressive"
        elif t.mean() < 0.15:
            interp = "Aggressive"
        elif t.mean() < 0.22:
            interp = "Balanced"
        else:
            interp = "Conservative"
        print(f"{adv:<6} {t.mean():>10.4f} {t.std():>8.4f} "
              f"{t.min():>8.4f} {t.max():>8.4f}  {interp}")

    print(f"\n--- pCTR DISTRIBUTION ---")
    print(f"{'Agent':<6} {'Mean':>10} {'% zeros':>9} {'% ones':>9}")
    print("-" * 40)
    for adv in ADV_IDS:
        p     = log_df[f"pctr_{adv}"]
        zeros = (p == 0).sum() / len(p) * 100
        ones  = (p == 1).sum() / len(p) * 100
        print(f"{adv:<6} {p.mean():>10.4f} "
              f"{zeros:>8.1f}% {ones:>8.1f}%")

    print(f"\n--- WINNER DISTRIBUTION ---")
    wc = log_df[log_df["winner"] != "None"]["winner"].value_counts()
    for adv, count in wc.items():
        clicks = log_df[log_df["winner"] == adv]["click"].sum()
        cost   = log_df[log_df["winner"] == adv]["cost_paid"].sum()
        ctr    = clicks / count * 100 if count > 0 else 0
        print(f"  {adv}: {count:,} wins | "
              f"{clicks} clicks | "
              f"CTR={ctr:.1f}% | "
              f"Cost={cost:,.0f}")

    print(f"\n--- BUDGET DEPLETION & PACING ---")
    for adv in ADV_IDS:
        idx    = ADV_IDS.index(adv)
        start  = BUDGETS[idx]
        end    = log_df[f"budget_{adv}"].iloc[-1]
        spent  = start - end
        util   = spent / start * 100
        half   = log_df[log_df[f"budget_{adv}"] <= start * 0.5].index
        h_step = half[0] if len(half) > 0 else "never"
        avg_pe = log_df[f"pacing_err_{adv}"].abs().mean()
        flag   = "OK" if util >= 80 else "LOW"
        print(f"  {adv}: Budget={start:.0f} | "
              f"Spent={spent:.0f} ({util:.1f}%) | "
              f"50% at step={h_step} | "
              f"Avg pacing err={avg_pe:.3f} {flag}")

    # ======================================================
    # EPISODE SUMMARY
    # ======================================================
    print(f"\n{'='*70}")
    print(f" EPISODE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Adv':<6} {'Budget':>8} {'Clicks':>8} "
          f"{'Cost':>10} {'CPC':>8} {'Util%':>8} {'PaceErr':>9}")
    print(f"{'─'*70}")

    total_clicks = 0
    total_cost   = 0.0
    for i, adv in enumerate(ADV_IDS):
        clicks = int(env.clicks[i])
        cost   = float(env.costs[i])
        cpc    = cost / clicks if clicks > 0 else 0
        util   = (cost / BUDGETS[i]) * 100
        pace   = log_df[f"pacing_err_{adv}"].abs().mean()
        flag   = "OK" if util >= 80 else "LOW"
        print(f"{adv:<6} {int(BUDGETS[i]):>8} {clicks:>8} "
              f"{cost:>10.1f} {cpc:>8.2f} "
              f"{util:>7.1f}% {pace:>9.3f} {flag}")
        total_clicks += clicks
        total_cost   += cost

    total_util = (total_cost / sum(BUDGETS)) * 100
    avg_cpc    = total_cost / total_clicks if total_clicks > 0 else 0
    print(f"{'─'*70}")
    print(f"{'TOTAL':<6} {int(sum(BUDGETS)):>8} {total_clicks:>8} "
          f"{total_cost:>10.1f} {avg_cpc:>8.2f} "
          f"{total_util:>7.1f}%")
    print(f"{'='*70}")

    # CSV summary
    print(f"\n{'='*70}")
    print(f" BIDDING LOG CSV SUMMARY")
    print(f"{'='*70}")
    print(f"File    : {log_file}")
    print(f"Rows    : {len(log_rows):,} (one per auction step)")
    print(f"Columns : {len(log_df.columns)}")
    print(f"\nColumn groups:")
    print(f"  Impression  : step, market_price, hour, weekday, slot_w, slot_h")
    print(f"  Outcome     : winner, cost_paid, click")
    print(f"  Per agent   : pctr, threshold, decision, budget, budget_pct,")
    print(f"                pacing_err, reward, cum_clicks, cum_cost")
    print(f"  Total cols  : 9 shared + {NUM_AGENTS}×9 agent = "
          f"{9 + NUM_AGENTS*9} columns")
    print(f"\nDemo complete!")


if __name__ == "__main__":
    main()