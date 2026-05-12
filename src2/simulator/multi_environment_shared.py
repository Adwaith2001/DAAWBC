"""
multi_environment_shared.py
TRUE shared auction environment.

Reward function v2:
1. Click reward        : +1.0
2. No-click penalty    : -0.001 × price
3. Lost auction        : -0.01 × pCTR
4. Skip                : 0.0 (neutral)
5. Pacing penalty      : -(budget_ratio - time_ratio) × 0.5
6. End episode penalty : -(1 - util) × 20 if util < 80%

Episode reset: dataset shuffled each episode
→ Agent learns general strategy not sequence memorization
"""

import numpy as np
import pandas as pd


class MultiRTBEnvironmentShared:

    def __init__(
        self,
        shared_data_path,
        budgets,
        adv_ids,
        max_steps     = 2000,
        reserve_price = 1.0,
        state_dim     = 14,
    ):
        self.num_agents    = len(budgets)
        self.max_steps     = max_steps
        self.reserve_price = reserve_price
        self.state_dim     = state_dim
        self.adv_ids       = adv_ids
        self.budgets       = np.array(budgets, dtype=np.float32)

        print(f"Loading shared auction log...")
        self.df_original = pd.read_csv(shared_data_path, sep="\t")
        self.n           = len(self.df_original)

        print(f"  Rows        : {self.n:,}")
        print(f"  Avg Price   : {self.df_original['market_price'].mean():.1f}")
        print(f"  Max steps   : {self.max_steps}")
        print(f"  Shuffle/ep  : YES ← prevents sequence memorization")

        for adv in self.adv_ids:
            col = f"pctr_{adv}"
            assert col in self.df_original.columns, f"Missing: {col}"
            print(f"  pctr_{adv}  : mean={self.df_original[col].mean():.4f} ✅")

        print(f"State dim   : {self.state_dim}")
        self.reset()

    def reset(self):
        # ======================================================
        # SHUFFLE dataset every episode
        # Each episode sees same rows in DIFFERENT order
        # → Agent learns general bidding not sequence memorization
        # ======================================================
        self.df = self.df_original.sample(
            frac=1
        ).reset_index(drop=True)

        self.ptr              = 0
        self.t                = 0
        self.costs            = np.zeros(self.num_agents, dtype=np.float32)
        self.clicks           = np.zeros(self.num_agents, dtype=np.int32)
        self.remaining_budget = self.budgets.copy()
        self.recent_wins      = {i: [] for i in range(self.num_agents)}
        self.win_rates        = np.zeros(self.num_agents, dtype=np.float32)
        return self._get_states()

    def _get_state_for_agent(self, i):
        if self.ptr >= self.n:
            return np.zeros(self.state_dim, dtype=np.float32)

        row          = self.df.iloc[self.ptr]
        budget_ratio = self.remaining_budget[i] / self.budgets[i]
        time_ratio   = 1.0 - (self.t / self.max_steps)
        urgency      = budget_ratio * (1.0 - time_ratio)
        pctr         = float(row[f"pctr_{self.adv_ids[i]}"])

        return np.array([
            budget_ratio,
            time_ratio,
            pctr,
            float(row["market_price"]),
            float(row["hour"]) / 23.0,
            float(row["weekday"]) / 6.0,
            float(row["slot_w"]) / 1000.0,
            float(row["slot_h"]) / 1000.0,
            self.win_rates[i],
            self.costs[i] / self.budgets[i],
            float(row.get("region", 0)) / 100.0,
            float(row.get("device_type", 0)) / 5.0,
            float(row.get("usertag_count", 0)) / 50.0,
            urgency,
        ], dtype=np.float32)

    def _get_states(self):
        return [self._get_state_for_agent(i)
                for i in range(self.num_agents)]

    def step(self, pctr_thresholds):
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        if self.ptr >= self.n:
            return None, rewards, True

        row          = self.df.iloc[self.ptr]
        market_price = float(row["market_price"])

        # ======================================================
        # AUCTION: all agents evaluate same impression
        # ======================================================
        valid_bidders = [
            i for i in range(self.num_agents)
            if float(row[f"pctr_{self.adv_ids[i]}"]) >= pctr_thresholds[i]
            and self.remaining_budget[i] >= self.reserve_price
        ]

        winner = None
        if valid_bidders:
            winner = max(
                valid_bidders,
                key=lambda i: float(row[f"pctr_{self.adv_ids[i]}"])
            )

            if self.remaining_budget[winner] >= market_price:
                cost  = market_price
                self.remaining_budget[winner] -= cost
                self.costs[winner]            += cost

                pctr  = float(row[f"pctr_{self.adv_ids[winner]}"])
                click = 1 if np.random.rand() < pctr else 0
                self.clicks[winner] += click

                # REWARD 1: Click reward
                if click == 1:
                    rewards[winner] = +1.0
                else:
                    rewards[winner] = -0.001 * market_price

                self.recent_wins[winner].append(1)

        # REWARD 2: Non-winners
        for i in range(self.num_agents):
            if i == winner:
                continue
            pctr = float(row[f"pctr_{self.adv_ids[i]}"])
            if pctr >= pctr_thresholds[i]:
                rewards[i] = -0.01 * pctr  # bid but lost
            else:
                rewards[i] = 0.0            # skip = neutral
            self.recent_wins[i].append(0)

        # Update win rates
        for i in range(self.num_agents):
            if len(self.recent_wins[i]) > 100:
                self.recent_wins[i].pop(0)
            if self.recent_wins[i]:
                self.win_rates[i] = np.mean(self.recent_wins[i])

        # REWARD 3: Budget pacing signal
        for i in range(self.num_agents):
            budget_ratio = self.remaining_budget[i] / self.budgets[i]
            time_ratio   = 1.0 - (self.t / self.max_steps)
            pacing_error = budget_ratio - time_ratio
            if pacing_error > 0.1:
                rewards[i] -= pacing_error * 0.5

        self.ptr += 1
        self.t   += 1

        done = (
            self.ptr >= self.n
            or self.t  >= self.max_steps
        )

        # REWARD 4: End of episode penalty
        if done:
            for i in range(self.num_agents):
                util = self.costs[i] / self.budgets[i]
                if util < 0.8:
                    rewards[i] -= (1.0 - util) * 20.0

        return (self._get_states() if not done else None), rewards, done