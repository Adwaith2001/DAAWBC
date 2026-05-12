"""
multi_environment_v5.py
Multi-agent RTB environment v5

Changes from v4 (multi_environment_v2.py stays UNTOUCHED):
1. Budget utilization penalty at episode end (stronger: x20)
2. Urgency signal added to state (14 features)
"""

import numpy as np
import pandas as pd


class MultiRTBEnvironmentV5:

    def __init__(
        self,
        data_paths,
        budgets,
        max_steps     = 10000,
        reserve_price = 1.0,
        state_dim     = 14,
    ):
        self.num_agents    = len(budgets)
        self.max_steps     = max_steps
        self.reserve_price = reserve_price
        self.state_dim     = state_dim

        assert len(data_paths) == self.num_agents

        print("Loading datasets...")
        self.dfs = {}
        for i, path in data_paths.items():
            df = pd.read_csv(path, sep="\t")
            self.dfs[i] = df
            print(f"  Agent {i}: {len(df):,} rows | "
                  f"cols={len(df.columns)} | "
                  f"has_pctr={'pctr' in df.columns}")

        self.budgets = np.array(budgets, dtype=np.float32)
        print(f"State dimension : {self.state_dim} (13 enhanced + urgency)")
        self.reset()

    def reset(self):
        self.t                = 0
        self.indices          = {i: 0 for i in range(self.num_agents)}
        self.costs            = np.zeros(self.num_agents, dtype=np.float32)
        self.clicks           = np.zeros(self.num_agents, dtype=np.int32)
        self.remaining_budget = self.budgets.copy()
        self.recent_wins      = {i: [] for i in range(self.num_agents)}
        self.win_rates        = np.zeros(self.num_agents, dtype=np.float32)
        return self._get_states()

    def _get_state_for_agent(self, i):
        df  = self.dfs[i]
        idx = self.indices[i]

        if idx >= len(df):
            return np.zeros(self.state_dim, dtype=np.float32)

        row = df.iloc[idx]

        budget_ratio = self.remaining_budget[i] / self.budgets[i]
        time_ratio   = 1.0 - (self.t / self.max_steps)

        # Urgency: high budget remaining + little time = must spend now
        urgency = budget_ratio * (1.0 - time_ratio)

        state = [
            budget_ratio,
            time_ratio,
            float(row["pctr"]),
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
            urgency,                               # 14th feature
        ]

        return np.array(state, dtype=np.float32)

    def _get_states(self):
        return [self._get_state_for_agent(i) for i in range(self.num_agents)]

    def step(self, pctr_thresholds):
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        valid_bidders = [
            i for i in range(self.num_agents)
            if self.indices[i] < len(self.dfs[i])
            and float(self.dfs[i].iloc[self.indices[i]]["pctr"])
               >= pctr_thresholds[i]
            and self.remaining_budget[i] >= self.reserve_price
        ]

        winner = None
        if valid_bidders:
            sorted_bidders = sorted(
                valid_bidders,
                key=lambda i: float(
                    self.dfs[i].iloc[self.indices[i]]["pctr"]),
                reverse=True
            )
            winner = sorted_bidders[0]

            row          = self.dfs[winner].iloc[self.indices[winner]]
            market_price = float(row["market_price"])
            pctr         = float(row["pctr"])

            if self.remaining_budget[winner] >= market_price:
                cost = market_price
                self.remaining_budget[winner] -= cost
                self.costs[winner]            += cost

                click = 1 if np.random.rand() < pctr else 0
                self.clicks[winner] += click

                if click == 1:
                    rewards[winner] = +1.0
                else:
                    rewards[winner] = -0.001 * market_price

                self.recent_wins[winner].append(1)

        for i in range(self.num_agents):
            if self.indices[i] >= len(self.dfs[i]):
                continue
            if i == winner:
                continue

            row  = self.dfs[i].iloc[self.indices[i]]
            pctr = float(row["pctr"])

            if pctr >= pctr_thresholds[i]:
                rewards[i] = -0.01 * pctr
            else:
                rewards[i] = +0.001

            self.recent_wins[i].append(0)

        for i in range(self.num_agents):
            if len(self.recent_wins[i]) > 100:
                self.recent_wins[i].pop(0)
            if self.recent_wins[i]:
                self.win_rates[i] = np.mean(self.recent_wins[i])

        self.t += 1
        for i in range(self.num_agents):
            if self.indices[i] < len(self.dfs[i]):
                self.indices[i] += 1

        done = (
            self.t >= self.max_steps
            or all(
                self.indices[i] >= len(self.dfs[i])
                for i in range(self.num_agents)
            )
        )

        # ======================================================
        # V5: Stronger budget utilization penalty (x20)
        # ======================================================
        if done:
            for i in range(self.num_agents):
                utilization = self.costs[i] / self.budgets[i]
                if utilization < 0.8:
                    penalty = (1.0 - utilization) * 20.0  # ← stronger
                    rewards[i] -= penalty

        return (self._get_states() if not done else None), rewards, done