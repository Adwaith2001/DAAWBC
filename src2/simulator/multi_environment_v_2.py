"""
multi_environment_v2.py
Multi-agent RTB environment v2

Core concept: Agent learns WHEN to bid based on pCTR threshold
Reward: Simplified RLIB 4-function reward system
"""

import numpy as np
import pandas as pd


class MultiRTBEnvironmentV2:

    def __init__(
        self,
        data_paths,
        budgets,
        max_steps=10000,
        reserve_price=1.0,
    ):
        self.num_agents    = len(budgets)
        self.max_steps     = max_steps
        self.reserve_price = reserve_price

        assert len(data_paths) == self.num_agents

        self.dfs     = {i: pd.read_csv(path, sep="\t") for i, path in data_paths.items()}
        self.budgets = np.array(budgets, dtype=np.float32)
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
            return np.zeros(10, dtype=np.float32)

        row = df.iloc[idx]

        return np.array([
            self.remaining_budget[i] / self.budgets[i],
            1.0 - (self.t / self.max_steps),
            float(row["pctr"]),
            float(row["market_price"]),
            float(row["hour"]) / 23.0,
            float(row["weekday"]) / 6.0,
            float(row["slot_w"]) / 1000.0,
            float(row["slot_h"]) / 1000.0,
            self.win_rates[i],
            self.costs[i] / self.budgets[i],
        ], dtype=np.float32)

    def _get_states(self):
        return [self._get_state_for_agent(i) for i in range(self.num_agents)]

    def step(self, pctr_thresholds):
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        # Valid bidders — those whose pCTR >= their threshold
        valid_bidders = [
            i for i in range(self.num_agents)
            if self.indices[i] < len(self.dfs[i])
            and float(self.dfs[i].iloc[self.indices[i]]["pctr"]) >= pctr_thresholds[i]
            and self.remaining_budget[i] >= self.reserve_price
        ]

        winner = None
        if valid_bidders:
            # Winner = agent with highest pCTR among bidders
            sorted_bidders = sorted(
                valid_bidders,
                key=lambda i: float(self.dfs[i].iloc[self.indices[i]]["pctr"]),
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

                # ==========================================
                # RLIB REWARDS — simplified scale
                # ==========================================
                if click == 1:
                    # Win + Click → strong positive
                    rewards[winner] = +1.0
                else:
                    # Win + No Click → small penalty
                    rewards[winner] = -0.001 * market_price

                self.recent_wins[winner].append(1)

        # Non-winners
        for i in range(self.num_agents):
            if self.indices[i] >= len(self.dfs[i]):
                continue
            if i == winner:
                continue

            row   = self.dfs[i].iloc[self.indices[i]]
            pctr  = float(row["pctr"])
            mktpr = float(row["market_price"])

            if pctr >= pctr_thresholds[i]:
                # Tried to bid but lost auction → small penalty
                rewards[i] = -0.01 * pctr
            else:
                # Correctly skipped low pCTR → tiny reward
                rewards[i] = +0.001

            self.recent_wins[i].append(0)

        # Update win rates
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
            or all(self.indices[i] >= len(self.dfs[i]) for i in range(self.num_agents))
        )

        return (self._get_states() if not done else None), rewards, done