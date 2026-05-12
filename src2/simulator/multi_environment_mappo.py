"""
multi_environment_mappo.py
Multi-agent RTB environment for MAPPO

Changes from v5:
1. Returns global state (all agents combined) for centralized critic
2. step() returns (local_states, global_state, rewards, done)
3. Everything else same as v5
"""

import random
import numpy as np
import pandas as pd


class MultiRTBEnvironmentMAPPO:

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
        self.global_dim    = state_dim * self.num_agents  # 14×5 = 70

        assert len(data_paths) == self.num_agents

        print("Loading datasets...")
        self.dfs = {}
        for i, path in data_paths.items():
            df = pd.read_csv(path, sep="\t")
            self.dfs[i] = df
            print(f"  Agent {i}: {len(df):,} rows | "
                  f"has_pctr={'pctr' in df.columns}")

        self.budgets = np.array(budgets, dtype=np.float32)
        print(f"Local state dim  : {self.state_dim}")
        print(f"Global state dim : {self.global_dim} ({self.num_agents} agents × {self.state_dim})")
        self.reset()

    def reset(self):
        self.t                = 0
        self.indices          = {i: 0 for i in range(self.num_agents)}
        self.costs            = np.zeros(self.num_agents, dtype=np.float32)
        self.clicks           = np.zeros(self.num_agents, dtype=np.int32)
        self.remaining_budget = self.budgets.copy()
        self.recent_wins      = {i: [] for i in range(self.num_agents)}
        self.win_rates        = np.zeros(self.num_agents, dtype=np.float32)

        local_states = self._get_local_states()
        global_state = self._get_global_state()
        return local_states, global_state

    def _get_state_for_agent(self, i):
        df  = self.dfs[i]
        idx = self.indices[i]

        if idx >= len(df):
            return np.zeros(self.state_dim, dtype=np.float32)

        row = df.iloc[idx]

        budget_ratio = self.remaining_budget[i] / self.budgets[i]
        time_ratio   = 1.0 - (self.t / self.max_steps)
        urgency      = budget_ratio * (1.0 - time_ratio)

        return np.array([
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
            urgency,
        ], dtype=np.float32)

    def _get_local_states(self):
        """Individual state for each agent's actor"""
        return [self._get_state_for_agent(i) for i in range(self.num_agents)]

    def _get_global_state(self):
        """
        Concatenated state of ALL agents for centralized critic
        Shape: (num_agents × state_dim,) = (5 × 14 = 70,)
        Critic sees everything — handles non-stationarity
        """
        all_states = [self._get_state_for_agent(i)
                      for i in range(self.num_agents)]
        return np.concatenate(all_states).astype(np.float32)

    def step(self, pctr_thresholds):
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        # ======================================================
        # CORE: Valid bidders — pCTR >= threshold
        # ======================================================
        valid_bidders = [
            i for i in range(self.num_agents)
            if self.indices[i] < len(self.dfs[i])
            and float(self.dfs[i].iloc[self.indices[i]]["pctr"])
               >= pctr_thresholds[i]
            and self.remaining_budget[i] >= self.reserve_price
        ]

        winner = None
        if valid_bidders:
            # Highest pCTR wins auction
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

        # Non-winners
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
            or all(
                self.indices[i] >= len(self.dfs[i])
                for i in range(self.num_agents)
            )
        )

        # Budget utilization penalty at episode end
        if done:
            for i in range(self.num_agents):
                utilization = self.costs[i] / self.budgets[i]
                if utilization < 0.8:
                    rewards[i] -= (1.0 - utilization) * 20.0

        # Return local states AND global state
        if not done:
            local_states = self._get_local_states()
            global_state = self._get_global_state()
        else:
            local_states = [np.zeros(self.state_dim, dtype=np.float32)
                           for _ in range(self.num_agents)]
            global_state = np.zeros(self.global_dim, dtype=np.float32)

        return local_states, global_state, rewards, done
