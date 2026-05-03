"""
multi_environment_v2.py
Multi-agent RTB environment v2

Supports both v3 (10 features) and v4 (13 features)
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
        max_steps     = 10000,
        reserve_price = 1.0,
    ):
        self.num_agents    = len(budgets)
        self.max_steps     = max_steps
        self.reserve_price = reserve_price

        assert len(data_paths) == self.num_agents

        print("Loading datasets...")
        self.dfs     = {}
        for i, path in data_paths.items():
            df = pd.read_csv(path, sep="\t")
            self.dfs[i] = df
            print(f"  Agent {i}: {len(df):,} rows | "
                  f"cols={len(df.columns)} | "
                  f"has_pctr={'pctr' in df.columns}")

        self.budgets  = np.array(budgets, dtype=np.float32)

        # Detect state dimension from first dataset
        self._detect_state_dim()
        print(f"State dimension: {self.state_dim}")

        self.reset()

    def _detect_state_dim(self):
        """Auto-detect state dimension based on available columns"""
        df = self.dfs[0]
        # Enhanced features (v4): 13 features
        enhanced_cols = [
            "region", "slotvisibility", "slotformat",
            "device_type", "usertag_count"
        ]
        has_enhanced = all(c in df.columns for c in enhanced_cols)
        self.state_dim    = 13 if has_enhanced else 10
        self.has_enhanced = has_enhanced

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

        # Base 10 features (v3)
        state = [
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
        ]

        # ✅ Enhanced features (v4) — 3 extra features
        if self.has_enhanced:
            state += [
                float(row.get("region", 0)) / 100.0,
                float(row.get("device_type", 0)) / 5.0,
                float(row.get("usertag_count", 0)) / 50.0,
            ]

        return np.array(state, dtype=np.float32)

    def _get_states(self):
        return [self._get_state_for_agent(i) for i in range(self.num_agents)]

    def step(self, pctr_thresholds):
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        # ======================================================
        # CORE: Only valid bidders (pCTR >= threshold)
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

                # RLIB rewards
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

            row   = self.dfs[i].iloc[self.indices[i]]
            pctr  = float(row["pctr"])
            mktpr = float(row["market_price"])

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

        return (self._get_states() if not done else None), rewards, done