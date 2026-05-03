"""
environment_v2.py
Single-agent RTB environment v2

Core concept: Agent learns WHEN to bid based on pCTR threshold
Reward: Simplified RLIB 4-function reward system
"""

import numpy as np
import pandas as pd


class RTBEnvironmentV2:

    def __init__(
        self,
        data_path,
        budget=20000.0,
        max_steps=10000,
    ):
        self.data_path = data_path
        self.budget    = float(budget)
        self.max_steps = max_steps
        self._load_data()
        self.reset()

    def _load_data(self):
        self.df = pd.read_csv(self.data_path, sep="\t")
        self.n  = len(self.df)

    def reset(self):
        self.ptr              = 0
        self.steps            = 0
        self.remaining_budget = float(self.budget)
        self.cost             = 0.0
        self.total_clicks     = 0
        self.recent_wins      = []
        self.win_rate         = 0.0
        return self._get_state()

    def _get_state(self):
        if self.ptr >= self.n:
            return np.zeros(10, dtype=np.float32)

        row = self.df.iloc[self.ptr]

        return np.array([
            self.remaining_budget / self.budget,
            1.0 - (self.steps / self.max_steps),
            float(row["pctr"]),
            float(row["market_price"]),
            float(row["hour"]) / 23.0,
            float(row["weekday"]) / 6.0,
            float(row["slot_w"]) / 1000.0,
            float(row["slot_h"]) / 1000.0,
            self.win_rate,
            self.cost / self.budget,
        ], dtype=np.float32)

    def step(self, pctr_threshold):
        """
        Action = pCTR threshold (0 to 1)
        Agent bids only if current impression pCTR >= threshold
        """
        if self.ptr >= self.n or self.steps >= self.max_steps:
            return self._get_state(), 0.0, True

        row          = self.df.iloc[self.ptr]
        market_price = float(row["market_price"])
        pctr         = float(row["pctr"])
        reward       = 0.0

        # ======================================================
        # CORE CONCEPT: Only bid if pCTR >= threshold
        # ======================================================
        if pctr >= pctr_threshold and self.remaining_budget >= market_price:

            cost = market_price
            self.remaining_budget -= cost
            self.cost             += cost

            click = 1 if np.random.rand() < pctr else 0
            self.total_clicks += click

            # ======================================================
            # RLIB REWARDS — simplified scale
            # ======================================================
            if click == 1:
                # Win + Click → strong positive
                reward = +1.0
            else:
                # Win + No Click → small penalty
                reward = -0.001 * market_price

            self.recent_wins.append(1)

        else:
            if pctr >= pctr_threshold:
                # Tried to bid but no budget → small penalty
                reward = -0.01 * pctr
            else:
                # Correctly skipped low pCTR → tiny reward
                reward = +0.001

            self.recent_wins.append(0)

        if len(self.recent_wins) > 100:
            self.recent_wins.pop(0)
        self.win_rate = np.mean(self.recent_wins)

        self.ptr   += 1
        self.steps += 1

        done = self.remaining_budget <= 0

        return self._get_state(), reward, done