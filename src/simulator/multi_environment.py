import numpy as np
import pandas as pd
from pathlib import Path


class MultiRTBEnvironment:
    """
    Advertiser-specific multi-agent RTB environment.

    - Each agent has its own dataset (advertiser-specific)
    - Second-price auction with reserve price
    - Independent budgets
    - pCTR-driven click simulation
    """

    def __init__(
        self,
        data_paths,            # dict: agent_id -> path to final_sample_log_with_pctr.txt
        budgets,               # list of budgets, one per agent
        max_steps=5000,
        lambda_init=0.005,
        reserve_price=0.1,
    ):
        self.num_agents = len(budgets)
        assert len(data_paths) == self.num_agents, "Mismatch between agents and datasets"

        # Load advertiser-specific datasets
        self.dfs = {
            i: pd.read_csv(path, sep="\t")
            for i, path in data_paths.items()
        }

        self.max_steps = max_steps
        self.reserve_price = reserve_price

        # Budgets
        self.budgets = np.array(budgets, dtype=np.float32)
        self.remaining_budget = self.budgets.copy()

        # Lagrangian penalty (one per advertiser)
        self.lambda_penalty = np.ones(self.num_agents, dtype=np.float32) * lambda_init

        self.reset()

    # ======================================================
    # RESET
    # ======================================================
    def reset(self):
        self.t = 0

        # Individual time indices per advertiser
        self.indices = {i: 0 for i in range(self.num_agents)}

        self.costs = np.zeros(self.num_agents, dtype=np.float32)
        self.clicks = np.zeros(self.num_agents, dtype=np.int32)
        self.remaining_budget = self.budgets.copy()

        return self._get_state()

    # ======================================================
    # STATE
    # ======================================================
    def _get_state(self):
        states = []

        for i in range(self.num_agents):
            df = self.dfs[i]
            idx = self.indices[i]

            if idx >= len(df):
                # End of this advertiser's data
                states.append(
                    np.zeros(4, dtype=np.float32)
                )
                continue

            row = df.iloc[idx]

            pctr = row["pctr"]
            market_price = row["market_price"]

            time_ratio = self.t / self.max_steps
            budget_ratio = self.remaining_budget[i] / self.budgets[i]

            states.append(
                np.array(
                    [budget_ratio, time_ratio, pctr, market_price],
                    dtype=np.float32,
                )
            )

        return states

    # ======================================================
    # STEP
    # ======================================================
    def step(self, bids):
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        # --------------------------------------------------
        # VALID BIDDERS (reserve price + budget constraint)
        # --------------------------------------------------
        valid_bidders = [
            i for i in range(self.num_agents)
            if bids[i] >= self.reserve_price
            and self.remaining_budget[i] >= bids[i]
            and self.indices[i] < len(self.dfs[i])
        ]

        if valid_bidders:
            # Second-price auction
            sorted_bidders = sorted(valid_bidders, key=lambda i: bids[i], reverse=True)
            winner = sorted_bidders[0]

            if len(sorted_bidders) > 1:
                second_price = bids[sorted_bidders[1]]
            else:
                second_price = bids[winner]

            cost = min(second_price, self.remaining_budget[winner])

            # Deduct cost
            self.remaining_budget[winner] -= cost
            self.costs[winner] += cost

            # Click sampling from winner's dataset
            row = self.dfs[winner].iloc[self.indices[winner]]
            click = np.random.rand() < row["pctr"]

            self.clicks[winner] += int(click)

            rewards[winner] = int(click) - self.lambda_penalty[winner] * cost

        # --------------------------------------------------
        # ADVANCE TIME (ALL ADVERTISERS)
        # --------------------------------------------------
        self.t += 1
        for i in range(self.num_agents):
            if self.indices[i] < len(self.dfs[i]):
                self.indices[i] += 1

        done = (
            self.t >= self.max_steps
            or all(self.indices[i] >= len(self.dfs[i]) for i in range(self.num_agents))
        )

        next_states = self._get_state() if not done else None
        return next_states, rewards, done
