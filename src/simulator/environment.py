import numpy as np
import pandas as pd


class RTBEnvironment:
    def __init__(
        self,
        data_path,
        budget=300.0,
        max_steps=10000,
        lambda_init=0.013,
    ):
        self.data_path = data_path
        self.budget = float(budget)
        self.max_steps = max_steps
        self.lambda_penalty = lambda_init

        self._load_data()
        self.reset()

    def _load_data(self):
        self.df = pd.read_csv(self.data_path, sep="\t")
        self.n = len(self.df)

    def reset(self):
        self.ptr = 0
        self.steps = 0

        self.remaining_budget = float(self.budget)
        self.cost = 0.0
        self.total_clicks = 0

        return self._get_state()

    def _get_state(self):
        if self.ptr >= self.n:
            return np.zeros(4, dtype=np.float32)

        row = self.df.iloc[self.ptr]

        budget_ratio = self.remaining_budget / self.budget
        time_ratio = 1.0 - (self.steps / self.max_steps)
        pctr = float(row["pctr"])
        market_price = float(row["market_price"])

        return np.array(
            [budget_ratio, time_ratio, pctr, market_price],
            dtype=np.float32,
        )

    def step(self, bid):
        done = False
        reward = 0.0

        if self.ptr >= self.n or self.steps >= self.max_steps:
            return self._get_state(), 0.0, True

        row = self.df.iloc[self.ptr]
        market_price = float(row["market_price"])
        pctr = float(row["pctr"])

        click = 0
        cost = 0.0

        # Win auction
        if bid >= market_price and self.remaining_budget >= market_price:
            cost = market_price
            self.remaining_budget -= cost
            self.cost += cost

            # ✅ CRITICAL: stochastic click
            click = 1 if np.random.rand() < pctr else 0
            self.total_clicks += click

            reward = click - self.lambda_penalty * cost
        else:
            reward = 0.0

        self.ptr += 1
        self.steps += 1

        if self.remaining_budget <= 0:
            done = True

        return self._get_state(), reward, done
