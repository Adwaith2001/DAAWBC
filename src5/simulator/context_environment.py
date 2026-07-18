"""
src5/simulator/context_environment.py
=====================================

Multi-agent shared 2nd-price auction environment using CONTEXT-PROPENSITY
as the bid signal (not pCTR).

Key differences from src4's multi_environment_strategic.py:
  - Bid signal is `propensity(row)` from a frozen logistic-regression
    fitted on weekday {3,4} context features. pctr_raw is NOT used.
  - Click outcome is READ FROM the file's `click` column (not sampled
    from pctr at step time). This avoids re-introducing circular click
    generation.
  - Filtered to a single weekday range at load time (training: {3,4},
    eval: {5}). The propensity model was fit on the train range; the
    agent evaluates only on the held-out range.

Action space (per agent, per impression):
  - Threshold head: 51 bins over propensity space. Skip bid if
    propensity(row) < threshold.
  - Residual head: 11 bins over [-0.3, +0.3]. Multiplicative adjustment
    to bid_base.

Bid formula:
    if propensity(row) < threshold:
        bid = 0
    else:
        bid_base = alpha_scale * propensity(row) / lambda_t
        bid     = bid_base * (1 + residual)
        bid     = clip(bid, 0, bid_cap)

alpha_scale is calibrated at env init so mean bid_base ≈ mean market
price on the loaded data, keeping the auction economically sensible.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Default action grids
# ----------------------------------------------------------------------
DEFAULT_THRESHOLD_GRID = np.linspace(0.0, 0.6, 51, dtype=np.float32)  # propensity space
DEFAULT_RESIDUAL_GRID = np.linspace(-0.3, 0.3, 11, dtype=np.float32)
DEFAULT_BID_CAP = 300.0

ADV_IDS: list[str] = ["1458", "2259", "3386", "2997", "3476"]


class ContextRTBEnvironment:
    """Shared multi-agent 2nd-price auction with context-propensity bidding."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        df: pd.DataFrame,
        budgets: Sequence[float],
        threshold_grid: np.ndarray = DEFAULT_THRESHOLD_GRID,
        residual_grid: np.ndarray = DEFAULT_RESIDUAL_GRID,
        bid_cap: float = DEFAULT_BID_CAP,
        lambda_init: float = 1.0,
        reserve_price: float = 0.1,
        episode_rows: int | None = None,
        click_reward_scale: float | None = None,
        target_ctr_design: float = 0.15,
        verbose: bool = True,
    ):
        # df MUST already have a `propensity` column (computed once via
        # ContextPropensityModel.attach_to_dataframe).
        if "propensity" not in df.columns:
            raise ValueError(
                "df must have a `propensity` column. Use "
                "ContextPropensityModel.attach_to_dataframe first."
            )
        if "click" not in df.columns:
            raise ValueError("df must have a `click` column")
        if "market_price" not in df.columns:
            raise ValueError("df must have a `market_price` column")

        self.df = df.reset_index(drop=True)
        self.n_rows = len(self.df)

        # Cache numpy views for fast access
        self._propensity = self.df["propensity"].values.astype(np.float32)
        self._market_price = self.df["market_price"].values.astype(np.float32)
        self._click = self.df["click"].values.astype(np.int32)

        # Agents
        self.n_agents = len(budgets)
        self.budgets = np.array(budgets, dtype=np.float32)

        # Action grids
        self.threshold_grid = np.asarray(threshold_grid, dtype=np.float32)
        self.residual_grid = np.asarray(residual_grid, dtype=np.float32)
        self.n_threshold = len(self.threshold_grid)
        self.n_residual = len(self.residual_grid)
        self.bid_cap = float(bid_cap)
        self.reserve_price = float(reserve_price)

        # Lambda — initialized per-agent, updated by adaptive controller
        self.lambda_init = float(lambda_init)
        # click_reward_scale derivation (frozen per Round-9 review §1):
        #   scale = mean_market_price / target_CTR_design
        # Interpretable as: "one click is worth approximately the cost of
        # the impressions you bid to get it under the dataset's design CTR."
        # Computed from training-data mean market price at env init so
        # it's not a tunable knob after seeing results.
        self.target_ctr_design = float(target_ctr_design)

        # Episode length — default to full file size, or user-specified
        self.episode_rows = int(episode_rows) if episode_rows else self.n_rows
        if self.episode_rows > self.n_rows:
            self.episode_rows = self.n_rows

        # Calibrate alpha_scale ONCE so mean bid_base ≈ mean market price
        mean_prop = float(self._propensity.mean())
        mean_mp = float(self._market_price.mean())
        if mean_prop <= 0:
            raise ValueError("Mean propensity is non-positive")
        self.alpha_scale = mean_mp / mean_prop

        # Now finalize click_reward_scale (Round-9 §1):
        if click_reward_scale is None:
            self.click_reward_scale = mean_mp / self.target_ctr_design
            scale_source = "derived"
        else:
            self.click_reward_scale = float(click_reward_scale)
            scale_source = "explicit override"

        if verbose:
            print(f"  ContextRTBEnvironment initialized")
            print(f"    n_rows         : {self.n_rows:,}")
            print(f"    episode_rows   : {self.episode_rows:,}")
            print(f"    n_agents       : {self.n_agents}")
            print(f"    budgets        : {self.budgets.tolist()}")
            print(f"    mean propensity: {mean_prop:.4f}")
            print(f"    mean mkt price : {mean_mp:.2f}")
            print(f"    alpha_scale    : {self.alpha_scale:.2f}")
            print(f"    threshold grid : {self.n_threshold} bins "
                  f"in [{self.threshold_grid.min():.3f}, {self.threshold_grid.max():.3f}]")
            print(f"    residual grid  : {self.n_residual} bins "
                  f"in [{self.residual_grid.min():+.3f}, {self.residual_grid.max():+.3f}]")
            print(f"    bid cap        : {self.bid_cap}")
            print(f"    click reward scale: {self.click_reward_scale:.2f} "
                  f"({scale_source}; target_CTR_design={self.target_ctr_design})")
            print(f"    reward formula : click_reward_scale * click  (R9g: cost penalty removed)")

        # Per-episode state (set by reset)
        self.t = 0
        self.cursor = 0
        self.cost = np.zeros(self.n_agents, dtype=np.float32)
        self.clicks = np.zeros(self.n_agents, dtype=np.int32)
        self.wins = np.zeros(self.n_agents, dtype=np.int32)
        self.remaining_budget = self.budgets.copy()
        self.lam = np.full(self.n_agents, self.lambda_init, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, start_row: int | None = None) -> list[np.ndarray]:
        self.t = 0
        # Random start if not given (avoids overfitting to one slice)
        if start_row is None:
            max_start = max(1, self.n_rows - self.episode_rows)
            self.cursor = int(np.random.randint(0, max_start))
        else:
            self.cursor = int(start_row)

        self.cost[:] = 0
        self.clicks[:] = 0
        self.wins[:] = 0
        self.remaining_budget = self.budgets.copy()
        self.lam[:] = self.lambda_init

        return self._get_state()

    # ------------------------------------------------------------------
    def _get_state(self) -> list[np.ndarray]:
        """Per-agent state: [budget_ratio, time_ratio, propensity, mp_norm].

        Identical layout to src4 — only the third entry's meaning changed
        from pctr to propensity.
        """
        if self.cursor >= self.n_rows:
            zero = np.zeros(4, dtype=np.float32)
            return [zero.copy() for _ in range(self.n_agents)]

        propensity = self._propensity[self.cursor]
        mp = self._market_price[self.cursor]
        # Crude market-price normalizer ~ [0, ~5]
        mp_norm = mp / 100.0

        states = []
        for i in range(self.n_agents):
            budget_ratio = self.remaining_budget[i] / max(1e-6, self.budgets[i])
            time_ratio = 1.0 - (self.t / max(1, self.episode_rows))
            s = np.array([budget_ratio, time_ratio, propensity, mp_norm],
                         dtype=np.float32)
            states.append(s)
        return states

    # ------------------------------------------------------------------
    def step(
        self,
        threshold_actions: Sequence[int],
        residual_actions: Sequence[int],
    ):
        """One auction step. Returns (next_states, rewards, done, info)."""
        if self.cursor >= self.n_rows or self.t >= self.episode_rows:
            zero = np.zeros(4, dtype=np.float32)
            return ([zero.copy()] * self.n_agents,
                    np.zeros(self.n_agents, dtype=np.float32),
                    True,
                    {"end_of_data": True})

        prop = float(self._propensity[self.cursor])
        mp = float(self._market_price[self.cursor])
        click = int(self._click[self.cursor])

        # Per-agent bid computation
        bids = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            th = self.threshold_grid[threshold_actions[i]]
            if prop < th:
                bids[i] = 0.0
                continue
            res = self.residual_grid[residual_actions[i]]
            bid_base = self.alpha_scale * prop / max(1e-6, self.lam[i])
            bid = bid_base * (1.0 + res)
            bid = max(0.0, min(self.bid_cap, bid))
            # Budget hard guard
            if bid > self.remaining_budget[i]:
                bid = 0.0
            bids[i] = bid

        # Auction: 2nd-price among bidders above reserve
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        sold = False
        winner = -1
        price_paid = 0.0

        bidders = [i for i in range(self.n_agents) if bids[i] >= self.reserve_price]
        if bidders:
            sorted_bidders = sorted(bidders, key=lambda i: bids[i], reverse=True)
            top1 = sorted_bidders[0]
            top1_bid = bids[top1]
            # 2nd-price = max(2nd-highest bid, market_price, reserve)
            if len(sorted_bidders) >= 2:
                top2_bid = bids[sorted_bidders[1]]
                payable = max(top2_bid, mp, self.reserve_price)
            else:
                payable = max(mp, self.reserve_price)

            if top1_bid >= payable:
                winner = top1
                price_paid = float(payable)
                price_paid = min(price_paid, self.remaining_budget[winner])
                self.remaining_budget[winner] -= price_paid
                self.cost[winner] += price_paid
                self.wins[winner] += 1
                self.clicks[winner] += click
                # Reward = click_reward_scale * click - lambda * cost
                # click_reward_scale should approximate eCPC so click bonus
                # is comparable in magnitude to cost penalty per win.
                # R9g: click-only reward (cost penalty removed).
                # The original lambda*cost term was creating a non-Markov reward
                # signal: lambda swung 0.8 -> 99.99 across episodes, making the same
                # (state, action) tuple produce rewards spanning +61 to -2513.
                # Budget is still enforced as a hard constraint via the bid cap on
                # remaining_budget; lambda still scales bids for pacing. Standard
                # RTB-RL formulation (HiBid, DRLB).
                rewards[winner] = self.click_reward_scale * float(click)
                sold = True

        # Time advances
        self.t += 1
        self.cursor += 1

        # Adaptive lambda update (per agent, slow)
        # Push lambda up when overspending pace; down when underspending.
        time_ratio = self.t / max(1, self.episode_rows)
        for i in range(self.n_agents):
            spent_ratio = (self.budgets[i] - self.remaining_budget[i]) / max(1e-6, self.budgets[i])
            pace_err = spent_ratio - time_ratio
            # Gentle exponential update
            self.lam[i] *= float(np.exp(0.05 * pace_err))
            self.lam[i] = float(np.clip(self.lam[i], 1e-3, 100.0))

        done = (self.t >= self.episode_rows) or (self.cursor >= self.n_rows)
        info = {
            "winner": winner,
            "price_paid": price_paid,
            "click_label": click,
            "sold": sold,
            "propensity": prop,
            "market_price": mp,
        }

        return self._get_state(), rewards, done, info


    # ------------------------------------------------------------------
    def step_slot(
        self,
        threshold_actions: Sequence[int],
        residual_actions: Sequence[int],
        slot_size: int,
    ):
        """Time-slot wrapper: run slot_size impressions with FIXED actions.

        One AC decision per agent per slot (vs per-impression). This is the
        architecture ported from src4 per R9b ruling — the per-impression
        original was computationally infeasible (~35 min/ep on RTX 3050
        laptop due to per-call CUDA round-trip overhead).

        Per-slot reward = sum of per-impression rewards in the slot. Since
        click_reward_scale * click is non-trivial, slot rewards can be
        10-50x per-impression magnitudes (R9b §5 concern). The training
        loop should verify gradient stability under these magnitudes via
        the smoke test.

        Returns:
          next_states: list of per-agent states at slot boundary
          slot_rewards: per-agent reward summed over the slot
          done: True if episode finished
          slot_info: dict with slot_clicks, slot_wins, slot_cost per agent
        """
        slot_rewards = np.zeros(self.n_agents, dtype=np.float32)
        slot_clicks_start = self.clicks.copy()
        slot_wins_start = self.wins.copy()
        slot_cost_start = self.cost.copy()

        done = False
        steps_taken = 0

        for _ in range(slot_size):
            _, r, done, _ = self.step(threshold_actions, residual_actions)
            slot_rewards += r
            steps_taken += 1
            if done:
                break

        slot_info = {
            "slot_clicks": (self.clicks - slot_clicks_start).tolist(),
            "slot_wins": (self.wins - slot_wins_start).tolist(),
            "slot_cost": (self.cost - slot_cost_start).tolist(),
            "steps_taken": steps_taken,
        }
        next_states = self._get_state()
        return next_states, slot_rewards, done, slot_info

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        """End-of-episode summary."""
        return {
            "wins": self.wins.tolist(),
            "clicks": self.clicks.tolist(),
            "cost": self.cost.tolist(),
            "remaining_budget": self.remaining_budget.tolist(),
            "utilization_pct": [
                100.0 * (1.0 - r / max(1e-6, b))
                for r, b in zip(self.remaining_budget, self.budgets)
            ],
            "lambda_final": self.lam.tolist(),
            "steps": int(self.t),
        }