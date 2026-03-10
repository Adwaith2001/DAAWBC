import random
# Used to control stochasticity for reproducibility across seeds

import numpy as np
# Used for numerical aggregation (mean, std) across seeds

from pathlib import Path
# Provides robust, OS-independent path handling

import pandas as pd
# Imported for completeness (not explicitly used here, but useful for extension)

from simulator.environment import RTBEnvironment
# RTB simulation environment shared with RL agent


# =========================
# CONFIGURATION
# =========================

# Resolve project root directory
ROOT = Path(__file__).resolve().parents[1]

# Dataset augmented with cached pCTR values
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"

# Fixed campaign budget used across all baselines
BUDGET = 300.0

# Fixed bid value for constant bidding strategy
FIXED_BID = 50.0

# Scaling factor for linear pCTR-based bidding
LINEAR_ALPHA = 300.0

# Random seeds for multi-run robustness evaluation
SEEDS = [0, 1, 2, 3, 4]


# =========================
# UTILITY FUNCTIONS
# =========================

def set_seed(seed):
    """
    Fix random seeds for reproducibility.
    This ensures that stochastic click outcomes are comparable
    across different baseline strategies.
    """
    random.seed(seed)
    np.random.seed(seed)


# =========================
# BASELINE STRATEGIES
# =========================

def run_fixed_bid(seed):
    """
    Executes a fixed-bid baseline strategy.

    The agent submits the same bid value for every impression,
    regardless of context or predicted click probability.
    """

    # Set seed for this run
    set_seed(seed)

    # Initialize RTB environment
    env = RTBEnvironment(
        data_path=str(DATA_FILE),
        budget=BUDGET,
        max_steps=10000,
    )

    # Reset environment to initial state
    env.reset()

    done = False

    # Process impressions sequentially
    while not done:
        # Always bid the same fixed amount
        _, _, done = env.step(FIXED_BID)

    # Return total clicks and total cost incurred
    return env.total_clicks, env.cost


def run_linear_pctr(seed):
    """
    Executes a linear pCTR-based bidding strategy.

    Bid formula:
        bid = α × pCTR

    where α is a manually tuned scaling constant.
    """

    # Set seed for reproducibility
    set_seed(seed)

    # Initialize RTB environment
    env = RTBEnvironment(
        data_path=str(DATA_FILE),
        budget=BUDGET,
        max_steps=10000,
    )

    # Reset environment
    env.reset()

    done = False

    # Sequentially process impressions
    while not done:

        # Access current impression
        row = env.df.iloc[env.ptr]

        # Compute bid as a linear function of pCTR
        bid = LINEAR_ALPHA * row["pctr"]

        # Submit bid to environment
        _, _, done = env.step(bid)

    # Return performance metrics
    return env.total_clicks, env.cost


# =========================
# MAIN EVALUATION LOOP
# =========================

if __name__ == "__main__":

    print("\n==============================")
    print(" MULTI-SEED BASELINE COMPARISON")
    print("==============================\n")

    fixed_results = []
    linear_results = []

    # Run each baseline for all seeds
    for seed in SEEDS:

        # Fixed bid strategy
        c, cost = run_fixed_bid(seed)
        fixed_results.append((c, cost))

        # Linear pCTR strategy
        c, cost = run_linear_pctr(seed)
        linear_results.append((c, cost))

    # Extract clicks only for statistical comparison
    fixed_clicks = np.array([x[0] for x in fixed_results])
    linear_clicks = np.array([x[0] for x in linear_results])

    # ----------------------------------------------------
    # Report results
    # ----------------------------------------------------
    print("Fixed Bid (mean ± std)")
    print(f"Clicks : {fixed_clicks.mean():.2f} ± {fixed_clicks.std():.2f}")

    print("\nLinear pCTR (mean ± std)")
    print(f"Clicks : {linear_clicks.mean():.2f} ± {linear_clicks.std():.2f}")
