import numpy as np
# Used for numerical operations and random sampling (click simulation)

import pandas as pd
# Used to load and manage the RTB log data (tab-separated format)


class RTBEnvironment:
    """
    Real-Time Bidding (RTB) simulation environment.

    This class models a single-advertiser RTB process as an episodic
    Markov Decision Process (MDP), where each step corresponds to one
    ad impression opportunity.
    """

    def __init__(
        self,
        data_path,
        budget=300.0,
        max_steps=10000,
        lambda_init=0.013,
    ):
        """
        Initialize the RTB environment.

        Parameters
        ----------
        data_path : str
            Path to the processed RTB log file containing pCTR and market prices.
        budget : float
            Total campaign budget available to the advertiser.
        max_steps : int
            Maximum number of impressions processed per episode.
        lambda_init : float
            Initial Lagrangian multiplier used for budget constraint penalization.
        """

        # Path to the offline RTB dataset
        self.data_path = data_path

        # Fixed total campaign budget
        self.budget = float(budget)

        # Upper bound on episode length (safety constraint)
        self.max_steps = max_steps

        # Lagrangian penalty coefficient for budget-aware reward shaping
        self.lambda_penalty = lambda_init

        # Load dataset into memory
        self._load_data()

        # Reset environment state
        self.reset()

    def _load_data(self):
        """
        Load the RTB dataset from disk.

        The dataset is expected to contain, at minimum:
        - pctr : predicted click-through rate (cached from a supervised model)
        - market_price : auction clearing price

        Data is stored as a Pandas DataFrame for sequential access.
        """
        self.df = pd.read_csv(self.data_path, sep="\t")
        self.n = len(self.df)

    def reset(self):
        """
        Reset the environment at the start of a new episode.

        This corresponds to starting a new advertising campaign
        with full budget and zero accumulated cost/clicks.
        """

        # Pointer to current impression in the dataset
        self.ptr = 0

        # Number of steps taken in the current episode
        self.steps = 0

        # Remaining campaign budget
        self.remaining_budget = float(self.budget)

        # Total cost incurred so far
        self.cost = 0.0

        # Total number of clicks obtained
        self.total_clicks = 0

        # Return the initial state
        return self._get_state()

    def _get_state(self):
        """
        Construct the current environment state vector.

        State representation (4-dimensional):
        1. budget_ratio   : remaining_budget / total_budget
        2. time_ratio     : remaining steps normalized by max_steps
        3. pctr           : predicted CTR for current impression
        4. market_price   : auction clearing price

        This state is Markovian under the assumption that:
        - pCTR is precomputed and fixed
        - market prices are revealed per impression
        """

        # If dataset is exhausted, return a zero state
        if self.ptr >= self.n:
            return np.zeros(4, dtype=np.float32)

        # Fetch current impression row
        row = self.df.iloc[self.ptr]

        # Normalize remaining budget
        budget_ratio = self.remaining_budget / self.budget

        # Normalize remaining time
        time_ratio = 1.0 - (self.steps / self.max_steps)

        # Cached predicted click-through rate
        pctr = float(row["pctr"])

        # Auction market price (second-highest bid)
        market_price = float(row["market_price"])

        return np.array(
            [budget_ratio, time_ratio, pctr, market_price],
            dtype=np.float32,
        )

    def step(self, bid):
        """
        Execute one environment step given the agent's bid.

        Parameters
        ----------
        bid : float
            Bid price submitted by the agent for the current impression.

        Returns
        -------
        next_state : np.ndarray
            Next environment state.
        reward : float
            Reward received for this step.
        done : bool
            Whether the episode has terminated.
        """

        done = False
        reward = 0.0

        # Terminate if dataset or step limit is exceeded
        if self.ptr >= self.n or self.steps >= self.max_steps:
            return self._get_state(), 0.0, True

        # Retrieve impression information
        row = self.df.iloc[self.ptr]
        market_price = float(row["market_price"])
        pctr = float(row["pctr"])

        click = 0
        cost = 0.0

        # --------------------------------------------------
        # Auction outcome (Second-Price Auction assumption)
        # --------------------------------------------------
        # Agent wins the auction if:
        # 1. Bid >= market price
        # 2. Sufficient remaining budget exists
        if bid >= market_price and self.remaining_budget >= market_price:

            # Pay the market price
            cost = market_price
            self.remaining_budget -= cost
            self.cost += cost

            # ----------------------------------------------
            # Stochastic user response
            # ----------------------------------------------
            # Click is sampled from Bernoulli(pCTR)
            # This introduces stochasticity into the reward
            click = 1 if np.random.rand() < pctr else 0
            self.total_clicks += click

            # ----------------------------------------------
            # Budget-aware reward shaping
            # ----------------------------------------------
            # Reward trades off clicks against budget usage
            reward = click - self.lambda_penalty * cost
        else:
            # No auction win → no cost, no click
            reward = 0.0

        # Advance to next impression
        self.ptr += 1
        self.steps += 1

        # Episode terminates if budget is exhausted
        if self.remaining_budget <= 0:
            done = True

        return self._get_state(), reward, done
