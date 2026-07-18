"""
src6 — MAPPO policy networks.

Multi-Agent PPO with Centralized Training, Decentralized Execution (CTDE).
Yu et al. (2022) "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games" — NeurIPS 2022.

Architecture
------------
- Per-agent ACTOR (1 per advertiser, 5 total):
    Input  : that agent's own state (state_dim=4)
    Output : (threshold logits, residual logits)
    Same shape as src5's actor head — decentralized at execution time.

- CENTRALIZED CRITIC (1, shared across all agents):
    Input  : ALL agents' states concatenated (state_dim * n_agents = 20)
    Output : per-agent value estimate (n_agents = 5)
    Used ONLY during training. Never queried at eval/execution.

CTDE means: at training time we use the centralized critic to compute
per-agent advantages with full multi-agent context (each agent's
advantage estimate is conditioned on what the OTHER agents observed,
which empirically reduces non-stationarity and stabilizes learning).
At eval time we throw the critic away — actors run independently on
their own local observations.
"""

import torch
import torch.nn as nn


class MAPPOActor(nn.Module):
    """
    Decentralized actor. One instance per agent.

    Identical interface to src5's StrategicActorCritic actor head, so
    eval-side action selection is interchangeable. Output is two logit
    vectors (threshold over 51 bins, residual over 11 bins) which are
    sampled INDEPENDENTLY given the state — joint action log-prob is
    the sum of the two head log-probs.
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 128,
        n_threshold: int = 51,
        n_residual: int = 11,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_threshold = n_threshold
        self.n_residual = n_residual

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),  # PPO traditionally uses tanh; smoother gradients than ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.threshold_head = nn.Linear(hidden_dim, n_threshold)
        self.residual_head = nn.Linear(hidden_dim, n_residual)

        # Orthogonal init w/ small final-layer gain (PPO recipe)
        self._init_weights()

    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.41)  # sqrt(2)
                nn.init.zeros_(m.bias)
        # Small gain on policy heads — encourages near-uniform initial policy
        nn.init.orthogonal_(self.threshold_head.weight, gain=0.01)
        nn.init.zeros_(self.threshold_head.bias)
        nn.init.orthogonal_(self.residual_head.weight, gain=0.01)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, state):
        """
        state: (batch, state_dim) or (state_dim,)
        returns: (threshold_logits, residual_logits)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        z = self.trunk(state)
        return self.threshold_head(z), self.residual_head(z)


class CentralizedCritic(nn.Module):
    """
    Centralized value function V(s_1, s_2, ..., s_N).

    Input  : (batch, state_dim * n_agents) — concatenation of all agent states
    Output : (batch, n_agents) — value estimate per agent
             (cooperative-style: one critic head per agent, all conditioned
              on the global state — this is the standard MAPPO formulation)
    """

    def __init__(
        self,
        state_dim: int = 4,
        n_agents: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.global_dim = state_dim * n_agents

        self.net = nn.Sequential(
            nn.Linear(self.global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_agents),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.net[:-1]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.41)
                nn.init.zeros_(m.bias)
        # Final value head — small gain (standard PPO init)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, global_state):
        """
        global_state: (batch, state_dim * n_agents)
        returns:      (batch, n_agents) — per-agent value estimates
        """
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        return self.net(global_state)


def concat_global_state(states_per_agent, device=None):
    """
    Helper. Convert a list of per-agent state arrays into one flat
    global-state tensor for the centralized critic.

    Input : list of n_agents 1D arrays/tensors of shape (state_dim,)
    Output: 1D tensor of shape (state_dim * n_agents,)
    """
    import numpy as np
    if isinstance(states_per_agent[0], torch.Tensor):
        flat = torch.cat([s.flatten() for s in states_per_agent], dim=0)
    else:
        flat = torch.tensor(
            np.concatenate([np.asarray(s).flatten() for s in states_per_agent]),
            dtype=torch.float32,
        )
    if device is not None:
        flat = flat.to(device)
    return flat
