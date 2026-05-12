"""
policy_network_mappo.py
Networks for MAPPO:
- Actor: individual policy per agent (same as v2)
- CentralizedCritic: shared critic seeing all agents' states
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MAPPOActor(nn.Module):
    """
    Individual actor for each agent.
    Input: own local state (14 features)
    Output: action logits over threshold values
    """
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

    def get_action(self, x):
        logits = self.forward(x)
        dist   = Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_action(self, x, action):
        logits  = self.forward(x)
        dist    = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized critic for MAPPO.
    Input: ALL agents' states concatenated (num_agents × state_dim)
    Output: state value V(s_global)

    Sees the full global state → handles non-stationarity
    Each agent's actor is still decentralized (own state only)
    """
    def __init__(self, global_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, global_state):
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        return self.net(global_state)
