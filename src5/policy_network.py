"""
src5/policy_network.py
======================

Strategic Actor-Critic with two action heads:
  - threshold head: 51 logits (propensity threshold; pick which propensity
    threshold below which to skip bidding)
  - residual  head: 11 logits (bid multiplier residual in [-0.3, +0.3])

Shared trunk (state -> hidden) for both heads + a single critic head.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class StrategicActorCritic(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        n_threshold: int = 51,
        n_residual: int = 11,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_threshold = n_threshold
        self.n_residual = n_residual

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.threshold_head = nn.Linear(hidden_dim, n_threshold)
        self.residual_head = nn.Linear(hidden_dim, n_residual)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.trunk(x)
        logits_th = self.threshold_head(h)
        logits_res = self.residual_head(h)
        value = self.critic_head(h)
        return logits_th, logits_res, value
