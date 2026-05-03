"""
policy_network_v2.py
Upgraded Actor-Critic network for v3 training.

Changes from v1:
- Input dim: 4 → 10 (richer state space)
- Wider network: 128 → 256 units
- Deeper: 2 → 3 shared layers
- LayerNorm instead of BatchNorm (works with batch size 1)
- Separate deeper heads for actor and critic
"""

import torch
import torch.nn as nn


class ActorCriticNetworkV2(nn.Module):
    """
    Shared backbone with:
    - Actor head : policy logits over bid actions
    - Critic head: state value V(s)

    Improvements over v1:
    - 3 shared layers (was 2)
    - 256 hidden units (was 128)
    - LayerNorm after each shared layer (works with batch size 1)
    - Deeper actor and critic heads
    - Orthogonal weight initialization
    """

    def __init__(self, input_dim: int = 10, num_actions: int = 51):
        super().__init__()

        # --------------------------------------------------
        # SHARED BACKBONE
        # --------------------------------------------------
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # --------------------------------------------------
        # ACTOR HEAD
        # --------------------------------------------------
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

        # --------------------------------------------------
        # CRITIC HEAD
        # --------------------------------------------------
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h      = self.shared(x)
        logits = self.actor(h)
        value  = self.critic(h)

        return logits, value