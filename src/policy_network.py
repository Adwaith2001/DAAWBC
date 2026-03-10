import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    """
    Shared backbone with:
    - Actor head: policy logits
    - Critic head: state-value V(s)
    """

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)

        return logits, value
