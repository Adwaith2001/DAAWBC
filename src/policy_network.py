import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Simple MLP policy for discrete bidding.

    Input state vector:
        [budget_ratio, time_ratio, pctr, market_price]  -> size 4

    Output:
        logits over discrete bid actions (e.g., 21 actions in BID_GRID).
    """

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [input_dim] or [batch, input_dim]
        returns: logits tensor of shape [batch, num_actions]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, input_dim]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits
