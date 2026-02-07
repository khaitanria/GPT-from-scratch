from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.linear_layer_first = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.linear_layer_second = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, context_length, model_dim]
        returns: [batch, context_length, model_dim]
        """
        return self.dropout(self.linear_layer_second(self.relu(self.linear_layer_first(x))))