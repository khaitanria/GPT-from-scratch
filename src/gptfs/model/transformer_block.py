from __future__ import annotations

import torch
import torch.nn as nn

from gptfs.model.multi_head_self_attention import MultiHeadSelfAttention
from gptfs.model.feed_forward_neural_network import FeedForwardNeuralNetwork


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.norm_first = nn.LayerNorm(model_dim)
        self.mult_attn = MultiHeadSelfAttention(model_dim, num_heads)
        self.norm_second = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForwardNeuralNetwork(model_dim)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        embedded: [batch, context_length, model_dim]
        returns: [batch, context_length, model_dim]
        """
        first_norm = self.norm_first(embedded)
        embedded = embedded + self.mult_attn(first_norm) # add / skip connection
        
        second_norm = self.norm_second(embedded)
        embedded = embedded + self.feed_forward(second_norm) # add / skip connection
        return embedded