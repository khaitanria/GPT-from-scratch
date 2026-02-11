from __future__ import annotations

import torch
import torch.nn as nn

from gptfs.model.single_head_attention import SingleHeadAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )
        head_size = int(model_dim // num_heads)

        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(SingleHeadAttention(model_dim, head_size))

        self.compute = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        embedded: [batch, context_length, model_dim]
        returns:  [batch, context_length, model_dim]
        """
        outputs = []
        for head in self.attention_heads:
            outputs.append(head(embedded))  # each head [batch x context_length x head_size]

        concat_outputs = torch.cat(
            outputs, dim=-1
        )  # [batch x context_length x num_heads*head_size]
        return self.dropout(self.compute(concat_outputs))
