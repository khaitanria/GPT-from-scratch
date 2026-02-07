from __future__ import annotations

import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.key_layer = nn.Linear(model_dim, head_size, bias=False)
        self.query_layer = nn.Linear(model_dim, head_size, bias=False)
        self.value_layer = nn.Linear(model_dim, head_size, bias=False)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        embedded: [batch, context_length, model_dim]
        returns:  [batch, context_length, head_size]
        """
        # softmax((Q@K^T) / sqrt(attention_dim)) @ V
        keys = self.key_layer(embedded) # [batch x context_length x head_size]
        queries = self.query_layer(embedded) # [batch x context_length x head_size]
        attn_dim = keys.shape[-1] # head size
        attention_scores = (queries @ torch.transpose(keys, 1, 2)) / (attn_dim ** 0.5) # [batch x context_length x context_length]

        # masking
        cntxt_len = keys.shape[1]
        # mask on the same device as embedded to ensure GPU compatibility
        mask = torch.tril(torch.ones(cntxt_len, cntxt_len, device=embedded.device)) == 0
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        softmax_result = torch.nn.functional.softmax(attention_scores, dim=-1)
        values = self.value_layer(embedded) # [batch x context_length x head_size]
        return softmax_result @ values