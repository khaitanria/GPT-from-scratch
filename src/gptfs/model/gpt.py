from __future__ import annotations

import torch
import torch.nn as nn

from gptfs.model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        # embedding layers
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.positional_embeddings = nn.Embedding(context_length, model_dim)
        # transformer blocks
        self.transformers = nn.Sequential()
        for i in range(num_blocks):
            self.transformers.append(TransformerBlock(model_dim, num_heads))
        # final norm layer
        self.norm_final = nn.LayerNorm(model_dim)
        # linear layer vocabulary projection
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: [batch, context_length]
        returns: [batch, context_length, vocab_size]
        """
        cntxt_len = context.shape[1] # [batch x context_length]
        # positions on the same device as context to ensure GPU compatibility
        positions = torch.arange(cntxt_len, device=context.device)
        embeddings = self.token_embeddings(context) + self.positional_embeddings(positions) # [batch x context_length x model_dim]

        transformers_out = self.transformers(embeddings) # all transformer blocks executed sequentially

        final_norm = self.norm_final(transformers_out)
        result = self.vocab_projection(final_norm) # [batch x context_length x vocab_size]
        return result