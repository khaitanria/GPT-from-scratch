from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


def generate(model: nn.Module, new_chars: int, context: torch.Tensor, context_length: int, int_to_char: Mapping[int, str]) -> str:
    """
    context: assumes batch = 1
    returns: one string of length new_chars
    """
    outputs = []

    for i in range(new_chars):

        if context.size(1) > context_length:
            context = context[:, -context_length:]

        model_pred = model(context) # [batch x context_length x vocab_size]

        last_pred = model_pred[:, -1, :] # [batch x vocab_size]
        probabilities = nn.functional.softmax(last_pred, dim=-1)

        next_token = torch.multinomial(probabilities, num_samples=1) # [batch x 1]
        context = torch.cat((context, next_token), dim=-1)

        token_id = int(next_token.item())
        outputs.append(int_to_char[token_id])
       

    return "".join(outputs)