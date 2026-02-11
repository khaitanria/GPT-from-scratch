from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from gptfs.inference import generate


class DummyAlwaysZero(nn.Module):
    def __init__(self, vocab_size: int, expected_context_length: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.expected_context_length = expected_context_length

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.size(0)
        context_length = context.size(1)

        # Ensures generate() is trimming the context before calling the model
        assert context_length <= self.expected_context_length

        # Logits that force token 0 with probability ~1.0 at every position
        model_pred = torch.full(
            (batch, context_length, self.vocab_size),
            -1e9,
            device=context.device,
        )
        model_pred[:, :, 0] = 0.0
        return model_pred


def test_generate_returns_expected_length_and_chars() -> None:
    """
    Verifies generate returns the expected length and characters.
    Uses a deterministic dummy model so the output can be asserted reliably.
    output: one string of length new_chars
    """
    torch.manual_seed(0)

    vocab_size = 5
    context_length = 4
    new_chars = 12

    model = DummyAlwaysZero(vocab_size=vocab_size, expected_context_length=context_length)
    int_to_char: Mapping[int, str] = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

    context = torch.zeros((1, 1), dtype=torch.long)
    out = generate(
        model=model,
        new_chars=new_chars,
        context=context,
        context_length=context_length,
        int_to_char=int_to_char,
    )

    assert out == "a" * new_chars


def test_generate_trims_context_to_context_length() -> None:
    """
    Verifies generate trims the rolling context so the model never receives more than
    context_length tokens.
    """
    torch.manual_seed(0)

    vocab_size = 5
    context_length = 4
    new_chars = 6

    model = DummyAlwaysZero(vocab_size=vocab_size, expected_context_length=context_length)
    int_to_char: Mapping[int, str] = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

    # Start with an intentionally long context; generate should trim before calling the model
    context = torch.zeros((1, 10), dtype=torch.long)
    out = generate(
        model=model,
        new_chars=new_chars,
        context=context,
        context_length=context_length,
        int_to_char=int_to_char,
    )

    assert out == "a" * new_chars


def test_generate_device_matches_input() -> None:
    """
    Ensures generate works on GPU if available and does not create CPU tensors accidentally.
    (Skips if CUDA is unavailable.)
    """
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)

    vocab_size = 5
    context_length = 4
    new_chars = 8

    model = DummyAlwaysZero(vocab_size=vocab_size, expected_context_length=context_length).cuda()
    int_to_char: Mapping[int, str] = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

    context = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    out = generate(
        model=model,
        new_chars=new_chars,
        context=context,
        context_length=context_length,
        int_to_char=int_to_char,
    )

    assert out == "a" * new_chars
