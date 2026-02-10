from __future__ import annotations

import torch

from gptfs.inference import generate
from gptfs.model import GPT


def test_generate_with_gpt_runs_end_to_end() -> None:
    """
    Verifies generate(model=GPT(...)) runs end-to-end without error
    """
    torch.manual_seed(0)

    vocab_size = 5
    context_length = 4
    model_dim = 16
    num_blocks = 1
    num_heads = 1

    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    )
    model.eval()

    int_to_char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
    context = torch.zeros((1, 1), dtype=torch.long)

    out = generate(
        model=model,
        new_chars=10,
        context=context,
        context_length=context_length,
        int_to_char=int_to_char,
    )

    assert isinstance(out, str)


def test_generate_with_gpt_output_length_equals_new_chars() -> None:
    """
    Verifies generate returns an output string of length new_chars
    """
    torch.manual_seed(0)

    vocab_size = 5
    context_length = 4
    model_dim = 16
    num_blocks = 1
    num_heads = 1

    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    )
    model.eval()

    int_to_char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
    context = torch.zeros((1, 1), dtype=torch.long)

    new_chars = 20
    out = generate(
        model=model,
        new_chars=new_chars,
        context=context,
        context_length=context_length,
        int_to_char=int_to_char,
    )

    assert len(out) == new_chars


def test_generate_with_gpt_output_characters_are_valid() -> None:
    """
    Verifies all generated characters are contained in int_to_char.values()
    """
    torch.manual_seed(0)

    vocab_size = 5
    context_length = 4
    model_dim = 16
    num_blocks = 1
    num_heads = 1

    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    )
    model.eval()

    int_to_char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
    context = torch.zeros((1, 1), dtype=torch.long)

    out = generate(
        model=model,
        new_chars=20,
        context=context,
        context_length=context_length,
        int_to_char=int_to_char,
    )

    assert set(out).issubset(set(int_to_char.values()))


def test_generate_with_gpt_trims_context_to_context_length() -> None:
    """
    Verifies generate trims the rolling context so GPT is never given more than context_length tokens.

    This is an indirect integration check: if generate fails to trim, GPT should error when the
    input length exceeds context_length due to positional embedding shape mismatch.
    """
    torch.manual_seed(0)

    vocab_size = 5
    context_length = 4
    model_dim = 16
    num_blocks = 1
    num_heads = 1

    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    )
    model.eval()

    int_to_char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

    # Intentionally longer than context_length, GPT will raise an error if generate does not trim
    context = torch.zeros((1, 10), dtype=torch.long)

    out = generate(
        model=model,
        new_chars=10,
        context=context,
        context_length=context_length,
        int_to_char=int_to_char,
    )

    assert len(out) == 10


def test_generate_with_gpt_device_agreement_cuda() -> None:
    """
    Ensures generate works on CUDA if available and does not create CPU tensors in the loop
    (Skips if CUDA is unavailable.)
    """
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)

    vocab_size = 5
    context_length = 4
    model_dim = 16
    num_blocks = 1
    num_heads = 1

    device = torch.device("cuda")

    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    ).to(device)
    model.eval()

    int_to_char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
    context = torch.zeros((1, 10), dtype=torch.long, device=device)

    out = generate(
        model=model,
        new_chars=20,
        context=context,
        context_length=context_length,
        int_to_char=int_to_char,
    )

    assert len(out) == 20
    assert set(out).issubset(set(int_to_char.values()))