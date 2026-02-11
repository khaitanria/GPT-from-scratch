from __future__ import annotations

import torch

from gptfs.model import GPT


def test_gpt_shape() -> None:
    """
    Verifies GPT forward output shape:
    input:  [batch, context_length]
    output: [batch, context_length, vocab_size]
    """
    torch.manual_seed(0)

    batch = 2
    context_length = 8
    vocab_size = 50
    model_dim = 24
    num_blocks = 2
    num_heads = 3  # head_size = 8

    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    )
    model.eval()

    context = torch.randint(0, vocab_size, (batch, context_length), dtype=torch.long)
    out = model(context)

    assert out.shape == (batch, context_length, vocab_size)


def test_gpt_is_causal() -> None:
    """
    Checks the causal masking property:
    changing tokens after position t must not affect outputs at positions <= t.
    """
    torch.manual_seed(0)

    batch = 1
    context_length = 8
    vocab_size = 50
    model_dim = 24
    num_blocks = 2
    num_heads = 3

    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    )
    model.eval()

    context_a = torch.randint(0, vocab_size, (batch, context_length), dtype=torch.long)
    context_b = context_a.clone()

    # Modify the "future" portion (positions 4..7)
    context_b[:, 4:] = torch.randint(0, vocab_size, (batch, context_length - 4), dtype=torch.long)

    out_a = model(context_a)
    out_b = model(context_b)

    torch.testing.assert_close(out_a[:, :4, :], out_b[:, :4, :], rtol=0.0, atol=1e-6)


def test_gpt_device_matches_input() -> None:
    """
    Ensures the module works on GPU if available and does not create CPU tensors accidentally.
    (Skips if CUDA is unavailable.)
    """
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)

    batch = 2
    context_length = 8
    vocab_size = 50
    model_dim = 24
    num_blocks = 2
    num_heads = 3

    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    ).cuda()
    model.eval()

    context = torch.randint(0, vocab_size, (batch, context_length), dtype=torch.long, device="cuda")
    out = model(context)

    assert out.device.type == "cuda"
