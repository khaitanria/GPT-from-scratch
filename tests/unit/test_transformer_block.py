from __future__ import annotations

import torch

from gptfs.model import TransformerBlock


def test_transformer_block_shape() -> None:
    """
    Verifies TransformerBlock forward output shape:
    input:  [batch, context_length, model_dim]
    output: [batch, context_length, model_dim]
    """
    torch.manual_seed(0)

    batch = 2
    context_length = 5
    model_dim = 12
    num_heads = 3  # head_size = 4

    block = TransformerBlock(model_dim=model_dim, num_heads=num_heads)
    block.eval()

    embedded = torch.randn(batch, context_length, model_dim)
    out = block(embedded)

    assert out.shape == (batch, context_length, model_dim)


def test_transformer_block_is_causal() -> None:
    """
    Checks the causal masking property:
    changing tokens after position t must not affect outputs at positions <= t.
    """
    torch.manual_seed(0)

    batch = 1
    context_length = 6
    model_dim = 12
    num_heads = 3

    block = TransformerBlock(model_dim=model_dim, num_heads=num_heads)
    block.eval()

    embedded_a = torch.randn(batch, context_length, model_dim)
    embedded_b = embedded_a.clone()

    # Modify the "future" portion (positions 4 and 5)
    embedded_b[:, 4:, :] = torch.randn(batch, 2, model_dim)

    out_a = block(embedded_a)
    out_b = block(embedded_b)

    torch.testing.assert_close(out_a[:, :4, :], out_b[:, :4, :], rtol=0.0, atol=1e-6)


def test_transformer_block_device_matches_input() -> None:
    """
    Ensures the module works on GPU if available and does not create CPU tensors accidentally.
    (Skips if CUDA is unavailable.)
    """
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)

    batch = 2
    context_length = 5
    model_dim = 12
    num_heads = 3

    block = TransformerBlock(model_dim=model_dim, num_heads=num_heads).cuda()
    block.eval()

    embedded = torch.randn(batch, context_length, model_dim, device="cuda")
    out = block(embedded)

    assert out.device.type == "cuda"
