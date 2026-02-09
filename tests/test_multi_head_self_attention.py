from __future__ import annotations

import pytest
import torch

from gptfs.model import MultiHeadSelfAttention


def test_multi_head_self_attention_shape() -> None:
    """
    Verifies MultiHeadSelfAttention forward output shape:
    input:  [batch, context_length, model_dim]
    output: [batch, context_length, model_dim]
    """
    torch.manual_seed(0)

    batch = 2
    context_length = 5
    model_dim = 12
    num_heads = 3  # head_size = 4

    attn = MultiHeadSelfAttention(model_dim=model_dim, num_heads=num_heads)
    embedded = torch.randn(batch, context_length, model_dim)

    out = attn(embedded)

    assert out.shape == (batch, context_length, model_dim)


def test_multi_head_self_attention_requires_divisible_model_dim() -> None:
    """
    MultiHeadSelfAttention should raise if model_dim is not divisible by num_heads.
    """
    model_dim = 10
    num_heads = 3

    with pytest.raises(ValueError):
        _ = MultiHeadSelfAttention(model_dim=model_dim, num_heads=num_heads)


def test_multi_head_self_attention_is_causal() -> None:
    """
    Checks the causal masking property:
    changing tokens after position t must not affect outputs at positions <= t.
    """
    torch.manual_seed(0)

    batch = 1
    context_length = 6
    model_dim = 12
    num_heads = 3

    attn = MultiHeadSelfAttention(model_dim=model_dim, num_heads=num_heads)
    attn.eval()

    embedded_a = torch.randn(batch, context_length, model_dim)
    embedded_b = embedded_a.clone()

    # Modify the "future" portion (positions 4 and 5)
    embedded_b[:, 4:, :] = torch.randn(batch, 2, model_dim)

    out_a = attn(embedded_a)
    out_b = attn(embedded_b)

    # Outputs for positions [0..3] should match closely
    torch.testing.assert_close(out_a[:, :4, :], out_b[:, :4, :], rtol=0.0, atol=1e-6)


def test_multi_head_self_attention_device_matches_input() -> None:
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

    attn = MultiHeadSelfAttention(model_dim=model_dim, num_heads=num_heads).cuda()
    embedded = torch.randn(batch, context_length, model_dim, device="cuda")

    out = attn(embedded)

    assert out.device.type == "cuda"
