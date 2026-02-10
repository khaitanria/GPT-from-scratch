from __future__ import annotations

import torch

from gptfs.model import FeedForwardNeuralNetwork


def test_feed_forward_neural_network_shape() -> None:
    """
    Verifies FeedForwardNeuralNetwork forward output shape:
    input:  [batch, context_length, model_dim]
    output: [batch, context_length, model_dim]
    """
    torch.manual_seed(0)

    batch = 2
    context_length = 5
    model_dim = 12

    ff = FeedForwardNeuralNetwork(model_dim=model_dim)
    ff.eval()

    x = torch.randn(batch, context_length, model_dim)
    out = ff(x)

    assert out.shape == (batch, context_length, model_dim)


def test_feed_forward_neural_network_device_matches_input() -> None:
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

    ff = FeedForwardNeuralNetwork(model_dim=model_dim).cuda()
    ff.eval()

    x = torch.randn(batch, context_length, model_dim, device="cuda")
    out = ff(x)

    assert out.device.type == "cuda"