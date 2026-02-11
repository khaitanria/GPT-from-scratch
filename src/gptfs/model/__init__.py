"""
Model components for the GPT-from-scratch project.

This package exposes the stable public model API (core modules and GPT).
"""

from __future__ import annotations

from .feed_forward_neural_network import FeedForwardNeuralNetwork
from .gpt import GPT
from .multi_head_self_attention import MultiHeadSelfAttention
from .single_head_attention import SingleHeadAttention
from .transformer_block import TransformerBlock

__all__ = [
    "SingleHeadAttention",
    "MultiHeadSelfAttention",
    "FeedForwardNeuralNetwork",
    "TransformerBlock",
    "GPT",
]
