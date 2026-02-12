"""
Training components for the GPT-from-scratch project.

This package will contain training configuration, data utilities,
checkpointing, and the training entrypoint.
"""

from __future__ import annotations

from .train import train
from .training_config import TrainingConfig

__all__ = ["TrainingConfig", "train"]
