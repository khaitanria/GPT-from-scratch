from __future__ import annotations

from .training_config import TrainingConfig


def train(config: TrainingConfig) -> None:
    """
    Training entrypoint.

    To implement:
    - data loading + tokenization
    - model creation
    - optimizer + loss
    - training loop + evaluation
    - checkpointing
    """
    raise NotImplementedError("Training loop will be implemented.")
