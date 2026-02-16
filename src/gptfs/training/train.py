from __future__ import annotations

from gptfs.training.training_config import TrainingConfig
from gptfs.training.training_result import TrainingResult


def train(config: TrainingConfig) -> TrainingResult:
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
