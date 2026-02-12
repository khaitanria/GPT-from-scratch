from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """
    Immutable configuration for a training run.
    """

    data_dir: Path
    run_dir: Path

    seed: int = 1337

    context_length: int = 256
    model_dim: int = 384
    num_blocks: int = 6
    num_heads: int = 6

    batch_size: int = 64
    learning_rate: float = 3e-4
    max_steps: int = 2_000
