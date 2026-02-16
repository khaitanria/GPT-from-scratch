from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingResult:
    """
    Summary of a completed training run.
    """

    run_dir: Path
    config_path: Path
    vocab_path: Path
    latest_checkpoint_path: Path

    steps_completed: int
    tokens_processed: int

    final_train_loss: float
    final_val_loss: float | None
