from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """
    Immutable configuration for a training run.
    """

    raw_data_dir: Path = Path("data/raw/shakespeare")
    processed_data_dir: Path = Path("data/processed")
    runs_dir: Path = Path("artifacts/runs")
    run_name: str = "debug-run"

    device: str = "auto"
    seed: int = 1337

    context_length: int = 256
    model_dim: int = 384
    num_blocks: int = 6
    num_heads: int = 6

    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 2_000

    eval_interval: int = 200
    eval_steps: int = 20

    save_interval: int = 200
    save_step_checkpoints: bool = False

    @property
    def run_dir(self) -> Path:
        """
        Directory for this run's artifacts: artifacts/runs/<run_name>.
        """
        return self.runs_dir / self.run_name

    def to_dict(self) -> dict[str, object]:
        """
        Convert this config into a JSON-serializable dictionary.
        """
        data = asdict(self)
        data["raw_data_dir"] = str(self.raw_data_dir)
        data["processed_data_dir"] = str(self.processed_data_dir)
        data["runs_dir"] = str(self.runs_dir)
        data["run_dir"] = str(self.run_dir)
        return data
